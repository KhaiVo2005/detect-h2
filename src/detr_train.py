# detr_train.py — DETR (Transformers) finetune mạnh hơn
import os, time, glob, math, argparse, warnings, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF_WARNING"] = "1"
os.environ["USE_TF"] = "0"

import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR
from detr_dataset import YoloDetectionDataset, collate_fn
from transformers import DetrImageProcessor, DetrForObjectDetection

warnings.filterwarnings("ignore")

# -------------------- Utils --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _abs_path_from_yaml(yaml_path: str, p: str) -> str:
    p = p.replace("\\", "/")
    if os.path.isabs(p):
        return p
    base = os.path.dirname(os.path.abspath(yaml_path))
    return os.path.abspath(os.path.join(base, p)).replace("\\", "/")

def _resolve_yaml_paths(src_yaml: str, dst_yaml: str) -> dict:
    with open(src_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for k in ("train", "val", "valid", "test"):
        if k in cfg and cfg[k]:
            cfg[k] = _abs_path_from_yaml(src_yaml, cfg[k])

    if "valid" in cfg and "val" not in cfg:
        cfg["val"] = cfg["valid"]

    os.makedirs(os.path.dirname(dst_yaml), exist_ok=True)
    with open(dst_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return cfg

def _count_images(img_dir: str) -> int:
    pats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    return sum(len(glob.glob(os.path.join(img_dir, p))) for p in pats)

def to_coco_annotations(images, targets):
    """targets (xyxy) -> COCO dicts cho processor HF."""
    import torch as _torch
    coco_targets = []
    for img, t in zip(images, targets):
        boxes = t["boxes"]
        labels = t["labels"]
        anns = []
        for i in range(len(labels)):
            x1, y1, x2, y2 = boxes[i].tolist()
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            anns.append({
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "category_id": int(labels[i].item()),
                "area": float(w * h),
                "iscrowd": 0
            })
        image_id = t.get("image_id", None)
        if image_id is None:
            image_id = 0
        elif _torch.is_tensor(image_id):
            image_id = int(image_id.view(-1)[0].item())
        coco_targets.append({"image_id": int(image_id), "annotations": anns})
    return coco_targets

# -------------------- Training --------------------
def main():
    parser = argparse.ArgumentParser(description="Train DETR (HuggingFace) trên YOLO-format dataset (tối ưu)")
    parser.add_argument("--data", type=str, default="data/data.yaml")
    parser.add_argument("--device", type=str, default="0", help="GPU id hoặc 'cpu'")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=0.1, help="max grad norm")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0, help="đóng băng backbone N epoch đầu")
    parser.add_argument("--outdir", type=str, default="runs/detr_hf")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default="", help="đường dẫn checkpoint để resume (thư mục đã save_pretrained)")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    # Resolve yaml
    data_yaml = args.data
    if not os.path.isabs(data_yaml):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_yaml = os.path.abspath(os.path.join(project_root, args.data))
    resolved_yaml = os.path.join(os.path.dirname(data_yaml), "data_resolved.yaml")
    cfg = _resolve_yaml_paths(data_yaml, resolved_yaml)

    # Check data
    train_dir, val_dir = cfg.get("train"), (cfg.get("val") or cfg.get("valid"))
    if not train_dir or not os.path.isdir(train_dir): raise RuntimeError(f"Train images folder not found: {train_dir}")
    if not val_dir   or not os.path.isdir(val_dir):   raise RuntimeError(f"Val images folder not found: {val_dir}")

    n_train, n_val = _count_images(train_dir), _count_images(val_dir)
    print(f"[DATA] train: {train_dir} -> {n_train} images")
    print(f"[DATA] val  : {val_dir} -> {n_val} images")

    # Datasets / Loaders
    train_ds = YoloDetectionDataset(resolved_yaml, "train", transforms=None)
    val_ds   = YoloDetectionDataset(resolved_yaml, "val",   transforms=None)
    num_classes = train_ds.num_classes
    names = train_ds.names if train_ds.names else [str(i) for i in range(num_classes)]
    print(f"[DATA] Classes ({num_classes}): {names}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.workers,
                              collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers,
                              collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

    # Model & processor
    if args.resume and os.path.isdir(args.resume):
        print(f"[LOAD] Resume từ: {args.resume}")
        processor = DetrImageProcessor.from_pretrained(args.resume)
        model = DetrForObjectDetection.from_pretrained(args.resume, ignore_mismatched_sizes=True)
        # đảm bảo num_labels đúng:
        if model.config.num_labels != num_classes:
            model.config.num_labels = num_classes
    else:
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    model.to(device)

    # Param groups: backbone lr thấp hơn
    back_params, other_params = [], []
    for n, p in model.named_parameters():
        (back_params if "backbone" in n else other_params).append(p)
    optimizer = AdamW(
        [{"params": back_params, "lr": args.lr_backbone},
         {"params": other_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # LR schedule: warmup -> cosine
    total_steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.accum)))
    warmup_steps = args.warmup_epochs * total_steps_per_epoch
    total_steps  = args.epochs * total_steps_per_epoch

    def warmup_lambda(step):
        if warmup_steps == 0: return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[max(1, warmup_steps)])

    # Freeze backbone vài epoch đầu (tuỳ chọn)
    def set_backbone_requires_grad(req: bool):
        for n, p in model.named_parameters():
            if "backbone" in n:
                p.requires_grad = req

    os.makedirs(args.outdir, exist_ok=True)
    best_val = float("inf")
    best_dir = os.path.join(args.outdir, "best")
    latest_dir = os.path.join(args.outdir, "latest")

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def run_one_epoch(loader, epoch, train=True):
        phase = "TRAIN" if train else "VAL"
        model.train(train)
        running = 0.0
        num_batches = len(loader)

        if train and args.freeze_backbone_epochs > 0:
            set_backbone_requires_grad(epoch > args.freeze_backbone_epochs)

        print(f"\n🟢 Epoch {epoch}/{args.epochs} — {phase} | batches={num_batches} | accum={args.accum}")
        optimizer.zero_grad(set_to_none=True)

        for bidx, (images, targets) in enumerate(loader):
            # Chuẩn bị input cho HF
            coco_targets = to_coco_annotations(images, targets)
            inputs = processor(images=images, annotations=coco_targets, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            labels = []
            for lab in inputs["labels"]:
                labels.append({
                    "class_labels": lab["class_labels"].to(device),
                    "boxes": lab["boxes"].to(device)
                })

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss / max(1, args.accum)

            if train:
                scaler.scale(loss).backward()
                if (bidx + 1) % args.accum == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            running += float(loss.item()) * max(1, args.accum)

            if (bidx + 1) % 20 == 0 or (bidx + 1) == num_batches:
                print(f"   ▸ {phase} batch {bidx+1}/{num_batches} | loss: {running/(bidx+1):.4f}")

        avg = running / max(1, num_batches)
        print(f"✅ Done {phase}: avg_loss={avg:.4f}")
        return avg

    print("✅ Model loaded, bắt đầu training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_one_epoch(train_loader, epoch, train=True)
        val_loss   = run_one_epoch(val_loader,   epoch, train=False)
        dt = time.time() - t0
        print(f"📈 Epoch {epoch:03d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | {dt:.1f}s")

        # Save latest mỗi epoch
        model.save_pretrained(latest_dir)
        processor.save_pretrained(latest_dir)

        # Save best theo val_loss
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"💾 Saved BEST → {best_dir} (val_loss={best_val:.4f})")

        # Save mỗi N epoch
        if args.save_every and epoch % args.save_every == 0:
            save_dir = os.path.join(args.outdir, f"epoch{epoch:03d}")
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)

    print(f"🎉 Training xong. Best val loss: {best_val:.4f} | dir: {best_dir}")

if __name__ == "__main__":
    main()
