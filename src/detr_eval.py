# detr_eval.py — Evaluate HuggingFace DETR on YOLO-format dataset (YOLO-style outputs/plots)
# Features:
# - mAP@0.5 and mAP@0.5:0.95 (COCO style, 101-pt interpolated)
# - Per-class AP table
# - Precision/Recall/F1 vs confidence curves
# - PR curve (overall)
# - Confusion Matrix (with background row/col)
# - Debug images: GT (green) vs Pred (red), confidence shown
# - metrics.txt / metrics.json / class_ap.csv saved to outdir
#
# Compatible with training script that saves with model.save_pretrained(processor.save_pretrained)

import os
# Silence TF logs if TF present
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF_WARNING"] = "1"
os.environ["USE_TF"] = "0"

import argparse
import json
import math
import time
import glob
import shutil
from typing import List, Dict, Tuple

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from transformers import DetrForObjectDetection, DetrImageProcessor

# Our dataset/loader (from your repo)
from detr_dataset import YoloDetectionDataset, collate_fn


# ------------------------------ Utils ------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_outdir(d: str):
    os.makedirs(d, exist_ok=True)

def xyxy_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [Na,4], b:[Nb,4] in xyxy
    if a.size == 0 or b.size == 0:
        return np.zeros(a.shape[0], b.shape[0], dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)  # [Na]
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)  # [Nb]

    lt = np.maximum(a[:, None, :2], b[None, :, :2])  # [Na,Nb,2]
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])  # [Na,Nb,2]
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [Na,Nb]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou.astype(np.float32)

def voc_ap(rec, prec):
    # 101-point interpolation (COCO-like averaging on recall samples 0..1)
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    # precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    # 101-point sampling
    xs = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(xs, mrec, mpre), xs)
    return float(ap)

def match_detections_to_gts(det_boxes, det_scores, det_labels, gt_boxes, gt_labels, iou_thr):
    """
    For a given image, greedy match detections to GTs by IoU.
    Return: matches (len=Nd) -> (tp:bool, matched_gt_idx or -1)
    """
    Nd = len(det_boxes)
    Ng = len(gt_boxes)
    matches = [(-1, False)] * Nd
    if Nd == 0 or Ng == 0:
        return matches

    iou = xyxy_iou(det_boxes, gt_boxes)  # [Nd, Ng]
    gt_used = np.zeros(Ng, dtype=bool)
    order = np.argsort(-det_scores)  # high score first

    for di in order:
        cls_d = det_labels[di]
        # mask only gts of same class
        cand = np.where(gt_labels == cls_d)[0]
        if cand.size == 0:
            continue
        ious = iou[di, cand]
        best = int(np.argmax(ious))
        best_gt = cand[best]
        if iou[di, best_gt] >= iou_thr and not gt_used[best_gt]:
            gt_used[best_gt] = True
            matches[di] = (best_gt, True)
        else:
            matches[di] = (-1, False)
    return matches

def draw_sample(img_pil, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, names, save_path, conf_thr=0.25):
    im = np.array(img_pil)[:, :, ::-1].copy()  # BGR
    # GT in green
    for b, c in zip(gt_boxes, gt_labels):
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(im, (x1,y1), (x2,y2), (0,255,0), 2)
        txt = names[c] if 0 <= c < len(names) else str(c)
        cv2.putText(im, f"GT:{txt}", (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    # Pred in red
    for b, c, s in zip(pred_boxes, pred_labels, pred_scores):
        if s < conf_thr: 
            continue
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(im, (x1,y1), (x2,y2), (0,0,255), 2)
        txt = names[c] if 0 <= c < len(names) else str(c)
        cv2.putText(im, f"{txt} {s:.2f}", (x1, min(y2-5, im.shape[0]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, im)

def confusion_matrix(num_classes: int, dets, gts, iou_thr=0.5, conf_thr=0.25):
    """
    Build confusion matrix with background (last index).
    dets/gts: list of per-image dicts with 'boxes','labels','scores' (dets) and 'boxes','labels' (gts)
    Return: (C+1)x(C+1)
    """
    C = num_classes
    M = np.zeros((C+1, C+1), dtype=np.int64)  # rows=gt, cols=pred; last index = background

    for det, gt in zip(dets, gts):
        dsel = det['scores'] >= conf_thr
        db = det['boxes'][dsel]
        dl = det['labels'][dsel]
        ds = det['scores'][dsel]
        gb = gt['boxes']
        gl = gt['labels']

        # match by IoU per prediction
        if db.shape[0] == 0 and gb.shape[0] == 0:
            continue
        if db.shape[0] == 0 and gb.shape[0] > 0:
            # all gt -> FN (background prediction)
            for c in gl:
                gi = int(c)
                M[gi, C] += 1
            continue

        iou = xyxy_iou(db, gb) if gb.shape[0] else np.zeros((db.shape[0], 0), dtype=np.float32)
        matched_gt = np.zeros(gb.shape[0], dtype=bool)

        order = np.argsort(-ds)
        for i in order:
            if gb.shape[0] == 0:
                # all detections are FP -> predict background from gt pov
                pi = int(dl[i])
                M[C, min(pi, C)] += 1
                continue

            # best gt of same class
            cand = np.where(gl == dl[i])[0]
            best_gt = -1
            if cand.size:
                ious = iou[i, cand]
                j = int(np.argmax(ious))
                gj = cand[j]
                if iou[i, gj] >= iou_thr and not matched_gt[gj]:
                    best_gt = gj

            if best_gt >= 0:
                # TP
                gi = int(gl[best_gt])
                pi = int(dl[i])
                M[min(gi, C), min(pi, C)] += 1
                matched_gt[best_gt] = True
            else:
                # FP (no gt match)
                pi = int(dl[i])
                M[C, min(pi, C)] += 1

        # Remaining unmatched GTs -> FN
        for k, used in enumerate(matched_gt):
            if not used:
                gi = int(gl[k])
                M[min(gi, C), C] += 1
    return M

def save_confusion_matrix(M, names, save_png):
    import matplotlib.pyplot as plt
    C = len(names)
    fig, ax = plt.subplots(figsize=(8, 7))
    all_names = names + ["background"]
    im = ax.imshow(M, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(np.arange(C+1))
    ax.set_yticks(np.arange(C+1))
    ax.set_xticklabels(all_names, rotation=45, ha='right')
    ax.set_yticklabels(all_names)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # annotate
    for i in range(C+1):
        for j in range(C+1):
            ax.text(j, i, str(M[i, j]), ha='center', va='center', fontsize=8, color='w' if M[i, j] > M.max()/2 else 'black')
    plt.tight_layout()
    fig.savefig(save_png, dpi=200)
    plt.close(fig)


# ------------------------------ Evaluation Core ------------------------------

@torch.no_grad()
def run_eval(weights_dir: str,
             data_yaml: str,
             split: str = "val",
             device_str: str = "0",
             batch_size: int = 8,
             workers: int = 2,
             iou_main: float = 0.5,
             conf_steps: int = 101,
             cm_score_thr: float = 0.25,
             outdir: str = "runs/detr_hf/eval",
             save_imgs: int = 1,
             max_det: int = 300):

    # Device
    device = torch.device(f"cuda:{device_str}" if device_str != "cpu" and torch.cuda.is_available() else "cpu")

    # Data
    cfg = load_yaml(data_yaml)
    names = cfg.get("names") or []
    nc = len(names) if names else int(cfg.get("nc", 1))
    if split == "val":
        split_key = "val" if "val" in cfg else "valid"
    else:
        split_key = split
    print(f"[EVAL] Split = {split} | IoU(main) = {iou_main:.2f} | Outdir = {outdir}")
    print(f"[EVAL] Classes: {names if names else list(range(nc))}")

    dataset = YoloDetectionDataset(data_yaml, split=split_key, transforms=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

    # Model
    processor = DetrImageProcessor.from_pretrained(weights_dir)
    model = DetrForObjectDetection.from_pretrained(weights_dir).to(device).eval()

    make_outdir(outdir)
    img_out = os.path.join(outdir, "images")
    if save_imgs:
        if os.path.exists(img_out):
            shutil.rmtree(img_out)
        os.makedirs(img_out, exist_ok=True)

    # Collect all predictions and GTs
    all_dets = []  # each: {'boxes': Nx4, 'labels': N, 'scores': N}
    all_gts  = []  # each: {'boxes': Mx4, 'labels': M}
    sample_images = []  # PIL.Image

    t0 = time.time()
    for batch_i, (images, targets) in enumerate(loader):
        # target_sizes needed for post_process
        target_sizes = []
        for t in targets:
            # boxes in absolute px already; image sizes are from PIL Image
            # but our 'images' are PIL.Image objects (from dataset)
            # Ensure we can get width/height:
            # When DataLoader collates, images remain PIL in our collate_fn
            w, h = images[0].size if hasattr(images[0], "size") else (t["boxes"][:, [0, 2]].max().item(), t["boxes"][:, [1, 3]].max().item())
            target_sizes.append((h, w))
        # Better: read sizes from PIL directly for each image
        target_sizes = []
        for im in images:
            w, h = im.size
            target_sizes.append((h, w))
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        outputs = model(pixel_values=pixel_values)
        processed = processor.post_process_object_detection(outputs, threshold=0.0,
                                                            target_sizes=torch.tensor(target_sizes, device=outputs.logits.device))
        # For every image of batch
        for i, (im, t, p) in enumerate(zip(images, targets, processed)):
            # preds
            boxes = p["boxes"].detach().cpu().numpy()  # xyxy in px
            scores = p["scores"].detach().cpu().numpy()
            labels = p["labels"].detach().cpu().numpy().astype(int)
            if boxes.shape[0] > max_det:
                keep = np.argsort(-scores)[:max_det]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # gts
            gt_boxes = t["boxes"].detach().cpu().numpy()
            gt_labels = t["labels"].detach().cpu().numpy().astype(int)

            all_dets.append({"boxes": boxes, "scores": scores, "labels": labels})
            all_gts.append({"boxes": gt_boxes, "labels": gt_labels})
            sample_images.append(im)

        if (batch_i + 1) % 10 == 0:
            print(f"  Inferred {batch_i+1}/{len(loader)} batches...")

    dt = time.time() - t0
    print(f"[EVAL] Inference done in {dt:.1f}s over {len(dataset)} images.")

    # ---------------- AP calculation (COCO-like) ----------------
    iou_thresholds = np.arange(0.5, 0.95 + 1e-9, 0.05)
    ap_per_class_iou = np.zeros((len(iou_thresholds), nc), dtype=np.float64)

    # Flatten detections/GTs per class across dataset
    for ci in range(nc):
        # Build a big list of (img_idx, score, box) for detections of class ci
        det_list = []
        gt_count_per_img = []
        gt_by_img = []
        for img_idx, (det, gt) in enumerate(zip(all_dets, all_gts)):
            # detections
            m = (det["labels"] == ci)
            for b, s in zip(det["boxes"][m], det["scores"][m]):
                det_list.append((img_idx, float(s), b))
            # gts
            k = int((gt["labels"] == ci).sum())
            gt_count_per_img.append(k)
            gt_by_img.append(gt["boxes"][gt["labels"] == ci])

        if sum(gt_count_per_img) == 0:
            # No GTs of this class -> AP=0
            continue

        # Sort detections by score desc
        det_list.sort(key=lambda x: -x[1])
        if len(det_list) == 0:
            continue

        # Precompute IoUs image-wise for speed
        iou_cache = {}
        for img_idx in set([d[0] for d in det_list]):
            dboxes = np.stack([d[2] for d in det_list if d[0] == img_idx], axis=0) if any(d[0] == img_idx for d in det_list) else np.zeros((0,4), np.float32)
            gboxes = gt_by_img[img_idx]
            iou_cache[img_idx] = xyxy_iou(dboxes, gboxes) if gboxes.shape[0] else np.zeros((dboxes.shape[0], 0), np.float32)

        # For each IoU threshold, compute PR curve & AP
        for it, thr in enumerate(iou_thresholds):
            tp = np.zeros(len(det_list), dtype=np.float32)
            fp = np.zeros(len(det_list), dtype=np.float32)
            gt_used_flags = {img_idx: np.zeros(gt_by_img[img_idx].shape[0], dtype=bool) for img_idx in range(len(all_gts))}

            row_offsets = {}
            # Build quick index mapping to row in iou_cache per image
            for img_idx in iou_cache.keys():
                # map global det index -> row index within that image subset
                row_offsets[img_idx] = []
            # second pass to mark rows
            img_row_counter = {}
            for k, (img_idx, _, _) in enumerate(det_list):
                if img_idx not in img_row_counter:
                    img_row_counter[img_idx] = 0
                row_offsets[img_idx].append(img_row_counter[img_idx])
                img_row_counter[img_idx] += 1

            # Evaluate detections
            per_img_det_counts = {}
            for k, (img_idx, s, box) in enumerate(det_list):
                if gt_by_img[img_idx].shape[0] == 0:
                    fp[k] = 1.0
                    continue
                row_idx = row_offsets[img_idx][ per_img_det_counts.get(img_idx, 0) ]
                per_img_det_counts[img_idx] = per_img_det_counts.get(img_idx, 0) + 1
                ious = iou_cache[img_idx][row_idx]  # [Ng]
                j = int(np.argmax(ious))
                if ious[j] >= thr and not gt_used_flags[img_idx][j]:
                    tp[k] = 1.0
                    gt_used_flags[img_idx][j] = True
                else:
                    fp[k] = 1.0

            # Compute precision-recall
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / (sum(gt_count_per_img) + 1e-9)
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)
            ap = voc_ap(recall, precision)
            ap_per_class_iou[it, ci] = ap

    # Summaries
    map_50 = float(np.nanmean(ap_per_class_iou[0])) if nc > 0 else 0.0
    map_5095 = float(np.nanmean(ap_per_class_iou)) if nc > 0 else 0.0

    print("\n========== METRICS ==========")
    print(f"Classes: {names if names else list(range(nc))}")
    print(f"mAP@0.5: {map_50:.4f}")
    print(f"mAP@0.5:0.95: {map_5095:.4f}\n")

    print("Per-class AP@0.5:")
    for ci in range(nc):
        print(f"  {names[ci] if names else ci}: {ap_per_class_iou[0, ci]:.4f}")

    # Save per-class AP CSV
    class_ap_csv = os.path.join(outdir, "class_ap.csv")
    with open(class_ap_csv, "w", encoding="utf-8") as f:
        f.write("class,AP@0.5,AP@0.5:0.95\n")
        for ci in range(nc):
            f.write(f"{names[ci] if names else ci},{ap_per_class_iou[0,ci]:.6f},{float(np.mean(ap_per_class_iou[:,ci])):.6f}\n")

    # ---------------- Curves vs confidence (IoU=0.5) ----------------
    conf_grid = np.linspace(0, 1, conf_steps)
    P_curve, R_curve, F1_curve = [], [], []
    for thr in conf_grid:
        TP = FP = FN = 0
        for det, gt in zip(all_dets, all_gts):
            m = det["scores"] >= thr
            db = det["boxes"][m]
            dl = det["labels"][m]
            gb = gt["boxes"]
            gl = gt["labels"]
            if db.shape[0] == 0 and gb.shape[0] == 0:
                continue
            if db.shape[0] == 0 and gb.shape[0] > 0:
                FN += gb.shape[0]
                continue
            iou = xyxy_iou(db, gb) if gb.shape[0] else np.zeros((db.shape[0], 0), dtype=np.float32)
            gt_used = np.zeros(gb.shape[0], dtype=bool)

            order = np.argsort(-det["scores"][m])
            for i in order:
                if gb.shape[0] == 0:
                    FP += 1
                    continue
                cand = np.where(gl == dl[i])[0]
                best = -1
                if cand.size > 0:
                    ious = iou[i, cand]
                    j = int(np.argmax(ious))
                    gj = cand[j]
                    if iou[i, gj] >= 0.5 and not gt_used[gj]:
                        best = gj
                if best >= 0:
                    TP += 1
                    gt_used[best] = True
                else:
                    FP += 1
            FN += int((~gt_used).sum())
        P = TP / (TP + FP + 1e-9)
        R = TP / (TP + FN + 1e-9)
        F1 = 2 * P * R / (P + R + 1e-9)
        P_curve.append(P); R_curve.append(R); F1_curve.append(F1)

    # Save curves
    def plot_and_save(x, y, title, xlabel, ylabel, path):
        fig = plt.figure(figsize=(6,4))
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    plot_and_save(conf_grid, P_curve, "Precision vs Confidence", "Confidence", "Precision", os.path.join(outdir, "precision_curve.png"))
    plot_and_save(conf_grid, R_curve, "Recall vs Confidence", "Confidence", "Recall", os.path.join(outdir, "recall_curve.png"))
    plot_and_save(conf_grid, F1_curve, "F1 vs Confidence", "Confidence", "F1", os.path.join(outdir, "f1_curve.png"))

    # PR curve (overall @0.5 by sweeping confidence)
    fig = plt.figure(figsize=(6,5))
    plt.plot(R_curve, P_curve)
    plt.title("PR Curve (@0.5)")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "pr_curve.png"), dpi=200)
    plt.close(fig)

    # ---------------- Confusion Matrix ----------------
    M = confusion_matrix(nc, all_dets, all_gts, iou_thr=iou_main, conf_thr=cm_score_thr)
    save_confusion_matrix(M, names if names else [str(i) for i in range(nc)],
                          os.path.join(outdir, "confusion_matrix.png"))

    # ---------------- Save sample debug images ----------------
    if save_imgs:
        vis_num = min(50, len(sample_images))
        for i in range(vis_num):
            det = all_dets[i]
            gt = all_gts[i]
            save_p = os.path.join(img_out, f"{i:05d}.jpg")
            draw_sample(sample_images[i], gt["boxes"], gt["labels"],
                        det["boxes"], det["labels"], det["scores"],
                        names if names else [str(i) for i in range(nc)], save_p, conf_thr=cm_score_thr)

    # ---------------- Save metrics summary ----------------
    # Find best F1 point
    best_idx = int(np.argmax(F1_curve))
    best_conf = float(conf_grid[best_idx])
    best_f1, best_p, best_r = float(F1_curve[best_idx]), float(P_curve[best_idx]), float(R_curve[best_idx])

    summary = {
        "classes": names if names else [str(i) for i in range(nc)],
        "map_50": map_50,
        "map_50_95": map_5095,
        "ap_per_class@50": { (names[i] if names else str(i)): float(ap_per_class_iou[0, i]) for i in range(nc) },
        "best_f1": {"f1": best_f1, "precision": best_p, "recall": best_r, "confidence": best_conf},
        "iou_main": iou_main,
        "conf_matrix_threshold": cm_score_thr,
        "num_images": len(dataset),
        "runtime_sec": dt
    }
    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Classes: {summary['classes']}\n")
        f.write(f"mAP@0.5: {map_50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map_5095:.4f}\n\n")
        f.write("Per-class AP@0.5:\n")
        for k, v in summary["ap_per_class@50"].items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nBest F1 point (@0.5 sweep):\n")
        f.write(f"  F1={best_f1:.3f}, P={best_p:.3f}, R={best_r:.3f}, thr={best_conf:.3f}\n")

    print("\n========== SUMMARY ==========")
    print(f"mAP@0.5: {map_50:.4f}")
    print(f"mAP@0.5:0.95: {map_5095:.4f}")
    print("Per-class AP@0.5:")
    for ci in range(nc):
        print(f"  {names[ci] if names else ci}: {ap_per_class_iou[0, ci]:.4f}")
    print(f"Best F1: {best_f1:.3f} at conf {best_conf:.3f} (P={best_p:.3f}, R={best_r:.3f})")
    print(f"Saved results to: {outdir}")


# ------------------------------ CLI ------------------------------

def main():
    ap = argparse.ArgumentParser("Evaluate HuggingFace DETR on YOLO-format dataset")
    ap.add_argument("--weights", type=str, required=True, help="weights dir saved by save_pretrained (contains config.json, model.safetensors, preprocessor_config.json)")
    ap.add_argument("--data", type=str, default="data/data.yaml")
    ap.add_argument("--split", type=str, default="val", choices=["test", "val", "valid", "train"])
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf_steps", type=int, default=101)
    ap.add_argument("--cm_score_thr", type=float, default=0.25)
    ap.add_argument("--outdir", type=str, default="runs/detr_hf/eval")
    ap.add_argument("--save_imgs", type=int, default=1)
    ap.add_argument("--max_det", type=int, default=300)
    args = ap.parse_args()

    # Normalize split key (accept 'valid' alias)
    split = "val" if args.split == "valid" else args.split

    run_eval(
        weights_dir=args.weights,
        data_yaml=args.data,
        split=split,
        device_str=args.device,
        batch_size=args.batch,
        workers=args.workers,
        iou_main=args.iou,
        conf_steps=args.conf_steps,
        cm_score_thr=args.cm_score_thr,
        outdir=args.outdir,
        save_imgs=int(args.save_imgs) == 1,
        max_det=args.max_det
    )

if __name__ == "__main__":
    main()
