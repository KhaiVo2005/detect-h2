import argparse, os, cv2, torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np

# Vẽ bounding boxes lên ảnh
def draw_boxes(im_bgr, boxes, labels, scores, class_names, score_thr=0.25):
    for box, lab, sc in zip(boxes, labels, scores):
        if sc < score_thr:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = class_names[lab] if 0 <= lab < len(class_names) else str(lab)
        cv2.putText(im_bgr, f"{name} {sc:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return im_bgr


def infer_one_image(img_bgr, model, processor, device, class_names, score_thr):
    # Chuyển BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Tiền xử lý
    inputs = processor(images=img_rgb, return_tensors="pt").to(device)

    # Dự đoán
    with torch.no_grad():
        outputs = model(**inputs)

    # Hậu xử lý để lấy boxes, labels, scores
    processed = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([img_rgb.shape[:2]]).to(device)
    )[0]

    boxes = processed["boxes"].cpu().numpy()
    labels = processed["labels"].cpu().numpy()
    scores = processed["scores"].cpu().numpy()

    return draw_boxes(img_bgr, boxes, labels, scores, class_names, score_thr)


def main():
    parser = argparse.ArgumentParser(description="HuggingFace DETR Inference")
    parser.add_argument("--model_dir", type=str, default="runs/detr_hf/best", help="Folder chứa model đã train (best)")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Để lấy tên lớp từ dataset")
    parser.add_argument("--source", type=str, default="0", help="0 (webcam) | path/to/image | path/to/video")
    parser.add_argument("--score_thr", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0", help="GPU id hoặc 'cpu'")
    args = parser.parse_args()

    # Thiết lập device
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"🖥 Using device: {device}")

    # Lấy class names từ dataset (đọc trực tiếp YAML)
    import yaml
    with open(args.data, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    class_names = cfg["names"] if "names" in cfg else [str(i) for i in range(cfg["nc"])]

    # Load model & processor
    print(f"📦 Loading model from {args.model_dir} ...")
    processor = DetrImageProcessor.from_pretrained(args.model_dir)
    model = DetrForObjectDetection.from_pretrained(args.model_dir).to(device)
    model.eval()
    print("✅ Model loaded successfully!")

    # Webcam
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            raise RuntimeError("❌ Cannot open webcam")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = infer_one_image(frame, model, processor, device, class_names, args.score_thr)
            cv2.imshow("DETR Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Image or Video
    else:
        ext = os.path.splitext(args.source)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv"]:
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                raise FileNotFoundError(f"❌ Cannot open video: {args.source}")

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = infer_one_image(frame, model, processor, device, class_names, args.score_thr)
                cv2.imshow("DETR Inference", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            img = cv2.imread(args.source)
            if img is None:
                raise FileNotFoundError(f"❌ Cannot read image: {args.source}")

            img = infer_one_image(img, model, processor, device, class_names, args.score_thr)
            cv2.imshow("DETR Inference", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
