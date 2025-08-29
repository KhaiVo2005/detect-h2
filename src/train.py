import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, 'cpu' for CPU)")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Train
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )

if __name__ == "__main__":
    main()
