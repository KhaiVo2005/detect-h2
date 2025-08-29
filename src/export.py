import argparse
from ultralytics import YOLO
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "openvino", "engine", "coreml", "tflite", "torchscript", "pb"],
                        help="Export format")
    parser.add_argument("--outdir", type=str, default="exports", help="Folder to save exported model")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # Export
    exported_file = model.export(format=args.format)
    
    exported_path = Path(exported_file)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    target_path = outdir / exported_path.name
    shutil.move(str(exported_path), str(target_path))

    print(f"✅ Model exported to: {target_path}")

if __name__ == "__main__":
    main()
