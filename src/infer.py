import argparse
from ultralytics import YOLO
import cv2

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference")
    parser.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt", help="Path to trained model weights (.pt)")
    parser.add_argument("--source", type=str, default="0", help="Source: 0=webcam, path/to/image, path/to/video, or RTSP URL")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--save", action="store_true", help="Save output to runs/predict/")
    parser.add_argument("--show", action="store_true", help="Show window with predictions")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.weights)

    # Run inference
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show
    )

    for r in results:
        print(r)
    
    for r in results:
        im_bgr = r.plot()
        
        cv2.imshow("YOLOv8 Detection", im_bgr)

        if args.source.isdigit() or args.source.endswith((".mp4", ".avi")):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
