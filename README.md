# Detect H₂ Leak with YOLOv8

## Overview

This project aims to **detect hydrogen leaks in thermal images** using the YOLOv8 object detection framework.  
The dataset is prepared with **Roboflow** annotations and trained with **Ultralytics YOLO**.  

The workflow includes:
1. Dataset download
2. Training
3. Inference
4. Exporting the trained model for deployment

Representative figures and results are saved under `runs/detect/train/` and selected files can be copied into `assets/report_images/` for inclusion in publications.

---

## Use Case

* **Goal:** Identify potential hydrogen leak points (`point_leak`) and containers (`container`) in thermal images.
* **Dataset:** `h2detect-vrdju` (downloaded via Roboflow API)
* **Model:** YOLOv8n (custom trained)
* **Performance (sample run):**
  * mAP50 = **0.832**
  * mAP50-95 = **0.612**

---

## Project Structure

```
├── data/
│   └── data.yaml                 
├── datasets/
│   └── h2detect-vrdju/           
├── runs/
│   └── detect/train/             
│       ├── weights/best.pt       
│       ├── results.png           
│       ├── confusion_matrix.png  
│       └── ...
├── src/
│   ├── roboflow_download.py      
│   ├── train.py                  
│   ├── infer.py                  
│   └── export.py                 
├── requirements.txt              
├── .gitignore                    
└── README.md                     
```

---

## Installation & Setup

### 1. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate      # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Download dataset

```bash
python src/roboflow_download.py
```

Expected output:

```
Dataset and data.yaml have been successfully downloaded, organized, and cleaned.
```

Dataset will be saved under `datasets/h2detect-vrdju/` and config at `data/data.yaml`.

---

### Step 2 — Train the model

```bash
python src/train.py --model yolov8n.pt --data data/data.yaml --epochs 100 --batch 16 --imgsz 640 --device 0
```

* Results: `runs/detect/train/`
* Best weights: `runs/detect/train/weights/best.pt`
* Figures: `runs/detect/train/results.png`, `runs/detect/train/confusion_matrix.png`

---

### Step 3 — Inference

Run on single image:

```bash
python src/infer.py --weights runs/detect/train/weights/best.pt --source datasets/h2detect-vrdju/test/images/example.jpg --save
```

Run on folder:

```bash
python src/infer.py --weights runs/detect/train/weights/best.pt --source datasets/h2detect-vrdju/test/images --save
```

---

### Step 4 — Export model

```bash
python src/export.py --weights runs/detect/train/weights/best.pt --format onnx
```

* Output: `best.onnx` (for deployment).
* The exported model will be saved under the exports/ folder (e.g., exports/best.onnx).

---

# Results & Preparing for Paper

After training, key metrics were obtained:

| Class       | Images | Instances | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
|-------------|--------|-----------|---------------|------------|-------|----------|
| all         | 35     | 47        | 0.822         | 0.836      | 0.832 | 0.612    |
| container   | 35     | 37        | 0.955         | 0.973      | 0.993 | 0.793    |
| point_leak  | 10     | 10        | 0.689         | 0.700      | 0.671 | 0.430    |

---

## Save Training Figures

Copy results into `assets/report_images/` for later use in reports/papers:

```bash
mkdir -p assets/report_images
cp runs/detect/train/results.png assets/report_images/training_curves.png
cp runs/detect/train/confusion_matrix.png assets/report_images/confusion_matrix.png
cp runs/detect/predict/example.jpg assets/report_images/sample_inference.png
```
---

## Citation (if needed)

If you use this repository in your research, please cite YOLOv8 and Roboflow.

* Ultralytics YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* Roboflow: [https://roboflow.com/](https://roboflow.com/)

---

## License

This project is for **academic and research use** only.  
Check individual licenses for Roboflow datasets and YOLO framework.
