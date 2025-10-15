import os, glob, yaml
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from PIL import Image

def _load_yaml(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _make_abs_path(yaml_path: str, p: str) -> str:
    """If p is relative, convert to absolute relative to yaml file location."""
    p = p.replace("\\", "/")
    if os.path.isabs(p):
        return p
    base = os.path.dirname(os.path.abspath(yaml_path))
    return os.path.abspath(os.path.join(base, p))

def _yolo_txt_to_boxes(txt_path: str, img_w: int, img_h: int):
    boxes, labels = [], []
    if not os.path.exists(txt_path):
        return torch.zeros((0,4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            cx, cy, w, h = cx*img_w, cy*img_h, w*img_w, h*img_h
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w-1, x2), min(img_h-1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    if not boxes:
        return torch.zeros((0,4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

class YoloDetectionDataset(Dataset):
    def __init__(self, data_yaml="data/data.yaml", split="train", transforms=None):
        self.data_yaml = data_yaml
        cfg = _load_yaml(data_yaml)

        # Resolve image directory
        if split == "train":
            self.img_dir = cfg["train"]
        elif split in ["val", "valid"]:
            self.img_dir = cfg.get("val") or cfg.get("valid")
        elif split == "test":
            self.img_dir = cfg.get("test", None)
            if self.img_dir is None:
                raise ValueError("No 'test' path in data.yaml")
        else:
            raise ValueError("split must be one of ['train','val','test']")

        # ✅ Convert to absolute path
        self.img_dir = _make_abs_path(data_yaml, self.img_dir)

        # Names & classes
        self.names = cfg.get("names") or cfg.get("class_names") or []
        self.num_classes = len(self.names) if self.names else (cfg.get("nc") or 1)
        self.transforms = transforms

        # Load image files
        exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
        self.images = []
        for e in exts:
            self.images.extend(glob.glob(os.path.join(self.img_dir, e)))
        self.images = sorted(self.images)
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        # Labels path (../images -> ../labels)
        self.label_dir = self.img_dir.replace("/images", "/labels").replace("\\images", "\\labels")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Any, Dict[str, torch.Tensor]]:
        img_path = self.images[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.label_dir, base + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes, labels = _yolo_txt_to_boxes(lbl_path, w, h)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]) if boxes.numel() else torch.tensor([]),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64)
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)
