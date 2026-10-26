import os
import shutil
import random
from pathlib import Path

# Set seeds for reproducibility
random.seed(42)

# Point to raw color directory in PlantVillage
BASE_DIR = Path("PlantVillage-Dataset/raw/color")
OUTPUT_DIR = Path("dataset_rgb")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

if not BASE_DIR.exists():
    # Alternative check if path differs
    alt_path = Path("PlantVillage/raw/color")
    if alt_path.exists():
        BASE_DIR = alt_path
    else:
        print(f"[!] Warning: Could not find {BASE_DIR}. Please verify your color folder path.")

print(f"[*] Processing RGB images from: {BASE_DIR}")

classes = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]
print(f"[*] Found {len(classes)} classes.")

for split in ["train", "val"]:
    for cls in classes:
        (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

for cls in classes:
    imgs = list((BASE_DIR / cls).glob("*.JPG")) + list((BASE_DIR / cls).glob("*.jpg")) + list((BASE_DIR / cls).glob("*.png"))
    random.shuffle(imgs)
    
    split_idx = int(len(imgs) * TRAIN_RATIO)
    train_imgs = imgs[:split_idx]
    val_imgs = imgs[split_idx:]
    
    for img in train_imgs:
        shutil.copy(img, OUTPUT_DIR / "train" / cls / img.name)
        
    for img in val_imgs:
        shutil.copy(img, OUTPUT_DIR / "val" / cls / img.name)

print(f"[?] Pure RGB train/val split created at: {OUTPUT_DIR.resolve()}")
