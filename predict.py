import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ----------------- Configuration & Preprocessing ----------------- #
SPLIT_CSV_PATH = Path("data/dataset_splits.csv")
MODEL_PATH = Path("models/resnet18_baseline_cpu.pth")

eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_resources():
    if not SPLIT_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing split CSV at {SPLIT_CSV_PATH.resolve()}")
    df = pd.read_csv(SPLIT_CSV_PATH)
    classes = sorted(df["class_name"].unique())

    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing weights file at {MODEL_PATH.resolve()}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, classes, device

def calculate_severity(pil_img):
    img_rgb = np.array(pil_img)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # Foliage isolation
    lower_foliage = np.array([15, 30, 30])
    upper_foliage = np.array([95, 255, 255])
    leaf_mask = cv2.inRange(img_hsv, lower_foliage, upper_foliage)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

    # Lesion isolation via LAB a-channel
    a_channel = img_lab[:, :, 1]
    _, lesion_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bounded_lesion = cv2.bitwise_and(lesion_mask, lesion_mask, mask=leaf_mask)

    total_leaf_pixels = np.count_nonzero(leaf_mask)
    lesion_pixels = np.count_nonzero(bounded_lesion)

    if total_leaf_pixels == 0:
        return 0.0, "Undetermined"

    infection_pct = (lesion_pixels / total_leaf_pixels) * 100.0
    if infection_pct < 3.0:
        stage = "Healthy / Negligible"
    elif infection_pct < 15.0:
        stage = "Mild Infection"
    elif infection_pct < 35.0:
        stage = "Moderate Severity"
    else:
        stage = "Severe / Advanced Necrosis"

    return round(float(infection_pct), 2), stage

def predict(image_path: str, top_k: int = 3):
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"Image not found at {img_file.resolve()}")

    model, classes, device = load_resources()
    image = Image.open(img_file).convert("RGB")

    tensor = eval_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, top_k)

    top_preds = []
    for p, idx in zip(top_probs, top_indices):
        class_str = classes[idx.item()]
        plant, pathology = class_str.split("___") if "___" in class_str else ("Unknown", class_str)
        top_preds.append({
            "raw_class": class_str,
            "crop": plant.replace("_", " "),
            "condition": pathology.replace("_", " "),
            "confidence_pct": round(float(p.item() * 100), 2)
        })

    infection_pct, stage = calculate_severity(image)

    result = {
        "file": str(img_file.name),
        "primary_prediction": top_preds[0],
        "top_k_predictions": top_preds,
        "damage_assessment": {
            "infection_surface_pct": infection_pct,
            "severity_stage": stage
        }
    }
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Diagnostic Inference for Plant Disease Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input RGB leaf image")
    parser.add_argument("--top_k", type=int, default=3, help="Top K predictions to output")
    args = parser.parse_args()

    diagnosis = predict(args.image, args.top_k)
    print(json.dumps(diagnosis, indent=2))