from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_resources():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data" / "dataset_splits.csv"
    
    if csv_path.exists() and csv_path.stat().st_size > 0:
        df = pd.read_csv(csv_path)
        classes = sorted(df["class_name"].unique())
    else:
        raise FileNotFoundError("data/dataset_splits.csv missing or empty.")

    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    
    weights_path = base_dir / "models" / "resnet18_baseline_cpu.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, classes, device