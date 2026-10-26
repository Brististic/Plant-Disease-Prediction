import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
import json

# 1. Dataset class using dataset_splits.csv
class PlantVillageSubsetDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        label = row["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

def train():
    split_path = Path("data/dataset_splits.csv")
    if not split_path.exists():
        raise FileNotFoundError("Run create_multimodal_splits.py first to generate data/dataset_splits.csv")

    df = pd.read_csv(split_path)
    classes = sorted(df["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["class_name"].map(class_to_idx)

    # Save class indices for inference sync
    with open("class_indices.json", "w") as f:
        json.dump(classes, f, indent=2)
    print(f"[✓] Saved class mapping ({len(classes)} classes) to class_indices.json")

    # Filter for the 10% stratified train/val splits
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    print(f"[*] Training on {len(train_df)} stratified samples (Validation: {len(val_df)}) on device: cpu")

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        PlantVillageSubsetDataset(train_df, train_transform),
        batch_size=64, # Batch size 64 speeds up CPU throughput
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        PlantVillageSubsetDataset(val_df, val_transform),
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze convolutional feature layers for rapid CPU transfer learning
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    epochs = 2
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_acc = correct / total

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_acc = v_correct / v_total
        print(f"--> Epoch {epoch+1} Complete | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%\n")

    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "models/resnet18_baseline_cpu.pth")
    print("[✓] Model saved successfully to models/resnet18_baseline_cpu.pth")

if __name__ == "__main__":
    train()