from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from predict import load_resources, eval_transform

model, classes, dev = load_resources()
df = pd.read_csv('data/dataset_splits.csv')

test_df = df[df['split'] == 'test'] if 'split' in df.columns else df

# Check performance across modalities if 'modality' column is present, otherwise sample 5 test images
if 'modality' in test_df.columns:
    modalities = [m for m in ['color', 'grayscale', 'segmented'] if m in test_df['modality'].unique()]
    for mod in modalities:
        sample = test_df[test_df['modality'] == mod].iloc[0]
        img = Image.open(sample['filepath']).convert('RGB')
        tensor = eval_transform(img).unsqueeze(0).to(dev)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            top_prob, top_idx = torch.topk(probs, 1)

        pred_class = classes[top_idx[0].item()]
        conf = top_prob[0].item() * 100
        match = "✅" if pred_class == sample['class_name'] else "❌"
        print(f"{match} [{mod.upper()}] True: {sample['class_name']} | Pred: {pred_class} ({conf:.2f}%)")
else:
    for idx, sample in test_df.sample(5, random_state=42).iterrows():
        img = Image.open(sample['filepath']).convert('RGB')
        tensor = eval_transform(img).unsqueeze(0).to(dev)

        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            top_prob, top_idx = torch.topk(probs, 1)

        pred_class = classes[top_idx[0].item()]
        conf = top_prob[0].item() * 100
        match = "✅" if pred_class == sample['class_name'] else "❌"
        print(f"{match} True: {sample['class_name']} | Pred: {pred_class} ({conf:.2f}%)")
