import pandas as pd
from PIL import Image
import torch
from predict import load_resources, eval_transform

model, classes, dev = load_resources()
df = pd.read_csv('data/dataset_splits.csv')
test_df = df[df['split'] == 'test'] if 'split' in df.columns else df

print(f"[*] Running RGB evaluation on 10 held-out test samples...")
correct = 0
for idx, sample in test_df.sample(10, random_state=42).iterrows():
    img = Image.open(sample['filepath']).convert('RGB')
    tensor = eval_transform(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        top_prob, top_idx = torch.topk(probs, 1)

    pred_class = classes[top_idx[0].item()]
    conf = top_prob[0].item() * 100
    is_correct = (pred_class == sample['class_name'])
    correct += int(is_correct)
    match = "✅" if is_correct else "❌"
    print(f"{match} True: {sample['class_name']} | Pred: {pred_class} ({conf:.2f}%)")

print(f"\nFinal RGB Test Accuracy: {correct}/10 ({correct * 10}%)")
