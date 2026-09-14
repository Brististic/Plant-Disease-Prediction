from pathlib import Path
import urllib.request
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

# --- Page Config & Styling ---
st.set_page_config(page_title="AgriVision AI - Plant Pathology Hub", layout="wide")
st.title("🌿 AgriVision: AI Plant Disease & Severity Diagnostic Suite")
st.markdown("Automated classification, visual explainability, and quantitative lesion estimation.")

# --- Load Model & Indexing ---
@st.cache_resource
def load_model_and_classes():
    # Base directory relative to app.py location
    base_dir = Path(__file__).resolve().parent

    # 1. Resolve CSV split dynamically with a complete 38-class fallback
    split_csv_path = base_dir / "data" / "dataset_splits.csv"
    if split_csv_path.exists():
        df = pd.read_csv(split_csv_path)
        classes = sorted(df["class_name"].unique())
    else:
        classes = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
            'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    # 2. Check model weights relative to repository structure
    possible_paths = [
        base_dir / "models" / "resnet18_baseline_cpu.pth",
        base_dir / "resnet18_baseline_cpu.pth",
        Path("models/resnet18_baseline_cpu.pth")
    ]

    weights_path = next((p for p in possible_paths if p.exists()), None)
    loaded_ok = False

    # Optional: Auto-download weights if deploying remotely where weights were gitignored
    if weights_path is None:
        models_dir = base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        target_path = models_dir / "resnet18_baseline_cpu.pth"
        
        # Example release URL if hosted on GitHub Releases:
        # url = "https://github.com/<YOUR_USER>/<REPO>/releases/download/v1.0.0/resnet18_baseline_cpu.pth"
        # try:
        #     urllib.request.urlretrieve(url, target_path)
        #     weights_path = target_path
        # except Exception:
        #     pass

    if weights_path is not None and weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        loaded_ok = True

    model.eval()
    return model, classes, device, loaded_ok


# Call loader and notify outside the cache function
model, classes, device, is_loaded = load_model_and_classes()

if not is_loaded:
    st.warning("⚠️ Trained model weights (`resnet18_baseline_cpu.pth`) not found. Model is using untrained weights. Upload your checkpoint to the `models/` folder to run live inferences.")

# Image transforms
eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Helper: DIP Disease Severity Calculator ---
def compute_severity(pil_img):
    img_rgb = np.array(pil_img)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # 1. Isolate total leaf foliage
    lower_foliage = np.array([15, 30, 30])
    upper_foliage = np.array([95, 255, 255])
    leaf_mask = cv2.inRange(img_hsv, lower_foliage, upper_foliage)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

    # 2. Extract necrotic lesions via LAB a-channel
    a_channel = img_lab[:, :, 1]
    _, lesion_mask = cv2.threshold(a_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bounded_lesion = cv2.bitwise_and(lesion_mask, lesion_mask, mask=leaf_mask)

    total_leaf_pixels = np.count_nonzero(leaf_mask)
    lesion_pixels = np.count_nonzero(bounded_lesion)

    if total_leaf_pixels == 0:
        return 0.0, bounded_lesion, "Undetermined"

    infection_percentage = (lesion_pixels / total_leaf_pixels) * 100.0

    if infection_percentage < 3.0:
        stage = "Healthy / Negligible"
    elif infection_percentage < 15.0:
        stage = "Mild Infection"
    elif infection_percentage < 35.0:
        stage = "Moderate Severity"
    else:
        stage = "Severe / Advanced Necrosis"

    return infection_percentage, bounded_lesion, stage

# --- Helper: Grad-CAM Explainability ---
def generate_gradcam(pil_img, model, target_class_idx):
    gradients, activations = [], []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1].conv2
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)
    output = model(input_tensor)

    model.zero_grad()
    score = output[0, target_class_idx]
    score.backward()

    h1.remove()
    h2.remove()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    act = activations[0].squeeze(0)
    for i in range(act.size(0)):
        act[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(act, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    orig_np = np.array(pil_img)
    heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(orig_np, 0.6, heatmap_colored, 0.4, 0)
    return overlay

# --- UI Layout ---
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.subheader("1. Ingest Sample")
    uploaded_file = st.file_uploader("Upload Leaf RGB Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Target Image", use_container_width=True)

with col_right:
    if uploaded_file is not None:
        st.subheader("2. Diagnostic Results")
        
        # Classification
        input_tensor = eval_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_prob, top_idx = torch.topk(probs, 3)

        pred_class = classes[top_idx[0].item()]
        plant, disease = pred_class.split("___") if "___" in pred_class else ("Plant", pred_class)

        st.metric(label="Predicted Pathology", value=f"{disease.replace('_', ' ')}", delta=f"{plant.replace('_', ' ')}")
        st.write(f"**Model Confidence:** `{top_prob[0].item() * 100:.2f}%`")

        # Severity Estimation
        pct, lesion_mask, stage = compute_severity(image)
        st.divider()
        st.subheader("3. Quantitative Damage Assessment")
        st.write(f"**Infected Surface Area:** `{pct:.2f}%`")
        st.write(f"**Severity Class:** `{stage}`")
        st.progress(min(int(pct), 100))

st.divider()

if uploaded_file is not None:
    st.subheader("4. Algorithmic Explainability & Computer Vision Diagnostics")
    tab1, tab2, tab3 = st.tabs(["Grad-CAM AI Attention", "Segmented Lesions (DIP)", "Top-3 Probabilities"])

    with tab1:
        cam_overlay = generate_gradcam(image, model, top_idx[0].item())
        st.image(cam_overlay, caption="Grad-CAM Saliency Map (Highlighting AI Focus Regions)", use_container_width=True)

    with tab2:
        # Colored pseudo-colormap directly rendered via OpenCV
        colored_lesions = cv2.applyColorMap(lesion_mask, cv2.COLORMAP_HOT)
        colored_lesions = cv2.cvtColor(colored_lesions, cv2.COLOR_BGR2RGB)
        st.image(colored_lesions, caption="Otsu-Extracted Necrotic Spots (Heatmap)", use_container_width=True)

    with tab3:
        chart_data = pd.DataFrame({
            "Condition": [classes[i].replace("___", " - ") for i in top_idx.tolist()],
            "Confidence": [p.item() * 100 for p in top_prob]
        })
        st.bar_chart(chart_data.set_index("Condition"))