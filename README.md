# AgriVision: AI Plant Disease Detection & Quantitative Severity Suite

An end-to-end computer vision and deep learning system that diagnoses 38 distinct crop pathologies across 14 plant species from the PlantVillage dataset. The project benchmarks classical Digital Image Processing (DIP) techniques against modern Transfer Learning, provides explainability via Grad-CAM, and calculates quantitative damage severity using automated lesion thresholding.

---

## 📌 Key Architectural Highlights
- **Convolutional Neural Network Backbone**: ResNet-18 pretrained on ImageNet, adapted with a custom linear classification head for 38 disease categories.
- **DIP vs. Deep Learning Benchmark**: Direct empirical comparison showing that naive color-space segmentation drops test performance from **83.42%** down to **77.15%** due to necrotic lesion clipping.
- **Formal DIP Syllabus Integration**: Hands-on implementations of 2D Discrete Fourier Transforms (DFT), CLAHE enhancement, Laplacian spatial sharpening, Wiener & Median noise restoration, Canny/Sobel edge extraction, K-Means clustering, and lossless Run-Length Encoding (RLE).
- **Interactive UI Dashboard**: Streamlit diagnostic interface featuring real-time Grad-CAM saliency heatmaps and an automated infected surface area calculator.

---

## 📊 Empirical Evaluation & Benchmarks

All models were evaluated on an identical 10% stratified holdout split under CPU constraints:

| Pipeline | Features Fed to Model | Val Accuracy | Test Accuracy | Observations |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline CNN** | Raw RGB Images (128x128) | **81.82%** | **83.42%** | Preserves context, serrated edges, and brown necrotic lesions. |
| **DIP + CNN** | HSV Segmented Leaves | **80.59%** | **77.15%** | Fails on necrotic diseases (*Late Blight*, *Black Rot*) due to color-clipping. |

---

## 🛠️ Project Directory Structure

```text
plantdiseaseprediction/
│
├── data/
│   ├── dataset_splits.csv              # 70/15/15 stratified leak-free splits index
│   └── PlantVillage-Dataset/           # Raw plant foliage images (38 categories)
│
├── models/
│   └── resnet18_baseline_cpu.pth       # Saved model checkpoint
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb       # Imbalance checks, distribution, split generation
│   ├── 02_dip_experiments.ipynb       # Color models (HSV/LAB), Otsu thresholding
│   ├── 03_cnn_baseline.ipynb          # ResNet-18 CPU training (83.42% test acc)
│   ├── 04_dip_cnn_comparison.ipynb     # DIP-segmented training, failure modes, confusion matrix
│   └── 05_advanced_dip_curriculum.ipynb # DFT, CLAHE, Wiener, Canny, K-Means, RLE
│
├── app.py                              # Streamlit interactive diagnostics dashboard
├── predict.py                          # Headless CLI inference tool
├── requirements.txt                    # Project dependencies
└── README.md


🚀 Installation & Quickstart
1. Clone & Set Up Virtual Environment
Bash
git clone [https://github.com/](https://github.com/)<your-username>/plantdiseaseprediction.git
cd plantdiseaseprediction
python -m venv venv
venv\Scripts\activate      # On Windows
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Run Web Dashboard
Bash
streamlit run app.py
4. Run CLI Prediction
Bash
python predict.py --image "data/sample_leaf.JPG"