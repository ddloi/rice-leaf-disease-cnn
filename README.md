# Rice Leaf Disease Detection using Neural Networks & Data Augmentation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
</p>

<p align="center">
  An academic research project applying Convolutional Neural Networks (CNN) and data augmentation techniques to detect rice leaf diseases from image data.
</p>

---

## Research Overview

Rice is one of the most important staple crops in Southeast Asia. Early detection of leaf diseases can prevent significant crop losses. This project develops a deep learning pipeline that:

1. Collects and preprocesses rice leaf image datasets
2. Applies **data augmentation** to address class imbalance
3. Trains a **CNN-based classifier** to distinguish between healthy and diseased leaves
4. Evaluates model performance with accuracy, precision, recall, and F1-score

---

## Disease Categories

| Class | Description |
|---|---|
| Healthy | Normal, disease-free rice leaves |
| Bacterial Leaf Blight | Caused by *Xanthomonas oryzae* |
| Brown Spot | Caused by *Bipolaris oryzae* |
| Leaf Smut | Caused by *Entyloma oryzae* |

---

## Methodology

### 1. Data Collection & Preprocessing
- Dataset sourced from publicly available rice disease image repositories
- Images resized to a uniform resolution (e.g., 224x224)
- Normalization applied (pixel values scaled to [0, 1])

### 2. Data Augmentation
To combat data imbalance and improve generalization:
- Random horizontal/vertical flip
- Random rotation (up to 30 degrees)
- Brightness and contrast jitter
- Zoom and crop transformations

### 3. Model Architecture
- **Base Model:** Custom CNN / Transfer Learning (e.g., MobileNetV2 / VGG16)
- **Layers:** Convolutional, MaxPooling, BatchNormalization, Dropout, Dense
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy

### 4. Evaluation Metrics
- Accuracy
- Precision / Recall / F1-Score (per class)
- Confusion Matrix
- Training & Validation loss/accuracy curves

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| TensorFlow / Keras | Deep learning framework |
| OpenCV | Image processing & augmentation |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| scikit-learn | Evaluation metrics |
| Jupyter Notebook | Development environment |

---

## How to Run

### Prerequisites

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run the Notebook

```bash
git clone https://github.com/DataZenLab/rice-leaf-disease-detection.git
cd rice-leaf-disease-detection
jupyter notebook rice_leaf_disease_detection.ipynb
```

### Dataset Setup

1. Download the dataset (link in the notebook)
2. Place images in a folder structure:
```
dataset/
├── healthy/
├── bacterial_leaf_blight/
├── brown_spot/
└── leaf_smut/
```

---

## Results

> Detailed results including accuracy graphs and confusion matrices are available inside the notebook.

---

## Academic Context

- **Course:** Introduction to Deep Learning / Computer Vision
- **Institution:** University of Information Technology (UIT), Vietnam
- **Team:** Group 8

---

## Author

**DataZenLab**
- GitHub: [@DataZenLab](https://github.com/DataZenLab)

---

## License

This project is for academic purposes. Dataset credits go to the original data providers.
