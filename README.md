# Efficient Facial Emotion Recognition Using MobileNetV3 and Transfer Learning

A lightweight deep learning framework for Facial Emotion Recognition (FER) using MobileNetV3-Small with transfer learning. The proposed system achieves high accuracy while maintaining low computational complexity, making it suitable for real-time and resource-constrained environments.

📌 Project Overview

Facial Emotion Recognition (FER) is a key task in affective computing and human-computer interaction. Traditional deep CNN models achieve strong performance but are computationally expensive.

This project implements:

- A **Baseline CNN model (trained from scratch)**
- A **MobileNetV3-Small model with transfer learning**

The goal is to compare performance and demonstrate how lightweight architectures can achieve better efficiency without sacrificing accuracy.

🚀 Key Results

| Model                  | Accuracy (%) | F1-Score |
|------------------------|--------------|----------|
| Baseline CNN           | 70.59       | 0.7102   |
| MobileNetV3-Small      | 80.78       | 0.7965   |

✔ ~10% accuracy improvement  
✔ ~95% model size reduction  
✔ Suitable for edge/mobile deployment  

🧠 Model Architecture

### 1️⃣ Baseline CNN
- Convolution + ReLU + MaxPooling layers
- Fully connected classification head
- Trained from scratch on FER2013

### 2️⃣ MobileNetV3-Small
- Pretrained on ImageNet
- Depthwise separable convolutions
- Squeeze-and-Excitation blocks
- Fine-tuned for 7 emotion classes

📊 Dataset

- **FER2013 Dataset**
- 48×48 grayscale facial images
- 7 emotion classes:
  - Anger
  - Disgust
  - Fear
  - Happiness
  - Sadness
  - Surprise
  - Neutral

Dataset source:
https://www.kaggle.com/datasets/msambare/fer2013

⚙️ Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📂 Project Structure

FER-Lightweight/
│
├── models/
│ ├── baseline_cnn.py
│ ├── mobilenetv3_model.py
│
├── training/
│ ├── dataset.py
│ ├── train_baseline.py
│ ├── train_mobilenetv3.py
│ ├── evaluate_baseline.py
│ ├── evaluate_mobilenetv3.py
│
├── preprocessing/
│ ├── fer2013_to_images.py
│
├── .gitignore
└── README.md


---

## 📈 Evaluation Metrics

- Accuracy
- F1-Score
- Confusion Matrix

---

## 🖥️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
python training/train_baseline.py
python training/train_mobilenetv3.py
python training/evaluate_mobilenetv3.py
