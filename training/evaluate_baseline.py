import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

from training.dataset import FER2013Dataset
from models.baseline_cnn import BaselineCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset
test_dataset = FER2013Dataset("data/FER2013/images")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = BaselineCNN(num_classes=7)
model.load_state_dict(torch.load("training/baseline_cnn.pth", map_location=device))
model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred)

print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)
