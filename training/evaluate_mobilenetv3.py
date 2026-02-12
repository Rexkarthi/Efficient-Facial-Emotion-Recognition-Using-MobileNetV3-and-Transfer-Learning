import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from training.dataset import FER2013Dataset
from models.mobilenetv3_model import MobileNetV3FER

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset
test_dataset = FER2013Dataset("data/FER2013/images")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
model = MobileNetV3FER(num_classes=7).to(device)
model.load_state_dict(
    torch.load("training/mobilenetv3_fer.pth", map_location=device)
)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average="macro")
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)
