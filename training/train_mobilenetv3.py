import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from training.dataset import FER2013Dataset
from models.mobilenetv3_model import MobileNetV3FER

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset
dataset = FER2013Dataset("data/FER2013/images")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = MobileNetV3FER(num_classes=7).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "training/mobilenetv3_fer.pth")
print("MobileNetV3 model saved to training/mobilenetv3_fer.pth")
