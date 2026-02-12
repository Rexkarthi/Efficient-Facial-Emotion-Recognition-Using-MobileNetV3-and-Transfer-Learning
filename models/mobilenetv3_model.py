import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class MobileNetV3FER(nn.Module):
    def __init__(self, num_classes=7):
        super(MobileNetV3FER, self).__init__()

        # Load pretrained MobileNetV3-Small
        self.backbone = mobilenet_v3_small(pretrained=True)

        # Replace classifier for FER (7 classes)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Convert grayscale → RGB (1 channel → 3 channels)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)
