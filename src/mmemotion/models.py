from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class AudioCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        z = self.features(x).flatten(1)
        return self.classifier(z)


def build_face_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_video_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = tvm.video.R3D_18_Weights.DEFAULT if pretrained else None
    model = tvm.video.r3d_18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
