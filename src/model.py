"""
A small CNN for 4-class image classification.

Architecture (deliberately simple — this is the point):
  Conv -> ReLU -> Pool   (block 1: learns edges/colors)
  Conv -> ReLU -> Pool   (block 2: learns shapes/textures)
  Conv -> ReLU -> Pool   (block 3: learns object-level patterns)
  Flatten -> Dropout -> Linear -> ReLU -> Linear (classifier head)

Dropout is a constructor argument so experiments can toggle it (0.0 = off).
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 input channels (RGB) -> 16 feature maps
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128x128 -> 64x64

            # Block 2: 16 -> 32 feature maps
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64 -> 32x32

            # Block 3: 32 -> 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # quick sanity check: does a fake batch flow through without shape errors?
    model = SmallCNN(num_classes=4, dropout=0.5)
    dummy_input = torch.randn(8, 3, 128, 128)  # batch of 8, RGB, 128x128
    output = model(dummy_input)
    print("Output shape:", output.shape)  # expect [8, 4]
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
