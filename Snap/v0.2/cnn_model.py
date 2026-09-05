"""
cnn_model.py — Custom lightweight CNN for corrosion detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → optional MaxPool"""

    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CorrosionCNN(nn.Module):
    """
    Custom CNN:
      Input  : (B, 3, 224, 224)
      Output : (B, num_classes)

    Architecture:
      Conv Block 1 : 3  → 32   → MaxPool → 112×112
      Conv Block 2 : 32 → 64   → MaxPool → 56×56
      Conv Block 3 : 64 → 128  → MaxPool → 28×28
      Conv Block 4 : 128→ 256  → MaxPool → 14×14
      Conv Block 5 : 256→ 512  → MaxPool → 7×7
      GAP → 512
      FC 512 → 256 → num_classes
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=True),
            ConvBlock(128, 256, pool=True),
            ConvBlock(256, 512, pool=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CorrosionCNN(num_classes=2)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {model.num_parameters():,}")
