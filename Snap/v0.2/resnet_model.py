"""
resnet_model.py — Transfer learning with ResNet50 backbone for corrosion detection
"""

import torch
import torch.nn as nn
from torchvision import models


class CorrosionResNet(nn.Module):
    """
    ResNet50 with custom classification head.
    Supports fine-tuning all layers or only the head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.4,
    ):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Feature extractor (everything before the FC layer)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = backbone.fc.in_features  # 2048

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )
        self._init_head()

    # ------------------------------------------------------------------
    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

    # ------------------------------------------------------------------
    def unfreeze_backbone(self, layers: int = None):
        """
        Unfreeze backbone layers for fine-tuning.
        layers=None → unfreeze all; layers=N → unfreeze last N children.
        """
        children = list(self.backbone.children())
        if layers is None:
            targets = children
        else:
            targets = children[-layers:]
        for child in targets:
            for p in child.parameters():
                p.requires_grad = True

    def num_parameters(self, trainable_only=True):
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad if trainable_only else True)
        )


# ---------------------------------------------------------------------------
# Alternative: EfficientNet-B3
# ---------------------------------------------------------------------------
class CorrosionEfficientNet(nn.Module):
    """Lightweight EfficientNet-B3 backbone."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b3(weights=weights)

        in_features = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.net = net

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = CorrosionResNet(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    print("Output:", model(x).shape)
    print("Trainable params:", f"{model.num_parameters():,}")
