"""
src/models/model.py
====================
All model architectures for NEU 6-class surface defect classification.

Supported:
  - CorrosionResNet   (ResNet-50, default ✅)
  - CorrosionResNet18 (ResNet-18, fast training)
  - CorrosionEffNet   (EfficientNet-B3, best accuracy/size tradeoff)
  - CorrosionCNN      (Custom lightweight CNN, no pretrained weights)
"""

import torch
import torch.nn as nn
from torchvision import models


# ── 1. ResNet-50  (recommended) ─────────────────────────────────────────────
class CorrosionResNet(nn.Module):
    """
    ResNet-50 backbone with a custom 3-layer head.
    Input : (B, 3, 224, 224)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.4,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Replace final FC; keep everything else
        in_feats = backbone.fc.in_features          # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        if freeze_backbone:
            self._set_backbone_grad(False)

        self.head = nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _set_backbone_grad(self, requires_grad: bool):
        for p in self.backbone.parameters():
            p.requires_grad = requires_grad

    def freeze_backbone(self):
        self._set_backbone_grad(False)

    def unfreeze_backbone(self, n_layers: int = None):
        """Unfreeze last n_layers children of backbone (None = all)."""
        children = list(self.backbone.children())
        targets = children if n_layers is None else children[-n_layers:]
        for child in targets:
            for p in child.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.head(self.backbone(x))

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 2. ResNet-18  (faster, good for quick experiments) ──────────────────────
class CorrosionResNet18(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_feats = backbone.fc.in_features           # 512
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


# ── 3. EfficientNet-B3  (best accuracy) ─────────────────────────────────────
class CorrosionEffNet(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True, dropout: float = 0.35):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b3(weights=weights)
        in_feats = net.classifier[1].in_features
        net.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_feats, num_classes),
        )
        self.model = net

    def forward(self, x):
        return self.model(x)


# ── 4. Custom lightweight CNN ────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
    def forward(self, x): return self.block(x)


class CorrosionCNN(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),   # → 112×112
            ConvBlock(32,  64),   # → 56×56
            ConvBlock(64,  128),  # → 28×28
            ConvBlock(128, 256),  # → 14×14
            ConvBlock(256, 512),  # → 7×7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


# ── Factory ─────────────────────────────────────────────────────────────────
def build_model(cfg: dict) -> nn.Module:
    arch         = cfg["model"]["architecture"]
    num_classes  = cfg["data"]["num_classes"]
    pretrained   = cfg["model"]["pretrained"]
    dropout      = cfg["model"]["dropout"]

    if arch == "resnet50":
        model = CorrosionResNet(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=(cfg["training"]["freeze_epochs"] > 0),
            dropout=dropout,
        )
    elif arch == "resnet18":
        model = CorrosionResNet18(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif arch == "efficientnet_b3":
        model = CorrosionEffNet(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    elif arch == "custom_cnn":
        model = CorrosionCNN(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model


if __name__ == "__main__":
    for name, cls in [
        ("ResNet-50",       CorrosionResNet),
        ("ResNet-18",       CorrosionResNet18),
        ("EfficientNet-B3", CorrosionEffNet),
        ("Custom CNN",      CorrosionCNN),
    ]:
        m = cls(num_classes=6, pretrained=False) if name != "Custom CNN" else cls(num_classes=6)
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        total = sum(p.numel() for p in m.parameters())
        print(f"{name:<20} output={tuple(out.shape)}  params={total:,}")
