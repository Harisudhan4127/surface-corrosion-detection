"""tests/test_neu_pipeline.py"""
import sys, pytest, torch, numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.model import CorrosionResNet, CorrosionCNN, CorrosionEffNet, build_model
from data.transforms import get_train_transform, get_val_transform
from utils.metrics import MetricsTracker

NUM_CLASSES = 6
DUMMY_BATCH = torch.randn(4, 3, 224, 224)
DUMMY_LABELS = torch.tensor([0, 2, 4, 1])


# ── Model output shapes ────────────────────────────────────────────────────
class TestModelShapes:
    def test_resnet50(self):
        m = CorrosionResNet(num_classes=6, pretrained=False)
        assert m(DUMMY_BATCH).shape == (4, 6)

    def test_custom_cnn(self):
        m = CorrosionCNN(num_classes=6)
        assert m(DUMMY_BATCH).shape == (4, 6)

    def test_factory_resnet50(self):
        cfg = {"model": {"architecture": "resnet50", "pretrained": False, "dropout": 0.4,
                         "freeze_epochs": 0},
               "data": {"num_classes": 6}, "training": {"freeze_epochs": 0}}
        m = build_model(cfg)
        assert m(DUMMY_BATCH).shape == (4, 6)


# ── Gradient flow ──────────────────────────────────────────────────────────
class TestGradients:
    def test_resnet_gradients(self):
        m = CorrosionResNet(num_classes=6, pretrained=False)
        loss = torch.nn.CrossEntropyLoss()(m(DUMMY_BATCH), DUMMY_LABELS)
        loss.backward()
        for name, p in m.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad: {name}"

    def test_cnn_gradients(self):
        m = CorrosionCNN(num_classes=6)
        loss = torch.nn.CrossEntropyLoss()(m(DUMMY_BATCH), DUMMY_LABELS)
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No grad: {name}"


# ── Freeze / unfreeze ──────────────────────────────────────────────────────
class TestFreezeUnfreeze:
    def test_freeze(self):
        m = CorrosionResNet(num_classes=6, pretrained=False, freeze_backbone=True)
        frozen = all(not p.requires_grad for p in m.backbone.parameters())
        assert frozen

    def test_unfreeze_all(self):
        m = CorrosionResNet(num_classes=6, pretrained=False, freeze_backbone=True)
        m.unfreeze_backbone()
        trainable = any(p.requires_grad for p in m.backbone.parameters())
        assert trainable


# ── Dataset ───────────────────────────────────────────────────────────────
class TestDataset:
    def test_val_transform_output(self):
        tf  = get_val_transform(224)
        img = Image.fromarray(np.random.randint(0,255,(200,200,3), dtype=np.uint8))
        t   = tf(img)
        assert t.shape == (3, 224, 224)

    def test_train_transform_output(self):
        tf  = get_train_transform(224)
        img = Image.fromarray(np.random.randint(0,255,(200,200,3), dtype=np.uint8))
        t   = tf(img)
        assert t.shape == (3, 224, 224)


# ── Metrics ───────────────────────────────────────────────────────────────
class TestMetrics:
    def test_perfect_accuracy(self):
        tracker = MetricsTracker()
        logits  = torch.eye(6).repeat(3, 1)     # perfect predictions
        labels  = torch.arange(6).repeat(3)
        tracker.update(logits, labels)
        m = tracker.compute()
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0)

    def test_auc_range(self):
        tracker = MetricsTracker()
        tracker.update(torch.randn(30, 6), torch.randint(0, 6, (30,)))
        m = tracker.compute()
        assert 0.0 <= m["macro_auc"] <= 1.0 or np.isnan(m["macro_auc"])


# ── Integration: single batch train step ────────────────────────────────
class TestIntegration:
    def test_train_step(self):
        m   = CorrosionCNN(num_classes=6)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        crit= torch.nn.CrossEntropyLoss()
        m.train()
        opt.zero_grad()
        loss = crit(m(DUMMY_BATCH), DUMMY_LABELS)
        loss.backward()
        opt.step()
        assert loss.item() > 0
        assert not torch.isnan(loss)
