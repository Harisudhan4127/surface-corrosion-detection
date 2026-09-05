#!/usr/bin/env python3
"""
scripts/export_onnx.py — Export trained model to ONNX for deployment.

Usage:
    python scripts/export_onnx.py --checkpoint models/checkpoints/best_model.pth
"""
import sys, argparse, yaml
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models.model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--output",     default="models/saved/corrosion_detector.onnx")
    p.add_argument("--opset",      type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["pretrained"] = False    # weights loaded from checkpoint

    model = build_model(cfg)
    ckpt  = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    # state_dict = ckpt.get("model_state_dict", ckpt)
    # model.load_state_dict(state_dict)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    dummy   = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"])
    out_path = str(ROOT / args.output)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, out_path,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"✓ ONNX model saved → {out_path}")

    # Quick verification
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path)
    out  = sess.run(["logits"], {"input": dummy.numpy()})
    print(f"✓ ONNX verified  output shape: {out[0].shape}")
    print(f"\nLoad in Python:\n"
          f"  import onnxruntime as ort, numpy as np\n"
          f"  sess = ort.InferenceSession('{out_path}')\n"
          f"  logits = sess.run(['logits'], {{'input': img_array}})[0]\n"
          f"  pred   = logits.argmax(axis=1)")


if __name__ == "__main__":
    main()
