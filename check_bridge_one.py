"""bridge 1 個の出力の大きさを測る。同一の入力を通して比較できるよう乱数種を固定する。"""
import argparse
import torch
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--bridge", required=True)
a = ap.parse_args()

w = load_file(a.bridge, device="cpu")
torch.manual_seed(0)
x = torch.randn(64, 3072)
y = (x @ w["layer1.weight"].float().T) @ w["layer2.weight"].float().T
print(f"  layer1 {w['layer1.weight'].float().norm():.3f} / "
      f"layer2 {w['layer2.weight'].float().norm():.3f} / "
      f"出力ノルム（1 本あたり平均）{y.norm(dim=-1).mean():.4f}")
