"""bridge の重みノルムを初期値 / 1k / 10k で比べる。

bridge_lr=1e-4 は論文（全体 2e-6）からの意図的な逸脱。10k ステップでその影響が
どれだけ積み上がったかを見る。注入ベクトルの大きさが増えていれば、
「正しいリファレンスを入れると何も入れないより悪化する」ことの説明候補になる。
"""
import torch
from safetensors.torch import load_file

B = "/groups/gcg51557/experiments/0374_japanese_kame"
paths = [("初期値", f"{B}/arc_bridge_init.safetensors"),
         ("1k (d3b)", f"{B}/ckpt_d3b/bridge.safetensors"),
         ("10k (d4)", f"{B}/ckpt_d4/bridge.safetensors")]

ws = {}
for name, p in paths:
    ws[name] = load_file(p, device="cpu")

keys = sorted(ws["初期値"].keys())
print(f"{'層':<28}{'初期値':>12}{'1k':>12}{'10k':>12}{'10k/初期':>10}")
for k in keys:
    a, b, c = (ws[n][k].float() for n in ("初期値", "1k (d3b)", "10k (d4)"))
    print(f"{k:<28}{a.norm():12.3f}{b.norm():12.3f}{c.norm():12.3f}{(c.norm()/a.norm()):10.3f}")

# 同じ入力を通したときの出力の大きさを比べる（注入ベクトルの実際の大きさ）
torch.manual_seed(0)
x = torch.randn(64, 3072)
print(f"\n同一の入力（正規乱数 64×3072）を通したときの出力ノルム（1 本あたりの平均）")
for name, _ in paths:
    w = ws[name]
    h = x @ w["layer1.weight"].float().T
    y = h @ w["layer2.weight"].float().T
    print(f"  {name:<12}{y.norm(dim=-1).mean():.4f}  (出力次元 {y.shape[-1]})")
