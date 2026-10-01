"""D-1 の検証：射影層を挟んだ注入が正しく、かつ射影層に勾配が流れるか。

確認する 3 点
  1. 因果性：注入の開始フレームより前で text_logits の差分が完全にゼロ
  2. 位置：差分が出た最初のフレームが注入開始と一致
  3. 勾配：損失を逆伝播したとき射影層（layer1 / layer2）に勾配が入る
     ここが 0 なら射影層が学習されず、5c を直した意味がなくなる
"""
import sys, json, glob
sys.path.insert(0, "/groups/gcg51557/experiments/0374_japanese_kame/moshi-rag/moshi")
sys.path.insert(0, "/groups/gcg51557/experiments/0374_japanese_kame/moshi_kame_finetune")
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from safetensors.torch import load_file
from moshi.models.lm import LMModel
from moshi.conditioners.base import ConditionFuser, ConditionType
from moshirag_bridge import make_bridge, build_streaming_sum

B = "/groups/gcg51557/experiments/0374_japanese_kame"
D = f"{B}/llmjp_moshi_v1_ragfmt"
REF = "reference_with_time"


def main():
    kw = json.load(open(f"{D}/moshi_lm_kwargs.json"))
    fuser = ConditionFuser({"sum": [], "prepend": [], "cross": [], "streaming_sum": [REF]})
    lm = LMModel(**kw, fuser=fuser, device="cuda", dtype=torch.bfloat16)
    miss, unexp = lm.load_state_dict(load_file(f"{D}/model.safetensors", device="cpu"), strict=False)
    assert not miss and not unexp
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)          # 勾配の到達を射影層だけで見るため一旦止める

    bridge = make_bridge(f"{B}/arc_bridge_init.safetensors", dtype=torch.float32)
    print(f"  射影層: layer1 {tuple(bridge.layer1.weight.shape)} / layer2 {tuple(bridge.layer2.weight.shape)}",
          flush=True)

    df = pd.read_parquet(sorted(glob.glob(f"{B}/stage6_dataset/*.parquet"))[0])
    row = next(df.iloc[i] for i in range(len(df))
               if len(np.frombuffer(df.iloc[i]["ret_frames"], np.int32)) > 0)
    c = np.frombuffer(row["codes"], np.int32).reshape(row["codes_shape"])
    T = min(c.shape[1], 600)
    codes = torch.from_numpy(c[:, :T].astype(np.int64)).unsqueeze(0).cuda()
    rf = np.frombuffer(row["ret_frames"], np.int32)
    dl = np.frombuffer(row["d_lead_frames"], np.int32)
    off = np.frombuffer(row["ref_offsets"], np.int32)
    val = np.frombuffer(row["ref_values"], np.float16).reshape(-1, row["ref_dim"])
    print(f"  {row['dialogue_id']} T={T} / <ret> {len(rf)} 件 / ref_dim={row['ref_dim']}", flush=True)
    assert row["ref_dim"] == 3072, f"ref_dim が 3072 でない: {row['ref_dim']}"

    cond, starts = build_streaming_sum(bridge, rf, dl, off, val, T, np.random.default_rng(0))
    print(f"  注入開始 {starts} / 非ゼロ {int((cond[0].abs().sum(-1) > 0).sum())} フレーム", flush=True)
    mask = torch.ones(1, T, dtype=torch.bool, device="cuda")

    # --- 1, 2：因果性と位置 ---
    with torch.no_grad():
        base = lm(codes)
        inj = lm(codes, condition_tensors={REF: ConditionType(cond.detach(), mask)})
    d = (inj.text_logits.float() - base.text_logits.float()).abs()[0, 0].max(-1).values.cpu().numpy()
    s0 = starts[0]
    print(f"  開始前（0〜{s0-1}）の差分 最大 {d[:s0].max():.6f}  ← 0 なら因果性 OK", flush=True)
    print(f"  開始後の差分 最大 {d[s0:].max():.4f}", flush=True)
    nz = np.nonzero(d > 1e-4)[0]
    print(f"  差分が出た最初のフレーム {int(nz[0]) if len(nz) else None}（期待 {s0}）", flush=True)

    # --- 3：勾配が射影層に届くか ---
    cond2, _ = build_streaming_sum(bridge, rf, dl, off, val, T, np.random.default_rng(0))
    out = lm(codes, condition_tensors={REF: ConditionType(cond2, mask)})
    tl = out.text_logits[:, 0]
    tgt = codes[:, 0, : tl.shape[1]]
    m = out.text_mask[:, 0] & torch.isfinite(tl).all(-1)
    loss = F.cross_entropy(tl[m].float(), tgt[m])
    loss.backward()
    g1 = bridge.layer1.weight.grad
    g2 = bridge.layer2.weight.grad
    print(f"  損失 {loss.item():.4f}", flush=True)
    print(f"  layer1 の勾配ノルム {0.0 if g1 is None else g1.norm().item():.6f}", flush=True)
    print(f"  layer2 の勾配ノルム {0.0 if g2 is None else g2.norm().item():.6f}", flush=True)
    assert g1 is not None and g1.norm() > 0, "射影層に勾配が届いていない"


if __name__ == "__main__":
    main()
