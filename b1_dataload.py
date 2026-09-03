"""B-1：parquet からバッチを作り、元実装の LMModel.forward が通るかを確認する。

参照の注入はまだ入れない。素の順伝播が通ることを先に確かめる
（混ぜると落ちたときにどちらが原因か切り分けられない）。

padding には zero_token_id を使う。元実装の forward が
    logits_mask &= codes[:, audio_offset:audio_offset+dep_q] != zero_token_id
    text_logits_mask &= codes[:, :1] != zero_token_id
として損失から除外しているため（lm.py:372-374）、この値で埋めれば padding が
自動的に損失計算から外れる。
"""
import sys, json, glob
sys.path.insert(0, "/groups/gcg51557/experiments/0374_japanese_kame/moshi-rag/moshi")
import numpy as np, pandas as pd, torch
from safetensors.torch import load_file
from moshi.models.lm import LMModel

B = "/groups/gcg51557/experiments/0374_japanese_kame"
D = f"{B}/llmjp_moshi_v1_ragfmt"


def load_model(device="cuda", dtype=torch.bfloat16):
    kw = json.load(open(f"{D}/moshi_lm_kwargs.json"))
    lm = LMModel(**kw, device=device, dtype=dtype)
    miss, unexp = lm.load_state_dict(load_file(f"{D}/model.safetensors", device="cpu"), strict=False)
    assert not miss and not unexp, f"重みが噛み合っていない: 未{len(miss)} 余{len(unexp)}"
    return lm


def make_batch(rows, zero_id: int, max_frames: int):
    """[B, 17, T] を作る。T はバッチ内最長に合わせ、足りない分は zero_token_id で埋める。"""
    arrs = []
    for r in rows:
        c = np.frombuffer(r["codes"], np.int32).reshape(r["codes_shape"])
        arrs.append(c[:, :max_frames])
    T = max(a.shape[1] for a in arrs)
    out = np.full((len(arrs), 17, T), zero_id, np.int64)
    for i, a in enumerate(arrs):
        out[i, :, : a.shape[1]] = a
    return torch.from_numpy(out)


def main():
    lm = load_model()
    print(f"  zero_token_id={lm.zero_token_id} / initial={lm.initial_token_id} / "
          f"text_initial={lm.text_initial_token_id} / ungenerated={lm.ungenerated_token_id}", flush=True)
    print(f"  num_codebooks={lm.num_codebooks} / dep_q={lm.dep_q} / context={lm.context}", flush=True)

    f = sorted(glob.glob(f"{B}/stage6_dataset/*.parquet"))[0]
    df = pd.read_parquet(f, columns=["dialogue_id", "codes", "codes_shape", "ret_frames"])
    rows = [df.iloc[i] for i in range(2)]
    codes = make_batch(rows, lm.zero_token_id, lm.context).cuda()
    print(f"  バッチ codes {tuple(codes.shape)}", flush=True)

    with torch.no_grad():
        out = lm(codes)
    print(f"  logits      {tuple(out.logits.shape)}      （期待 [B, dep_q=8, T, card=2048]）", flush=True)
    print(f"  text_logits {tuple(out.text_logits.shape)} （期待 [B, 1, T, text_card=32000]）", flush=True)
    print(f"  mask 有効率 音声 {out.mask.float().mean():.3f} / テキスト {out.text_mask.float().mean():.3f}", flush=True)

    # 損失が計算できるか（元実装の mask を使う）
    import torch.nn.functional as F
    tl = out.text_logits[:, 0]                      # [B, T, text_card]
    tgt = codes[:, 0, : tl.shape[1]]                # [B, T]
    m = out.text_mask[:, 0]
    valid = m & torch.isfinite(tl).all(-1)
    loss = F.cross_entropy(tl[valid], tgt[valid])
    print(f"  テキスト損失 = {loss.item():.4f}（有効 {int(valid.sum())} フレーム）", flush=True)
    print(f"  参考：一様分布なら ln(32000) = {np.log(32000):.4f}", flush=True)


if __name__ == "__main__":
    main()
