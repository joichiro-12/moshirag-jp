"""学習対象の射影層（bridge）と、参照注入テンソルの組み立て。

## 役割の分担（論文 §4.2 より。2026-09-03 に arXiv:2604.12928v2 で確認）

    all parameters trainable except for the reference text encoder
    projected via a one-layer trainable linear layer

    凍結  ARC-Encoder 本体（embedder）      … 工程 5c で事前計算し 3072 次元を保存
    学習  射影層（bridge_module）           … ここで持つ

初版は 5c で bridge を通した 4096 次元を保存していたが、それでは学習対象が
固定値になってしまう。3072 次元を保存し、学習時にこの射影層を通す形に直した。

## 論文と実装の食い違い

論文は "one-layer trainable linear layer" と書くが、元実装の `EmbProjector` は 2 層
（3072 → 2048 → 4096、bias なし）。配布重み `kyutai/ARC4_Encoder_Llama` に
`bridge_module.layer1.weight` と `layer2.weight` の両方が入っているため、
**実装（2 層）に合わせ、同梱重みを初期値に使う**。ゼロから学習するより有利で、
かつ推論時に元実装へそのまま戻せる。

## 推論との互換

元実装の推論では、射影層は
    condition_provider.conditioners.reference_with_time.bridge_module.layer{1,2}.weight
に置かれる（`arc_encoder.py:477` の `ArcEncoderConditioner._init_modules`）。
学習後にこのキー名で書き出せば、元実装の推論経路にそのまま載る。そのため
ここでも元実装の `EmbProjector` クラスをそのまま使い、パラメータ名を変えない。
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, "/groups/gcg51557/experiments/0374_japanese_kame/moshi-rag/moshi")
from moshi.conditioners.arc_encoder import EmbProjector  # noqa: E402

FRAME_RATE = 12.5
MAIN_BRANCH_MIN_D_LEAD = 2.0      # これ未満だと主分岐に入らない（論文 式 3）
FALLBACK_PROB = 0.2               # p < 0.2 で U(0, d_lead) に落とす
IN_DIM, HIDDEN_DIM, OUT_DIM = 3072, 2048, 4096


def make_bridge(init_path: str | None = None, device="cuda", dtype=torch.float32) -> EmbProjector:
    """射影層を作る。init_path があれば ARC 同梱重みで初期化する。"""
    br = EmbProjector(in_dim=IN_DIM, out_dim=OUT_DIM, hidden_dim=HIDDEN_DIM)
    if init_path:
        from safetensors.torch import load_file
        sd = load_file(init_path, device="cpu")
        miss, unexp = br.load_state_dict(sd, strict=False)
        assert not miss and not unexp, f"bridge 初期値が噛み合わない: 未{miss} 余{unexp}"
    return br.to(device=device, dtype=dtype)


def sample_delay_frames(d_lead_frames: int, rng: np.random.Generator) -> int:
    """検索遅延 d' をフレーム数で引く（論文 式 3）。毎回引き直してデータ拡張とする。"""
    d_lead = d_lead_frames / FRAME_RATE
    if d_lead < MAIN_BRANCH_MIN_D_LEAD or rng.random() < FALLBACK_PROB:
        d = rng.uniform(0.0, d_lead)
    else:
        d = rng.uniform(1.0, d_lead - 1.0)
    return int(round(d * FRAME_RATE))


def build_streaming_sum(bridge: EmbProjector, ret_frames, d_lead_frames,
                        ref_offsets, ref_values_3072, n_frames: int,
                        rng: np.random.Generator, device="cuda", dtype=torch.bfloat16):
    """[1, n_frames, 4096] の注入テンソルを作る。

    参照が無いフレームはゼロ。**射影層を通すため勾配が流れる**（学習対象）。
    戻り値は (テンソル, 注入開始フレームのリスト)。
    """
    out = torch.zeros(1, n_frames, OUT_DIM, device=device, dtype=dtype)
    starts = []
    if len(ret_frames) == 0:
        return out, starts
    # 参照をまとめて射影層に通す（会話ごとに 1 回で済ませる）
    v = torch.from_numpy(np.ascontiguousarray(ref_values_3072)).to(device=device, dtype=bridge.layer1.weight.dtype)
    proj = bridge(v).to(dtype)                     # [sum(L), 4096]
    for k, rf in enumerate(ret_frames):
        a, b = int(ref_offsets[k]), int(ref_offsets[k + 1])
        if b <= a:
            continue
        s = int(rf) + sample_delay_frames(int(d_lead_frames[k]), rng)
        if s >= n_frames:
            continue                                # 会話の末尾を超える分は置けない
        e = min(s + (b - a), n_frames)
        out[0, s:e] = proj[a:a + (e - s)]
        starts.append(s)
    return out, sorted(starts)
