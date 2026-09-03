"""参照埋め込みの注入テンソルを組み立てる（工程 6 の C）。

仕様は元実装の推論コードから確定させた（2026-09-03）。

  推論（moshi-rag lm.py:748-765 apply_pending_streaming_sum_condition）
    参照は [T_ref, dim] の時系列テンソル。1 フレーム進むごとに先頭 1 本を消費し、
    残りを 1 つ削る。尽きたらゼロを足す。加算先はテキスト埋め込みと音声埋め込みの和
    （lm.py:404-410 forward_text）。

  学習（非公開のため論文 §4.2 から実装）
    <ret> の位置から検索遅延 d' だけ待って参照が届く、という推論時の状況を再現する。
        d' = U(1.0, d_lead - 1.0)   d_lead >= 2 かつ p >= 0.2 のとき
             U(0, d_lead)           それ以外            … 論文 式 3
    開始フレーム s = ret_frame + round(d' * frame_rate) とし、
    参照の第 i 本目を第 s+i フレームに置く（i = 0 .. T_ref-1）。

d' は毎回引き直す（データ拡張として働かせるため）。そのため d_lead は parquet に
持たせてある。
"""
from __future__ import annotations
import numpy as np

FRAME_RATE = 12.5
MAIN_BRANCH_MIN_D_LEAD = 2.0      # これ未満だと主分岐に入らない（論文 式 3）
FALLBACK_PROB = 0.2               # p < 0.2 で U(0, d_lead) に落とす


def sample_delay_frames(d_lead_frames: int, rng: np.random.Generator) -> int:
    """検索遅延 d' をフレーム数で引く。論文 式 3。"""
    d_lead = d_lead_frames / FRAME_RATE
    if d_lead < MAIN_BRANCH_MIN_D_LEAD or rng.random() < FALLBACK_PROB:
        d = rng.uniform(0.0, d_lead)
    else:
        d = rng.uniform(1.0, d_lead - 1.0)
    return int(round(d * FRAME_RATE))


def build_streaming_sum(ret_frames, d_lead_frames, ref_offsets, ref_values,
                        n_frames: int, rng: np.random.Generator, dim: int = 4096):
    """[n_frames, dim] の注入テンソルを作る。参照が無いフレームはゼロ。

    戻り値は (テンソル, 実際に置いた区間のリスト)。区間は検証用。
    """
    out = np.zeros((n_frames, dim), np.float32)
    spans = []
    for k, rf in enumerate(ret_frames):
        vec = ref_values[ref_offsets[k]:ref_offsets[k + 1]]
        if len(vec) == 0:
            continue
        s = int(rf) + sample_delay_frames(int(d_lead_frames[k]), rng)
        e = min(s + len(vec), n_frames)
        if s >= n_frames:
            continue                       # 会話の末尾を超える場合は置けない
        out[s:e] = vec[: e - s]
        spans.append((s, e))
    return out, spans
