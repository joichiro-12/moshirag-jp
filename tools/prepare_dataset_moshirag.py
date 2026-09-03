"""工程 5 の成果物を、学習用の parquet に束ねる（MoshiRAG 版）。

KAME の tools/prepare_dataset.py は text と audio を merge するところまでで、
MoshiRAG に必要な次の 2 つを持たない。ここで足す。

1. `<ret>` の埋め込み
   論文 §4.2：lead 部分の最初のテキストトークンの**直前**のトークンを `<ret>` に置換する。
   Moshi の式 5 からそこは必ず EPAD になる。`rag_token_id` は 4
   （`llm-jp-moshi-v1` の未使用 ID。元実装 moshika-rag と同じ値）。

   **位置は工程 4 が出した ret_frame をそのまま使わない。** ret_frame は
   `int(lead_onset * 12.5) - 1` で求めた値で、実測すると EPAD に当たるのは 35.3% だけだった
   （残り 64.7% は 1 フレーム手前の PAD）。テキストストリーム上で lead 最初のテキスト
   トークンを探し、その直前に置く規則に変えると 99.5%（5,116/5,144）が EPAD に着地する。

2. 参照埋め込みの持ち回り
   ARC-Encoder の出力（4096 次元）を、`<ret>` の位置と対にして parquet に載せる。
   KAME の oracle 列と同じく、可変長を values + offsets で平坦化して持つ。

チャネル配置は元実装に合わせる（`lm.py` の num_codebooks = n_q + 1 = 17、
audio_offset = 1、needed_tokens = num_codebooks - dep_q - 1 = 8 から確定）。

    行 0      テキスト（Moshi の inner monologue。1 本のみ。ユーザ側のテキストは持たない）
    行 1-8    Moshi の音声（Mimi 8 層）。生成対象
    行 9-16   ユーザの音声（Mimi 8 層）。条件

**話者ごとに 9 行ずつ分ける形にしてはいけない。** 初版は KAME の prepare_dataset.py を
踏襲して A / B に 9 行ずつ入れたが、Moshi は 1 本の系列に両話者を載せる設計であり、
モデルの期待（17）と合わなかった。delays が 17 要素であることが設定側の根拠。

我々のデータでは channel 0（左 / S1）が人間役、channel 1（右 / S2）が Moshi 役。
工程 5 の出力では A = 人間役、B = Moshi 役に対応する。

出力列
    dialogue_id                             … 識別子
    codes, codes_shape                      … [17, T] int32
    ret_frames                              … <ret> を置いたフレーム位置 [R]
    d_lead_frames                           … 各 <ret> の d_lead（フレーム数）[R]
                                              学習時の遅延サンプリング d' に使う。
                                              d' は毎エポック引き直すので学習時に必要
    ref_offsets                             … 各参照の埋め込み開始位置 [R+1]
    ref_values                              … 埋め込みを平坦化したもの [sum(L), 4096] float16
"""
from __future__ import annotations

import argparse, os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

PAD_DEFAULT, EPAD_DEFAULT, RET_DEFAULT = 3, 0, 4


def fit_text(text_ids: np.ndarray, n_frames: int, text_padding_id: int) -> np.ndarray:
    """テキストストリームを音声のフレーム数に合わせる（切り詰め or PAD 延長）。"""
    if text_ids.shape[0] > n_frames:
        return text_ids[:n_frames]
    if text_ids.shape[0] < n_frames:
        return np.concatenate([text_ids, np.full(n_frames - text_ids.shape[0], text_padding_id)])
    return text_ids


def place_ret(text_b: np.ndarray, ret_frames, pad: int, epad: int, ret_id: int,
              search_back: int = 2, search_fwd: int = 8):
    """lead 最初のテキストトークンの直前（EPAD）を <ret> に置換する。

    戻り値は (置換後のストリーム, 実際に置いた位置のリスト, 置けなかった数)。
    置けなかったものは参照側からも落とす（対応が崩れるため）。
    """
    out = text_b.copy(); placed = []; miss = 0
    for f0 in ret_frames:
        first = None
        for f in range(max(int(f0) - search_back, 0), min(int(f0) + search_fwd, len(out))):
            if out[f] not in (pad, epad):
                first = f
                break
        if first is None or first - 1 < 0 or out[first - 1] != epad:
            miss += 1; placed.append(-1); continue
        out[first - 1] = ret_id
        placed.append(first - 1)
    return out, placed, miss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenized_text_dir", required=True)
    p.add_argument("--tokenized_audio_dir", required=True)
    p.add_argument("--ref_embed_dir", required=True)
    p.add_argument("--ret_dir", required=True, help="工程 4 の出力（d_lead を読む）")
    p.add_argument("--stems", required=True)
    p.add_argument("--output_prefix", required=True)
    p.add_argument("--text_padding_id", type=int, default=PAD_DEFAULT)
    p.add_argument("--end_of_text_padding_id", type=int, default=EPAD_DEFAULT)
    p.add_argument("--rag_token_id", type=int, default=RET_DEFAULT)
    p.add_argument("--num_examples_per_parquet", type=int, default=2000)
    a = p.parse_args()

    stems = [l.strip() for l in open(a.stems) if l.strip()]
    os.makedirs(os.path.dirname(a.output_prefix) or ".", exist_ok=True)
    nshard = (len(stems) + a.num_examples_per_parquet - 1) // a.num_examples_per_parquet
    tot_ret = tot_miss = 0

    for i in range(nshard):
        part = stems[i * a.num_examples_per_parquet:(i + 1) * a.num_examples_per_parquet]
        rows = []
        for stem in tqdm(part, desc=f"parquet {i+1}/{nshard}"):
            t = np.load(os.path.join(a.tokenized_text_dir, f"{stem}.npz"))
            au = np.load(os.path.join(a.tokenized_audio_dir, f"{stem}.npz"))
            rf = np.load(os.path.join(a.ref_embed_dir, f"{stem}.npz"))
            ev = json.load(open(os.path.join(a.ret_dir, f"{stem}.json"), encoding="utf-8"))["ret_events"]
            dlf = np.array([e["d_lead_frames"] for e in ev], np.int32)

            frames, lens, embeds = rf["ret_frames"], rf["lengths"], rf["embeds"]
            # 両話者で長さを揃える（Mimi は同じ音声長から作っているので通常一致する）
            T = min(au["A"].shape[-1], au["B"].shape[-1])

            # テキストを音声長に合わせてから <ret> を置く（順序が重要。逆にすると位置がずれる）
            tb = fit_text(t["B"], T, a.text_padding_id)
            tb, placed, miss = place_ret(tb, frames, a.text_padding_id,
                                         a.end_of_text_padding_id, a.rag_token_id)
            # 行 0 = テキスト、行 1-8 = Moshi（B）、行 9-16 = ユーザ（A）
            codes = np.concatenate([tb[None, :], au["B"][:, :T], au["A"][:, :T]], axis=0)
            assert codes.shape[0] == 17, codes.shape

            keep = [k for k, v in enumerate(placed) if v >= 0]
            tot_ret += len(keep); tot_miss += miss
            off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            vals = [embeds[off[k]:off[k + 1]] for k in keep]
            new_off = np.concatenate([[0], np.cumsum([len(v) for v in vals])]).astype(np.int32)
            new_val = (np.concatenate(vals, 0) if vals
                       else np.zeros((0, embeds.shape[1] if embeds.size else 4096), np.float16))

            rows.append({
                "dialogue_id": f"{os.path.basename(a.output_prefix)}/{stem}",
                "codes": codes.astype(np.int32).tobytes(),
                "codes_shape": list(codes.shape),
                "ret_frames": np.array([placed[k] for k in keep], np.int32).tobytes(),
                "d_lead_frames": dlf[keep].tobytes() if len(keep) else np.zeros(0, np.int32).tobytes(),
                "ref_offsets": new_off.tobytes(),
                "ref_values": new_val.astype(np.float16).tobytes(),
                "ref_dim": int(new_val.shape[1]) if new_val.size else 4096,
            })
        out = f"{a.output_prefix}-{i+1:03d}-of-{nshard:03d}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"  書き出し {out}（{len(rows)} 会話）", flush=True)
    print(f"完了 {len(stems)} 会話 / <ret> 配置 {tot_ret} 件 / 置けず {tot_miss} 件", flush=True)


if __name__ == "__main__":
    main()
