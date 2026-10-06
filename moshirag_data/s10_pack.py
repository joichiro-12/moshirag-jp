"""テキスト・音声トークン、<ret> 位置、参照埋め込みを学習用の parquet に束ねる。

<ret>（rag_token_id 4）は lead の最初のテキストトークンの直前（EPAD）に置く。
codes は [17, T]：行 0 = Moshi のテキスト、1-8 = Moshi の音声、9-16 = ユーザの音声
（A = ユーザ、B = Moshi）。

Columns: dialogue_id, codes, codes_shape, ret_frames, d_lead_frames,
         ref_offsets [R+1], ref_values [sum(L), 3072] float16
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
    p.add_argument("--ref_dim", type=int, default=3072,
                   help="参照埋め込みの次元。参照を持たない会話の既定値にも使う")
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
                       else np.zeros((0, embeds.shape[1] if embeds.size else a.ref_dim), np.float16))

            rows.append({
                "dialogue_id": f"{os.path.basename(a.output_prefix)}/{stem}",
                "codes": codes.astype(np.int32).tobytes(),
                "codes_shape": list(codes.shape),
                "ret_frames": np.array([placed[k] for k in keep], np.int32).tobytes(),
                "d_lead_frames": dlf[keep].tobytes() if len(keep) else np.zeros(0, np.int32).tobytes(),
                "ref_offsets": new_off.tobytes(),
                "ref_values": new_val.astype(np.float16).tobytes(),
                "ref_dim": int(new_val.shape[1]) if new_val.size else a.ref_dim,
            })
        out = f"{a.output_prefix}-{i+1:03d}-of-{nshard:03d}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"  書き出し {out}（{len(rows)} 会話）", flush=True)
    print(f"完了 {len(stems)} 会話 / <ret> 配置 {tot_ret} 件 / 置けず {tot_miss} 件", flush=True)


if __name__ == "__main__":
    main()
