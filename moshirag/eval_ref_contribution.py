"""参照埋め込みの寄与を body フレームで測る（3 条件の比較）。

設計は dev/llm-jp/260906_参照寄与の測定/実験設計.md に実行前に確定させてある。

条件
  correct  その <ret> に対応する参照
  none     注入しない（ゼロ）
  wrong    別の会話の参照（長さを合わせる）

body フレームのテキスト損失だけを測る。会話全体で測ると body が全体の 10.5% しか
ないため、参照の効果が会話ごとの難易度差に埋もれる。

判定は条件間の差が会話ごとの標準誤差の 2 倍以上あるかで行う（目視で決めない）。
"""
from __future__ import annotations
import argparse, glob, json, sys
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
FR = 12.5


def body_frames(stem: str, n_frames: int) -> np.ndarray:
    """body ターンが占めるフレームの真偽配列を返す。"""
    ev = json.load(open(f"{B}/pilot5k_ret/{stem}.json", encoding="utf-8"))["ret_events"]
    turns = {t["index"]: t for t in
             json.load(open(f"{B}/pilot5k_audio/{stem}.manifest.json", encoding="utf-8"))["turns"]}
    m = np.zeros(n_frames, bool)
    for e in ev:
        t = turns.get(e["body_turn_index"])
        if t is None:
            continue
        s = int(t["onset"] * FR); en = min(int((t["onset"] + t["duration"]) * FR), n_frames)
        if s < n_frames:
            m[s:en] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default=None, help="評価する会話の一覧（held-out）")
    ap.add_argument("--n_conv", type=int, default=80)
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--ckpt", default=None, help="学習後の重み。省略時は学習前")
    ap.add_argument("--bridge_ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    kw = json.load(open(f"{D}/moshi_lm_kwargs.json"))
    fuser = ConditionFuser({"sum": [], "prepend": [], "cross": [], "streaming_sum": [REF]})
    lm = LMModel(**kw, fuser=fuser, device="cuda", dtype=torch.bfloat16)
    lm.load_state_dict(load_file(a.ckpt or f"{D}/model.safetensors", device="cpu"), strict=False)
    lm.eval()
    bridge = make_bridge(a.bridge_ckpt or f"{B}/arc_bridge_init.safetensors", dtype=torch.float32)
    print(f"  重み: {'学習後 ' + a.ckpt if a.ckpt else '学習前（llm-jp-moshi-v1 + ARC 同梱の射影層）'}", flush=True)

    # <ret> を持つ会話だけを集める
    want = {l.strip() for l in open(a.stems)} if a.stems else None
    rows = []
    for f in sorted(glob.glob(f"{B}/stage6_dataset/*.parquet")):
        df = pd.read_parquet(f)
        for i in range(len(df)):
            r = df.iloc[i]
            if len(np.frombuffer(r["ret_frames"], np.int32)) == 0:
                continue
            if want is not None and r["dialogue_id"].split("/")[-1] not in want:
                continue
            rows.append(r)
            if len(rows) >= a.n_conv:
                break
        if len(rows) >= a.n_conv:
            break
    print(f"  評価会話 {len(rows)} 件（すべて <ret> を持つ）", flush=True)

    per = {"correct": [], "none": [], "wrong": []}
    nbody = 0
    for i, r in enumerate(rows):
        stem = r["dialogue_id"].split("/")[-1]
        c = np.frombuffer(r["codes"], np.int32).reshape(r["codes_shape"])
        T = min(c.shape[1], a.max_frames)
        codes = torch.from_numpy(c[:, :T].astype(np.int64)).unsqueeze(0).cuda()
        bm = torch.from_numpy(body_frames(stem, T)).cuda()
        if int(bm.sum()) == 0:
            continue
        nbody += int(bm.sum())
        rf = np.frombuffer(r["ret_frames"], np.int32)
        dl = np.frombuffer(r["d_lead_frames"], np.int32)
        off = np.frombuffer(r["ref_offsets"], np.int32)
        val = np.frombuffer(r["ref_values"], np.float16).reshape(-1, r["ref_dim"])
        # 誤った参照：別の会話から取る（長さは自前の offsets に合わせて切る）
        o = rows[(i + len(rows) // 2) % len(rows)]
        oval = np.frombuffer(o["ref_values"], np.float16).reshape(-1, o["ref_dim"])
        need = int(off[-1])
        wval = np.resize(oval, (need, val.shape[1])) if len(oval) else val

        for name, v in (("correct", val), ("wrong", wval)):
            cond, _ = build_streaming_sum(bridge, rf, dl, off, v, T, np.random.default_rng(a.seed))
            with torch.no_grad():
                out = lm(codes, condition_tensors={
                    REF: ConditionType(cond, torch.ones(1, T, dtype=torch.bool, device="cuda"))})
            per[name].append(body_loss(out, codes, bm))
        with torch.no_grad():
            out = lm(codes)
        per["none"].append(body_loss(out, codes, bm))
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(rows)}", flush=True)

    print(f"\n  body フレーム合計 {nbody}", flush=True)
    print(f"  {'条件':<10}{'損失':>9}{'標準誤差':>10}")
    for k in ("correct", "none", "wrong"):
        x = np.array(per[k]); se = x.std(ddof=1) / np.sqrt(len(x))
        print(f"  {k:<10}{x.mean():9.4f}{se:10.4f}", flush=True)
    c_, n_, w_ = (np.array(per[k]) for k in ("correct", "none", "wrong"))
    for lab, d in (("correct − none", c_ - n_), ("correct − wrong", c_ - w_)):
        se = d.std(ddof=1) / np.sqrt(len(d))
        sig = "有意（標準誤差の 2 倍以上）" if abs(d.mean()) >= 2 * se else "区別できない"
        print(f"  {lab}: {d.mean():+.4f} ± {se:.4f}  → {sig}", flush=True)


def body_loss(out, codes, bm) -> float:
    tl = out.text_logits[:, 0]
    m = out.text_mask[:, 0] & torch.isfinite(tl).all(-1) & bm[None, : tl.shape[1]]
    if int(m.sum()) == 0:
        return float("nan")
    return F.cross_entropy(tl[m].float(), codes[:, 0, : tl.shape[1]][m]).item()


if __name__ == "__main__":
    main()
