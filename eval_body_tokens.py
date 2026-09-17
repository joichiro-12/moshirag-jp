"""body のテキストトークンを復号して、リファレンスの内容が出力に現れるかを見る。

## なぜ必要か

これまでの測定（eval_ref_contribution.py）は body の**損失**を見ていた。損失が下がることと、
リファレンスの内容を実際に話すことは別である。ここではトークン単位の正答率と、復号した
文字列そのものを見る。

## 測り方（段 A：教師強制での読み出し）

推論では `<ret>` が出ないと検索が起動しないが、その出力確率は現状きわめて低い
（eval_ret_emission.py）。したがって自由生成を回しても検索は走らず、リファレンスの効果は
測れない。ここでは `<ret>` を教師強制で与えたうえで、body の各フレームでモデルが 1 位に
置くトークンを読み出す。

フレーム t の予測が使える文脈はフレーム < t までなので、**正解トークン自体は前置きに
含まれない**。したがって「その位置で何と言うか」の測定として成立する。

条件は 2 つ。正しいリファレンス / 別の会話のリファレンス。後者が対照であり、
両条件で正答率が変わらなければ、当たっているのはリファレンスの内容ではなく
文脈や事前知識によるものだと判断できる。

## リファレンス由来トークンの定義

正解 body のテキストトークンのうち、その断片（先頭の ▁ を除く）がリファレンス文に
現れ、**かつ lead に現れない**もの。lead は教師強制の前置きに入っているため、
そこから写せるトークンを除く。

body 全体で測ると助詞や定型表現に薄まるため、この部分集合を主指標とする。
"""
from __future__ import annotations
import argparse, glob, json, sys
sys.path.insert(0, "/groups/gcg51557/experiments/0374_japanese_kame/moshi-rag/moshi")
import numpy as np, pandas as pd, torch
import sentencepiece as spm
from safetensors.torch import load_file
from moshi.models.lm import LMModel
from moshi.conditioners.base import ConditionFuser, ConditionType
from moshirag_bridge import make_bridge, build_streaming_sum

B = "/groups/gcg51557/experiments/0374_japanese_kame"
D = f"{B}/llmjp_moshi_v1_ragfmt"
REF = "reference_with_time"
FR = 12.5
RET_ID = 4
PAD, EPAD = 3, 0
SPECIAL = {0, 1, 2, 3, 4}


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


def ref_and_lead_text(stem: str) -> tuple[str, str]:
    ev = json.load(open(f"{B}/pilot5k_ret/{stem}.json", encoding="utf-8"))["ret_events"]
    return ("".join(e.get("reference", "") for e in ev),
            "".join(e.get("lead", "") for e in ev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default=None)
    ap.add_argument("--n_conv", type=int, default=200)
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bridge_ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_show", type=int, default=5, help="復号して表示する会話数")
    a = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(f"{B}/tokenizer_spm_32k_3.model")

    kw = json.load(open(f"{D}/moshi_lm_kwargs.json"))
    fuser = ConditionFuser({"sum": [], "prepend": [], "cross": [], "streaming_sum": [REF]})
    lm = LMModel(**kw, fuser=fuser, device="cuda", dtype=torch.bfloat16)
    lm.load_state_dict(load_file(a.ckpt or f"{D}/model.safetensors", device="cpu"), strict=False)
    lm.eval()
    bridge = make_bridge(a.bridge_ckpt or f"{B}/arc_bridge_init.safetensors", dtype=torch.float32)
    bridge.eval()
    print(f"  重み: {'学習後' if a.ckpt else '学習前'}", flush=True)

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
    print(f"  評価会話 {len(rows)} 件", flush=True)

    # hit[cond] = 正解した数、tot[cond] = 対象トークン数
    hit = {"correct": 0, "wrong": 0}; tot = 0
    hit_r = {"correct": 0, "wrong": 0}; tot_r = 0
    # 会話ごとの正答率（対応のある差の標準誤差を出すため。判定基準は損失側と同じ 2×SE）
    per = {"correct": [], "wrong": []}; per_r = {"correct": [], "wrong": []}
    # McNemar 用：2 条件で判定が食い違ったトークン数
    disc = {"c_only": 0, "w_only": 0}; disc_r = {"c_only": 0, "w_only": 0}
    shown = 0
    for i, r in enumerate(rows):
        stem = r["dialogue_id"].split("/")[-1]
        c = np.frombuffer(r["codes"], np.int32).reshape(r["codes_shape"])
        T = min(c.shape[1], a.max_frames)
        codes = torch.from_numpy(c[:, :T].astype(np.int64)).unsqueeze(0).cuda()
        bm = body_frames(stem, T)
        tb = c[0, :T]
        # 対象：body フレームかつ正解が実トークン（PAD/EPAD/<ret> を除く）
        sel = np.where(bm & ~np.isin(tb, list(SPECIAL)))[0]
        if len(sel) == 0:
            continue
        ref_t, lead_t = ref_and_lead_text(stem)
        is_ref = np.array([
            (p := sp.id_to_piece(int(tb[f])).replace("▁", "")) != ""
            and p in ref_t and p not in lead_t
            for f in sel])

        rf = np.frombuffer(r["ret_frames"], np.int32)
        dl = np.frombuffer(r["d_lead_frames"], np.int32)
        off = np.frombuffer(r["ref_offsets"], np.int32)
        val = np.frombuffer(r["ref_values"], np.float16).reshape(-1, r["ref_dim"])
        o = rows[(i + len(rows) // 2) % len(rows)]
        oval = np.frombuffer(o["ref_values"], np.float16).reshape(-1, o["ref_dim"])
        wval = np.resize(oval, (int(off[-1]), val.shape[1])) if len(oval) else val

        pred = {}
        for name, v in (("correct", val), ("wrong", wval)):
            cond, _ = build_streaming_sum(bridge, rf, dl, off, v, T, np.random.default_rng(a.seed))
            with torch.no_grad():
                out = lm(codes, condition_tensors={
                    REF: ConditionType(cond, torch.ones(1, T, dtype=torch.bool, device="cuda"))})
            pred[name] = out.text_logits[0, 0].float().argmax(-1).cpu().numpy()

        gold = tb[sel]
        tot += len(sel); tot_r += int(is_ref.sum())
        okc = {}
        for name in ("correct", "wrong"):
            ok = pred[name][sel] == gold
            okc[name] = ok
            hit[name] += int(ok.sum()); hit_r[name] += int((ok & is_ref).sum())
            per[name].append(ok.mean())
            if int(is_ref.sum()):
                per_r[name].append(ok[is_ref].mean())
        disc["c_only"] += int((okc["correct"] & ~okc["wrong"]).sum())
        disc["w_only"] += int((okc["wrong"] & ~okc["correct"]).sum())
        disc_r["c_only"] += int((okc["correct"] & ~okc["wrong"] & is_ref).sum())
        disc_r["w_only"] += int((okc["wrong"] & ~okc["correct"] & is_ref).sum())

        if shown < a.n_show:
            shown += 1
            dec = lambda ids: sp.decode([int(x) for x in ids if int(x) not in SPECIAL])
            print(f"\n--- 例 {shown}（{stem}）---")
            print(f"  リファレンス : {ref_t[:110]}")
            print(f"  正解 body    : {dec(gold)}")
            print(f"  予測(正しい) : {dec(pred['correct'][sel])}")
            print(f"  予測(誤り)   : {dec(pred['wrong'][sel])}")
            print(f"  リファレンス由来トークン {int(is_ref.sum())}/{len(sel)} 件", flush=True)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(rows)}", flush=True)

    print(f"\n  body の実テキストトークン {tot} 件（うちリファレンス由来 {tot_r} 件）")
    print(f"  {'条件':<10}{'全体の正答率':>14}{'リファレンス由来':>18}")
    for name in ("correct", "wrong"):
        print(f"  {name:<10}{100*hit[name]/max(tot,1):13.2f}%{100*hit_r[name]/max(tot_r,1):17.2f}%")
    d = 100 * (hit["correct"] - hit["wrong"]) / max(tot, 1)
    dr = 100 * (hit_r["correct"] - hit_r["wrong"]) / max(tot_r, 1)
    print(f"  差（正しい − 誤り）: 全体 {d:+.2f} ポイント / リファレンス由来 {dr:+.2f} ポイント")

    # 対応のある差の検定。判定基準は損失側と同じ「会話間ばらつきの 2 倍」
    print("\n  対応のある差（会話ごとに取り、会話間の標準誤差で判定）")
    for lab, p in (("全体", per), ("リファレンス由来", per_r)):
        if not p["correct"]:
            continue
        x = 100 * (np.array(p["correct"]) - np.array(p["wrong"]))
        se = x.std(ddof=1) / np.sqrt(len(x))
        sig = "有意（2×SE 以上）" if abs(x.mean()) >= 2 * se else "区別できない"
        print(f"    {lab:<18}{x.mean():+.3f} ± {se:.3f} ポイント（n={len(x)} 会話） → {sig}")
    # McNemar：判定が食い違ったトークンのみで見る
    print("  判定が食い違ったトークン（McNemar）")
    for lab, dd in (("全体", disc), ("リファレンス由来", disc_r)):
        b, c = dd["c_only"], dd["w_only"]
        if b + c == 0:
            print(f"    {lab:<18}食い違いなし"); continue
        z = (b - c) / np.sqrt(b + c)
        sig = "有意（|z|≥2）" if abs(z) >= 2 else "区別できない"
        print(f"    {lab:<18}正しいのみ正解 {b} / 誤りのみ正解 {c} → z={z:+.2f} {sig}")


if __name__ == "__main__":
    main()
