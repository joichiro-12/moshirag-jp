"""モデルが `<ret>` を自分で出せるかを測る。

## なぜ別に測る必要があるか

これまでの測定（eval_ref_contribution.py）は **`<ret>` を教師強制で与えていた**。
入力の該当フレームに既に `<ret>`（ID 4）が入っている状態で、リファレンスの注入が
body の予測を助けるかを見たもの。

推論では順序が違う。
    ① モデルが自分で `<ret>` を出す（検索が必要だと判断する）
    ② 検索が走る
    ③ リファレンスが注入される

**① が起きなければ ②③ は起こらない。** 注入が効くかとは独立に、`<ret>` を正しい位置で
出せることが必須。ここを測っていなかった。

## 測り方

教師強制で 1 回順伝播し、各フレームでモデルが ID 4 に与える確率を読む。
`<ret>` の位置にリファレンスはまだ注入されていない（注入は ret_frame + d' から）ので、
この測定は注入の有無に依存しない。

  再現率側：本来 `<ret>` がある位置での ID 4 の確率と順位
  誤検出側：`<ret>` が無い位置（Moshi が話しているフレーム）での ID 4 の確率

学習前と学習後を同じ会話で比べる。
"""
from __future__ import annotations
import argparse, glob, json, sys
sys.path.insert(0, "/groups/gcg51557/experiments/0374_japanese_kame/moshi-rag/moshi")
import numpy as np, pandas as pd, torch
from safetensors.torch import load_file
from moshi.models.lm import LMModel
from moshi.conditioners.base import ConditionFuser

B = "/groups/gcg51557/experiments/0374_japanese_kame"
D = f"{B}/llmjp_moshi_v1_ragfmt"
REF = "reference_with_time"
RET_ID = 4
PAD, EPAD = 3, 0
# 元実装 moshi-rag の LMGen 既定値。検索の起動は「サンプリングされたトークン == ID 4」で判定される
# （moshi/moshi/inference_utils/channel.py:208）。素の softmax 確率ではなくこの方策下の確率が効く。
TEMP_TEXT, TOP_K_TEXT = 0.7, 25


def emit_prob(lg):
    """元実装 sample_token(use_sampling=True, temp_text, top_k_text) の下で ID 4 が選ばれる確率。

    sample_token は softmax(logits / temp) を取り、sample_top_k で上位 k に切って
    再正規化してから多項サンプリングする（moshi/moshi/utils/sampling.py:84）。
    lg: [T, card] -> [T]
    """
    p = torch.softmax(lg / TEMP_TEXT, dim=-1)
    topv, topi = p.topk(TOP_K_TEXT, dim=-1)
    hit = (topi == RET_ID)
    return (topv * hit).sum(-1) / topv.sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default=None)
    ap.add_argument("--n_conv", type=int, default=200)
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    kw = json.load(open(f"{D}/moshi_lm_kwargs.json"))
    fuser = ConditionFuser({"sum": [], "prepend": [], "cross": [], "streaming_sum": [REF]})
    lm = LMModel(**kw, fuser=fuser, device="cuda", dtype=torch.bfloat16)
    lm.load_state_dict(load_file(a.ckpt or f"{D}/model.safetensors", device="cpu"), strict=False)
    lm.eval()
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

    p_at_ret, rank_at_ret, p_elsewhere = [], [], []
    e_at_ret, e_conv_any, e_fp_per_conv, top1_at_ret = [], [], [], []
    for r in rows:
        c = np.frombuffer(r["codes"], np.int32).reshape(r["codes_shape"])
        T = min(c.shape[1], a.max_frames)
        codes = torch.from_numpy(c[:, :T].astype(np.int64)).unsqueeze(0).cuda()
        rf = [int(x) for x in np.frombuffer(r["ret_frames"], np.int32) if int(x) < T]
        if not rf:
            continue
        with torch.no_grad():
            out = lm(codes)                       # 注入なし。<ret> の位置は注入前なので影響しない
        lg = out.text_logits[0, 0].float()        # [T, text_card]
        pr = torch.softmax(lg, -1)
        ep = emit_prob(lg)                        # 元実装のサンプリング方策下の実効確率 [T]
        for f_ in rf:
            p_at_ret.append(pr[f_, RET_ID].item())
            rank_at_ret.append(int((lg[f_] > lg[f_, RET_ID]).sum().item()) + 1)
            e_at_ret.append(ep[f_].item())
            top1_at_ret.append(int(lg[f_].argmax().item()))
        # 会話あたり「1 回以上 <ret> を出す」確率
        e_conv_any.append(1.0 - float(np.prod([1.0 - ep[f_].item() for f_ in rf])))
        # 誤検出：<ret> 以外の全フレームでの実効確率の総和 = 1 会話あたりの期待誤起動回数
        other = torch.ones(T, dtype=torch.bool); other[rf] = False
        e_fp_per_conv.append(ep[other.cuda()].sum().item())
        # 素の確率での誤検出側（従来指標。Moshi が話しているフレームのみ）
        tb = codes[0, 0, :T].cpu().numpy()
        spk = np.where((tb != PAD) & (tb != EPAD) & (tb != RET_ID))[0]
        if len(spk):
            sel = spk[:: max(len(spk) // 20, 1)][:20]
            p_elsewhere += pr[sel, RET_ID].tolist()

    pr_ = np.array(p_at_ret); rk = np.array(rank_at_ret); pe = np.array(p_elsewhere)
    ea = np.array(e_at_ret); ec = np.array(e_conv_any); ef = np.array(e_fp_per_conv)
    print(f"\n  [素の softmax]  <ret> の位置（n={len(pr_)}）")
    print(f"    ID 4 の確率 : 中央 {np.median(pr_):.4f} / 平均 {pr_.mean():.4f}")
    print(f"    ID 4 の順位 : 中央 {np.median(rk):.0f} 位 / 1 位が {100*(rk==1).mean():.1f}% / 10 位以内が {100*(rk<=10).mean():.1f}% / 25 位以内が {100*(rk<=25).mean():.1f}%")
    print(f"  [素の softmax]  <ret> が無い位置（n={len(pe)}）")
    print(f"    ID 4 の確率 : 中央 {np.median(pe):.6f} / 平均 {pe.mean():.6f}")
    print(f"  比（<ret> 位置 / それ以外）: {pr_.mean()/max(pe.mean(),1e-12):.1f} 倍")
    print(f"\n  [元実装のサンプリング下 temp_text={TEMP_TEXT} / top_k_text={TOP_K_TEXT}]")
    print(f"    <ret> 位置での実効出力確率 : 中央 {np.median(ea):.4f} / 平均 {ea.mean():.4f}")
    print(f"    会話あたり 1 回以上出る確率 : 平均 {ec.mean():.4f}（n={len(ec)} 会話）")
    print(f"    誤起動の期待回数 : {ef.mean():.3f} 回/会話（<ret> 以外の全フレーム）")

    # <ret> の位置で 1 位を占めているトークン。そこは元のストリームでは EPAD の枠なので、
    # 競合が padding なら学習不足、実テキストなら <ret> の配置自体を疑う根拠になる。
    import collections
    names = {0: "EPAD", 1: "<1>", 2: "<2>", 3: "PAD", RET_ID: "<ret>"}
    cnt = collections.Counter(top1_at_ret)
    print(f"    <ret> 位置で 1 位のトークン（n={len(top1_at_ret)}）:", end="")
    for tid, n in cnt.most_common(5):
        nm = names.get(tid) or ("ID%d" % tid)
        print(" %s=%.1f%%" % (nm, 100 * n / len(top1_at_ret)), end="")
    print()


if __name__ == "__main__":
    main()
