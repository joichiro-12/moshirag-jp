"""日本語 MoshiRAG の学習ループ（D-3：小規模で loss が下がるかを見る段階）。

モデル定義は元実装 kyutai-labs/moshi-rag をそのまま使う（D-28）。KAME の finetune.py は
使わない。学習ループは公開されていないため、論文 §4.2 と Moshi 論文の定義から実装する。

## 学習設定（論文 §4.2。D-34）

    学習率 2e-6 / バッチ 32 / 100k 更新 / 参照ごとにドロップアウト 0.2
    学習対象：参照テキストエンコーダ以外の全パラメータ

参照エンコーダ（ARC-Encoder）は工程 5c で事前計算済みのため、この時点で凍結されている。
射影層（bridge）は**学習対象**（D-32、論文 §4.2「projected via a one-layer trainable
linear layer」）。

## 損失

Moshi 論文（Défossez et al. 2024）に従い、テキストと音声の交差エントロピーを足す。
元実装は `LMOutput` の mask で無効位置を除外する設計なので、それをそのまま使う。
padding は zero_token_id（-1）で埋めており、mask がこれを自動的に外す（lm.py:372-374）。

## 学習率を 2 群に分ける（元実装からの逸脱。理由を明記する）

論文は全体で 2e-6。ただし射影層（bridge）は **ARC の Llama 向け初期値から Moshi 空間への
写像を学ぶ**必要があり、他のパラメータ（既に日本語 Moshi として学習済み）とは事情が違う。
2e-6 では動きが遅く、短い学習で「参照が使えない」という**偽陰性**を招く。

そこで LM 本体は 2e-6（論文どおり）、射影層のみ `--bridge_lr`（既定 1e-4）とする。
新しく適応させる層に高い学習率を当てる一般的な扱いだが、**元実装の記述にはない**。
本番学習でこの分離を残すかは、参照寄与の測定結果を見て決める。

## 未実装（別途）

- 挨拶削除 p=0.3、冒頭無音の前置（◆15）
- 音声の前処理（80 ms 窓で RMS −65 dBFS 未満をゼロ化）
"""
from __future__ import annotations
import argparse, glob, json, sys, time
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
REF_DROPOUT = 0.2          # 論文 §4.2「A dropout probability of 0.2 is applied to each reference document」


def load_rows(n_conv: int, stems_file: str | None = None):
    """会話を読む。stems_file があればその一覧に含まれるものだけを使う
    （評価に使う会話を学習から除くため）。"""
    want = None
    if stems_file:
        want = {l.strip() for l in open(stems_file) if l.strip()}
    rows = []
    for f in sorted(glob.glob(f"{B}/stage6_dataset/*.parquet")):
        df = pd.read_parquet(f)
        for i in range(len(df)):
            r = df.iloc[i]
            if want is not None and r["dialogue_id"].split("/")[-1] not in want:
                continue
            rows.append(r)
            if len(rows) >= n_conv:
                return rows
    return rows


def make_batch(rows, zero_id: int, max_frames: int, bridge, rng, device="cuda"):
    """codes [B,17,T] と注入テンソル [B,T,4096] を作る。"""
    arrs = [np.frombuffer(r["codes"], np.int32).reshape(r["codes_shape"])[:, :max_frames] for r in rows]
    T = max(a.shape[1] for a in arrs)
    codes = np.full((len(arrs), 17, T), zero_id, np.int64)
    for i, a in enumerate(arrs):
        codes[i, :, : a.shape[1]] = a
    cond = torch.zeros(len(arrs), T, 4096, device=device, dtype=torch.bfloat16)
    nref = ndrop = 0
    for i, r in enumerate(rows):
        rf = np.frombuffer(r["ret_frames"], np.int32)
        if len(rf) == 0:
            continue
        dl = np.frombuffer(r["d_lead_frames"], np.int32)
        off = np.frombuffer(r["ref_offsets"], np.int32)
        val = np.frombuffer(r["ref_values"], np.float16).reshape(-1, r["ref_dim"])
        # 参照ごとのドロップアウト（論文 §4.2）。落とした参照は注入しない
        keep = [k for k in range(len(rf)) if rng.random() >= REF_DROPOUT]
        ndrop += len(rf) - len(keep); nref += len(rf)
        if not keep:
            continue
        sub_off = np.concatenate([[0], np.cumsum([off[k+1]-off[k] for k in keep])]).astype(np.int32)
        sub_val = np.concatenate([val[off[k]:off[k+1]] for k in keep], 0)
        c, _ = build_streaming_sum(bridge, rf[keep], dl[keep], sub_off, sub_val, T, rng, device=device)
        cond[i] = c[0]
    return torch.from_numpy(codes).to(device), cond, nref, ndrop


def save_ckpt(lm, bridge, path):
    """重みを path に書き出す。推論時のキー名に合わせて bridge を別ファイルにする。"""
    import os
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    save_file({k: v.contiguous() for k, v in lm.state_dict().items()}, f"{path}/model.safetensors")
    save_file({k: v.contiguous() for k, v in bridge.state_dict().items()}, f"{path}/bridge.safetensors")
    print(f"  保存: {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default=None, help="使う会話の一覧。省略時は先頭から")
    ap.add_argument("--n_conv", type=int, default=100)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-6)          # 論文 §4.2
    ap.add_argument("--bridge_lr", type=float, default=1e-4)   # 逸脱。上の説明を参照
    ap.add_argument("--save", default=None, help="学習後の重みの保存先")
    ap.add_argument("--save_every", type=int, default=0,
                    help="この間隔で途中の重みも保存する（0 で無効）。学習は止めず、傾きを後から測るための副産物")
    ap.add_argument("--max_frames", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    kw = json.load(open(f"{D}/moshi_lm_kwargs.json"))
    fuser = ConditionFuser({"sum": [], "prepend": [], "cross": [], "streaming_sum": [REF]})
    lm = LMModel(**kw, fuser=fuser, device="cuda", dtype=torch.bfloat16, gradient_checkpointing=True)
    miss, unexp = lm.load_state_dict(load_file(f"{D}/model.safetensors", device="cpu"), strict=False)
    assert not miss and not unexp
    lm.train()
    bridge = make_bridge(f"{B}/arc_bridge_init.safetensors", dtype=torch.float32)

    lm_params = [p for p in lm.parameters() if p.requires_grad]
    br_params = list(bridge.parameters())
    params = lm_params + br_params
    print(f"  学習対象: LM {sum(p.numel() for p in lm_params)/1e9:.2f}B（lr {a.lr}） + "
          f"射影層 {sum(p.numel() for p in br_params)/1e6:.1f}M（lr {a.bridge_lr}）", flush=True)
    opt = torch.optim.AdamW([{"params": lm_params, "lr": a.lr},
                             {"params": br_params, "lr": a.bridge_lr}])

    rows = load_rows(a.n_conv, a.stems)
    print(f"  会話 {len(rows)} 件 / バッチ {a.batch} / {a.steps} ステップ / 学習率 {a.lr}", flush=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time(); hist = []
    for step in range(a.steps):
        idx = rng.choice(len(rows), a.batch, replace=False)
        codes, cond, nref, ndrop = make_batch([rows[i] for i in idx], lm.zero_token_id,
                                              a.max_frames, bridge, rng)
        mask = torch.ones(cond.shape[0], cond.shape[1], dtype=torch.bool, device="cuda")
        out = lm(codes, condition_tensors={REF: ConditionType(cond, mask)})

        tl = out.text_logits[:, 0]
        tm = out.text_mask[:, 0] & torch.isfinite(tl).all(-1)
        loss_t = F.cross_entropy(tl[tm].float(), codes[:, 0, : tl.shape[1]][tm])
        al = out.logits.permute(0, 2, 1, 3)                      # [B,T,K,card]
        am = out.mask.permute(0, 2, 1) & torch.isfinite(al).all(-1)
        tgt_a = codes[:, 1 : 1 + lm.dep_q, : al.shape[1]].permute(0, 2, 1)
        loss_a = F.cross_entropy(al[am].float(), tgt_a[am])
        loss = loss_t + loss_a

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append((loss_t.item(), loss_a.item()))
        if step % 5 == 0 or step == a.steps - 1:
            print(f"  step {step:3d}  text {loss_t.item():.4f}  audio {loss_a.item():.4f}  "
                  f"参照 {nref} 件（{ndrop} 件 drop）  {time.time()-t0:.0f}s", flush=True)
        # 途中の重みを残す。学習は止めない（更新回数と <ret> 出力確率の関係を後から測るため）
        if a.save and a.save_every and (step + 1) % a.save_every == 0 and (step + 1) < a.steps:
            save_ckpt(lm, bridge, f"{a.save}_step{step+1}")

    h = np.array(hist)
    k = max(len(h) // 5, 1)
    print(f"\n  テキスト損失: 最初の {k} ステップ平均 {h[:k,0].mean():.4f} → 最後 {k} ステップ平均 {h[-k:,0].mean():.4f}", flush=True)
    print(f"  音声損失    : 最初の {k} ステップ平均 {h[:k,1].mean():.4f} → 最後 {k} ステップ平均 {h[-k:,1].mean():.4f}", flush=True)
    if a.save:
        save_ckpt(lm, bridge, a.save)


if __name__ == "__main__":
    main()
