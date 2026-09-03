"""C の検証：参照の注入が意図した位置以降だけを変えているかを数値で確かめる。

条件は元実装の経路（ConditionFuser の streaming_sum）で渡す。学習時も同じ経路を使う。
検証は注入なし／ありで forward を 2 回回して text_logits の差分を取る。

期待する結果
  1. 注入の開始フレーム以降に差分が出る（因果的な Transformer なので影響は後方へ伝播する）
  2. 開始より前は差分が完全にゼロ。破れていたら未来の情報が過去に漏れている
"""
import sys, json, glob
sys.path.insert(0, "/groups/gcg51557/experiments/0374_japanese_kame/moshi-rag/moshi")
sys.path.insert(0, "/groups/gcg51557/experiments/0374_japanese_kame/moshi_kame_finetune")
import numpy as np, pandas as pd, torch
from safetensors.torch import load_file
from moshi.models.lm import LMModel
from moshi.conditioners.base import ConditionFuser, ConditionType
from moshirag_inject import build_streaming_sum

B = "/groups/gcg51557/experiments/0374_japanese_kame"
D = f"{B}/llmjp_moshi_v1_ragfmt"
REF = "reference_with_time"          # 元実装 loaders.py の conditioner 名


def main():
    kw = json.load(open(f"{D}/moshi_lm_kwargs.json"))
    # streaming_sum だけの fuser。first_speaker（prepend）は D で扱う
    fuser = ConditionFuser({"sum": [], "prepend": [], "cross": [], "streaming_sum": [REF]})
    lm = LMModel(**kw, fuser=fuser, device="cuda", dtype=torch.bfloat16)
    miss, unexp = lm.load_state_dict(load_file(f"{D}/model.safetensors", device="cpu"), strict=False)
    assert not miss and not unexp, (len(miss), len(unexp))
    lm.eval()

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
    print(f"  {row['dialogue_id']}  T={T} / <ret> {len(rf)} 件 @ {rf.tolist()}", flush=True)

    ss, spans = build_streaming_sum(rf, dl, off, val, T, np.random.default_rng(0), dim=row["ref_dim"])
    spans = sorted(s for s, _ in spans)
    print(f"  注入の開始フレーム {spans} / 非ゼロ {int((np.abs(ss).sum(1) > 0).sum())} フレーム", flush=True)

    cond = torch.from_numpy(ss).unsqueeze(0).to("cuda", torch.bfloat16)
    mask = torch.ones(1, T, dtype=torch.bool, device="cuda")

    with torch.no_grad():
        base = lm(codes)                                                   # 注入なし
        inj = lm(codes, condition_tensors={REF: ConditionType(cond, mask)})  # 注入あり

    d = (inj.text_logits.float() - base.text_logits.float()).abs()[0, 0].max(-1).values  # [T]
    d = d.cpu().numpy()
    s0 = spans[0]
    print(f"  差分の最大値 全体 {d.max():.4f}", flush=True)
    print(f"  開始より前（0〜{s0-1}）: 最大 {d[:s0].max():.6f}  ← 0 でなければ実装の誤り", flush=True)
    print(f"  開始以降（{s0}〜）    : 最大 {d[s0:].max():.4f}", flush=True)
    nz = np.nonzero(d > 1e-4)[0]
    print(f"  差分が出た最初のフレーム {int(nz[0]) if len(nz) else None}（期待 {s0}）", flush=True)


if __name__ == "__main__":
    main()
