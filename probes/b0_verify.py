"""B-0 の検証：変換した重みが moshi-rag の LMModel に載り、順伝播が通るか。"""
import sys, json, torch
sys.path.insert(0, "/groups/gcg51557/experiments/0374_japanese_kame/moshi-rag/moshi")
from safetensors.torch import load_file
from moshi.models.lm import LMModel

D="/groups/gcg51557/experiments/0374_japanese_kame/llmjp_moshi_v1_ragfmt"
kw=json.load(open(f"{D}/moshi_lm_kwargs.json"))
lm=LMModel(**kw, device="cuda", dtype=torch.bfloat16)
sd=load_file(f"{D}/model.safetensors", device="cpu")
missing, unexpected = lm.load_state_dict(sd, strict=False)
print(f"  未読み込み {len(missing)} / 余り {len(unexpected)}")
for k in list(missing)[:5]: print(f"    未: {k}")
for k in list(unexpected)[:5]: print(f"    余: {k}")
print(f"  rag_token_id = {lm.rag_token_id} / num_codebooks = {lm.num_codebooks}")

# 実データ 1 会話で順伝播
import pandas as pd, numpy as np, glob
f=sorted(glob.glob("/groups/gcg51557/experiments/0374_japanese_kame/stage6_dataset/*.parquet"))[0]
df=pd.read_parquet(f, columns=["dialogue_id","B","B_shape","ret_frames"])
r=df.iloc[2]
Bm=np.frombuffer(r["B"],np.int32).reshape(r["B_shape"])
T=min(Bm.shape[1], 400)
codes=torch.from_numpy(Bm[:,:T].astype(np.int64)).unsqueeze(0).cuda()   # [1, 9, T]
print(f"  入力 codes {tuple(codes.shape)}（K=9 は text 1 + Mimi 8）")
# モデルは n_q=16（両話者分）を期待する。片側だけなら足りないので確認
print(f"  モデルが期待する codebook 数: {lm.num_codebooks}")
