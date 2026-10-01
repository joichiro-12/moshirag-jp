import glob
import numpy as np, pandas as pd

B = "/groups/gcg51557/experiments/0374_japanese_kame"
n_all = n_ret = 0
for f in sorted(glob.glob(f"{B}/stage6_dataset/*.parquet")):
    df = pd.read_parquet(f)
    for i in range(len(df)):
        n_all += 1
        if len(np.frombuffer(df.iloc[i]["ret_frames"], np.int32)) > 0:
            n_ret += 1
print(f"全会話 {n_all:,} 件 / <ret> を持つ会話 {n_ret:,} 件（{100*n_ret/n_all:.1f}%）")
tr = sum(1 for _ in open(f"{B}/split_train.txt"))
ev = sum(1 for _ in open(f"{B}/split_eval.txt"))
print(f"split_train {tr:,} / split_eval {ev:,} / 合計 {tr+ev:,}")
