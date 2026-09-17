import glob, collections
import numpy as np, pandas as pd

B = "/groups/gcg51557/experiments/0374_japanese_kame"
tot = 0
cnt = collections.Counter()
nconv = 0
for f in sorted(glob.glob(f"{B}/stage6_dataset/*.parquet"))[:2]:
    df = pd.read_parquet(f)
    for i in range(min(len(df), 300)):
        r = df.iloc[i]
        c = np.frombuffer(r["codes"], np.int32).reshape(r["codes_shape"])
        tb = c[0]
        tot += len(tb)
        nconv += 1
        cnt["ret"] += int((tb == 4).sum())
        cnt["epad"] += int((tb == 0).sum())
        cnt["pad"] += int((tb == 3).sum())
cnt["real"] = tot - cnt["epad"] - cnt["pad"] - cnt["ret"]
print(f"会話 {nconv} 件 / テキストフレーム {tot:,}")
for k in ("ret", "epad", "pad", "real"):
    print(f"  {k:6s} {cnt[k]:10,}  {100*cnt[k]/tot:7.3f}%")
print(f"  <ret> は 1 会話あたり {cnt['ret']/nconv:.2f} 件")
