"""ABCI 上の ja_wiki を調査する。

トピック供給源として使えるかを判断するため、記事数・本文長の分布・meta のキーを見る。
"""
import gzip, glob, json, collections

P = "/groups/gcg51557/experiments/0118_dedup_corpusv4_ja/data/all/cleaned/ja_wiki"
files = sorted(glob.glob(f"{P}/*.jsonl.gz"))
print(f"ファイル {len(files)} 個")

n = 0
lens = []
meta_keys = collections.Counter()
samples = []
for f in files:
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            n += 1
            lens.append(len(d["text"]))
            meta_keys.update(d.get("meta", {}).keys())
            if len(samples) < 5 and 200 < len(d["text"]) < 2000:
                samples.append((d["meta"].get("title"), d["text"][:120]))

lens.sort()
print(f"記事数: {n:,}")
print(f"meta のキー: {dict(meta_keys)}")
print("本文長（文字）:")
for q, lab in ((0, "最小"), (10, "10%"), (25, "25%"), (50, "中央"),
               (75, "75%"), (90, "90%"), (100, "最大")):
    i = min(int(n * q / 100), n - 1)
    print(f"  {lab:>4}: {lens[i]:,}")
print(f"  平均: {sum(lens)//n:,}")
for th in (200, 500, 1000, 2000):
    c = sum(1 for x in lens if x >= th)
    print(f"  本文 {th:,} 文字以上: {c:,} 件（{100*c/n:.1f}%）")

print("\n本文 200〜2000 文字の例:")
for t, s in samples:
    print(f"  【{t}】{s}...")
