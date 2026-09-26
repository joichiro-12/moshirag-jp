"""本文長の帯ごとに記事タイトルを抜き、長さが知名度の代理になるかを確かめる。

閾値を上げる提案をする前に、「長い記事ほど人が話題にするもの」という仮定が
実際に成り立っているかを見る。成り立たないなら、長さ以外の基準が要る。
"""
import json, random
from pathlib import Path

B = Path("/groups/gcg51557/experiments/0374_japanese_kame/moshi_kame_finetune/data/japanese_kame/qa_pairs")
bands = {"500-999": [], "1000-1999": [], "2000-4999": [], "5000+": []}
for l in (B / "wiki_candidates.jsonl").open(encoding="utf-8"):
    d = json.loads(l)
    c = d["n_chars"]
    k = ("500-999" if c < 1000 else "1000-1999" if c < 2000
         else "2000-4999" if c < 5000 else "5000+")
    if len(bands[k]) < 20000:      # 先頭に偏らないよう各帯から余裕を持って溜める
        bands[k].append(d["title"])

rng = random.Random(0)
for k, v in bands.items():
    print(f"\n=== 本文 {k} 文字（{len(v):,} 件から 12 件を無作為抽出）===")
    for t in rng.sample(v, min(12, len(v))):
        print(f"  {t}")
