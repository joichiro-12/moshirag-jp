"""採用された QA の元記事が、どれくらいの長さの記事だったかを測る。

## なぜ必要か

パイロットで採用された話題に、人が自発的に尋ねるとは考えにくいものが混ざっている
（「黒田茂助（漆器商）」「実践倫理宏正会研修会館」など）。論文のトピックは
Natural Questions 由来で実際の検索質問なので、分布が違う。

**記事の長さを知名度の代理指標として使う**（長い記事ほど加筆され、参照されていると
考えられる。ただしこれは仮定であり、検証していない）。閾値を上げたときに
必要数を確保できるかを見る。
"""
import json
from pathlib import Path

B = Path("/groups/gcg51557/experiments/0374_japanese_kame/moshi_kame_finetune/data/japanese_kame/qa_pairs")

chars = {}
for l in (B / "wiki_candidates.jsonl").open(encoding="utf-8"):
    d = json.loads(l)
    chars[d["wiki_id"]] = d["n_chars"]

rows = [json.loads(l) for l in (B / "wiki_qa_pilot.jsonl").open(encoding="utf-8") if l.strip()]
lens = sorted(chars.get(r["wiki_id"], 0) for r in rows)
n = len(lens)
print(f"採用 {n:,} 件の元記事の本文長（文字）")
for q, lab in ((10, "10%"), (25, "25%"), (50, "中央"), (75, "75%"), (90, "90%")):
    print(f"  {lab:>4}: {lens[int(n*q/100)]:,}")
print(f"  平均: {sum(lens)//n:,}")

print("\n本文長の下限を上げたときに残る候補数（全 740,916 件に対して）")
allc = sorted(chars.values())
m = len(allc)
import bisect
for th in (500, 1000, 1500, 2000, 3000, 5000):
    left = m - bisect.bisect_left(allc, th)
    print(f"  {th:>5,} 文字以上: {left:7,} 件（{100*left/m:4.1f}%） / 必要 478,500 に対し {left/478500:.2f} 倍")
