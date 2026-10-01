"""--num_samples を 5000 から 35687 に上げても既存の 5,000 会話が活きるかを実データで確認する。

生成器と同じ手順（seed でシャッフル → 先頭 num_samples 件）を再現し、
各添字の question が、既に生成済みの JSON に記録された seed_question と一致するかを見る。
一致しなければ集合が変わっており、作り直しになる。
"""
import json, random
from pathlib import Path

B = "/groups/gcg51557/experiments/0374_japanese_kame/moshi_kame_finetune"
SEED = 1
rows = [json.loads(l) for l in open(f"{B}/data/japanese_kame/qa_pairs/jaquad.jsonl",
                                    encoding="utf-8") if l.strip()]
print(f"入力 {len(rows):,} 行")

random.Random(SEED).shuffle(rows)          # 生成器と同一の処理
full = rows[:35687]

out = Path(f"{B}/data/moshirag_jp/pilot5k")
checked = mismatch = missing = 0
for i in range(5000):
    pj = out / f"jaquad_{i:06d}.json"
    if not pj.exists():
        missing += 1
        continue
    rec = json.load(open(pj, encoding="utf-8"))
    if rec.get("seed_question") != full[i]["question"]:
        mismatch += 1
        if mismatch <= 3:
            print(f"  不一致 i={i}")
            print(f"    既存: {rec.get('seed_question')!r}")
            print(f"    新 　: {full[i]['question']!r}")
    checked += 1

print(f"照合 {checked:,} 件 / 不一致 {mismatch} 件 / 欠落 {missing} 件")
print("判定:", "既存分はそのまま活きる" if mismatch == 0 else "集合が変わる。作り直しが要る")

# シャード分割の網羅性も確認する
NSHARD = 8
cover = {}
for sh in range(NSHARD):
    cover[sh] = [i for i in range(len(full)) if i % NSHARD == sh]
total = sum(len(v) for v in cover.values())
overlap = len(full) - len(set(i for v in cover.values() for i in v))
print(f"\nシャード {NSHARD} 分割: 合計 {total:,} 件 / 全 {len(full):,} 件 / 重複 {overlap} 件")
print(f"  1 シャードあたり {min(len(v) for v in cover.values()):,}〜"
      f"{max(len(v) for v in cover.values()):,} 件")
