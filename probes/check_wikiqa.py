"""パイロットで作った QA の質を、機械的に測れる範囲で確認する。

## 見たいこと

会話生成器は問いをそのままユーザ発話の種にする。**問いだけで答えが一意に定まらないと、
リファレンスを読んでも「正しく答えた」と判定できない**。

記事から作った問いは、記事を暗黙の文脈にしてしまうことがある。
例：「フランス人精神分析家として有名だったのは誰ですか」→ 答えは多数ありうる。

機械的に切り分けられるのは次の 2 型である。

- **属性型**：問いが記事の主題に言及している（「情報処理学会はいつ設立されたか」）。
  主題が問いに入っているので自己完結しやすい
- **定義型**：答えが記事タイトルそのもの（「〜を指す言葉は何か」→ ボーンベッド）。
  説明が十分に限定的でないと答えが一意にならない

型の判定は自己完結性の代理指標にすぎない。定義型がすべて駄目なわけではない。
"""
import json, sys
from collections import Counter

path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
print(f"件数 {len(rows):,}")

c = Counter()
ans_len = []
for r in rows:
    q, a, t = r["question"], r["answer"], r["title"]
    ans_len.append(len(a))
    core = t.split("（")[0].split(" (")[0].strip()
    in_q = bool(core) and core in q
    is_def = bool(core) and (core in a or a in core)
    if in_q and not is_def:
        c["属性型（主題が問いにある）"] += 1
    elif is_def:
        c["定義型（答えがタイトル）"] += 1
    else:
        c["どちらでもない"] += 1

for k, v in c.most_common():
    print(f"  {k}: {v:,}（{100*v/len(rows):.1f}%）")

ans_len.sort()
print(f"\n答えの長さ（文字）: 中央 {ans_len[len(ans_len)//2]} / "
      f"最大 {ans_len[-1]} / 20 文字超 {sum(1 for x in ans_len if x > 20):,} 件")

print("\n--- 定義型の例（自己完結していない懸念があるもの） ---")
n = 0
for r in rows:
    core = r["title"].split("（")[0].split(" (")[0].strip()
    if core and (core in r["answer"] or r["answer"] in core) and core not in r["question"]:
        print(f"  【{r['title']}】{r['question']} → {r['answer']}")
        n += 1
        if n >= 8:
            break

print("\n--- 属性型の例 ---")
n = 0
for r in rows:
    core = r["title"].split("（")[0].split(" (")[0].strip()
    if core and core in r["question"] and core not in r["answer"]:
        print(f"  【{r['title']}】{r['question']} → {r['answer']}")
        n += 1
        if n >= 5:
            break
