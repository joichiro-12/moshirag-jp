"""既定版と厳しい版を、同一の 1,000 件に対する結果として比較する。

## なぜ必要か

厳しい版の採用率は 36.0%（既定 98.4%）まで下がったが、**落ちた 64% が狙ったものか**は
目視 12 件では判断できない。先に問題として挙げた型が実際に減ったかを数える。

## 数える型

いずれも「人が会話で持ち出すとは考えにくい」と判断した型である。
正規表現による粗い分類であり、取りこぼしと誤検出を含む。

1. **循環**：答えが記事タイトルと完全一致
2. **日付だけ**：問いが発売日・配信日などを尋ね、答えが日付
3. **住所・番号**：答えが郵便番号・住所・電話番号の形
4. **経歴の細部**：問いが出身地・生年・在籍先など、人物の属性を尋ねる
"""
import json, re
from pathlib import Path

B = Path("/groups/gcg51557/experiments/0374_japanese_kame/moshi_kame_finetune/data/japanese_kame/qa_pairs")

DATE_Q = re.compile(r"(発売|配信|公開|放送)(日|開始|され)|いつ(発売|配信|公開|放送)")
DATE_A = re.compile(r"\d{4}年|\d{1,2}月\d{1,2}日")
ADDR_A = re.compile(r"\d{3}-\d{4}|[都道府県].{0,12}[市区町村].{0,20}\d|丁目|番地")
BIO_Q = re.compile(r"(出身|生まれ|どこで生まれ|何歳|正式な名前|本名)")


def stats(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(rows)
    c = {"循環": 0, "日付だけ": 0, "住所・番号": 0, "経歴の細部": 0}
    for r in rows:
        q, a, t = r["question"], r["answer"], r["title"].strip()
        if a.strip() == t:
            c["循環"] += 1
        if DATE_Q.search(q) and DATE_A.search(a):
            c["日付だけ"] += 1
        if ADDR_A.search(a):
            c["住所・番号"] += 1
        if BIO_Q.search(q):
            c["経歴の細部"] += 1
    return n, c, {r["wiki_id"] for r in rows}


n1, c1, ids1 = stats(B / "wiki_qa_pilot.jsonl")
n2, c2, ids2 = stats(B / "wiki_qa_strict.jsonl")

print(f"{'':<14}{'既定':>16}{'厳しい版':>18}")
print(f"{'採用件数':<14}{n1:>10,} 件{'':>4}{n2:>10,} 件")
print(f"{'採用率':<14}{100*n1/1000:>13.1f}%{'':>4}{100*n2/1000:>13.1f}%")
print()
print("問題として挙げた型の件数（採用分に占める割合）")
for k in c1:
    p1, p2 = 100 * c1[k] / n1, 100 * c2[k] / n2
    mark = " ←改善" if p2 < p1 * 0.7 else (" ←悪化" if p2 > p1 * 1.3 else "")
    print(f"  {k:<10}{c1[k]:>5,}（{p1:4.1f}%）{'':>4}{c2[k]:>5,}（{p2:4.1f}%）{mark}")

print(f"\n両方が採用した記事: {len(ids1 & ids2):,} 件")
print(f"既定のみ採用      : {len(ids1 - ids2):,} 件")
print(f"厳しい版のみ採用  : {len(ids2 - ids1):,} 件")

print("\n必要数への到達")
CAND = 740_916
for lab, rate in (("既定", n1 / 1000), ("厳しい版", n2 / 1000)):
    got = int(CAND * rate)
    print(f"  {lab:<8}{got:>9,} トピック / 必要 478,500 に対し {got/478500:.2f} 倍"
          f" → 4 スタイルで {got*4:,} 会話（論文比 {100*got*4/1_901_376:.0f}%）")
