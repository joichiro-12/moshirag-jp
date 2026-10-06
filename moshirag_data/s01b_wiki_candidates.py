"""ja_wiki から、QA 生成（s01c）に渡す候補記事を抜き出す。

本文が短い記事（--min_chars 未満）と、タイトルから題材にならないと分かる記事
（一覧・曖昧さ回避など）を落とす。最終的な取捨は s01c の LLM に任せる。

Each line: {"title": "...", ..., "lead": "本文冒頭（--lead_chars 文字）"}
"""
from __future__ import annotations
import argparse, glob, gzip, json, re

SRC = "/groups/gcg51557/experiments/0118_dedup_corpusv4_ja/data/all/cleaned/ja_wiki"

# タイトルで落とすもの。会話の題材になりにくいことが題名から分かるものに限る。
DROP_TITLE = re.compile(
    r"(一覧$|の一覧|曖昧さ回避|"                      # 一覧・曖昧さ回避
    r"^\d{3,4}年(代)?の|"                            # 「2007年の…」形式の年度別記事
    r"第\d+回|"                                      # 「第23回…大会」
    r"級\)$|"                                        # 「…（…級）」
    r"^(ISO|JIS|Category|Template|Portal|Wikipedia)[:：]|"
    r"駅$|"                                          # 個別の駅（数が多く内容が定型的）
    r"^(国道|県道|都道|府道)\d|"
    r"(小学校|中学校|高等学校|交通安全協会)$)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_chars", type=int, default=500, help="本文長の下限")
    ap.add_argument("--lead_chars", type=int, default=1000, help="切り出す冒頭の長さ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_title_filter", action="store_true",
                    help="タイトルによる除外を行わない（効果の測定用）")
    a = ap.parse_args()

    n = short = dropped = kept = 0
    dropped_examples = []
    with open(a.out, "w", encoding="utf-8") as w:
        for f in sorted(glob.glob(f"{SRC}/*.jsonl.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    n += 1
                    text = d["text"]
                    if len(text) < a.min_chars:
                        short += 1
                        continue
                    m = d["meta"]
                    title = m.get("title", "")
                    if not a.no_title_filter and DROP_TITLE.search(title):
                        dropped += 1
                        if len(dropped_examples) < 15:
                            dropped_examples.append(title)
                        continue
                    kept += 1
                    w.write(json.dumps({
                        "wiki_id": m.get("id"),
                        "title": title,
                        "url": m.get("url"),
                        "lead": text[: a.lead_chars],
                        "n_chars": len(text),
                    }, ensure_ascii=False) + "\n")

    print(f"全記事          {n:,}")
    print(f"  本文 {a.min_chars} 文字未満で除外 {short:,}（{100*short/n:.1f}%）")
    print(f"  タイトルで除外              {dropped:,}（{100*dropped/n:.1f}%）")
    print(f"  **残り**                    {kept:,}（{100*kept/n:.1f}%）")
    print(f"  必要な 478,500 件に対して   {kept/478500:.2f} 倍")
    if dropped_examples:
        print("\nタイトルで落とした例:")
        for t in dropped_examples:
            print(f"  {t}")
    print(f"\n出力: {a.out}")


if __name__ == "__main__":
    main()
