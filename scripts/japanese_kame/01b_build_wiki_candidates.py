"""ja_wiki から、会話トピックの候補になりうる記事を抜き出す。

## なぜ必要か

JaQuAD は 35,687 件しかなく、論文のトピック数（約 478,500）に 13.4 倍足りない。
日本語 Wikipedia から問いと答えの組を作って補う（大規模化の計画を参照）。

このスクリプトは**その前段**で、LLM に渡す候補記事を絞る。LLM 呼び出しは高価なので、
機械的に落とせるものは先に落とす。

## 落とす基準

1. **本文が短い記事**（既定 500 文字未満）。内容が薄く、問いと答えを作れない。
   全 1,200,785 件のうち 500 文字以上は 783,413 件（65.2%）
2. **タイトルから題材にならないと分かる記事**。一覧・曖昧さ回避・年度別の個別イベントなど。
   ここで落とせるのは明らかなものだけで、残りは LLM の判定に委ねる

**この 2 段で「会話の題材になる」ことは保証できない。** 本文長は内容の濃さの代理指標に
すぎず、タイトルの正規表現も網羅的ではない。最終的な取捨は LLM に判定させる。

## 出力

1 行 1 記事の JSONL。`lead` は本文の冒頭（既定 1,000 文字）で、QA 生成の入力になる。
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
