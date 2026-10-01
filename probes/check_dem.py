"""問いが記事を指示語で参照している割合を測る。

「このシングルの正式なタイトルは何ですか」のような問いは、記事を文脈として
前提にしており、会話のユーザ発話としては成立しない。機械的に検出できる欠陥である。
"""
import json, re, sys

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
dem = re.compile(
    r"(この|その|当該|本)(作品|シングル|曲|アルバム|映画|番組|人物|会社|企業|団体|駅|"
    r"学校|施設|建物|山|川|島|町|村|市|事件|戦い|条約|法律|制度|技術|装置|機体|艦|"
    r"車両|生物|種|属|科|目|記事|項目|話|巻|章|大会|試合|チーム|選手|キャラ|バンド|"
    r"番組|漫画|小説|ゲーム|アニメ)")
hit = [r for r in rows if dem.search(r["question"])]
print(f"指示語で記事を指す問い: {len(hit)} / {len(rows)}（{100*len(hit)/len(rows):.1f}%）")
for r in hit[:8]:
    t, q = r["title"], r["question"]
    print(f"  【{t}】{q}")
