"""評価用 200 会話（split_eval）と同じ Wikipedia 記事から作られた会話が、学習候補にどれだけあるかを数える。

学習候補 = JaQuAD 35,687 会話（評価 200 を除く）＋ Wikipedia 由来 64,313 会話。
会話の起点の問い（seed_question）から記事名を引く。JaQuAD は元データ（記事名つき）、
Wikipedia 由来は wiki_qa.jsonl の title を使う。
"""
import glob, json
from collections import Counter
import pandas as pd

B = "/groups/gcg51557/experiments/0374_japanese_kame"
W = f"{B}/moshi_kame_finetune"
snap = glob.glob("/home/aci18656mm/.cache/huggingface/hub/datasets--SkelterLabsInc--JaQuAD/snapshots/*/default")[0]
jq = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{snap}/*/0000.parquet")])
q2t_jq = dict(zip(jq["question"].str.strip(), jq["title"]))
q2t_wk = {}
for l in open(f"{W}/data/japanese_kame/qa_pairs/wiki_qa.jsonl", encoding="utf-8"):
    r = json.loads(l)
    q2t_wk[r["question"].strip()] = r["title"]


def titles(dirname, q2t):
    out, miss = {}, 0
    for f in glob.glob(f"{W}/data/moshirag_jp/{dirname}/*.json"):
        d = json.load(open(f, encoding="utf-8"))
        t = q2t.get(d["seed_question"].strip())
        if t is None:
            miss += 1
        out[f.rsplit("/", 1)[1][:-5]] = t
    return out, miss


jt, jmiss = titles("pilot5k_fixed", q2t_jq)
wt, wmiss = titles("wiki_main", q2t_wk)
print(f"記事名が引けなかった会話：JaQuAD {jmiss:,} / {len(jt):,}、Wikipedia 由来 {wmiss:,} / {len(wt):,}")

ev = [l.strip() for l in open(f"{B}/split_eval.txt") if l.strip()]
E = Counter(jt[s] for s in ev if jt.get(s))
print(f"評価 200 会話の記事：{len(E)} 記事（記事名が引けたもの {sum(E.values())} 会話）")

ev_set = set(ev)
jq_hit = [s for s, t in jt.items() if s not in ev_set and t in E]
wk_hit = [s for s, t in wt.items() if t in E]
print(f"同じ記事から作られた学習候補：JaQuAD {len(jq_hit):,} / {len(jt) - len(ev):,} 会話、"
      f"Wikipedia 由来 {len(wk_hit):,} / {len(wt):,} 会話")

per = Counter()
for s in jq_hit + wk_hit:
    per[jt.get(s) or wt.get(s)] += 1
covered = sum(1 for t in E if per[t] > 0)
print(f"評価の記事のうち、学習候補に同じ記事の会話があるもの：{covered} / {len(E)} 記事")
print("  同じ記事の会話が多い評価記事（上位 5）:", per.most_common(5))
ev_hit = sum(1 for s in ev if jt.get(s) and per[jt[s]] > 0)
print(f"評価 200 会話のうち、同じ記事の会話が学習候補にあるもの：{ev_hit} / {len(ev)} 会話")
