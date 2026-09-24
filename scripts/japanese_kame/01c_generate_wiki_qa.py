"""Wikipedia の記事から、会話の種になる問いと答えの組を作る。

## なぜ必要か

会話生成器（02e）は入力の `question` と `answer` しか使わない。参照文はそこから
LLM が作る（D-1）。したがって**正解段落つきの QA データセットである必要がなく**、
問いと答えの組さえあればよい。

JaQuAD は 35,687 件で論文のトピック数（約 478,500）に 13.4 倍足りないので、
Wikipedia から作って補う。

## 設計

**記事の適否判定と QA 生成を 1 回の呼び出しにまとめる。** 2 段に分けると呼び出し数が
倍になる。題材にならない記事には「スキップ」と答えさせる。

入力は `01b_build_wiki_candidates.py` の出力（本文冒頭つき）。
出力は `jaquad.jsonl` と同じ形（`question` / `answer` / `source`）に、
追跡用の記事情報を足したもの。そのまま 02e の `--input_file` に渡せる。

## 作る問いの条件

- **記事に書かれている事実に基づく。** 記事に無いことを書かせない
- **記事を読んでいない人が口に出しそうな問い。** Wikipedia は網羅性に偏っており、
  そのままでは人が尋ねない話題が多く混ざる
- **答えは短く。** 後段でトピック句を導出する際、答えを含まない話題に落とす必要がある
"""
from __future__ import annotations
import argparse, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# API の呼び出しと思考混入の検出は 02e の実装をそのまま使う。
# reasoning_effort の渡し方はサーバ実装に依存し、02e は 3 通り試して当たったものを
# 覚える作りになっている。複製すると片方だけ直す事故が起きるので import する。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
_v5 = __import__("02e_generate_moshirag_v5")
call = _v5.call

QA_ROLE = """\
Wikipedia の記事の冒頭を渡します。その内容から、**会話の中で自然に出てきそうな問い**と、
その答えを 1 組作ってください。

- 答えは**記事に書かれている事実**にする。記事に無いことを書かない
- 答えは短くする。固有名詞・数値・短い語句
- 問いは、記事を読んでいない人が口に出しそうなものにする
- 問いに答えを含めない

**次のような記事は題材にならないので、問いを作らず「スキップ」とだけ書いてください。**

- 自治体の外郭団体、学校、駅など、内容が定型的で話題にならないもの
- 統計や年表だけで構成され、人が尋ねる内容が無いもの
- 個別の試合結果・大会の回次など、その年その回にしか意味がないもの
- 記事が短すぎて事実を 1 つも取り出せないもの

出力は次の 2 行だけ。スキップの場合は「スキップ」の 1 行だけ。

問い: ...
答え: ...

例：

記事タイトル: 手塚治虫
記事冒頭: 手塚 治虫（てづか おさむ、1928年11月3日 - 1989年2月9日）は、日本の漫画家、
アニメーター、アニメーション監督。医師免許取得者であり、医学博士。戦後日本において
ストーリー漫画の第一人者として...

問い: 漫画家で医師免許も持っていた人っていましたよね、誰でしたっけ
答え: 手塚治虫

記事タイトル: 岡山県交通安全協会
記事冒頭: 一般財団法人岡山県交通安全協会は、岡山県内の地区交通安全協会を統括する...

スキップ"""


def parse(out: str) -> tuple[str, str] | None:
    """「問い: … / 答え: …」を取り出す。スキップまたは解釈できない形なら None。"""
    if "スキップ" in out[:20]:
        return None
    q = a = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(("問い:", "問い：")):
            q = line.split(":", 1)[-1].split("：", 1)[-1].strip()
        elif line.startswith(("答え:", "答え：")):
            a = line.split(":", 1)[-1].split("：", 1)[-1].strip()
    if not q or not a or len(a) > 60:
        return None
    return q, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="01b の出力")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--num_samples", type=int, default=0, help="0 で全件")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--model", required=True)
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--api_key", default=None)
    a = ap.parse_args()

    client = OpenAI(api_key=a.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
                    base_url=a.llm_base_url or None)

    rows = [json.loads(l) for l in open(a.input_file, encoding="utf-8") if l.strip()]
    # 02e と同じ方針：乱数種のみでシャッフルし前方を切る。件数を増やしても既存分が活きる
    random.Random(a.seed).shuffle(rows)
    if a.num_samples:
        rows = rows[: a.num_samples]
    # 添字は分割前のまま保つ（再開時の同一性のため）
    targets = [(i, r) for i, r in enumerate(rows) if i % a.nshard == a.shard]
    print(f"  候補 {len(rows):,} 件 / シャード {a.shard}/{a.nshard} 担当 {len(targets):,} 件",
          flush=True)

    # 既に書いた記事は飛ばす（--resume 相当）
    done = set()
    out_path = Path(a.output_file)
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["wiki_id"])
            except Exception:
                pass
        print(f"  既存 {len(done):,} 件をスキップ", flush=True)

    def work(idx_row):
        i, r = idx_row
        if r["wiki_id"] in done:
            return "skip_done", None
        try:
            out = call(client, a.model, QA_ROLE,
                       f"記事タイトル: {r['title']}\n記事冒頭: {r['lead']}",
                       a.effort)
        except Exception as e:  # noqa: BLE001
            return f"FAIL {e}", None
        qa = parse(out)
        if qa is None:
            return "skip_llm", None
        q, ans = qa
        return "ok", {"question": q, "answer": ans, "source": "ja_wiki",
                      "category": "wiki_generated", "wiki_id": r["wiki_id"],
                      "title": r["title"], "url": r["url"], "index": i}

    n_ok = n_skip_llm = n_skip_done = n_fail = 0
    import threading
    lock = threading.Lock()
    with out_path.open("a", encoding="utf-8") as w, \
         ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, t) for t in targets]
        for k, fu in enumerate(as_completed(futs), 1):
            status, rec = fu.result()
            if status == "ok":
                n_ok += 1
                with lock:
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    w.flush()
            elif status == "skip_llm":
                n_skip_llm += 1
            elif status == "skip_done":
                n_skip_done += 1
            else:
                n_fail += 1
                if n_fail <= 3:
                    print(f"    {status}", flush=True)
            if k % 200 == 0:
                print(f"    {k:,}/{len(targets):,}  採用 {n_ok:,} / "
                      f"スキップ {n_skip_llm:,} / 失敗 {n_fail:,}", flush=True)

    tried = n_ok + n_skip_llm + n_fail
    print(f"\n  採用 {n_ok:,} / LLM スキップ {n_skip_llm:,} / 既存 {n_skip_done:,} / 失敗 {n_fail:,}")
    if tried:
        print(f"  採用率 {100*n_ok/tried:.1f}%（既存を除く {tried:,} 件に対して）")


if __name__ == "__main__":
    main()
