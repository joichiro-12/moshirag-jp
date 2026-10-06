"""日本語 Wikipedia（ja_wiki）の記事から、LLM で問いと答えの組を作る。

どんな話題を聞かれても答えられるモデルにするため、記事を事前に絞り込まず、
ja_wiki の記事を乱数種でシャッフルして前から順に LLM に渡す。
LLM には記事の冒頭（--lead_chars 文字）を渡し、記事の適否判定と QA 生成を 1 回の呼び出しで行う。
事実を 1 つも取り出せない記事などは、LLM が「スキップ」と返し、出力に残らない。
出力はそのまま s02 の --input_file に渡せる。

入力（ja_wiki の *.jsonl.gz の各行）: {"text": "本文", "meta": {"id": ..., "title": ..., "url": ...}}
出力の各行: {"question": "...", "answer": "...", "source": "ja_wiki", "category": "wiki_generated",
            "wiki_id": "...", "title": "...", "url": "...", "index": ...}

Usage:
    （本番は jobs/qa_gen/wikiqa_main.pbs。vLLM のサーバを立ててから呼ぶ）
    $VENV/bin/python moshirag_data/s01c_wiki_qa.py \
        --wiki_dir /groups/gcg51557/experiments/0118_dedup_corpusv4_ja/data/all/cleaned/ja_wiki \
        --output_file data/japanese_kame/qa_pairs/wiki_qa.jsonl --num_samples 200000 \
        --model "$NAME" --llm_base_url "http://localhost:${PORT}/v1"
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# API の呼び出しと思考混入の検出は s02（旧 02e）の実装をそのまま使う。
# reasoning_effort の渡し方はサーバ実装に依存し、s02 は 3 通り試して当たったものを
# 覚える作りになっている。複製すると片方だけ直す事故が起きるので import する。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
_v5 = __import__("s02_dialogue")
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
    ap.add_argument("--wiki_dir", required=True, help="ja_wiki の *.jsonl.gz を置いたディレクトリ")
    ap.add_argument("--lead_chars", type=int, default=1000, help="LLM に渡す本文の冒頭の長さ")
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

    # 本文は冒頭だけ持つ（全記事を読むので、本文全体を持つとメモリが足りなくなる）
    rows = []
    files = sorted(glob.glob(os.path.join(a.wiki_dir, "*.jsonl.gz")))
    if not files:
        sys.exit(f"{a.wiki_dir} に *.jsonl.gz が無い")
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                m = d["meta"]
                rows.append({"wiki_id": m.get("id"), "title": m.get("title", ""),
                             "url": m.get("url"), "lead": d["text"][: a.lead_chars]})
    # s02 と同じ方針：乱数種のみでシャッフルし前方を切る。件数を増やしても既存分が活きる
    random.Random(a.seed).shuffle(rows)
    if a.num_samples:
        rows = rows[: a.num_samples]
    # 添字は分割前のまま保つ（再開時の同一性のため）
    targets = [(i, r) for i, r in enumerate(rows) if i % a.nshard == a.shard]
    print(f"  記事 {len(rows):,} 件 / "
          f"シャード {a.shard}/{a.nshard} 担当 {len(targets):,} 件", flush=True)

    # 出力はシャードごとに分ける。共有ファイルへの並行追記はノードをまたぐと
    # 保護できないため（スレッドロックはプロセス内にしか効かない）。
    out_path = Path(a.output_file)
    if a.nshard > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}.s{a.shard:03d}of{a.nshard:03d}{out_path.suffix}")
        print(f"  出力: {out_path.name}", flush=True)

    # 既に書いた記事は飛ばす（--resume 相当）
    done = set()
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
