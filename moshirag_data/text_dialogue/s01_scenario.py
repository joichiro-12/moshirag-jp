"""日本語 Wikipedia（ja_wiki）の記事から、LLM で会話のシナリオを作る。

入力（ja_wiki の *.jsonl.gz の各行）: {"text": "本文", "meta": {"id": ..., "title": ..., "url": ...}}
出力の各行: {"wiki_id": ..., "title": ..., "url": ..., "index": ..., "source": "ja_wiki",
            "article_chars": ..., "persona": {...}, "type": "...", "scenario": {...}}

Usage:
    $VENV/bin/python moshirag_data/text_dialogue/s01_scenario.py \
        --wiki_dir /groups/gcg51557/experiments/0118_dedup_corpusv4_ja/data/all/cleaned/ja_wiki \
        --output_file data/japanese_kame/qa_pairs/wiki_scenario.jsonl --num_samples 200000 \
        --model "$NAME" --llm_base_url "http://localhost:${PORT}/v1"
"""
from __future__ import annotations
import argparse, json, os, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from libs import llm, persona, wiki

SCENARIO_KEYS = ["状況", "最初の関心", "派生しうる関心", "なぜ関心があるのか"]

SCENARIO_ROLE = persona.load_prompt("0_scenario.txt")


def parse(out: str) -> dict | None:
    """4 項目を取り出す。項目が欠けていれば None。"""
    sc = {}
    for line in out.splitlines():
        line = line.strip().lstrip("-・ ")
        for k in SCENARIO_KEYS:
            if line.startswith((f"{k}:", f"{k}：")):
                sc[k] = line[len(k) + 1:].strip()
    if any(not sc.get(k) for k in SCENARIO_KEYS):
        return None
    return {k: sc[k] for k in SCENARIO_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki_dir", required=True, help="ja_wiki の *.jsonl.gz を置いたディレクトリ")
    ap.add_argument("--article_chars", type=int, default=6000,
                    help="LLM に渡す本文の長さ（先頭から）。s02 の 02-2 も同じ長さで切る")
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

    # 1 周目は記事の ID とタイトルだけを持ち、本文は担当分だけ 2 周目で読む
    # （全記事の本文を持つとメモリが足りなくなる）
    rows = [{"wiki_id": d["meta"].get("id"), "title": d["meta"].get("title", ""),
             "url": d["meta"].get("url")} for d in wiki.iter_wiki(a.wiki_dir)]
    # 乱数種のみでシャッフルし前方を切る。件数を増やしても既存分が活きる
    random.Random(a.seed).shuffle(rows)
    if a.num_samples:
        rows = rows[: a.num_samples]
    # 添字は分割前のまま保つ（再開時の同一性のため）
    targets = [(i, r) for i, r in enumerate(rows) if i % a.nshard == a.shard]
    print(f"  記事 {len(rows):,} 件 / "
          f"シャード {a.shard}/{a.nshard} 担当 {len(targets):,} 件", flush=True)

    out_path = Path(a.output_file)
    if a.nshard > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}.s{a.shard:03d}of{a.nshard:03d}{out_path.suffix}")
        print(f"  出力: {out_path.name}", flush=True)

    # 既に書いた記事は飛ばす
    done = set()
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["wiki_id"])
            except Exception:
                pass
        print(f"  既存 {len(done):,} 件をスキップ", flush=True)

    texts = wiki.read_articles(a.wiki_dir, {r["wiki_id"] for _, r in targets} - done, a.article_chars)

    def work(idx_row):
        i, r = idx_row
        if r["wiki_id"] in done:
            return "skip_done", None
        rng = random.Random(a.seed * 1000003 + i)
        user_persona = persona.sample_persona(rng)
        ty = persona.sample_type(rng)
        article = texts[r["wiki_id"]]
        try:
            out = llm.call(client, a.model,
                           SCENARIO_ROLE.format(moshi=persona.MOSHI_NAME_ROLE,
                                                type_desc=persona.TYPES[ty]["desc"]),
                           f"記事タイトル: {r['title']}\n記事: {article}\n"
                           f"人物: {persona.render_persona(user_persona)}\n会話の型: {ty}\n",
                           a.effort)
        except Exception as e:  # noqa: BLE001
            return f"FAIL {e}", None
        sc = parse(out)
        if sc is None:
            return f"FAIL 4 項目がそろわない: {out[:60]!r}", None
        return "ok", {"wiki_id": r["wiki_id"], "title": r["title"], "url": r["url"],
                      "index": i, "source": "ja_wiki", "article_chars": a.article_chars,
                      "persona": user_persona, "type": ty, "scenario": sc}

    n_ok = n_skip_done = n_fail = 0
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
            elif status == "skip_done":
                n_skip_done += 1
            else:
                n_fail += 1
                if n_fail <= 3:
                    print(f"    {status}", flush=True)
            if k % 200 == 0:
                print(f"    {k:,}/{len(targets):,}  作成 {n_ok:,} / 失敗 {n_fail:,}", flush=True)

    print(f"\n  作成 {n_ok:,} / 既存 {n_skip_done:,} / 失敗 {n_fail:,}")


if __name__ == "__main__":
    main()
