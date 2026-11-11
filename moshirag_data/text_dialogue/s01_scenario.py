"""日本語 Wikipedia（ja_wiki）の記事から、LLM で会話のシナリオを作る。

入力（ja_wiki の *.jsonl.gz の各行）: {"text": "本文", "meta": {"id": ..., "title": ..., "url": ...}}
出力の各行: {"wiki_id": ..., "title": ..., "url": ..., "index": ..., "source": "ja_wiki",
            "article_chars": ..., "persona": {...}, "user_goal": "...", "scenario": {...}}

--source tasks では記事を使わず、constants/task_seeds.json の依頼の種類・お題・場面・名字からシナリオを作る。
出力の各行: {"wiki_id": null, "task_id": "task_000000", "title": 依頼の種類, "url": null, "index": ...,
            "source": "tasks", "task": {"種類": ..., "お題": ..., "場面": ..., "名字": ...}, "article_chars": 0,
            "persona": {...}, "user_goal": "...", "scenario": {...}}
目的は --goals で絞る（既定は 長い依頼）。読み上げる一節が要る目的（読解）は使えない。scenario_rule が記事を前提に
書かれた目的（作業依頼・今の情報・探す など）は、記事なしでは指示が食い違うので使わない。

    $VENV/bin/python moshirag_data/text_dialogue/s01_scenario.py --source tasks \
        --output_file data/moshirag_jp/task_scenario.jsonl --num_samples 10000 \
        --goals 長い依頼 --model "$NAME" --llm_base_url "http://localhost:${PORT}/v1"

Usage:
    $VENV/bin/python moshirag_data/text_dialogue/s01_scenario.py \
        --wiki_dir /groups/gcg51557/experiments/0118_dedup_corpusv4_ja/data/all/cleaned/ja_wiki \
        --output_file data/japanese_kame/qa_pairs/wiki_scenario.jsonl --num_samples 200000 \
        --model "$NAME" --llm_base_url "http://localhost:${PORT}/v1"
"""
from __future__ import annotations
import argparse, json, os, random, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from libs import llm, persona, wiki

SCENARIO_KEYS = ["状況", "最初の関心", "派生しうる関心", "なぜ関心があるのか"]

SCENARIO_ROLE = persona.load_prompt("0_scenario.txt")
# 記事を使わないシナリオ（--source tasks）：依頼の種類・お題・場面・名字を種にする
TASK_SCENARIO_ROLE = persona.load_prompt("0_scenario_task.txt")
TASK_SEEDS: dict = persona.load_constant("task_seeds.json")
TASK_DEFAULT_GOALS = ["長い依頼"]


def sample_task(rng: random.Random) -> dict:
    """依頼の種類（重みで）、場面・お題・名字（同じ確率で）を選ぶ。
    お題と名字は後から足したので、種類と場面の選び方（乱数を引く順）は変えずに、その後で引く。"""
    kinds = TASK_SEEDS["依頼の種類"]
    kind = rng.choices(list(kinds), weights=[kinds[k]["weight"] for k in kinds])[0]
    scene = rng.choice(TASK_SEEDS["場面"])
    return {"種類": kind, "お題": rng.choice(kinds[kind]["お題"]), "場面": scene, "名字": rng.choice(TASK_SEEDS["名字"])}


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
    ap.add_argument("--source", choices=["wiki", "tasks"], default="wiki",
                    help="wiki：記事からシナリオを作る。tasks：記事を使わず、依頼の種類と場面から作る")
    ap.add_argument("--goals", default="",
                    help="この目的の中だけから選ぶ（カンマ区切り、重みは user_goals.json のまま）。"
                         "空なら、wiki ではすべての目的、tasks では " + "・".join(TASK_DEFAULT_GOALS))
    ap.add_argument("--wiki_dir", default=None, help="ja_wiki の *.jsonl.gz を置いたディレクトリ（--source wiki で要る）")
    ap.add_argument("--article_chars", type=int, default=6000,
                    help="LLM に渡す本文の長さ（先頭から）。s02 の 02-2 も同じ長さで切る")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--num_samples", type=int, default=0, help="0 で全件（--source tasks では作る件数で、要る）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--model", required=True)
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--api_key", default=None)
    a = ap.parse_args()

    goals = [g.strip() for g in re.split(r"[,、]", a.goals) if g.strip()] or None
    if a.source == "tasks":
        goals = goals or TASK_DEFAULT_GOALS
    unknown = [g for g in goals or [] if g not in persona.USER_GOALS]
    if unknown:
        raise SystemExit(f"user_goals.json に無い目的: {unknown}")
    if a.source == "tasks":
        # 読解のように記事の一節を読み上げる目的は、記事が無いと作れない
        need_article = [g for g in goals if persona.USER_GOALS[g].get("passage")]
        if need_article:
            raise SystemExit(f"--source tasks では、記事の一節が要る目的は使えない: {need_article}")
        if not a.num_samples:
            raise SystemExit("--source tasks では --num_samples（作る件数）が要る")
    elif not a.wiki_dir:
        raise SystemExit("--source wiki では --wiki_dir が要る")

    from openai import OpenAI
    client = OpenAI(api_key=a.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
                    base_url=a.llm_base_url or None)

    if a.source == "tasks":
        # 種は会話ごとの添字 i から選ぶので、件数を増やしても既存分が活きる
        rows = [{"task_id": f"task_{i:06d}"} for i in range(a.num_samples)]
    else:
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
    print(f"  {'依頼の種' if a.source == 'tasks' else '記事'} {len(rows):,} 件 / "
          f"シャード {a.shard}/{a.nshard} 担当 {len(targets):,} 件", flush=True)

    out_path = Path(a.output_file)
    if a.nshard > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}.s{a.shard:03d}of{a.nshard:03d}{out_path.suffix}")
        print(f"  出力: {out_path.name}", flush=True)

    # 既に書いた記事（tasks では依頼の種）は飛ばす
    done = set()
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            try:
                rec = json.loads(l)
                done.add(rec.get("wiki_id") or rec["task_id"])
            except Exception:
                pass
        print(f"  既存 {len(done):,} 件をスキップ", flush=True)

    texts = {} if a.source == "tasks" else \
        wiki.read_articles(a.wiki_dir, {r["wiki_id"] for _, r in targets} - done, a.article_chars)

    def work(idx_row):
        i, r = idx_row
        if (r.get("wiki_id") or r["task_id"]) in done:
            return "skip_done", None
        rng = random.Random(a.seed * 1000003 + i)
        user_persona = persona.sample_persona(rng)
        goal = persona.sample_user_goal(rng, goals)
        if a.source == "tasks":
            task = sample_task(rng)
            role = TASK_SCENARIO_ROLE
            head = (f"依頼の種類: {task['種類']}（{TASK_SEEDS['依頼の種類'][task['種類']]['中身']}）\n"
                    f"お題: {task['お題']}\n場面: {task['場面']}\n人名を出すときの名字: {task['名字']}\n")
        else:
            role = SCENARIO_ROLE
            head = f"記事タイトル: {r['title']}\n記事: {texts[r['wiki_id']]}\n"
        # 目的ごとのシナリオの書き方（user_goals.json の scenario_rule）は、その目的が選ばれたときだけ渡す
        rule = persona.USER_GOALS[goal].get("scenario_rule")
        try:
            out = llm.call(client, a.model,
                           role.format(moshi=persona.MOSHI_NAME_ROLE, goal=persona.USER_GOALS[goal]["goal"]),
                           head + f"人物: {persona.render_persona(user_persona)}\nユーザの目的: {goal}\n"
                           + (f"この目的でのシナリオの書き方: {rule}\n" if rule else ""),
                           a.effort)
        except Exception as e:  # noqa: BLE001
            return f"FAIL {e}", None
        sc = parse(out)
        if sc is None:
            return f"FAIL 4 項目がそろわない: {out[:60]!r}", None
        if a.source == "tasks":
            return "ok", {"wiki_id": None, "task_id": r["task_id"], "title": task["種類"], "url": None,
                          "index": i, "source": "tasks", "task": task,
                          "article_chars": 0, "persona": user_persona, "user_goal": goal, "scenario": sc}
        return "ok", {"wiki_id": r["wiki_id"], "title": r["title"], "url": r["url"],
                      "index": i, "source": "ja_wiki", "article_chars": a.article_chars,
                      "persona": user_persona, "user_goal": goal, "scenario": sc}

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
