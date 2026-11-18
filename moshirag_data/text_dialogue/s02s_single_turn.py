"""QA データセットの質問から、1 ターンの会話（ユーザの質問 1 つ → モシの応答）を LLM で作る。

MoshiRAG 論文（arXiv:2604.12928）の single-turn subset に合わせた工程。論文は、QA データセットの質問を
そのままユーザの発話にし、参照（reference LLM）とモシの応答（Moshi LLM）を LLM で作った（§4.1.2・付録 A）。
複数ターンの会話（s02）は挨拶や前置きから始まることがあるが、この工程の会話は挨拶なしで質問から始まる。

s02 と同じ 02-1（lead）・02-2（参照チャンク）・03（body）・04（相づち）のプロンプトを使い、出力も s02 と同じ形
（<stem>.json と <stem>.txt）にするので、s03 以降はそのまま使える。

入力（--input_file、1 行 1 問の jsonl）:
    {"id": "...", "question": "日本で二番目に大きな湖は？",
     "answers": ["霞ヶ浦"],            # 任意。--answer_hint のときだけ 02-2 に渡す
     "source": "...",                  # 任意。出所の名前（記録用）
     "wiki_id": "...", "title": "..."} # 任意。--wiki_dir があれば記事を 02-2 に渡す

検索の要否（--need）:
    always  すべて検索する（既定。論文の single-turn subset は、どの会話にも参照がある）
    judge   s02 と同じ 02 の判定に従う（不要なら参照なしで応答する。比較用）

Usage:
    python moshirag_data/text_dialogue/s02s_single_turn.py \\
        --input_file <質問の jsonl> --output_dir <出力先> --model "$NAME" \\
        --llm_base_url "http://localhost:${PORT}/v1" --num_samples 1000 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from libs.llm import call, first_line, strip_label
from libs.persona import (MOSHI_KNOWLEDGE, MOSHI_PERSONA, SPOKEN_STYLE, render_fillers,
                          sample_persona)
from libs.wiki import read_articles
from s02_dialogue import (BACKCHANNEL_ROLE, BODY_ROLE, CLARIFY_RULE, CODE_COMMIT, CODE_DIRTY,
                          LEAD_ROLE, NEED_ROLE, OPENINGS, REF_ROLE, build_script, parse_need,
                          parse_need_kind, parse_ref, plain_script, render_history, render_lines,
                          render_script, turn_lines)

_print_lock = threading.Lock()

# 話し始めのパターンは「質問」に固定する（constants/openings.json）。モシの応じ方もそこから取る
OPENING = "質問"
USER_GOAL = "1 ターンの質問"   # s02 の user_goals.json の目的ではない（記録用の名前）


def generate(item: dict, article: str, rng: random.Random, client, model, args) -> dict:
    q = " ".join(str(item["question"]).split())
    if not q:
        raise RuntimeError("質問が空")
    hist: list[tuple[str, str]] = [("user", q)]
    turn: dict = {"user": q}

    # ---- 02 検索の要否 ------------------------------------------------------------------
    if args.need == "always":
        turn.update(need=True, need_raw="（1 ターンの質問：常に検索する）", need_kind="")
    else:
        need_raw = call(client, model, NEED_ROLE.format(knowledge=MOSHI_KNOWLEDGE),
                        f"会話:\n{render_history(hist)}\n判定:", args.effort_need)
        verdict = parse_need(need_raw)
        turn.update(need=verdict == "要", need_raw=need_raw)
        if turn["need"]:
            turn["need_kind"] = parse_need_kind(need_raw)
        if verdict == "聞き返し":
            turn.update(clarify=True, reply_rule=CLARIFY_RULE)
    if not turn.get("reply_rule") and OPENINGS[OPENING].get("moshi"):
        turn["reply_rule"] = OPENINGS[OPENING]["moshi"]

    lead = ref = hint = ""
    basis: list[str] = []
    if turn["need"]:
        # ---- 02-1 lead -------------------------------------------------------------------
        lead_sys = LEAD_ROLE.format(persona=MOSHI_PERSONA, spoken_style=SPOKEN_STYLE,
                                    fillers=render_fillers())
        lead = strip_label(first_line(call(client, model, lead_sys,
                                           f"ここまでの会話:\n{render_history(hist)}\n前置き:",
                                           args.effort_lead)), "前置き", "モシ")
        # ---- 02-2 参照チャンク：記事があれば渡す。--answer_hint ならデータセットの正解も渡す ------
        ref_head = (f"記事タイトル: {item.get('title') or 'なし'}\n"
                    f"記事: {article or 'なし（記事は渡されていない。知識と推論で書く）'}\n")
        if args.answer_hint and item.get("answers"):
            hint = ("データセットの正解（参照チャンクはこれと矛盾しないように書く）: "
                    + "、".join(map(str, item["answers"])) + "\n")
        ref, basis = parse_ref(call(client, model, REF_ROLE,
                                    f"{ref_head}{hint}会話:\n{render_history(hist)}\n参照チャンク:",
                                    args.effort_ref))
        if not lead or not ref:
            raise RuntimeError("lead か参照チャンクが空")

    # ---- 03 body / 応答 ---------------------------------------------------------------------
    body_sys = BODY_ROLE.format(persona=MOSHI_PERSONA, spoken_style=SPOKEN_STYLE)
    p = f"参照チャンク: {ref or 'なし'}\n"
    if turn.get("reply_rule"):
        p += f"応じ方: {turn['reply_rule']}\n"
    p += f"ここまでの会話:\n{render_history(hist)}\n"
    if lead:
        p += f"モシ（話し始めている前置き）: {lead}\n"
    body = strip_label(first_line(call(client, model, body_sys, p + "続き:", args.effort_body)),
                       "続き", "モシ")
    if not body:
        raise RuntimeError("モシの応答が空")
    turn.update(lead=lead, reference=ref, body=body, ref_has_unknown="不明" in ref, ref_basis=basis)

    # ---- 04 相づち（s02 と同じ。失敗したら相づち無し）----------------------------------------
    lines = turn_lines(0, turn)
    if args.backchannel:
        try:
            raw = call(client, model, BACKCHANNEL_ROLE,
                       f"ここまでの会話:\n{render_script([])}\n\nこの往復の台本:\n{render_lines(lines)}\n\n出力:",
                       args.effort_backchannel)
            script, rejected, dropped = build_script(lines, raw)
            failed = False
        except Exception as e:  # noqa: BLE001
            raw, script, rejected, dropped, failed = f"（失敗）{e}", plain_script(lines), [], [], True
    else:
        raw, script, rejected, dropped, failed = "（相づちなし）", plain_script(lines), [], [], False
    turn["backchannel"] = {"raw": raw, "failed": failed, "n_lines": len(lines),
                           "n_backchannel": sum(1 for x in script if x["role"] == "backchannel"),
                           "rejected_lines": rejected, "dropped": dropped}

    record = [f"Human: {q}"]
    if turn["need"]:
        record += ["(augmented)", f"moshi (lead): {lead}", f"Reference: {ref}", f"moshi (body): {body}"]
    else:
        record += ["(unaugmented)", f"moshi: {body}"]

    return {
        "single_turn": True,
        "qa_id": item.get("id", ""),
        "answers": item.get("answers", []),
        "wiki_id": item.get("wiki_id", ""),
        "title": item.get("title", ""),
        "url": item.get("url"),
        "source": item.get("source", ""),
        "topic": item.get("title") or q,       # s03 が struct.json に写す
        "persona": sample_persona(rng),        # s03 が性別を struct.json の user_gender に写し、s04 が声を選ぶ
        "user_goal": USER_GOAL,
        "opening": OPENING,
        "scenario": {},
        "passage": "",
        "article_chars": len(article),
        "chunks": [ref] if turn["need"] else [],
        "turns": [turn],
        "record": record,
        "script": script,
        "n_backchannel": sum(1 for x in script if x["role"] == "backchannel"),
        "n_user_turns": 1,
        "n_augmented": int(turn["need"]),
        "n_clarify": int(bool(turn.get("clarify"))),
        "ended_by_eoc": False,
        "need_mode": args.need,
        "answer_hint": bool(hint),
        "model": model,
        "code_commit": CODE_COMMIT,
        "code_dirty": CODE_DIRTY,
    }


def main(args) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
                    base_url=args.llm_base_url or None)
    run(args, client)


def run(args, client) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rej = out / "rejected"
    with Path(args.input_file).open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.num_samples]
    targets = [(i, r) for i, r in enumerate(rows) if i % args.nshard == args.shard]
    print(f"  シャード {args.shard}/{args.nshard}: 担当 {len(targets)} 件 / 全 {len(rows)} 件", flush=True)
    articles: dict = {}
    if args.wiki_dir:
        ids = {r["wiki_id"] for _, r in targets if r.get("wiki_id")}
        articles = read_articles(args.wiki_dir, ids, args.article_chars) if ids else {}
        print(f"  記事 {len(articles)} 件を読んだ", flush=True)

    def work(i: int, item: dict):
        stem = f"{Path(args.input_file).stem}_single_{i:06d}"
        pj, pt = out / f"{stem}.json", out / f"{stem}.txt"
        if args.resume and (pj.exists() or (rej / f"{stem}.json").exists()):
            return stem, "skip"
        article = articles.get(item.get("wiki_id"), "")[: args.article_chars]
        for attempt in range(3):
            try:
                # ペルソナ（声の性別）は i から決める（再開しても同じになる）
                conv = generate(item, article, random.Random(args.seed * 1000 + i), client, args.model, args)
                conv["stem"] = stem
                pj.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
                pt.write_text("\n".join(conv["record"]) + "\n", encoding="utf-8")
                return stem, f"ok aug={conv['n_augmented']} bc={conv['n_backchannel']}"
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return stem, f"FAIL {e}"
        return stem, "FAIL"

    n = {"ok": 0, "skip": 0, "FAIL": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, r) for i, r in targets]
        for fu in as_completed(futs):
            stem, msg = fu.result()
            n[msg.split()[0]] += 1
            with _print_lock:
                print(f"[{msg.split()[0]}] {stem} {msg}", flush=True)
    print(f"\nDone: 作成 {n['ok']} / 既存 {n['skip']} / 失敗 {n['FAIL']} -> {out}")
    return n


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_file", required=True, help="質問の jsonl（1 行 1 問。書き方は冒頭の説明）")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--wiki_dir", default="", help="質問に wiki_id があるとき、記事を 02-2 に渡す（任意）")
    p.add_argument("--article_chars", type=int, default=6000, help="02-2 に渡す記事の長さ")
    p.add_argument("--need", choices=("always", "judge"), default="always",
                   help="always: すべて検索する（論文の single-turn subset と同じ）／ judge: 02 の判定に従う")
    p.add_argument("--answer_hint", action="store_true",
                   help="質問に answers があれば、データセットの正解として 02-2 に渡す")
    p.add_argument("--no_backchannel", dest="backchannel", action="store_false",
                   help="04 の相づちを書き足さない")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--model", default="llm-jp-4-32b-a3b-thinking")
    p.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default="")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--effort_need", default="low")
    p.add_argument("--effort_lead", default="low")
    p.add_argument("--effort_ref", default="low")
    p.add_argument("--effort_body", default="low")
    p.add_argument("--effort_backchannel", default="low")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
