"""シナリオ（s01）から、MoshiRAG 形式の日本語の会話記録を LLM で作る。

Usage:
    （本番は jobs/qa_gen/gen_wiki_main.pbs。vLLM のサーバを立ててから呼ぶ）
    $VENV/bin/python moshirag_data/text_dialogue/s02_dialogue.py \
        --input_file data/japanese_kame/qa_pairs/wiki_scenario.jsonl \
        --wiki_dir /groups/gcg51557/experiments/0118_dedup_corpusv4_ja/data/all/cleaned/ja_wiki \
        --output_dir data/moshirag_jp/wiki_main --model "$NAME" \
        --llm_base_url "http://localhost:${PORT}/v1" --resume
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from libs.llm import call, first_line, strip_label
from libs.persona import (SPOKEN_STYLE, MOSHI_KNOWLEDGE, MOSHI_NAME_ROLE, MOSHI_PERSONA, USER_GOALS,
                          load_constant, load_prompt, render_fillers, render_persona,
                          render_scenario)
from libs.wiki import read_articles

_print_lock = threading.Lock()


def _git(*a: str) -> str:
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


# Recorded into every output so that data can always be traced back to the exact
# prompts that produced it. data/ is gitignored, so a version string written by hand is
# not enough -- v2 was lost that way.
CODE_COMMIT = _git("rev-parse", "--short", "HEAD")
CODE_DIRTY = bool(_git("status", "--porcelain"))


# ======================================================================================
# 話し始めのパターン（会話ごとに乱択）
# ======================================================================================
# constants/openings.json。キーはパターン名
#   user: 01 の最初の発話に渡す指定。内容は縛らず、形だけを決める
#   moshi: 最初の応答の応じ方。Moshi の仕様なので明確に決める。null なら指定しない
#   search: 最初の応答の前に 02 で検索の要否を判定するか。false なら検索せずに moshi の応じ方で返す
#           （挨拶だけ・雑談など、答えるべき中身がまだ無いパターン）
#   weight: 出現の重み
OPENINGS: dict = load_constant("openings.json")

# 02 で「聞き返し」と判定したときの応じ方：constants/moshi_reply_rules.json
CLARIFY_RULE: str = load_constant("moshi_reply_rules.json")["聞き返し"]


def sample_opening(rng: random.Random) -> str:
    names = list(OPENINGS)
    return rng.choices(names, weights=[OPENINGS[n]["weight"] for n in names])[0]


# ======================================================================================
# プロンプト
# ======================================================================================
# ---- 01 ユーザ発話 ------------------------------------------------------------------
USER_ROLE = load_prompt("01_user.txt")

# ---- 02 検索の要否 ------------------------------------------------------------------
NEED_ROLE = load_prompt("02_need.txt")

# ---- 02-1 lead ----------------------------------------------------------------------
LEAD_ROLE = load_prompt("02-1_lead.txt")

# ---- 02-2 参照チャンク ----------------------------------------------------------------------
REF_ROLE = load_prompt("02-2_ref.txt")

# ---- 03 body / 応答 -----------------------------------------------------------------
BODY_ROLE = load_prompt("03_body.txt")

# ---- 04 相づち ----------------------------------------------------------------------
BACKCHANNEL_ROLE = load_prompt("04_backchannel.txt")



# ======================================================================================
# 04 相づち：1 往復ごとに、その往復の発話の途中に聞き手の相づちを LLM で書き足す
# ======================================================================================
# LLM には発話を［］の相づちで区切らせるだけで、発話の文字は変えさせない。結果は会話の "script" に書き、
# s03 はそれを台本にする（相づちは会話履歴 hist には入れないので、01〜03 には見えない）。［］を取り除いて元の発話と一致しない行と、前置きの行は、相づち無しの元の
# 発話にする（"rejected_lines"）。長すぎる相づち、前後の発話が短すぎる相づち、モシが打つ です・ます調で
# ない相づちは、その相づちだけを捨てて前後をつなぐ（"dropped" に理由を記録）。前置きと本題は別の行なので、その間に
# 相づちは入らない（s06 の d_lead が変わらない）

# 台本の行の話し手の書き方。role は s03 が lead・body の位置を求めるのに使う
BC_LABEL = {"user": "人", "lead": "モシ（前置き）", "body": "モシ（本題）", "reply": "モシ"}
BC_SPEAKER = {"user": "A", "lead": "B", "body": "B", "reply": "B"}   # A=人間 / B=moshi
BC_LISTENER = {"A": "B", "B": "A"}
_BC_LINE = re.compile(r"^\s*(\d+)\s*(?:" + "|".join(re.escape(v) for v in BC_LABEL.values()) + r")\s*[:：]\s*(.*)$")
_BC_MARK = re.compile(r"［([^］]*)］")
_SPACE = re.compile(r"\s+")
MAX_BC_CHARS = 7   # 相づちの長さの上限（プロンプトでは 1〜6 字。そうなんですね が入るように少し余裕を持たせる）
MIN_SEG_CHARS = 6  # 相づちの前後の発話の長さの下限（句読点を除く。プロンプトでは 8 字以上）
# モシが打ってよい相づち（ペルソナが です・ます調なので、うん・そうだね などは使わせない）
MOSHI_BACKCHANNELS = {"はい", "ええ", "はいはい", "なるほど", "へえ", "そうですね", "そうですよね",
                      "そうなんですね", "そうなんです"}
_NOT_COUNTED = re.compile(r"[、。？！…\s]")


def turn_lines(k: int, t: dict) -> list[dict]:
    """1 往復を台本の行にする。turn は s02 のターンの index。"""
    lines = [{"turn": k, "role": "user", "text": t["user"]}]
    if t.get("body") is None:
        return lines
    if t["need"]:
        return lines + [{"turn": k, "role": "lead", "text": t["lead"]},
                        {"turn": k, "role": "body", "text": t["body"]}]
    return lines + [{"turn": k, "role": "reply", "text": t["body"]}]


def render_lines(lines: list[dict]) -> str:
    return "\n".join(f"{i + 1} {BC_LABEL[l['role']]}: {l['text']}" for i, l in enumerate(lines))


def render_script(script: list[dict]) -> str:
    """相づちを書き足した台本を、相づちを［］に入れた形で 1 発話 1 行にする（04 に会話履歴として見せる）。"""
    out, key = [], None
    for x in script:
        if x["role"] == "backchannel":
            out[-1] += f"［{x['text']}］"
        elif (x["turn"], x["role"]) == key:
            out[-1] += x["text"]
        else:
            key = (x["turn"], x["role"])
            out.append(f"{BC_LABEL[x['role']]}: {x['text']}")
    return "\n".join(out) or "（まだ無い）"


def plain_script(lines: list[dict]) -> list[dict]:
    """相づち無しの台本（04 が失敗したとき、最後の EOC の発話）。"""
    return [{"speaker": BC_SPEAKER[l["role"]], "text": l["text"], "turn": l["turn"], "role": l["role"]}
            for l in lines]


def split_line(orig: str, out: str, allowed: set[str] | None = None) -> tuple[list[str], list[str]] | None:
    """LLM が区切った行を（[発話, 相づち, 発話, …], 捨てた相づちの理由の一覧）にする。
    allowed があれば、それ以外の相づちは捨てる。発話の文字が変わっていれば None。"""
    parts = _BC_MARK.split(out.strip())
    texts, bcs = parts[0::2], [b.strip() for b in parts[1::2]]
    if _SPACE.sub("", "".join(texts)) != _SPACE.sub("", orig):
        return None
    # 決まりに合わない相づちを捨てて、前後の発話をつなぐ。前から順に見て、
    # 相づちの前（つないだ後のもの）と後の発話が十分に長いものだけを残す
    out_parts, dropped = [texts[0]], []
    for b, t in zip(bcs, texts[1:]):
        if not 0 < len(b) <= MAX_BC_CHARS:
            why = "長さ"
        elif allowed is not None and b.rstrip("。、！") not in allowed:
            why = "口調"
        elif len(_NOT_COUNTED.sub("", out_parts[-1])) < MIN_SEG_CHARS:
            why = "前が短い"
        elif len(_NOT_COUNTED.sub("", t)) < MIN_SEG_CHARS:
            why = "後が短い"
        else:
            out_parts += [b, t]
            continue
        out_parts[-1] += t
        dropped.append(why)
    return out_parts, dropped


def build_script(lines: list[dict], out: str) -> tuple[list[dict], list[int], list[str]]:
    """LLM の出力から台本を組み立てる。戻り値は（台本, 元の発話に戻した行の番号, 捨てた相づちの理由の一覧）。"""
    got = {}
    for raw in out.splitlines():
        m = _BC_LINE.match(raw)
        if m:
            got.setdefault(int(m.group(1)) - 1, m.group(2))
    script, rejected, dropped = [], [], []
    for i, l in enumerate(lines):
        spk = BC_SPEAKER[l["role"]]
        allowed = MOSHI_BACKCHANNELS if BC_LISTENER[spk] == "B" else None
        res = split_line(l["text"], got[i], allowed) if i in got else None
        if res is None or (l["role"] == "lead" and len(res[0]) > 1):
            rejected.append(i + 1)
            parts = [l["text"]]
        else:
            parts = res[0]
            dropped += res[1]
        for j, p in enumerate(parts):
            if j % 2:
                script.append({"speaker": BC_LISTENER[spk], "text": p, "turn": l["turn"], "role": "backchannel"})
            else:
                script.append({"speaker": spk, "text": p, "turn": l["turn"], "role": l["role"]})
    return script, rejected, dropped


# ======================================================================================
# 1 会話の生成
# ======================================================================================
SPEAKER = {"user": "人", "moshi": "モシ"}


def render_history(hist: list[tuple[str, str]]) -> str:
    # 会話はユーザから始まるので、最初の 01 では履歴が空
    return "\n".join(f"{SPEAKER[w]}: {t}" for w, t in hist) or "（まだ無い）"


def parse_ref(out: str) -> tuple[str, list[str]]:
    """02-2 の出力から（参照チャンク, 根拠）を取り出す。根拠は「記事」「知識」「推論」の部分集合。
    根拠の行が無くても参照チャンクは使う（根拠は分析用の記録なので）。"""
    ref, basis = "", []
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("根拠:", "根拠：")):
            v = line[3:]
            basis = [b for b in ("記事", "知識", "推論") if b in v]
        elif not ref:
            ref = strip_label(line, "参照チャンク", "参照")
    return ref, basis


def parse_need(out: str) -> str:
    """「要」「不要」「聞き返し」のどれかを返す。"""
    s = first_line(strip_label(out, "判定"))
    for v in ("聞き返し", "不要", "要"):   # 「不要」は「要」を含むので先に見る
        if v in s:
            return v
    raise RuntimeError(f"検索の要否を解釈できない: {out[:60]}")


def generate(sc: dict, article: str, opening: str, client, model, args) -> dict:
    persona_txt = render_persona(sc["persona"])
    scenario_txt = render_scenario(sc["scenario"])
    goal = sc["user_goal"]
    user_sys = USER_ROLE.format(moshi=MOSHI_NAME_ROLE, end_condition=USER_GOALS[goal]["end_condition"],
                                spoken_style=SPOKEN_STYLE)
    user_head = f"人物: {persona_txt}\nユーザの目的: {goal}\nシナリオ:\n{scenario_txt}\n"
    need_sys = NEED_ROLE.format(knowledge=MOSHI_KNOWLEDGE)
    lead_sys = LEAD_ROLE.format(persona=MOSHI_PERSONA, spoken_style=SPOKEN_STYLE, fillers=render_fillers())
    body_sys = BODY_ROLE.format(persona=MOSHI_PERSONA, spoken_style=SPOKEN_STYLE)
    ref_head = f"記事タイトル: {sc['title']}\n記事: {article}\n"

    hist: list[tuple[str, str]] = []   # 会話はユーザから始める
    turns: list[dict] = []
    script: list[dict] = []            # 相づちを書き足した台本（04）
    ended = False
    for k in range(args.max_turns):
        # ---- 01 ユーザ発話：シナリオ・会話履歴（最初だけ話し始めの形を指定）--------------
        form = f"最初の発話の形: {opening}（{OPENINGS[opening]['user']}）\n" if k == 0 else ""
        u = first_line(call(client, model, user_sys,
                            f"{user_head}{form}ここまでの会話:\n{render_history(hist)}\n人:",
                            args.effort_user))
        u = strip_label(u, "人", "ユーザ", "人間")
        if u.rstrip().endswith("EOC"):
            u = u.rstrip()[:-3].rstrip("　 ")
            # --min_turns より前の EOC は無視して続ける
            if k >= args.min_turns:
                ended = True
                if u:
                    turns.append({"user": u})
                    hist.append(("user", u))
                break
        if not u:
            break
        hist.append(("user", u))
        turn = {"user": u}

        # ---- 02 検索の要否：会話履歴・知識の範囲 --------------------------------------
        # 最初の応答は話し始めのパターンの応じ方に従う。search が false のパターン（挨拶だけ・雑談など）は
        # 答えるべき中身がまだ無いので検索しない。true のパターン（質問など）は 02 で判定してから応じ方を渡す
        op = OPENINGS[opening] if k == 0 else {}
        rule = op.get("moshi")
        if rule and not op.get("search", False):
            turn.update(need=False, need_raw=f"（最初の応答・{opening}：検索しない）", reply_rule=rule)
        else:
            need_raw = call(client, model, need_sys,
                            f"会話:\n{render_history(hist)}\n判定:", args.effort_need)
            verdict = parse_need(need_raw)
            turn["need"] = verdict == "要"
            turn["need_raw"] = need_raw
            if verdict == "聞き返し":   # 検索せず、何のことかを聞き返す
                turn.update(clarify=True, reply_rule=CLARIFY_RULE)
            elif rule:
                turn["reply_rule"] = rule

        lead = ref = ""
        basis: list[str] = []
        if turn["need"]:
            # ---- 02-1 lead：会話履歴・ペルソナ ----------------------------------------
            lead = first_line(call(client, model, lead_sys,
                                   f"ここまでの会話:\n{render_history(hist)}\n前置き:",
                                   args.effort_lead))
            lead = strip_label(lead, "前置き", "モシ")
            # ---- 02-2 参照チャンク：直前のユーザ発話までの会話履歴・記事（lead は見せない）------
            ref, basis = parse_ref(call(client, model, REF_ROLE,
                                        f"{ref_head}会話:\n{render_history(hist)}\n参照チャンク:",
                                        args.effort_ref))
            if not lead or not ref:
                raise RuntimeError("lead か参照チャンクが空")

        # ---- 03 body / 応答：lead までの会話履歴・ペルソナ・参照チャンク ----------------------
        q = f"参照チャンク: {ref or 'なし'}\n"
        if turn.get("reply_rule"):
            q += f"応じ方: {turn['reply_rule']}\n"
        q += f"ここまでの会話:\n{render_history(hist)}\n"
        if lead:
            q += f"モシ（話し始めている前置き）: {lead}\n"
        body = first_line(call(client, model, body_sys, q + "続き:", args.effort_body))
        body = strip_label(body, "続き", "モシ")
        if not body:
            raise RuntimeError("モシの応答が空")
        turn.update(lead=lead, reference=ref, body=body, ref_has_unknown="不明" in ref,
                    ref_basis=basis)

        # ---- 04 相づち：相づち込みの会話履歴・この往復の発話 ----------------------------------
        # 失敗しても会話は作り直さず、この往復は相づち無しにする
        lines = turn_lines(k, turn)
        try:
            raw = call(client, model, BACKCHANNEL_ROLE,
                       f"ここまでの会話:\n{render_script(script)}\n\nこの往復の台本:\n{render_lines(lines)}\n\n出力:",
                       args.effort_backchannel)
            seg, rejected, dropped = build_script(lines, raw)
            failed = False
        except Exception as e:  # noqa: BLE001
            raw, seg, rejected, dropped = f"（失敗）{e}", plain_script(lines), [], []
            failed = True
        script += seg
        # rejected_lines：発話の文字が変わったので元に戻した行。dropped：捨てた相づちの理由
        turn["backchannel"] = {"raw": raw, "failed": failed, "n_lines": len(lines),
                               "n_backchannel": sum(1 for x in seg if x["role"] == "backchannel"),
                               "rejected_lines": rejected, "dropped": dropped}
        turns.append(turn)
        hist.append(("moshi", lead + body))

    # 最後の EOC の発話（モシの応答が無い）は相づちを入れずに台本に足す
    if turns and turns[-1].get("body") is None:
        script += plain_script(turn_lines(len(turns) - 1, turns[-1]))

    # ---- s03 が読む形式に組み立てる ----------------------------------------------------
    record = []
    for t in turns:
        record.append(f"Human: {t['user']}")
        if t.get("body") is None:
            continue
        if t["need"]:
            record += ["(augmented)", f"moshi (lead): {t['lead']}",
                       f"Reference: {t['reference']}", f"moshi (body): {t['body']}"]
        else:
            record += ["(unaugmented)", f"moshi: {t['body']}"]

    return {
        "wiki_id": sc["wiki_id"],
        "title": sc["title"],
        "url": sc.get("url"),
        "source": sc.get("source", ""),
        "topic": sc["title"],                 # s03 が struct.json に写す
        "persona": sc["persona"],
        "user_goal": goal,
        "opening": opening,
        "scenario": sc["scenario"],
        "article_chars": len(article),
        "chunks": [t["reference"] for t in turns if t.get("need")],   # s03 が struct.json に写す
        "turns": turns,
        "record": record,
        "script": script,
        "n_backchannel": sum(1 for x in script if x["role"] == "backchannel"),
        "n_user_turns": len(turns),
        "n_augmented": sum(1 for t in turns if t.get("need")),
        "n_clarify": sum(1 for t in turns if t.get("clarify")),
        "ended_by_eoc": ended,
        "model": model,
        "code_commit": CODE_COMMIT,
        "code_dirty": CODE_DIRTY,
    }


def main(args) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
                    base_url=args.llm_base_url or None)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 開発用の事後検査（dev/check_dialogue.py）で不合格になった会話の置き場。
    # 再開のときは、ここにある会話も作成済みとして飛ばす（作り直して LLM を呼び直さないため）
    rej = out / "rejected"

    with Path(args.input_file).open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.num_samples]

    # シャード分割。添字 i は分割前のものを保つ（stem が i から決まるため、
    # 分割の有無で生成結果が変わらない）。
    targets = [(i, sc) for i, sc in enumerate(rows) if i % args.nshard == args.shard]
    print(f"  シャード {args.shard}/{args.nshard}: 担当 {len(targets)} 件 / 全 {len(rows)} 件",
          flush=True)
    articles = read_articles(args.wiki_dir, {sc["wiki_id"] for _, sc in targets},
                             max([sc.get("article_chars", args.article_chars) for _, sc in targets],
                                 default=args.article_chars))
    print(f"  記事 {len(articles)} 件を読んだ", flush=True)

    def work(i: int, sc: dict):
        stem = f"{Path(args.input_file).stem}_{i:06d}"
        pj, pt = out / f"{stem}.json", out / f"{stem}.txt"
        if args.resume and (pj.exists() or (rej / f"{stem}.json").exists()):
            return stem, "skip"
        if sc["wiki_id"] not in articles:
            return stem, "FAIL 記事が見つからない"
        # 記事は s01 がシナリオを作ったときと同じ長さに切る
        article = articles[sc["wiki_id"]][: sc.get("article_chars", args.article_chars)]
        # 話し始めのパターンは i から決める（再開しても同じになる）
        opening = sample_opening(random.Random(args.seed * 1000 + i))
        for attempt in range(3):
            try:
                conv = generate(sc, article, opening, client, args.model, args)
                conv["stem"] = stem
                pj.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
                pt.write_text("\n".join(conv["record"]) + "\n", encoding="utf-8")
                return stem, (f"ok turns={conv['n_user_turns']} aug={conv['n_augmented']} clarify={conv['n_clarify']} "
                              f"eoc={conv['ended_by_eoc']} bc={conv['n_backchannel']}")
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return stem, f"FAIL {e}"
        return stem, "FAIL"

    n = {"ok": 0, "skip": 0, "FAIL": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, sc) for i, sc in targets]
        for fu in as_completed(futs):
            stem, msg = fu.result()
            n[msg.split()[0]] += 1
            with _print_lock:
                print(f"[{msg.split()[0]}] {stem} {msg}", flush=True)
    print(f"\nDone: 作成 {n['ok']} / 既存 {n['skip']} / 失敗 {n['FAIL']}"
          f" -> {out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", required=True, help="s01 の出力（シナリオの jsonl）")
    p.add_argument("--wiki_dir", required=True, help="ja_wiki の *.jsonl.gz を置いたディレクトリ（02-2 用）")
    p.add_argument("--article_chars", type=int, default=6000,
                   help="シナリオに article_chars が無いときに使う記事の長さ")
    p.add_argument("--shard", type=int, default=0,
                   help="担当するシャード番号（0 起算）")
    p.add_argument("--nshard", type=int, default=1,
                   help="シャード総数。i %% nshard == shard の会話だけを生成する")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--model", default="llm-jp-4-32b-a3b-thinking")
    p.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default="")
    p.add_argument("--min_turns", type=int, default=1,
                   help="これより前のユーザの EOC は無視する（往復数）")
    p.add_argument("--max_turns", type=int, default=8,
                   help="EOC が出なくてもここで打ち切る（往復数）")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--effort_user", default="high")
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
