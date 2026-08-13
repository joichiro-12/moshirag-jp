"""Generate MoshiRAG-style Japanese conversation scripts (line-oriented universal record).

Reimplements the data generation recipe of MoshiRAG (arXiv:2604.12928v2, Section 4.1.2)
for Japanese. Unlike 02_generate_dialogues.py (KAME, one LLM call per dialogue), this
maintains a single shared "universal record" and fills it **one line at a time**,
dispatching each line to the appropriate role (user / reference / moshi).

Design decisions are logged in the research repo:
  dev/llm-jp/260812_設計判断/日本語MoshiRAG_設計判断ログ.md   (D-1..D-10, D-5b)

Key structural invariants (do not break these):
  * The user role never sees Reference lines           -> T-3 (information leakage)
  * The lead is generated BEFORE this turn's Reference -> T-4 (lead is reference-free)
  * Reference: sits between the lead line and body line in the record (D-2)
  * lead contains no rough answer to the question      -> D-8

Usage:
    uv run --extra oracle python scripts/japanese_kame/02b_generate_moshirag_scripts.py \
        --input_file data/japanese_kame/qa_pairs/jaquad.jsonl \
        --output_dir data/moshirag_jp/v1_pilot \
        --num_samples 20 \
        --llm_base_url http://localhost:8000/v1 \
        --model llm-jp/llm-jp-4-32b-a3b-thinking
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
from pathlib import Path

_print_lock = threading.Lock()

# --------------------------------------------------------------------------------------
# Harmony / thinking-format stripping
# --------------------------------------------------------------------------------------
# llm-jp-4-*-thinking replies in the OpenAI Harmony format. When vLLM is started without
# --reasoning-parser the raw text contains the analysis channel, e.g.
#     analysis <thoughts> assistant final <answer>
# Keep only the final channel.
_HARMONY_FINAL = re.compile(r"assistant\s*final\s*", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    parts = _HARMONY_FINAL.split(text)
    out = parts[-1] if len(parts) > 1 else text
    # Some builds emit <think>...</think> instead.
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
    return out.strip()


def first_line(text: str) -> str:
    """Take the first non-empty line. Roles are asked for exactly one line."""
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            return line
    return ""


# --------------------------------------------------------------------------------------
# Shared style guidelines (Japanese counterpart of Table 15 "Guidelines")
# --------------------------------------------------------------------------------------
# NOTE on reading normalisation: the original prompt says "convert 25 to twenty-five".
# Transliterating every numeral into kana would fill Moshi's *text* stream with kana and
# damage the text modality, so for Japanese we instead require that the reading be
# unambiguous while keeping ordinary orthography. See ◆6 in the decision log.
COMMON_GUIDELINES = """\
【共通の書き方】
- 生成した文はそのまま音声合成に渡される。読み上げて自然な話し言葉にすること。
- 記号は日本語の語に開く（％→パーセント、&→と、℃→度、〜→から）。
- 読みが多義になる表記を避ける（「1日」→「一日」または「ついたち」、「3-4」→「三から四」）。
  ただし通常の漢数字・アラビア数字の表記は保つ。すべてを仮名に開かないこと。
- 使える記号は 、。？！ と三点リーダ（…）のみ。箇条書き・表・アスタリスク・括弧書きは使わない。
- 1 行は 40 文字程度まで。会話のテンポを保つ。
- 出力は指定された 1 行のみ。前置きも説明も付けない。"""

# --------------------------------------------------------------------------------------
# Persona (inherited from the 260806 pilot v2, 4 properties drawn per conversation)
# --------------------------------------------------------------------------------------
PERSONA_AXES = {
    "年齢層": {
        "10代": "若者言葉を使う。丁寧語は使わない",
        "20代": "くだけた口調。敬語は最小限",
        "30代": "常識的な口調。場面に応じて敬語を使う",
        "40代以上": "落ち着いた口調。丁寧語が基本",
    },
    "口調": {
        "タメ口": "友人に話すような口調",
        "丁寧語": "です・ます調で話す",
        "ぶっきらぼう": "短く言い切る。愛想は良くない",
    },
    "前提知識": {
        "詳しい": "この話題に詳しい。具体的な用語を使ってよい",
        "少し知っている": "うろ覚え。断定を避ける",
        "全然知らない": "この話題について何も知らない。専門用語は使わず素朴な言葉で聞く",
    },
    "乗り気度": {
        "興味あり": "強い関心がある。前のめりに聞く",
        "なんとなく": "軽い興味。深追いはしない",
        "確認だけ": "さらっと確認するだけ。深掘りはしない",
    },
}


def sample_persona(rng: random.Random) -> dict[str, str]:
    return {axis: rng.choice(list(opts)) for axis, opts in PERSONA_AXES.items()}


def render_persona(persona: dict[str, str]) -> str:
    lines = [
        f"- {axis}: {val}（{PERSONA_AXES[axis][val]}）" for axis, val in persona.items()
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
USER_SYSTEM = """\
あなたは、AI アシスタント「もしもし」と話している人間の次の 1 発話を書く役です。

【あなたが見えているもの】
- ここまでの会話（人間側・もしもし側の両方）
- 会話の話題

【人物設定】
以下の人物になりきって話してください。人物設定に合わない語彙・口調は使わないこと。
{persona}

【守ること】
- 話題に沿った、自然な人間の発話を 1 つ書く。
- 質問だけでなく、感想・言いさし・ためらい・軽い異論・相槌的な受けも自然な範囲で使ってよい。
- 会話は最低 {min_turns} 往復は続ける。自然ならもっと長くてよい。
- 会話を終えるときは、その行の末尾に EOC と書く。
- **答えを自分で言ってはいけない。** あなたは答えを知らない人物である。

{common}

出力形式（この 1 行だけを出力する）:
Human: 〈発話〉"""

REFERENCE_SYSTEM = """\
あなたは、AI アシスタント「もしもし」が次の発話で使うための参照文書を書く役です。

【あなたが見えているもの】
- ここまでの会話の全体（もしもしの前置きまで含む）

【守ること】
- 直前の人間の発話に答えるために必要な事実を含む、簡潔で客観的な文書を書く。
- 「もしもし」が次に話す内容の根拠になるように書く。
- 70 文字程度まで。1 行で書く。改行しない。
- 百科事典のような客観的な文体（だ・である調）で書く。
- 不確実な情報は書かない。断定できないことは書かない。
- これは音声にならない内部文書なので、話し言葉にしなくてよい。

出力形式（この 1 行だけを出力する）:
Reference: 〈参照文書〉"""

# --- moshi role: three modes -----------------------------------------------------------
MOSHI_BASE = """\
あなたは、気さくな AI アシスタント「もしもし」の次の 1 行を書く役です。

【あなたが見えているもの】
- ここまでの会話
- 参照文書（あれば）。会話の話題そのものは知らされない

【もしもしの応答の構造】
もしもしの各ターンは「参照が必要なターン」か「参照が不要なターン」のどちらかである。

- 参照が不要なターン（unaugmented）: 一般的な受け答え・相槌・感想。1 行で完結する。
- 参照が必要なターン（augmented）: 次の 3 つに分かれる。
    lead : 参照が届く前に話す前置き。外部の情報を一切必要としない内容に限る。
    body : 参照文書に基づいた本題の回答。
    tail : 締めの一言。空でもよい。

【口調】
- 相手の口調に合わせる。丁寧語の相手には丁寧語、タメ口の相手にはタメ口。
- 参照文書の表現をそのまま使わず、自分の言葉に言い換える。

{common}"""

MOSHI_LEAD_MODE = """\
【今回の仕事】
まず、直前の人間の発話に答えるのに外部の情報が必要かどうかを判断する。

■ 外部の情報が不要な場合（相槌・感想・一般的な受け答えで足りる場合）
出力形式（2 行）:
(unaugmented)
moshi: 〈発話〉

■ 外部の情報が必要な場合（固有名詞・年号・数値・具体的な事実を答える必要がある場合）
出力形式（2 行）:
(augmented)
moshi (lead): 〈前置き〉

【前置き（lead）の作り方 — ここが最も重要】
前置きは、参照文書が届くまでの時間を埋める。**2 秒以上しゃべる長さが必要**で、
日本語ではおおよそ 9 モーラ以上に相当する。

使える型（組み合わせて使う）:
  1. フィラー          えーと／あのー／うーん／あー
  2. 応答詞・評価      そうですね／なるほど／いい質問だね／それ気になるよね
  3. 復唱・言い換え    〈相手の質問を自分の言葉で言い直す〉  ← 長さを稼ぐ主役
  4. 話題の枠づけ      それはね／あれについてはね
  5. 記憶をたどる表明  なんだっけ／なんて言ったかな

標準の組み立ては「フィラー + 復唱」または「応答詞 + 復唱」。
復唱は相手の質問が複雑なほど長くなるので、長さはそれで調整する。

**やってはいけないこと**:
- 答えを言う、答えを推測する、答えをぼかして言う。
  ×「たしかけっこう古いはずだよ」「昭和のころだった気がするけど」
  前置きの時点であなたは答えを知らない。答えは body で参照文書に基づいて言う。
- 調べていることを相手に悟らせる表現。
  ×「ちょっと調べてみますね」「確認してみますね」「少し待ってね」
- フィラーを 3 つ以上並べて長さを稼ぐ。
  ×「えーと、あのー、そのー、なんか」
  長さは復唱で稼ぐ。フィラーは 1〜2 個まで。

良い例:
  (augmented)
  moshi (lead): えーと、日本で最初に三十分のテレビアニメになった作品ね。
  (augmented)
  moshi (lead): そうですね、紙の砦とどついたれのどちらが先か、ですか。"""

MOSHI_BODY_MODE = """\
【今回の仕事】
参照文書に基づいて本題を答える 1 行を書く。

- 参照文書に書かれている情報だけを根拠にする。書かれていないことを足さない。
- 参照文書の言い回しをそのまま写さず、話し言葉に言い換える。
- 直前の前置き（lead）から自然につながるように書く。
- 何についての回答かを表す短い語を subtopic に書く。

出力形式（この 1 行だけを出力する）:
moshi (body)(subtopic: 〈短い語〉): 〈回答〉"""

MOSHI_TAIL_MODE = """\
【今回の仕事】
締めの一言を書く。無くてよい場合は [empty] と書く。

- 参照文書に無い事実を足さない。
- 感想・軽い補足・相手への問い返しなど、外部の情報を必要としない内容にする。

出力形式（この 1 行だけを出力する）:
moshi (tail): 〈締め〉
または
moshi (tail): [empty]"""


# --------------------------------------------------------------------------------------
# Record views (who sees what)
# --------------------------------------------------------------------------------------
def view_for_user(record: list[str]) -> str:
    """The user role must not see Reference lines (T-3)."""
    return "\n".join(l for l in record if not l.startswith("Reference:"))


def view_full(record: list[str]) -> str:
    return "\n".join(record)


# --------------------------------------------------------------------------------------
# LLM call
# --------------------------------------------------------------------------------------
_EFFORT_STYLE: dict[str, str] = {}  # cache: which extra_body shape this server accepts


def _effort_bodies(effort: str) -> list[dict]:
    """Different vLLM versions accept reasoning_effort in different places."""
    return [
        {"reasoning_effort": effort},
        {"chat_template_kwargs": {"reasoning_effort": effort}},
        {},  # give up on controlling effort rather than failing the run
    ]


def call_role(
    client,
    model: str,
    system: str,
    record_text: str,
    effort: str,
) -> str:
    user_msg = "ここまでの会話:\n" + (record_text or "（まだ何も話していない）")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    candidates = _effort_bodies(effort)
    if effort in _EFFORT_STYLE:
        idx = _EFFORT_STYLE[effort]
        candidates = [candidates[idx]] + candidates
    last_err: Exception | None = None
    for i, body in enumerate(candidates):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, stream=False, extra_body=body
            )
            if effort not in _EFFORT_STYLE:
                _EFFORT_STYLE[effort] = i
                with _print_lock:
                    print(f"[effort] '{effort}' accepted via {body or 'no extra_body'}")
            return strip_thinking(response.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001 - probe the next shape
            last_err = e
    raise RuntimeError(f"all reasoning_effort shapes rejected: {last_err}")


# --------------------------------------------------------------------------------------
# One conversation
# --------------------------------------------------------------------------------------
def generate_conversation(
    topic: str,
    client,
    model: str,
    rng: random.Random,
    min_turns: int,
    max_turns: int,
    efforts: dict[str, str],
) -> dict:
    persona = sample_persona(rng)
    user_system = USER_SYSTEM.format(
        persona=render_persona(persona),
        min_turns=min_turns,
        common=COMMON_GUIDELINES,
    )
    moshi_base = MOSHI_BASE.format(common=COMMON_GUIDELINES)

    record: list[str] = []
    # multi-turn conversations always open with a Moshi greeting (paper Section 4.1.2).
    # Removal with p=0.3 is a *training-time* augmentation, so the greeting is kept in the
    # data and its span is annotated instead (requirement T-7).
    greeting = "moshi: こんにちは。何か気になることある？"
    record.append("(unaugmented)")
    record.append(greeting)
    greeting_lines = (0, 1)

    n_turns = 0
    ended = False
    for _ in range(max_turns):
        # ---- user turn --------------------------------------------------------------
        u = first_line(
            call_role(client, model, user_system + f"\n\n会話の話題: {topic}",
                      view_for_user(record), efforts["user"])
        )
        if not u.startswith("Human:"):
            u = "Human: " + u.lstrip()
        if u.rstrip().endswith("EOC"):
            ended = True
            u = u.rstrip()[:-3].rstrip()
            if u and u != "Human:":
                record.append(u)
            break
        record.append(u)
        n_turns += 1

        # ---- moshi: decide augmentation + first line --------------------------------
        head = call_role(client, model, moshi_base + "\n\n" + MOSHI_LEAD_MODE,
                         view_full(record), efforts["moshi"])
        head_lines = [l.strip() for l in head.splitlines() if l.strip()]
        label = next((l for l in head_lines if l in ("(augmented)", "(unaugmented)")), None)
        content = next((l for l in head_lines if l.startswith("moshi")), "")
        if label is None:
            label = "(augmented)" if "(lead)" in content else "(unaugmented)"
        if not content:
            raise ValueError(f"moshi head produced no content line: {head!r}")

        record.append(label)
        record.append(content)

        if label == "(unaugmented)":
            continue

        # ---- reference (generated AFTER the lead: this is the T-4 guarantee) ---------
        ref = first_line(call_role(client, model, REFERENCE_SYSTEM,
                                   view_full(record), efforts["reference"]))
        if not ref.startswith("Reference:"):
            ref = "Reference: " + ref.lstrip()
        record.append(ref)

        # ---- body -------------------------------------------------------------------
        body = first_line(call_role(client, model, moshi_base + "\n\n" + MOSHI_BODY_MODE,
                                    view_full(record), efforts["moshi"]))
        if not body.startswith("moshi (body)"):
            body = "moshi (body)(subtopic: 回答): " + body.lstrip()
        record.append(body)

        # ---- tail -------------------------------------------------------------------
        tail = first_line(call_role(client, model, moshi_base + "\n\n" + MOSHI_TAIL_MODE,
                                    view_full(record), efforts["moshi"]))
        if not tail.startswith("moshi (tail)"):
            tail = "moshi (tail): " + tail.lstrip()
        record.append(tail)

    return {
        "topic": topic,
        "persona": persona,
        "record": record,
        "greeting_lines": greeting_lines,   # T-7: annotated, not removed
        "n_user_turns": n_turns,
        "ended_by_eoc": ended,
        "model": model,
        "reasoning_effort": efforts,
        "prompt_version": "moshirag-jp-v1",
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI(
        api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
        base_url=args.llm_base_url or None,
    )
    efforts = {
        "user": args.effort_user,
        "reference": args.effort_reference,
        "moshi": args.effort_moshi,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.input_file).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][: args.num_samples]

    ok = fail = 0
    for i, row in enumerate(rows):
        stem = f"{Path(args.input_file).stem}_{i:06d}"
        out_json = out_dir / f"{stem}.json"
        out_txt = out_dir / f"{stem}.txt"
        if args.resume and out_json.exists():
            ok += 1
            continue

        # The topic is the seed question. Only the user role sees it (paper 4.1.2).
        topic = row["question"]
        rng = random.Random(args.seed + i)

        for attempt in range(3):
            try:
                conv = generate_conversation(
                    topic, client, args.model, rng,
                    args.min_turns, args.max_turns, efforts,
                )
                conv["source"] = row.get("source", "")
                out_json.write_text(
                    json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                out_txt.write_text("\n".join(conv["record"]) + "\n", encoding="utf-8")
                ok += 1
                with _print_lock:
                    print(f"[OK] {stem}  turns={conv['n_user_turns']} lines={len(conv['record'])}")
                break
            except Exception as e:
                if attempt < 2:
                    with _print_lock:
                        print(f"[RETRY {attempt + 1}/3] {stem}: {e}")
                else:
                    with _print_lock:
                        print(f"[WARN] {stem}: {e}")
                    fail += 1

    print(f"\nDone: {ok} ok / {fail} fail -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", required=True, help="QA pairs JSONL (topic seeds).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--model", default="llm-jp/llm-jp-4-32b-a3b-thinking")
    p.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default="")
    p.add_argument("--min_turns", type=int, default=3)
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument("--effort_user", default="high", help="D-5b: persona reasoning.")
    p.add_argument("--effort_reference", default="low")
    p.add_argument("--effort_moshi", default="low")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
