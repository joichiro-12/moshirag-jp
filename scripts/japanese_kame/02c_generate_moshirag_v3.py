"""Generate MoshiRAG-style Japanese conversation scripts (v2, two-axis design).

Replaces 02b_generate_moshirag_scripts.py. The v1 design generated the record
line-by-line with the reference produced *inside* the turn loop; that made every call
carry the whole task description and produced conversations that did not hold together
(see dev/llm-jp/260812_設計判断/スモーク生成v1_評価.md).

v2 splits the work along two axes:

  Axis A (independent of the conversation)
      QA {question, answer} -> 3 reference passages, one containing the answer and two
      covering other aspects of the same topic. Retrieval returns passages with surplus
      content, so the passages must have surplus too; otherwise the model learns to read
      out whatever it is handed.

  Axis B stage 1 : plain conversation. Two roles only (human / assistant), no markup.
                   The human never sees the passages (T-3). The assistant does, so the
                   causal direction is passage -> utterance.
  Axis B stage 2a: judge which passage each assistant utterance used ("none" = the turn
                   needs no retrieval, which is what supplies unaugmented turns).
  Axis B stage 2b: for augmented turns, write the lead from the preceding conversation
                   only -- it never sees the body or the passage, so T-4 holds
                   structurally. tail defaults to [empty].

Naming: `moshi:` is a speaker label in the record (following the original
implementation). It is NOT a character name -- nothing may address the assistant by
name, since "Moshi" is the model, not a persona.

Usage:
    uv run python scripts/japanese_kame/02c_generate_moshirag_v2.py \
        --input_file data/japanese_kame/qa_pairs/jaquad.jsonl \
        --output_dir data/moshirag_jp/v2_pilot \
        --num_samples 20 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_print_lock = threading.Lock()
_HARMONY_FINAL = re.compile(r"assistant\s*final\s*", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    parts = _HARMONY_FINAL.split(text)
    out = parts[-1] if len(parts) > 1 else text
    return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()


def first_line(text: str) -> str:
    for raw in text.splitlines():
        if raw.strip():
            return raw.strip()
    return ""


def strip_label(text: str, *labels: str) -> str:
    """Drop a leading 'label:' the model may have echoed, so we never double-prefix."""
    out = text.strip()
    for _ in range(3):
        for lab in labels:
            if out.startswith(lab):
                out = out[len(lab):].lstrip("：: ").strip()
                break
        else:
            break
    return out


# ======================================================================================
# Axis A : reference passages
# ======================================================================================
CHUNK_ANSWER = """\
QA データセットの質問と答えを渡します。
その答えを回答するための根拠になる参照文章を 1 つ作ってください。

- 60〜120 字、1 行で書く
- だ・である調
- 単独で読んで意味が通るように書く（「この作家は」「同年に」のような書き方は使わない）
- 確実に言えることだけ書く

例：

質問: 日本で一番長い川はどこ？
答え: 信濃川
参照: 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれ、新潟県に入って信濃川となる。

質問: 味噌に赤と白があるのはなぜ？
答え: 大豆の加熱方法と熟成期間の違い
参照: 味噌の色の違いは、主に大豆の加熱方法と熟成期間による。赤味噌は大豆を蒸して長く熟成させるため褐色化が進む。白味噌は煮た大豆を短期間で熟成させる。

質問: 花火大会はいつから行われているか？
答え: 江戸時代の一七三三年
参照: 日本の花火大会の起源は、江戸時代の一七三三年に隅田川で行われた水神祭とされる。前年の飢饉と疫病の死者を弔う目的があり、当時は両国の川開きと呼ばれた。"""

CHUNK_OTHER = """\
同じ話題について、渡された質問には答えていない参照文章を 1 つ作ってください。
すでに作った文章と内容が重ならないようにしてください。

- 60〜120 字、1 行で書く
- だ・である調
- 単独で読んで意味が通るように書く
- 確実に言えることだけ書く

例：

質問: 日本で一番長い川はどこ？
答え: 信濃川
すでに作った文章: 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。
別の文章: 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロで、信濃川に次ぐ長さを持つ。

質問: 味噌に赤と白があるのはなぜ？
答え: 大豆の加熱方法と熟成期間の違い
すでに作った文章: 味噌の色の違いは、主に大豆の加熱方法と熟成期間による。
別の文章: 味噌の好みには地域差があり、東海地方では豆味噌、関西では白味噌が好まれる。信州では米味噌が主流である。

質問: 花火大会はいつから行われているか？
答え: 江戸時代の一七三三年
すでに作った文章: 日本の花火大会の起源は、江戸時代の一七三三年に隅田川で行われた水神祭とされる。
別の文章: 隅田川花火大会は現在も毎年七月に行われ、二万発前後が打ち上げられる。観客数は九十万人を超える。"""

COMMON = """\
【書き方】
- 音声にして読み上げるので、話し言葉として自然にする
- 使える記号は 、。？！… だけ。括弧・箇条書き・アスタリスク・引用符は使わない
- 記号で省略しない（％→パーセント、1904〜1905年→1904年から1905年）
- 40 文字程度まで"""

# ======================================================================================
# Axis B stage 1 : plain conversation
# ======================================================================================
HUMAN_ROLE = """\
人と AI アシスタントの音声会話を作っています。あなたは人間側の次の 1 発話を書いてください。

- 聞きたいことはあなたの頭の中にあるだけで、相手は知らない。「その」「あの」で済ませず、誰の何についてかを自分の言葉で言う
- 書き言葉のまま読み上げない。試験問題のような言い方はしない
- 一度に一つのことだけ聞く。長い修飾を並べない
- 答えは知らない人物なので、答えを自分で言わない
- 相手を名前で呼ばない
- 質問だけでなく、感想・言いさし・ためらい・軽い異論も使ってよい
- 会話を終わらせたいときは行末に EOC と書く。最低 {min_turns} 往復は続ける

{common}

例：

聞きたいこと: 日本で最も長い河川の名称は何か
人物設定: 20代 / タメ口 / 少し知っている / 興味あり
ここまでの会話:
moshi: こんばんは。
人間: ねえ、日本で一番長い川ってどこだっけ？

聞きたいこと: 味噌の色に赤と白の差異が生じる要因は何か
人物設定: 40代以上 / 丁寧語 / 全然知らない / なんとなく
ここまでの会話:
moshi: こんにちは。
人間: 味噌に赤いのと白いのがありますよね。あれはどう違うんですか。

聞きたいこと: 日本の花火大会の起源とその目的は何か
人物設定: 10代 / タメ口 / 全然知らない / 確認だけ
ここまでの会話:
moshi: やあ。
人間: 花火大会って昔からあるもんなの？
moshi: 江戸時代の一七三三年に隅田川でやったのが始まりって言われてるよ。
人間: へえ、そんな前から。何のためにやったの？"""

ASSISTANT_ROLE = """\
人と AI アシスタントの音声会話を作っています。あなたはアシスタント側の次の 1 発話を書いてください。

- 直前の人間の発話に、話し言葉で自然に答える
- 資料に答えがあるなら、その部分だけを使う。資料の他の部分は話さない
- 資料に答えがないなら、資料を使わず普通に受け答えする。無理に情報を足さない
- **資料の存在を発話に出さない。**「資料には」「記載されていません」「情報がありません」と言わない。
  答えを知らないときは、人がそうするように「詳しくないんだ」「どうだったかな」と受ける
- 資料の文体をそのまま写さず、自分の言葉に言い換える
- 相手の口調に合わせる。丁寧語の相手には丁寧語、タメ口の相手にはタメ口

{common}

例：

手元の資料:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: ねえ、日本で一番長い川ってどこだっけ？
moshi: 信濃川だよ。長野から新潟まで流れて、三百六十七キロあるんだって。

手元の資料:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: へえ、そんなに長いんだ。
moshi: うん、意外だよね。

手元の資料:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: 一番大きい川とは違うの？
moshi: 流域面積だと利根川が日本で一番なんだよ。長さとは別なんだね。

手元の資料:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: 信濃川って昔から今の名前だったの？
moshi: うーん、どうだったかな。名前の由来までは詳しくないんだ。"""

# ======================================================================================
# Axis B stage 2a : judge
# ======================================================================================
JUDGE_ROLE = """\
完成した会話と、アシスタントが手元に持っていた資料を渡します。
アシスタントの各発話が、資料のどれを使っているかを判定してください。

- 資料の内容が発話に出ているなら、その番号を書く
- 言い換えられていても、内容が資料に由来するなら番号を書く
- 相槌・感想・一般的な受け答えだけで、資料の内容が入っていないなら「なし」と書く

出力は、アシスタントの発話ごとに 1 行。

例：

資料:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
会話:
moshi: こんばんは。
人間: ねえ、日本で一番長い川ってどこだっけ？
moshi: 信濃川だよ。長野から新潟まで流れて、三百六十七キロあるんだって。
人間: へえ、そんなに長いんだ。
moshi: うん、意外だよね。
人間: 一番大きい川とは違うの？
moshi: 流域面積だと利根川が日本で一番なんだよ。長さとは別なんだね。

判定:
1: なし
2: 1
3: なし
4: 2"""

# ======================================================================================
# Axis B stage 2b : lead
# ======================================================================================
LEAD_ROLE = """\
人と AI アシスタントの音声会話を作っています。
アシスタントが本題を話し始める直前の、短い前置きを 1 つ書いてください。

- 9〜18 モーラ。およそ 2〜4 秒で話せる長さ。短すぎても長すぎてもいけない
- この後アシスタントが何と答えるかは、あなたは知らない。答えを言わない、推測もしない
- 調べていることを相手に悟らせない
- フィラー（えーと、あのー、うーん、あー）は 1〜2 個まで
- 相手が聞いたことを言い直すときは、短い名詞句にする。疑問文をそのまま埋め込まない
- 相手の口調に合わせる

{common}

例：

ここまでの会話:
人間: ねえ、日本で一番長い川ってどこだっけ？
前置き: えーと、日本で一番長い川ね。

ここまでの会話:
人間: 味噌に赤いのと白いのがありますよね。あれはどう違うんですか。
前置き: はい、白いほうですね。

ここまでの会話:
人間: へえ、そんな前から。何のためにやったの？
前置き: うーん、何のためだったかな。"""

# ======================================================================================
PERSONA_AXES = {
    "年齢層": ["10代", "20代", "30代", "40代以上"],
    "口調": ["タメ口", "丁寧語", "ぶっきらぼう"],
    "前提知識": ["詳しい", "少し知っている", "全然知らない"],
    "乗り気度": ["興味あり", "なんとなく", "確認だけ"],
}
GREETINGS = ["こんにちは。", "こんばんは。", "やあ。", "どうも。"]


def sample_persona(rng: random.Random) -> str:
    return " / ".join(rng.choice(v) for v in PERSONA_AXES.values())


_EFFORT_SHAPE: dict = {}


def call(client, model: str, system: str, user: str, effort: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    shapes = [
        {"reasoning_effort": effort},
        {"chat_template_kwargs": {"reasoning_effort": effort}},
        {},
    ]
    if effort in _EFFORT_SHAPE:
        shapes = [shapes[_EFFORT_SHAPE[effort]]] + shapes
    err = None
    for i, body in enumerate(shapes):
        try:
            r = client.chat.completions.create(
                model=model, messages=msgs, stream=False, extra_body=body
            )
            _EFFORT_SHAPE.setdefault(effort, i)
            return strip_thinking(r.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"reasoning_effort rejected: {err}")


def make_chunks(client, model, qa, effort) -> list[str]:
    q, a = qa["question"], qa["answer"]
    c1 = first_line(
        call(client, model, CHUNK_ANSWER, f"質問: {q}\n答え: {a}\n参照:", effort)
    )
    c1 = strip_label(c1, "参照")
    chunks = [c1]
    for _ in range(2):
        prev = "／".join(chunks)
        out = first_line(
            call(client, model, CHUNK_OTHER,
                 f"質問: {q}\n答え: {a}\nすでに作った文章: {prev}\n別の文章:", effort)
        )
        chunks.append(strip_label(out, "別の文章", "参照"))
    return chunks


def render_chunks(chunks: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))


def generate(qa, client, model, rng, args) -> dict:
    persona = sample_persona(rng)
    chunks = make_chunks(client, model, qa, args.effort_chunk)
    ch = render_chunks(chunks)

    # ---- stage 1 : plain conversation -----------------------------------------------
    plain: list[tuple[str, str]] = [("moshi", rng.choice(GREETINGS))]
    ended = False
    for turn in range(args.max_turns):
        tr = "\n".join(f"{w}: {t}" for w, t in plain)
        h = first_line(call(
            client, model, HUMAN_ROLE.format(min_turns=args.min_turns, common=COMMON),
            f"聞きたいこと: {qa['question']}\n人物設定: {persona}\n"
            f"ここまでの会話:\n{tr}\n人間:", args.effort_human))
        h = strip_label(h, "人間")
        if h.rstrip().endswith("EOC"):
            h = h.rstrip()[:-3].rstrip("　 、。")
            # C4: an EOC before min_turns is ignored (v1 lost 3/20 conversations to it)
            if turn >= args.min_turns:
                ended = True
                if h:
                    plain.append(("人間", h))
                break
        if not h:
            break
        plain.append(("人間", h))

        tr = "\n".join(f"{w}: {t}" for w, t in plain)
        a = first_line(call(
            client, model, ASSISTANT_ROLE.format(common=COMMON),
            f"手元の資料:\n{ch}\nここまでの会話:\n{tr}\nmoshi:", args.effort_assistant))
        a = strip_label(a, "moshi", "アシスタント")
        if not a:
            break
        plain.append(("moshi", a))

    # ---- stage 2a : which passage did each assistant utterance use? ------------------
    tr = "\n".join(f"{w}: {t}" for w, t in plain)
    verdict_raw = call(client, model, JUDGE_ROLE,
                       f"資料:\n{ch}\n会話:\n{tr}\n\n判定:", args.effort_judge)
    used: dict[int, int | None] = {}
    for line in verdict_raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[:：]\s*(.+)", line)
        if not m:
            continue
        idx = int(m.group(1))
        val = m.group(2).strip()
        mm = re.search(r"\d+", val)
        used[idx] = int(mm.group()) if (mm and "なし" not in val) else None

    # ---- stage 2b : assemble the record --------------------------------------------
    record: list[str] = []
    greeting_lines = (0, 1)
    a_no = 0
    for pos, (who, text) in enumerate(plain):
        if who == "人間":
            record.append(f"Human: {text}")
            continue
        a_no += 1
        cidx = used.get(a_no)
        if a_no == 1 or cidx is None or not (1 <= cidx <= len(chunks)):
            record.append("(unaugmented)")
            record.append(f"moshi: {text}")
            continue
        # The lead role gets a PLAIN transcript of what was said up to this point.
        # It must not see Reference lines or record markup (that leak is what made the
        # v1 human role imitate the internal format), and it must not see this turn's
        # own body -- that is what keeps T-4 structural.
        hist = "\n".join(f"{w}: {t}" for w, t in plain[:pos])
        lead = first_line(call(client, model, LEAD_ROLE.format(common=COMMON),
                               f"ここまでの会話:\n{hist}\n前置き:", args.effort_lead))
        lead = strip_label(lead, "前置き", "moshi (lead)", "moshi")
        record.append("(augmented)")
        record.append(f"moshi (lead): {lead}")
        record.append(f"Reference: {chunks[cidx - 1]}")
        # subtopic dropped: it is generation-time metadata that never reaches the text
        # stream, and deriving it from the seed question produced garbage. The chunk
        # index is recorded in the JSON instead, which is more useful for analysis.
        record.append(f"moshi (body): {text}")
        record.append("moshi (tail): [empty]")

    return {
        "seed_question": qa["question"],
        "seed_answer": qa["answer"],
        "source": qa.get("source", ""),
        "persona": persona,
        "chunks": chunks,
        "plain": [{"speaker": w, "text": t} for w, t in plain],
        "judge_raw": verdict_raw,
        "chunk_used": {str(k): v for k, v in used.items()},
        "record": record,
        "greeting_lines": greeting_lines,
        "n_user_turns": sum(1 for w, _ in plain if w == "人間"),
        "ended_by_eoc": ended,
        "model": model,
        "prompt_version": "moshirag-jp-v3",
    }


def main(args) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
                    base_url=args.llm_base_url or None)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Path(args.input_file).open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    # F5: sample at random. Taking the head of jaquad.jsonl gave 20 topics that were all
    # from one Wikipedia article, so v1 had zero topic diversity.
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.num_samples]

    def work(i: int, qa: dict):
        stem = f"{Path(args.input_file).stem}_{i:06d}"
        pj, pt = out / f"{stem}.json", out / f"{stem}.txt"
        if args.resume and pj.exists():
            return stem, "skip"
        rng = random.Random(args.seed * 1000 + i)
        for attempt in range(3):
            try:
                conv = generate(qa, client, args.model, rng, args)
                pj.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
                pt.write_text("\n".join(conv["record"]) + "\n", encoding="utf-8")
                return stem, f"ok turns={conv['n_user_turns']} lines={len(conv['record'])}"
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return stem, f"FAIL {e}"
        return stem, "FAIL"

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, qa) for i, qa in enumerate(rows)]
        for fu in as_completed(futs):
            stem, msg = fu.result()
            ok += not msg.startswith("FAIL")
            fail += msg.startswith("FAIL")
            with _print_lock:
                print(f"[{msg.split()[0]}] {stem} {msg}", flush=True)
    print(f"\nDone: {ok} ok / {fail} fail -> {out}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--model", default="llm-jp-4-32b-a3b-thinking")
    p.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default="")
    p.add_argument("--min_turns", type=int, default=3)
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--effort_chunk", default="low")
    p.add_argument("--effort_human", default="high")
    p.add_argument("--effort_assistant", default="low")
    p.add_argument("--effort_judge", default="low")
    p.add_argument("--effort_lead", default="low")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
