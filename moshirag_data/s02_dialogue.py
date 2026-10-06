"""QA の組から、MoshiRAG 形式の日本語の会話記録を LLM で作る。

1. QA → 参照文 3 本（1 本は答えを含む）
2. 参照文を見た assistant と見ない human の素の会話
3. assistant の各発話が使った参照文を判定（none = 検索不要のターン）
4. 参照文を使ったターンに、直前の会話だけから lead を書く

Usage:
    （本番は jobs/qa_gen/gen_wiki_main.pbs。vLLM のサーバを立ててから呼ぶ）
    $VENV/bin/python moshirag_data/s02_dialogue.py \
        --input_file data/japanese_kame/qa_pairs/wiki_qa.jsonl \
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
_HARMONY_FINAL = re.compile(r"assistant\s*final\s*", re.IGNORECASE)


# Harmony 形式は analysis チャネルで思考し、assistantfinal 以降が答えになる。
# 生成が途中で打ち切られると assistantfinal が出ず、思考がそのまま返る。
# v5 の 5,000 会話では 81 ターン（1.58% の会話）でこれが起き、英語の独白が
# 台本に入って音声化まで通ってしまった。素通りさせず、検出して生成し直す。
_LOOKS_LIKE_THINKING = re.compile(
    r"^\s*analysis\b"
    r"|^\s*<think>"
    r"|\bNeed to produce\b"
    r"|\bLet's craft\b"
    r"|\bThe user (?:wants|is asking|gave)\b"
    r"|\bWe need to\b"
    r"|\bCount moras?\b",
    re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    parts = _HARMONY_FINAL.split(text)
    out = parts[-1] if len(parts) > 1 else text
    return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()


def looks_like_thinking(text: str) -> bool:
    """思考が剥がれずに残っているか。残っていれば生成をやり直す。"""
    return bool(_LOOKS_LIKE_THINKING.search(text))


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
# Axis B stage 0 : topic phrase (following the original implementation, which drives the
# conversation from a *topic*, not from the seed question itself)
# ======================================================================================
TOPIC_ROLE = """\
QA データセットの質問を渡します。その質問が扱っている話題を、短い言葉で表してください。

- 十語程度まで。名詞句にする
- 質問の形にしない
- **答えを含めない。** 質問の答えにあたる固有名詞は話題に書かない
- 会話の話題として自然な広さにする。狭すぎると会話が続かない

例：

質問: 戦後日本のストーリー漫画の第一人者で、医学博士の一面もある漫画家は誰?
話題: 戦後日本の漫画家

質問: 日本で一番長い川はどこ？
話題: 日本の川

質問: 味噌の色に赤と白の差異が生じる要因は何か
話題: 味噌の種類と作り方

質問: 日本の花火大会の起源とその目的は何か
話題: 花火大会の歴史"""

# ======================================================================================
# Axis B stage 1 : plain conversation
# ======================================================================================
HUMAN_ROLE = """\
人と AI アシスタントの音声会話を作っています。あなたは人間側の次の 1 発話を書いてください。

- 渡された話題について、あなたが素朴に気になることを聞く。決まった質問を再現するのではない
- 相手はあなたが何を気にしているか知らない。「その」「あの」で済ませず、誰の何についてかを自分の言葉で言う
- 書き言葉のまま読み上げない。試験問題のような言い方はしない
- 一度に一つのことだけ聞く。長い修飾を並べない
- 答えは知らない人物なので、答えを自分で言わない
- 相手を名前で呼ばない
- 質問だけでなく、感想・言いさし・ためらい・軽い異論も使ってよい
- 会話を終わらせたいときは行末に EOC と書く。最低 {min_turns} 往復は続ける

{common}

例：

話題: 日本の川
人物設定: 20代 / タメ口 / 少し知っている / 興味あり
ここまでの会話:
moshi: こんばんは。
人間: ねえ、日本で一番長い川ってどこだっけ？

話題: 味噌の種類と作り方
人物設定: 40代以上 / 丁寧語 / 全然知らない / なんとなく
ここまでの会話:
moshi: こんにちは。
人間: 味噌に赤いのと白いのがありますよね。あれはどう違うんですか。

話題: 花火大会の歴史
人物設定: 10代 / タメ口 / 全然知らない / 確認だけ
ここまでの会話:
moshi: やあ。
人間: 花火大会って昔からあるもんなの？
moshi: 江戸時代の一七三三年に隅田川でやったのが始まりって言われてるよ。
人間: へえ、そんな前から。何のためにやったの？"""

ASSISTANT_ROLE = """\
人と AI アシスタントの音声会話を作っています。あなたはアシスタント側の次の 1 発話を書いてください。

- 直前の人間の発話に、話し言葉で自然に答える
- 知っていることの中に答えがあるなら、その部分だけを使う。他の部分は話さない
- 知らないことは知らないと言う。ただし人がそう言うように言う
- 知らないときは、聞き返したり、知っている範囲に引き戻してもよい
- そのまま並べず、自分の言葉で話す
- 相手の口調に合わせる。丁寧語の相手には丁寧語、タメ口の相手にはタメ口

{common}

例：

あなたが知っていること:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: ねえ、日本で一番長い川ってどこだっけ？
moshi: 信濃川だよ。長野から新潟まで流れて、三百六十七キロあるんだって。

あなたが知っていること:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: へえ、そんなに長いんだ。
moshi: うん、意外だよね。

あなたが知っていること:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: 一番大きい川とは違うの？
moshi: 流域面積だと利根川が日本で一番なんだよ。長さとは別なんだね。

あなたが知っていること:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: 信濃川って昔から今の名前だったの？
moshi: うーん、そこまでは知らないなあ。

あなたが知っていること:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: 信濃川の水質ってどうなの？
moshi: 水質か。飲み水のこと？それとも魚とか？

あなたが知っていること:
[1] 信濃川は長野県から新潟県を流れる全長三百六十七キロの川で、日本で最も長い。長野県内では千曲川と呼ばれる。
[2] 利根川は関東地方を流れる川で、流域面積が日本最大である。全長は三百二十二キロ。
[3] 石狩川は北海道を流れる川で、全長は二百六十八キロ。かつては蛇行が激しかった。
ここまでの会話:
人間: 信濃川で釣りできる場所ってある？
moshi: 釣り場はわからないなあ。長野だと千曲川って呼ばれてる区間なんだけどね。"""

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


def call(client, model: str, system: str, user: str, effort: str, tries: int = 3) -> str:
    """思考が混ざった出力は捨てて引き直す。tries 回とも駄目なら例外にする
    （黙って思考混じりを返すより、その会話を落とす方が学習データとして安全）。"""
    last = ""
    for attempt in range(tries):
        out = _call_once(client, model, system, user, effort)
        if not looks_like_thinking(out):
            return out
        last = out
        print(f"    [警告] 思考の混入を検出、引き直す（{attempt + 1}/{tries}）: {out[:60]}",
              flush=True)
    raise RuntimeError(f"思考の混入が {tries} 回続いた: {last[:120]}")


def _call_once(client, model: str, system: str, user: str, effort: str) -> str:
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
    topic = first_line(call(client, model, TOPIC_ROLE,
                            f"質問: {qa['question']}\n話題:", args.effort_chunk))
    topic = strip_label(topic, "話題")
    chunks = make_chunks(client, model, qa, args.effort_chunk)
    ch = render_chunks(chunks)

    # ---- stage 1 : plain conversation -----------------------------------------------
    plain: list[tuple[str, str]] = [("moshi", rng.choice(GREETINGS))]
    ended = False
    for turn in range(args.max_turns):
        tr = "\n".join(f"{w}: {t}" for w, t in plain)
        h = first_line(call(
            client, model, HUMAN_ROLE.format(min_turns=args.min_turns, common=COMMON),
            f"話題: {topic}\n人物設定: {persona}\n"
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
            f"あなたが知っていること:\n{ch}\nここまでの会話:\n{tr}\nmoshi:", args.effort_assistant))
        a = strip_label(a, "moshi", "アシスタント")
        if not a:
            break
        plain.append(("moshi", a))

    # ---- stage 2a : which passage did each assistant utterance use? ------------------
    tr = "\n".join(f"{w}: {t}" for w, t in plain)
    verdict_raw = call(client, model, JUDGE_ROLE,
                       f"資料:\n{ch}\n会話:\n{tr}\n\n判定:", args.effort_judge)
    # 判定役は 1 つの発話に複数の資料を挙げることがある（例「2: 1,2」）。すべて拾う。
    # 以前は最初の番号しか拾わず、2 本目以降の資料の内容が body に出ているのに
    # Reference 行に載らなかった（augmented ターンの 16〜18% が該当。2026-09-26 に修正）。
    used: dict[int, list[int]] = {}
    for line in verdict_raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[:：]\s*(.+)", line)
        if not m:
            continue
        idx = int(m.group(1))
        val = m.group(2).strip()
        if "なし" in val:
            used[idx] = []
            continue
        nums = [int(x) for x in re.findall(r"\d+", val)]
        used[idx] = [n for n in dict.fromkeys(nums) if 1 <= n <= len(chunks)]

    # ---- stage 2b : assemble the record --------------------------------------------
    record: list[str] = []
    greeting_lines = (0, 1)
    a_no = 0
    for pos, (who, text) in enumerate(plain):
        if who == "人間":
            record.append(f"Human: {text}")
            continue
        a_no += 1
        cidx = used.get(a_no) or []
        if a_no == 1 or not cidx:
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
        record.append(f"Reference: {' '.join(chunks[c - 1] for c in cidx)}")
        # subtopic dropped: it is generation-time metadata that never reaches the text
        # stream, and deriving it from the seed question produced garbage. The chunk
        # index is recorded in the JSON instead, which is more useful for analysis.
        record.append(f"moshi (body): {text}")
        record.append("moshi (tail): [empty]")

    return {
        "seed_question": qa["question"],
        "topic": topic,
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
        "code_commit": CODE_COMMIT,
        "code_dirty": CODE_DIRTY,
    }


def main(args) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
                    base_url=args.llm_base_url or None)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with Path(args.input_file).open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    # F5: sample at random. Taking the head of the QA file gave 20 topics that were all
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
    # シャード分割。添字 i は分割前のものを保つ（stem と乱数種が i から決まるため、
    # 分割の有無で生成結果が変わらない）。
    targets = [(i, qa) for i, qa in enumerate(rows) if i % args.nshard == args.shard]
    print(f"  シャード {args.shard}/{args.nshard}: 担当 {len(targets)} 件 / 全 {len(rows)} 件",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, qa) for i, qa in targets]
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
    p.add_argument("--shard", type=int, default=0,
                   help="担当するシャード番号（0 起算）")
    p.add_argument("--nshard", type=int, default=1,
                   help="シャード総数。i %% nshard == shard の会話だけを生成する")
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
