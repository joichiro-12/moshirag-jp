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

from llm import call, first_line, strip_label
from persona import (SPOKEN_STYLE, MOSHI_KNOWLEDGE, MOSHI_NAME_ROLE, MOSHI_PERSONA, TYPES,
                     render_persona, render_scenario)
from wiki import read_articles

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
# プロンプト
# ======================================================================================
# ---- 01 ユーザ発話 ------------------------------------------------------------------
USER_ROLE = """\
音声対話 AI と人の会話を作っています。あなたは人の側の次の 1 発話を書いてください。
あなたはシナリオの人物になりきります。話し相手は {moshi}。

- 会話はあなたから始める。最初の発話は、状況に合った呼びかけやあいさつから入ってよい
- シナリオの状況と関心に沿って話す。ただし関心を一度に全部言わない
- 次に何を言うかは、相手の直前の応答を受けて決める。納得・驚き・聞き返し・異論・脱線など、相手の答えに反応する
- 相手がまだ言っていない事実を、自分から言わない。前提知識の程度を超える専門用語を使わない
- 相手はあなたの事情を知らない。必要なら、誰の何についてかを自分の言葉で言う
- 書き言葉のまま読み上げない。試験問題のような聞き方をしない
- 一度に一つのことだけ言う。40 文字程度まで
- 年齢・性別・職業に合った言葉づかいにする
- 会話の終わり方：{end}
- 会話を終えるときは、最後の発話の行末に EOC と書く

{spoken_style}

例：

人物: 年齢層: 30代 / 性別: 女性 / 職業: 会社員 / 前提知識: 全然知らない
会話の型: 目的達成
シナリオ:
- 状況: 週末の家族旅行の行き先を考えていて、車を運転しながら話しかけている
- 最初の関心: 記事の川の近くで、子どもと楽しめる場所があるか
- 派生しうる関心: その川がどうして有名なのか
- なぜ関心があるのか: 子どもが川遊びをしたがっている
ここまでの会話:
（まだ無い）
人: ねえ、信濃川の近くって、子ども連れで遊べるところあるかな。

人物: 年齢層: 60代以上 / 性別: 男性 / 職業: 退職者 / 前提知識: 少し知っている
会話の型: 確認
シナリオ:
- 状況: テレビの旅番組を見ながら、ふと思い出して話しかけている
- 最初の関心: 日本で一番長い川がどこだったか
- 派生しうる関心: 長野県での呼び名
- なぜ関心があるのか: 若いころ新潟に住んでいて、学校でそう習った気がする
ここまでの会話:
人: もしもし、ちょっと聞きたいことがあるんだけど。
モシ: はい、どうぞ。何でも聞いてください。
人: 日本で一番長い川って、信濃川で合ってたかな。
モシ: はい、信濃川で合っています。長さは三百六十七キロほどあります。
人: そうそう、やっぱりね。ありがとう。EOC"""

# ---- 02 検索の要否 ------------------------------------------------------------------
NEED_ROLE = """\
音声対話 AI「モシ」と人の会話を渡します。
モシが人の直前の発話に応じる前に、検索して参照チャンクを得る必要があるかを判定してください。

モシの知識の範囲:
{knowledge}

判定の基準：
- 要：よい応答に、モシが検索しないと正確に言えない具体的な事実が入る
- 不要：相槌・感想・あいさつ・お礼・別れ、一般常識で足りること、モシ自身のこと

出力は「要」か「不要」の 1 語だけ。

例：

会話:
人: もしもし、ちょっといい？
判定: 不要

会話:
人: 日本で一番長い川って、どこだったっけ。
判定: 要

会話:
モシ: はい、信濃川です。長さは三百六十七キロほどあります。
人: へえ、そんなに長いんだ。
判定: 不要

会話:
人: 明日の新潟の天気ってどうかな。
判定: 不要

会話:
人: ところで、モシって誰が作ったの？
判定: 不要"""

# ---- 02-1 lead ----------------------------------------------------------------------
LEAD_ROLE = """\
音声対話 AI「モシ」と人の会話を作っています。
モシが本題を話し始める直前の、短い前置きを 1 つ書いてください。

モシのペルソナ:
{persona}

- 9〜18 モーラ。およそ 2〜4 秒で話せる長さ。短すぎても長すぎてもいけない
- この後モシが何と答えるかは、あなたは知らない。答えを言わない、推測もしない
- 検索していることを相手に悟らせない
- フィラー（えーと、あのー、うーん、あー）は 1〜2 個まで
- 相手が聞いたことを言い直すときは、短い名詞句にする。疑問文をそのまま埋め込まない

{spoken_style}

例：

ここまでの会話:
人: ねえ、日本で一番長い川ってどこだっけ？
前置き: えーと、日本で一番長い川ですね。

ここまでの会話:
人: 味噌に赤いのと白いのがありますよね。あれはどう違うんですか。
前置き: ああ、赤味噌と白味噌ですね。

ここまでの会話:
モシ: 江戸時代の一七三三年に、隅田川で行われたのが始まりと言われています。
人: へえ、そんな前から。何のためにやったの？
前置き: うーん、何のためだったか、ですね。"""

# ---- 02-2 参照チャンク ----------------------------------------------------------------------
REF_ROLE = """\
音声対話 AI が使う検索の仕組みを模しています。
人の直前の発話に答えるための参照チャンクを、記事（Wikipedia の記事）から 1 つ作ってください。

- 記事に書かれていることだけを書く。記事に無いことは、一般に知られていることでも書かない
- 人が知りたいことに答える部分を中心にする。関連する周辺の事実も、記事に書かれていれば含めてよい
- 人が知りたいことが記事に書かれていなければ、そのことを「〇〇は不明。」と書く。推測や一般知識で埋めない
- 音声対話 AI は記事のことを知らないため、記事自体のことは記載しない
- 200 字まで、1 行で書く
- だ・である調
- 単独で読んで意味が通るように書く（「この作家は」「同年に」のような書き方は使わない）

出力は参照チャンクの 1 行だけ。

例：

記事タイトル: 信濃川
記事: 信濃川（しなのがわ）は、新潟県および長野県を流れ日本海に注ぐ一級河川。全長367kmは日本一の長さ。長野県では千曲川と呼ばれる。…
会話:
人: 日本で一番長い川って、どこだったっけ。
参照チャンク: 信濃川は長野県と新潟県を流れて日本海に注ぐ一級河川で、全長367kmは日本で最も長い。長野県内では千曲川と呼ばれ、新潟県に入ってから信濃川と呼ばれる。

記事タイトル: 信濃川
記事: 信濃川（しなのがわ）は、新潟県および長野県を流れ日本海に注ぐ一級河川。全長367kmは日本一の長さ。長野県では千曲川と呼ばれる。…
会話:
人: 信濃川って、昔から今の名前だったの？
参照チャンク: 信濃川がいつから今の名前で呼ばれているかは不明。長野県内では千曲川と呼ばれ、新潟県に入ってから信濃川と呼ばれる。"""

# ---- 03 body / 応答 -----------------------------------------------------------------
BODY_ROLE = """\
音声対話 AI「モシ」と人の会話を作っています。モシの次の 1 発話を書いてください。

モシのペルソナ:
{persona}

- 参照チャンクは、モシが検索して得た短い文書。検索したターンだけ渡される
- 人の直前の発話に、話し言葉で自然に応じる
- 参照チャンクが渡されたとき：具体的な事実は参照チャンクに書かれていることだけを使う。必要な部分だけを、自分の言葉で話す。参照チャンクに答えが無い、または「不明」と書かれていれば、分かりませんと言う
- 参照チャンクが無いとき：名前・年・数値などの具体的な事実を新しく言わない。一般常識と、会話にすでに出たことで応じる。分からないことは分からないと言う
- 前置きを話し始めているときは、その続きから話す。前置きを繰り返さない
- 自分の名前は、相手に聞かれたときだけ言う。あいさつや返事で名乗らない
- 「参照チャンク」「資料」「記事」「調べる」「検索」という言葉は使わない
- 1〜2 文、60 文字程度まで

{spoken_style}

例：

参照チャンク: 信濃川は長野県と新潟県を流れて日本海に注ぐ一級河川で、全長367kmは日本で最も長い。長野県内では千曲川と呼ばれる。
ここまでの会話:
人: 日本で一番長い川って、どこだったっけ。
モシ（話し始めている前置き）: えーと、日本で一番長い川ですね。
続き: 信濃川です。長野から新潟まで流れていて、三百六十七キロほどあります。

参照チャンク: なし
ここまでの会話:
モシ: 信濃川です。長野から新潟まで流れていて、三百六十七キロほどあります。
人: へえ、そんなに長いんだ。
続き: そうなんです、意外と長いですよね。

参照チャンク: 信濃川がいつから今の名前で呼ばれているかは不明。長野県内では千曲川と呼ばれ、新潟県に入ってから信濃川と呼ばれる。
ここまでの会話:
人: 信濃川って、昔から今の名前だったの？
モシ（話し始めている前置き）: うーん、昔の名前ですか。
続き: そこまでは分からないです。長野では千曲川と呼ばれている、というのは知っています。

参照チャンク: なし
ここまでの会話:
人: もしもし、ちょっと聞いてもいい？
続き: はい、どうぞ。何でも聞いてください。

参照チャンク: なし
ここまでの会話:
人: そういえば、あなたは何ていう名前なの？
続き: モシといいます。エルエルエムジェイピーという研究プロジェクトで作られました。

参照チャンク: なし
ここまでの会話:
人: 明日の新潟の天気ってどうかな。
続き: ごめんなさい、私はインターネットを見られないので、天気は分からないんです。"""

# ---- 04 事後検査 --------------------------------------------------------------------
CHECK_ROLE = """\
音声対話 AI「モシ」と人の会話の台本を渡します。学習データとして使えるかを検査してください。
モシの発話のうち、検索したターンには［前置き］［本題］の区切りと、そのとき得た［参照チャンク］を付けてあります。

モシのペルソナ:
{persona}

次のどれかに当てはまれば不合格です。
1. 人が、モシがまだ言っていない具体的な事実（答え）を先に言っている
2. 参照チャンクの無いターンで、モシが一般常識を超える具体的な事実（名前・年・数値など）を言っている
3. 参照チャンクのあるターンで、モシの本題が参照チャンクと食い違う、参照チャンクに無い具体的な事実を足している、または参照チャンクで不明とされていることに答えている
4. 前置きが答えを先に言っている、または本題とつながらない
6. 同じ内容の繰り返し、話のかみ合わない応答、途中で切れた発話がある

出力の 1 行目は「合格」か「不合格」。不合格のときは 2 行目以降に、当てはまる番号と該当箇所を短く書く。"""


# ======================================================================================
# 1 会話の生成
# ======================================================================================
SPEAKER = {"user": "人", "moshi": "モシ"}


def render_history(hist: list[tuple[str, str]]) -> str:
    # 会話はユーザから始まるので、最初の 01 では履歴が空
    return "\n".join(f"{SPEAKER[w]}: {t}" for w, t in hist) or "（まだ無い）"


def parse_need(out: str) -> bool:
    s = first_line(strip_label(out, "判定"))
    if "不要" in s:
        return False
    if "要" in s:
        return True
    raise RuntimeError(f"検索の要否を解釈できない: {out[:60]}")


def parse_check(out: str) -> bool:
    s = first_line(out)
    if "不合格" in s:
        return False
    if "合格" in s:
        return True
    raise RuntimeError(f"事後検査の結果を解釈できない: {out[:60]}")


def render_script(turns: list[dict]) -> str:
    """04 に渡す台本。検索したターンには前置き・本題の区切りと参照チャンクを付ける。"""
    lines = []
    for t in turns:
        lines.append(f"人: {t['user']}")
        if t.get("body") is None:
            continue
        if t["need"]:
            lines.append(f"モシ: ［前置き］{t['lead']}［本題］{t['body']}")
            lines.append(f"　［参照チャンク］{t['reference']}")
        else:
            lines.append(f"モシ: {t['body']}")
    return "\n".join(lines)


def generate(sc: dict, article: str, client, model, args) -> dict:
    persona_txt = render_persona(sc["persona"])
    scenario_txt = render_scenario(sc["scenario"])
    ty = sc["type"]
    user_sys = USER_ROLE.format(moshi=MOSHI_NAME_ROLE, end=TYPES[ty]["end"], spoken_style=SPOKEN_STYLE)
    user_head = f"人物: {persona_txt}\n会話の型: {ty}\nシナリオ:\n{scenario_txt}\n"
    need_sys = NEED_ROLE.format(knowledge=MOSHI_KNOWLEDGE)
    lead_sys = LEAD_ROLE.format(persona=MOSHI_PERSONA, spoken_style=SPOKEN_STYLE)
    body_sys = BODY_ROLE.format(persona=MOSHI_PERSONA, spoken_style=SPOKEN_STYLE)
    ref_head = f"記事タイトル: {sc['title']}\n記事: {article}\n"

    hist: list[tuple[str, str]] = []   # 会話はユーザから始める
    turns: list[dict] = []
    ended = False
    for k in range(args.max_turns):
        # ---- 01 ユーザ発話：シナリオ・会話履歴 ----------------------------------------
        u = first_line(call(client, model, user_sys,
                            f"{user_head}ここまでの会話:\n{render_history(hist)}\n人:",
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
        need_raw = call(client, model, need_sys,
                        f"会話:\n{render_history(hist)}\n判定:", args.effort_need)
        turn["need"] = parse_need(need_raw)
        turn["need_raw"] = need_raw

        lead = ref = ""
        if turn["need"]:
            # ---- 02-1 lead：会話履歴・ペルソナ ----------------------------------------
            lead = first_line(call(client, model, lead_sys,
                                   f"ここまでの会話:\n{render_history(hist)}\n前置き:",
                                   args.effort_lead))
            lead = strip_label(lead, "前置き", "モシ")
            # ---- 02-2 参照チャンク：直前のユーザ発話までの会話履歴・記事（lead は見せない）------
            ref = first_line(call(client, model, REF_ROLE,
                                  f"{ref_head}会話:\n{render_history(hist)}\n参照チャンク:",
                                  args.effort_ref))
            ref = strip_label(ref, "参照チャンク", "参照")
            if not lead or not ref:
                raise RuntimeError("lead か参照チャンクが空")

        # ---- 03 body / 応答：lead までの会話履歴・ペルソナ・参照チャンク ----------------------
        q = f"参照チャンク: {ref or 'なし'}\nここまでの会話:\n{render_history(hist)}\n"
        if lead:
            q += f"モシ（話し始めている前置き）: {lead}\n"
        body = first_line(call(client, model, body_sys, q + "続き:", args.effort_body))
        body = strip_label(body, "続き", "モシ")
        if not body:
            raise RuntimeError("モシの応答が空")
        turn.update(lead=lead, reference=ref, body=body, ref_has_unknown="不明" in ref)
        turns.append(turn)
        hist.append(("moshi", lead + body))

    # ---- 04 事後検査：台本全体 ----------------------------------------------------------
    script = render_script(turns)
    check_raw = call(client, model, CHECK_ROLE.format(persona=MOSHI_PERSONA),
                     f"台本:\n{script}\n\n検査:", args.effort_check)
    passed = parse_check(check_raw)

    # ---- s03 が読む形式に組み立てる ----------------------------------------------------
    record = []
    for t in turns:
        record.append(f"Human: {t['user']}")
        if t.get("body") is None:
            continue
        if t["need"]:
            record += ["(augmented)", f"moshi (lead): {t['lead']}",
                       f"Reference: {t['reference']}", f"moshi (body): {t['body']}",
                       "moshi (tail): [empty]"]
        else:
            record += ["(unaugmented)", f"moshi: {t['body']}"]

    return {
        "wiki_id": sc["wiki_id"],
        "title": sc["title"],
        "url": sc.get("url"),
        "source": sc.get("source", ""),
        "topic": sc["title"],                 # s03 が struct.json に写す
        "persona": sc["persona"],
        "type": ty,
        "scenario": sc["scenario"],
        "article_chars": len(article),
        "chunks": [t["reference"] for t in turns if t.get("need")],   # s03 が struct.json に写す
        "turns": turns,
        "check": {"passed": passed, "raw": check_raw},
        "record": record,
        "n_user_turns": len(turns),
        "n_augmented": sum(1 for t in turns if t.get("need")),
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
    rej = out / "rejected"   # 04 で不合格の会話。s03 は output_dir 直下しか読まない
    rej.mkdir(parents=True, exist_ok=True)

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
        for attempt in range(3):
            try:
                conv = generate(sc, article, client, args.model, args)
                conv["stem"] = stem
                if not conv["check"]["passed"]:
                    (rej / f"{stem}.json").write_text(
                        json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
                    return stem, f"reject turns={conv['n_user_turns']}"
                pj.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
                pt.write_text("\n".join(conv["record"]) + "\n", encoding="utf-8")
                return stem, (f"ok turns={conv['n_user_turns']} aug={conv['n_augmented']} "
                              f"eoc={conv['ended_by_eoc']}")
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return stem, f"FAIL {e}"
        return stem, "FAIL"

    n = {"ok": 0, "reject": 0, "skip": 0, "FAIL": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i, sc) for i, sc in targets]
        for fu in as_completed(futs):
            stem, msg = fu.result()
            n[msg.split()[0]] += 1
            with _print_lock:
                print(f"[{msg.split()[0]}] {stem} {msg}", flush=True)
    print(f"\nDone: 合格 {n['ok']} / 不合格 {n['reject']} / 既存 {n['skip']} / 失敗 {n['FAIL']}"
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
    p.add_argument("--effort_check", default="low")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
