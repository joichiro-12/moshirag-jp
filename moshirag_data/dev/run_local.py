"""Wikipedia の記事の束（例：data/wiki_raw_sample100.json）を、手元で s01〜s10 まで通して学習データにする。

Usage:
    # 何が流れるかだけを見る（何も呼ばない）
    .venv/bin/python moshirag_data/dev/run_local.py --input data/wiki_raw_sample100.json \\
        --work data/local_sample100 --dry_run
    # まず 5 件で s01〜s03 だけ
    .venv/bin/python moshirag_data/dev/run_local.py --input data/wiki_raw_sample100.json \\
        --work data/local_sample100 --limit 5 --steps s01-s03
    # 残り全部
    .venv/bin/python moshirag_data/dev/run_local.py --input data/wiki_raw_sample100.json \\
        --work data/local_sample100
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import statistics as st
import subprocess
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = ([str(REPO / "moshirag_data" / d) for d in ("text_dialogue", "speech_dialogue", "postprocess", "dev")]
                + [str(REPO)])

from libs.persona import CONSTANTS_DIR, PROMPTS_DIR  # noqa: E402  （上の sys.path を足してから読む）


# ======================================================================================
# LLM の使用量の計測。s01・s02・libs には手を入れず、OpenAI の SDK の create をこの script の中でだけ包む
# ======================================================================================
# (model, tag) → 回数・トークン数・秒。tag はプロンプトのファイル名（"01_user" など）で、
# system プロンプトの 1 行目を prompts/ の各ファイルの 1 行目と突き合わせて決める。
# 思考の混入で s01・s02 が捨てて引き直した応答も、呼び出しとして数える
_FIELDS = ("calls", "errors", "prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_tokens")
_USAGE: dict = {}
_USAGE_LOCK = threading.Lock()
_TAGS = {p.read_text(encoding="utf-8").split("\n", 1)[0]: p.stem
         for d in (PROMPTS_DIR, Path(__file__).resolve().parent / "prompts") for p in d.glob("*.txt")}


def _record(model: str, tag: str, r, seconds: float, error: bool = False) -> None:
    u = getattr(r, "usage", None)
    pd_ = getattr(u, "prompt_tokens_details", None)
    cd_ = getattr(u, "completion_tokens_details", None)
    with _USAGE_LOCK:
        row = _USAGE.setdefault((model, tag), {**dict.fromkeys(_FIELDS, 0), "seconds": 0.0})
        row["calls"] += 1
        row["errors"] += error
        row["seconds"] += seconds
        row["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        row["cached_tokens"] += getattr(pd_, "cached_tokens", 0) or 0
        row["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0   # 思考のトークンを含む
        row["reasoning_tokens"] += getattr(cd_, "reasoning_tokens", 0) or 0


def install_usage_meter() -> None:
    """openai の Completions.create を、時間とトークン数を数えるものに差し替える（1 回だけ）。"""
    from openai.resources.chat.completions import Completions
    orig = Completions.create
    if getattr(orig, "_metered", False):
        return

    def create(self, *args, **kw):
        msgs = kw.get("messages") or []
        first = (msgs[0].get("content") or "").split("\n", 1)[0] if msgs else ""
        model, tag = kw.get("model", "?"), _TAGS.get(first, "?")
        t0 = time.time()
        try:
            r = orig(self, *args, **kw)
        except Exception:
            _record(model, tag, None, time.time() - t0, error=True)
            raise
        _record(model, tag, r, time.time() - t0)
        return r

    create._metered = True
    Completions.create = create


def reset_usage() -> None:
    with _USAGE_LOCK:
        _USAGE.clear()


def usage_rows() -> list[dict]:
    with _USAGE_LOCK:
        rows = [{"model": m, "tag": t, **v} for (m, t), v in sorted(_USAGE.items())]
    for r in rows:
        r["seconds"] = round(r["seconds"], 1)
    return rows


def format_usage(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        lines.append(f"    {r['model']} {r['tag']:10s} {r['calls']:5d} 回（失敗 {r['errors']}）"
                     f" 入力 {r['prompt_tokens']:,}（キャッシュ {r['cached_tokens']:,}）"
                     f" 出力 {r['completion_tokens']:,}（思考 {r['reasoning_tokens']:,}）"
                     f" {r['seconds']:.0f} 秒")
    return "\n".join(lines)

# check は開発用の事後検査（dev/check_dialogue.py）。本番の工程には無い。--no_check で外せる
STEPS = ["s01", "s02", "check", "s03", "s04", "s05", "s06", "s07", "s08", "s09", "s10"]
FRAME_RATE = 12.5

# 外部環境（walkthrough.ipynb のセル 2 と同じ）
HOME = Path.home()
TTS_PKG = HOME / "moshirag_tts" / "src" / "zoom1-tts"
TTS_PY = TTS_PKG / ".venv" / "bin" / "python"
TTS_SYNTH = HOME / "moshirag_tts" / "synth_worker.py"
MFA_ENV = HOME / "miniforge3" / "envs" / "mfa"
MFA_ROOT = HOME / "moshirag_mfa" / "mfa_root"
BEAM_YAML = HOME / "mfa_beam.yaml"
TOK_PY = HOME / "moshirag_tok" / "moshirag-jp" / ".venv" / "bin" / "python"
NEEDS = {"s04": [TTS_PY, TTS_SYNTH], "s05": [MFA_ENV / "bin" / "mfa", MFA_ROOT, BEAM_YAML],
         "s07": [TOK_PY], "s09": [TOK_PY]}


def parse_steps(spec: str) -> list[str]:
    """「s01-s03」「s04,s06」「all」を工程の並びにする。"""
    if spec == "all":
        return STEPS
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        i, j = STEPS.index(a), STEPS.index(b or a)
        out += STEPS[i:j + 1]
    return [s for s in STEPS if s in out]


def pick_gpu() -> str:
    """空きメモリがいちばん多い GPU（ノートブックと同じ選び方）。"""
    if os.environ.get("WALKTHROUGH_GPU"):
        return os.environ["WALKTHROUGH_GPU"]
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                       stdout=subprocess.PIPE, text=True, check=True).stdout
    free = {i.strip(): int(m) for i, m in (l.split(",") for l in q.strip().splitlines())}
    return max(free, key=free.get)


def run_step(cmd, log: Path, env=None, cwd=None):
    """別環境のコマンドを実行し、出力を log に書く。失敗したら末尾を出して止める。"""
    cmd = [str(c) for c in cmd]
    print("  $", " ".join(cmd), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        r = subprocess.run(cmd, env={**os.environ, **(env or {})}, cwd=cwd,
                           stdout=f, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print("".join(log.read_text(encoding="utf-8").splitlines(keepends=True)[-30:]))
        raise SystemExit(f"!! 失敗（終了コード {r.returncode}）。全体は {log}")
    print(f"    → ログ {log}", flush=True)


def stems_in(d: Path, suffix: str) -> set[str]:
    return {p.name[: -len(suffix)] for p in d.glob(f"*{suffix}")} if d.is_dir() else set()


def write_stems(path: Path, stems) -> Path:
    path.write_text("".join(f"{s}\n" for s in sorted(stems)), encoding="utf-8")
    return path


def load_env_file(path: Path) -> None:
    """moshirag_data/.env を環境変数にする（ノートブックのセル 7 と同じ）。"""
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and not k.startswith("#"):
            os.environ[k.strip()] = v.strip().strip("\"'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help='{"records": [{"text": ..., "meta": {"id", "title", "url"}}, ...]} の JSON')
    ap.add_argument("--work", required=True, help="作業場所。消さずに使い続ける")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 件の記事だけ使う（0 で全件）")
    ap.add_argument("--steps", default="all", help="流す工程。例：all / s01-s03 / s04-s10 / s06,s10")
    ap.add_argument("--no_check", action="store_true", help="開発用の事後検査（check）を流さない")
    ap.add_argument("--summary_only", action="store_true",
                    help="工程は流さず、作業場所にある出力から summary.json だけを作り直す（LLM も呼ばない）")
    ap.add_argument("--dry_run", action="store_true", help="流す内容と LLM の呼び出し回数の見積もりだけを出す")
    ap.add_argument("--llm_backend", choices=["openai", "vllm"], default="openai")
    ap.add_argument("--llm_model", default="gpt-6-luna", help="openai のときのモデル（ノートブックと同じ）")
    ap.add_argument("--llm_base_url", default=None, help="省略時は openai / localhost:8000")
    ap.add_argument("--llm_workers", type=int, default=8, help="s01・s02 の並列数")
    ap.add_argument("--mfa_jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    steps = [s for s in parse_steps(a.steps) if not (a.no_check and s == "check")]
    work = Path(a.work).resolve()
    logs = work / "logs"
    S01, S02, S03 = work / "s01", work / "s02", work / "s03"
    AUDIO, S05_WORK, WORDS = work / "s04_audio", work / "s05_work", work / "s05_words"
    RET, TOK_AUDIO, TOK_TEXT = work / "s06_ret", work / "s07_tok_audio", work / "s08_tok_text"
    REF, S10 = work / "s09_ref_embed", work / "s10_parquet"
    WIKI = S01 / "ja_wiki"
    SC = S01 / "wiki_scenario.jsonl"     # s02 の stem は wiki_scenario_NNNNNN になる

    records = json.loads(Path(a.input).read_text(encoding="utf-8"))["records"]
    if a.limit:
        records = records[: a.limit]
    print(f"記事 {len(records)} 件 / 工程 {', '.join(steps)} / 作業場所 {work}", flush=True)

    if a.summary_only:
        write_summary(work)
        return

    if a.dry_run:
        # s02 は 1 往復に 01・02・03 の 3 回（挨拶・雑談で始まる最初の往復は 02 を呼ばず 2 回）、
        # 検索するときは 02-1・02-2 が足されて 5 回。最後に EOC を出す 01 が 1 回。
        # 最小は「挨拶 → 返事 → EOC」の 3 回、最大は 8 往復すべて検索した 40 回。check は 1 会話 1 回
        print(f"  s01 の LLM 呼び出し：{len(records)} 回（記事ごとに 1 回。失敗しても引き直しは 3 回まで）")
        print(f"  s02 の LLM 呼び出し：1 会話あたり 3〜40 回。{len(records)} 会話なら "
              f"{len(records) * 3:,}〜{len(records) * 40:,} 回（失敗した会話は 3 回まで作り直す）")
        if "check" in steps:
            print(f"  check の LLM 呼び出し：{len(records)} 回（会話ごとに 1 回）")
        for s in steps:
            miss = [str(p) for p in NEEDS.get(s, []) if not p.exists()]
            print(f"  {s}: {'環境 OK' if not miss else '見つからない: ' + ', '.join(miss)}")
        return

    for s in steps:
        for p in NEEDS.get(s, []):
            assert p.exists(), f"{s} に要るものが見つからない: {p}（README の前提の環境がこのマシンにあるか確認）"
    gpu = pick_gpu() if set(steps) & {"s04", "s07", "s09"} else None
    if gpu is not None:
        print(f"GPU {gpu}", flush=True)
    work.mkdir(parents=True, exist_ok=True)

    llm_args = []
    if set(steps) & {"s01", "s02", "check"}:
        install_usage_meter()
        if a.llm_backend == "openai":
            load_env_file(REPO / "moshirag_data" / ".env")
            assert os.environ.get("OPENAI_API_KEY"), "moshirag_data/.env に OPENAI_API_KEY が無い"
            base, model = a.llm_base_url or "https://api.openai.com/v1", a.llm_model
        else:
            import urllib.request
            base = a.llm_base_url or "http://localhost:8000/v1"
            model = json.loads(urllib.request.urlopen(base.rstrip("/") + "/models").read())["data"][0]["id"]
        llm_args = ["--model", model, "--llm_base_url", base]
        print(f"LLM {a.llm_backend} / {model} / {base}", flush=True)

    # ---- 記事を ja_wiki と同じ形（*.jsonl.gz）にする。s01・s02 の --wiki_dir に渡す ----------
    WIKI.mkdir(parents=True, exist_ok=True)
    with gzip.open(WIKI / "sample.jsonl.gz", "wt", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps({"text": r["text"], "meta": r["meta"]}, ensure_ascii=False) + "\n")

    t_all = time.time()
    run_id = time.strftime("%Y-%m-%dT%H:%M:%S")
    for s in steps:
        print(f"\n===== {s} =====", flush=True)
        before = count_outputs(work, s)
        reset_usage()
        t0 = time.time()
        ok = False
        try:

            if s == "s01":   # 記事 → シナリオ（ノートブックのセル 8）
                import s01_scenario as s01
                argv = ["s01_scenario.py", "--wiki_dir", str(WIKI), "--output_file", str(SC),
                        *llm_args, "--effort", "low", "--workers", str(a.llm_workers), "--seed", str(a.seed)]
                with mock.patch.object(sys, "argv", argv):
                    s01.main()

            elif s == "s02":   # シナリオ → 会話（ノートブックのセル 14。件数はシナリオの全件）
                import s02_dialogue as s02
                n_sc = sum(1 for l in SC.open(encoding="utf-8") if l.strip())
                s02.main(s02.build_parser().parse_args([
                    "--input_file", str(SC), "--wiki_dir", str(WIKI), "--output_dir", str(S02),
                    "--num_samples", str(n_sc), "--seed", str(a.seed), *llm_args,
                    "--min_turns", "1", "--max_turns", "8", "--workers", str(a.llm_workers),
                    "--effort_user", "high", "--effort_need", "low", "--effort_lead", "low",
                    "--effort_ref", "low", "--effort_body", "low", "--resume",
                ]))

            elif s == "check":   # 開発用の事後検査。不合格は S02/rejected/ に移るので s03 に進まない
                import check_dialogue
                check_dialogue.main(check_dialogue.build_parser().parse_args([
                    "--input_dir", str(S02), *llm_args, "--workers", str(a.llm_workers), "--effort", "low"]))

            elif s == "s03":   # 会話 → 台本と構造 JSON（セル 20）
                import s03_tts_input as s03
                s03.main(Namespace(input_dir=str(S02), output_dir=str(S03), limit=0))

            elif s == "s04":   # 台本 → 音声（セル 24）。音声がある会話は synth_worker が飛ばす
                run_step([TTS_PY, TTS_SYNTH, "--in_dir", S03, "--out_dir", AUDIO, "--oom_retries", "0"],
                         logs / "s04_synth.log", cwd=TTS_PKG,
                         env={"TTS_DIR": str(TTS_PKG), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                              "CUDA_VISIBLE_DEVICES": gpu})

            elif s == "s05":   # 語アライメント（セル 26・28・29。MFA は 1 回、beam 100 / retry 400）
                import s05_align as s05
                stems = sorted(stems_in(AUDIO, ".manifest.json"))
                bases = s05.build_corpus(AUDIO, stems, S05_WORK / "corpus")
                print(f"  コーパス {len(bases)} ターン / {len(stems)} 会話", flush=True)
                (S05_WORK / "out").mkdir(parents=True, exist_ok=True)
                run_step([MFA_ENV / "bin" / "mfa", "align", "--clean", "-j", str(a.mfa_jobs), "--output_format", "json",
                          "-c", BEAM_YAML, S05_WORK / "corpus", "japanese_mfa", "japanese_mfa", S05_WORK / "out"],
                         logs / "s05_mfa.log",
                         env={"MFA_ROOT_DIR": str(MFA_ROOT), "PATH": f"{MFA_ENV}/bin:{os.environ['PATH']}"})
                print(f"  {dict(s05.to_words(AUDIO, stems, [S05_WORK / 'out'], WORDS))}", flush=True)

            elif s == "s06":   # <ret> の位置と d_lead（セル 31）
                import s06_ret as s06
                s06.main(Namespace(manifest_dir=str(AUDIO), struct_dir=str(S03), output_dir=str(RET)))

            elif s == "s07":   # 音声トークン（セル 35）
                TOK_AUDIO.mkdir(parents=True, exist_ok=True)
                run_step([TOK_PY, "-m", "moshirag_data.postprocess.s07_tok_audio",
                          "--audio_dir", AUDIO, "--output_dir", TOK_AUDIO, "--num_workers", "1", "--resume"],
                         logs / "s07_tok_audio.log", cwd=REPO, env={"CUDA_VISIBLE_DEVICES": gpu})

            elif s == "s08":   # テキストトークン（セル 37）。語の時刻があり、<ret> の位置も求まった会話だけ
                import tools.tokenize_text as tok_text
                TOK_TEXT.mkdir(parents=True, exist_ok=True)
                stems = sorted((stems_in(WORDS, ".json") & stems_in(RET, ".json")) - stems_in(TOK_TEXT, ".npz"))
                print(f"  対象 {len(stems)} 会話", flush=True)
                args08 = Namespace(word_transcript_dir=str(WORDS), output_dir=str(TOK_TEXT),
                                   text_tokenizer_repo="llm-jp/j-moshi-v1", text_tokenizer_name="tokenizer_spm_32k_3.model",
                                   no_whitespace_before_word=True, text_padding_id=3, end_of_text_padding_id=0,
                                   audio_tokenizer_frame_rate=FRAME_RATE, num_workers=1, resume=False,
                                   allow_missing_speakers=False, allow_alignment_warnings=True)
                if stems:
                    tok_text.worker(0, stems, args08)

            elif s == "s09":   # 参照チャンクのベクトル（セル 40）。出力済みの会話は s09 が飛ばす
                REF.mkdir(parents=True, exist_ok=True)
                sf = write_stems(work / "stems_s09.txt", stems_in(RET, ".json"))
                run_step([TOK_PY, REPO / "moshirag_data" / "postprocess" / "s09_ref_embed.py",
                          "--ret_dir", RET, "--stems", sf, "--out", REF, "--cache", work / "_arc_cache"],
                         logs / "s09_ref_embed.log", cwd=REPO, env={"CUDA_VISIBLE_DEVICES": gpu})

            elif s == "s10":   # parquet（セル 42）。s06〜s09 がすべてそろった会話だけ
                import s10_pack as s10
                stems = (stems_in(TOK_TEXT, ".npz") & stems_in(TOK_AUDIO, ".npz")
                         & stems_in(REF, ".npz") & stems_in(RET, ".json"))
                sf = write_stems(work / "stems_s10.txt", stems)
                argv = ["s10_pack.py", "--tokenized_text_dir", str(TOK_TEXT), "--tokenized_audio_dir", str(TOK_AUDIO),
                        "--ref_embed_dir", str(REF), "--ret_dir", str(RET), "--stems", str(sf),
                        "--output_prefix", str(S10 / "moshirag_jp_local")]
                with mock.patch.object(sys, "argv", argv):
                    s10.main()

            ok = True
        finally:   # 失敗した工程も、かかった時間と LLM の使用量は残す
            rec = log_step(work, run_id, s, ok, time.time() - t0, count_outputs(work, s) - before)
            print(f"  {s} の経過 {rec['seconds']:.0f} 秒 / できた {rec['items']} 件"
                  + (f" / 1 件あたり {rec['seconds_per_item']:.1f} 秒" if rec["seconds_per_item"] else ""), flush=True)
            if rec["llm"]:
                print("  LLM の使用量:\n" + format_usage(rec["llm"]), flush=True)

    write_summary(work)
    print(f"全体の経過 {time.time() - t_all:.0f} 秒")


def write_summary(work: Path) -> None:
    summary = summarize(work)
    (work / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n===== 集計（{work / 'summary.json'}）=====")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def count_outputs(work: Path, step: str) -> int:
    """その工程の出力の件数（工程の前後の差を「できた件数」にする）。"""
    if step == "s01":
        f = work / "s01" / "wiki_scenario.jsonl"
        return sum(1 for l in f.open(encoding="utf-8") if l.strip()) if f.exists() else 0
    if step == "s02":
        return len(stems_in(work / "s02", ".json")) + len(stems_in(work / "s02" / "rejected", ".json"))
    if step == "check":
        return sum("check" in json.loads(p.read_text(encoding="utf-8"))
                   for d in (work / "s02", work / "s02" / "rejected") for p in d.glob("*.json"))
    if step == "s05":
        return len(stems_in(work / "s05_words", ".json")) + len(stems_in(Path(str(work / "s05_words") + "_excluded"), ".json"))
    if step == "s10":
        pq = list((work / "s10_parquet").glob("*.parquet"))
        if not pq:
            return 0
        import pandas as pd
        return sum(len(pd.read_parquet(p, columns=["dialogue_id"])) for p in pq)
    d, suffix = {"s03": ("s03", ".struct.json"), "s04": ("s04_audio", ".manifest.json"), "s06": ("s06_ret", ".json"),
                 "s07": ("s07_tok_audio", ".npz"), "s08": ("s08_tok_text", ".npz"), "s09": ("s09_ref_embed", ".npz")}[step]
    return len(stems_in(work / d, suffix))


STEP_COLUMNS = ["run_id", "step", "ok", "seconds", "items", "seconds_per_item",
                "llm_calls", "llm_errors", "prompt_tokens", "cached_tokens", "completion_tokens",
                "reasoning_tokens", "llm_seconds"]
LLM_COLUMNS = ["run_id", "step", "model", "tag", *_FIELDS, "seconds"]


def append_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """CSV に行を足す。ファイルが無ければ見出しから書く。None は空欄にする。"""
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerows({k: ("" if r.get(k) is None else r.get(k)) for k in columns} for r in rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def log_step(work: Path, run_id: str, step: str, ok: bool, seconds: float, items: int) -> dict:
    """1 工程の記録を step_log.csv（工程 1 行）と llm_usage.csv（工程 × プロンプト 1 行）に足す。
    s03・s06・s10 は毎回全件を作り直すので、2 回目以降の「できた件数」は新しく増えた分だけになる。"""
    rows = usage_rows()
    rec = {"run_id": run_id, "step": step, "ok": ok, "seconds": round(seconds, 1), "items": items,
           "seconds_per_item": round(seconds / items, 2) if items > 0 else None,
           "llm_calls": sum(r["calls"] for r in rows), "llm_errors": sum(r["errors"] for r in rows),
           **{k: sum(r[k] for r in rows) for k in ("prompt_tokens", "cached_tokens", "completion_tokens",
                                                   "reasoning_tokens")},
           "llm_seconds": round(sum(r["seconds"] for r in rows), 1)}
    append_csv(work / "step_log.csv", STEP_COLUMNS, [rec])
    if rows:
        append_csv(work / "llm_usage.csv", LLM_COLUMNS, [{"run_id": run_id, "step": step, **r} for r in rows])
    return {**rec, "llm": rows}


def summarize_log(work: Path) -> dict:
    """step_log.csv・llm_usage.csv の全実行を、工程ごと・プロンプトごとに合計する。"""
    recs = read_csv(work / "step_log.csv")
    if not recs:
        return {}
    steps: dict = {}
    for r in recs:
        a = steps.setdefault(r["step"], {"実行回数": 0, "失敗": 0, "秒": 0.0, "できた件数": 0})
        a["実行回数"] += 1
        a["失敗"] += r["ok"] != "True"
        a["秒"] += float(r["seconds"])
        a["できた件数"] += int(r["items"])
    for a in steps.values():
        a["秒"] = round(a["秒"], 1)
        a["1 件あたり秒"] = round(a["秒"] / a["できた件数"], 2) if a["できた件数"] > 0 else None
    usage: dict = {}
    for u in read_csv(work / "llm_usage.csv"):
        a = usage.setdefault((u["model"], u["tag"]), {**dict.fromkeys(_FIELDS, 0), "seconds": 0.0})
        for k in _FIELDS:
            a[k] += int(u[k])
        a["seconds"] += float(u["seconds"])
    rows = [{"model": m, "tag": t, **a, "seconds": round(a["seconds"], 1)} for (m, t), a in sorted(usage.items())]
    return {
        "工程ごと（全実行の合計）": {s: steps[s] for s in STEPS if s in steps},
        "LLM（全実行の合計、プロンプトごと）": rows,
    }


def summarize(work: Path) -> dict:
    """工程ごとに残った件数と、s02 の会話の中身の集計。"""
    S01, S02 = work / "s01", work / "s02"
    wiki = S01 / "ja_wiki" / "sample.jsonl.gz"
    n_articles = sum(1 for _ in gzip.open(wiki, "rt", encoding="utf-8")) if wiki.exists() else 0
    sc_file = S01 / "wiki_scenario.jsonl"
    convs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(S02.glob("*.json"))]
    rejected = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((S02 / "rejected").glob("*.json"))}
    turns = [t for c in convs for t in c["turns"] if t.get("body") is not None]
    aug = [t for t in turns if t.get("need")]
    d_leads = [e["d_lead"] for p in (work / "s06_ret").glob("*.json")
               for e in json.loads(p.read_text(encoding="utf-8"))["ret_events"]]
    pq = list((work / "s10_parquet").glob("*.parquet"))
    n_rows = 0
    if pq:
        import pandas as pd
        n_rows = sum(len(pd.read_parquet(p, columns=["dialogue_id"])) for p in pq)

    def dist(xs):
        return dict(collections.Counter(xs).most_common())

    return {
        "件数": {
            "記事": n_articles,
            "s01 シナリオ": sum(1 for l in sc_file.open(encoding="utf-8") if l.strip()) if sc_file.exists() else 0,
            "s02 会話（check で不合格になったものを除く）": len(convs),
            "check 合格": sum(1 for c in convs if c.get("check", {}).get("passed")),
            "check 不合格": len(rejected),
            "check 未実施": sum(1 for c in convs if "check" not in c),
            "s03 台本": len(stems_in(work / "s03", ".struct.json")),
            "s04 音声": len(stems_in(work / "s04_audio", ".manifest.json")),
            "s05 語の時刻": len(stems_in(work / "s05_words", ".json")),
            "s05 除外（取りこぼしのあるターンを含む）": len(stems_in(Path(str(work / "s05_words") + "_excluded"), ".json")),
            "s06 <ret> の位置": len(stems_in(work / "s06_ret", ".json")),
            "s07 音声トークン": len(stems_in(work / "s07_tok_audio", ".npz")),
            "s08 テキストトークン": len(stems_in(work / "s08_tok_text", ".npz")),
            "s09 参照チャンクのベクトル": len(stems_in(work / "s09_ref_embed", ".npz")),
            "s10 parquet の行": n_rows,
        },
        "s02（check で不合格になったものを除く会話）": {
            "ユーザの目的": dist(c["user_goal"] for c in convs),
            "話し始め": dist(c["opening"] for c in convs),
            "往復数の平均": round(st.mean(c["n_user_turns"] for c in convs), 2) if convs else None,
            "EOC で終わった割合": round(sum(c["ended_by_eoc"] for c in convs) / len(convs), 3) if convs else None,
            "<ret> を持つ会話の割合": round(sum(c["n_augmented"] > 0 for c in convs) / len(convs), 3) if convs else None,
            "Moshi の応答": len(turns),
            "うち検索した": len(aug),
            "うち聞き返した": sum(1 for t in turns if t.get("clarify")),
            "検索した応答のうち参照チャンクに「不明」を含む": sum(1 for t in aug if t.get("ref_has_unknown")),
            "参照チャンクの根拠（記事・知識・推論の組み合わせ）": dist("・".join(t.get("ref_basis") or []) or "なし"
                                                       for t in aug),
        },
        "check 不合格の理由（全文、会話ごと）": {stem: c["check"]["raw"] for stem, c in rejected.items()},
        "時間と LLM の使用量": summarize_log(work),
        "s06 d_lead（秒）": ({"件数": len(d_leads), "平均": round(st.mean(d_leads), 2),
                             "中央": round(st.median(d_leads), 2),
                             "2〜4 秒の割合": round(sum(2 <= d <= 4 for d in d_leads) / len(d_leads), 3)}
                            if d_leads else None),
    }


if __name__ == "__main__":
    main()
