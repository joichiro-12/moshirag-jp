"""開発用：s02 が作った会話を LLM で検査する（事後検査、工程 04）。本番では使わない。

s02 の出力ディレクトリにある会話（<stem>.json）を 1 件ずつ検査し、結果を会話の JSON の "check" に書く。
不合格の会話は rejected/ に移す（s03 は出力ディレクトリの直下しか読まないので、先へ進まない）。
検査の基準は dev/prompts/04_check.txt。

Usage:
    python moshirag_data/dev/check_dialogue.py --input_dir <s02 の出力> \\
        --model "$NAME" --llm_base_url "http://localhost:8000/v1"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text_dialogue"))
from libs.llm import call, first_line  # noqa: E402
from libs.persona import MOSHI_PERSONA  # noqa: E402

PROMPT = Path(__file__).resolve().parent / "prompts" / "04_check.txt"
_print_lock = threading.Lock()


def check_role() -> str:
    text = PROMPT.read_text(encoding="utf-8")
    text = text[:-1] if text.endswith("\n") else text
    return text.format(persona=MOSHI_PERSONA)


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


def main(args) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
                    base_url=args.llm_base_url or None)
    src = Path(args.input_dir)
    rej = src / "rejected"
    rej.mkdir(parents=True, exist_ok=True)
    system = check_role()

    # 検査済み（"check" がある）会話は飛ばす。rejected/ にあるものも検査済み
    targets = [p for p in sorted(src.glob("*.json"))
               if "check" not in json.loads(p.read_text(encoding="utf-8"))]
    print(f"  検査する会話 {len(targets)} 件（検査済みは飛ばす）", flush=True)

    def work(p: Path):
        conv = json.loads(p.read_text(encoding="utf-8"))
        for attempt in range(3):
            try:
                raw = call(client, args.model, system, f"台本:\n{render_script(conv['turns'])}\n\n検査:",
                           args.effort)
                passed = parse_check(raw)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return p.stem, f"FAIL {e}"
        conv["check"] = {"passed": passed, "raw": raw, "model": args.model}
        p.write_text(json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8")
        if passed:
            return p.stem, "ok"
        # 不合格は台本（.txt）ごと rejected/ に移す
        p.rename(rej / p.name)
        txt = p.with_suffix(".txt")
        if txt.exists():
            txt.rename(rej / txt.name)
        return p.stem, "reject"

    n = {"ok": 0, "reject": 0, "FAIL": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(work, p) for p in targets]):
            stem, msg = fu.result()
            n[msg.split()[0]] += 1
            with _print_lock:
                print(f"[{msg.split()[0]}] {stem} {msg}", flush=True)
    print(f"\nDone: 合格 {n['ok']} / 不合格 {n['reject']} / 失敗 {n['FAIL']} -> 不合格は {rej}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True, help="s02 の出力ディレクトリ")
    p.add_argument("--model", required=True)
    p.add_argument("--llm_base_url", default="http://localhost:8000/v1")
    p.add_argument("--api_key", default="")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--effort", default="low")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
