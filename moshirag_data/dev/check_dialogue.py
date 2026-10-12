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
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "text_dialogue"))
from libs.llm import call, first_line  # noqa: E402
from libs.persona import MOSHI_PERSONA  # noqa: E402

PROMPTS = Path(__file__).resolve().parent / "prompts"
_print_lock = threading.Lock()


def load(name: str) -> str:
    text = (PROMPTS / name).read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def check_role() -> str:
    return load("04_check.txt").format(persona=MOSHI_PERSONA)


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


def render_backchannel_script(script: list[dict]) -> str:
    """相づちの検査に渡す台本。相づちは話し手の発話の中に［番号 聞き手: 相づち］で入れる。"""
    name = {"A": "人", "B": "モシ"}
    lines, key, n = [], None, 0
    for x in script:
        if x["role"] == "backchannel":
            n += 1
            lines[-1] += f"［{n} {name[x['speaker']]}: {x['text']}］"
        elif (x["turn"], x["role"]) == key:
            lines[-1] += x["text"]
        else:
            key = (x["turn"], x["role"])
            lines.append(f"{name[x['speaker']]}: {x['text']}")
    return "\n".join(lines)


_ISSUE = re.compile(r"^\s*(\d+)\s*[:：]\s*([a-e])\b\s*(.*)$")


def parse_backchannel_check(out: str) -> tuple[bool, list[dict]]:
    """（合格か, 当てはまった相づちの一覧）。"""
    passed = parse_check(out)
    issues = [{"index": int(m.group(1)), "type": m.group(2), "reason": m.group(3).strip()}
              for m in map(_ISSUE.match, out.splitlines()[1:]) if m]
    return passed, issues


def main(args) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
                    base_url=args.llm_base_url or None)
    src = Path(args.input_dir)
    rej = src / "rejected"
    rej.mkdir(parents=True, exist_ok=True)
    system = check_role()
    bc_system = load("04_check_backchannel.txt")

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
        if "script" in conv:
            n_bc = sum(1 for x in conv["script"] if x["role"] == "backchannel")
            res = {"model": args.model, "n_backchannel": n_bc}
            if n_bc == 0:
                res.update(passed=True, raw="", issues=[])
            else:
                for attempt in range(3):
                    try:
                        bc_raw = call(client, args.model, bc_system,
                                      f"台本:\n{render_backchannel_script(conv['script'])}\n\n検査:", args.effort)
                        bc_passed, issues = parse_backchannel_check(bc_raw)
                        res.update(passed=bc_passed, raw=bc_raw, issues=issues)
                        res.pop("error", None)
                        break
                    except Exception as e:  # noqa: BLE001
                        res["error"] = str(e)
            conv["check_backchannel"] = res
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
