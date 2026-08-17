"""行指向レコード → 音声合成の入力一式（台本 / dialogue JSON / 構造注釈）。

MoshiRAG-JP の会話レコードから、工程 3 以降で必要になる 3 つを作る。

  <stem>.txt        FireRedTTS2 の台本（[S1]=人間 / [S2]=moshi）
  <stem>.json       04_word_alignment.py 用の dialogue JSON
                    [{"speaker": "A"|"B", "text": ...}, ...]  A=人間(L) / B=moshi(R)
  <stem>.struct.json  lead/body/tail の境界と参照文書。<ret> 配置に使う

台本では lead / body / tail を **別ターン**として出す。同一話者の連続ターンは
butt-join されるため音声は連続した一発話になり、かつ合成 manifest から各部の onset が
真値として取れる。struct.json はその turn index を保持し、08 側が manifest と突き合わせて
<ret> のフレーム位置と d_lead を求める。

以前は 1 ターンに結合し、境界を forced alignment で推定していた。B チャネルは 6 割以上が
無音のため stable-ts が破綻し、真値より 34 秒ずれた（d_lead が最大 43.68 秒になった）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_record(record: list[str]) -> tuple[list[dict], list[dict]]:
    """レコードを (台本ターン列, 構造情報) に分解する。"""
    turns: list[dict] = []          # {"speaker": "A"|"B", "text": ...}
    struct: list[dict] = []         # augmented ターンの境界情報
    pending: dict | None = None     # 組み立て中の moshi ターン

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        if not pending["augmented"]:
            if pending["body"]:
                turns.append({"speaker": "B", "text": pending["body"]})
            pending = None
            return
        # lead / body / tail を **別ターン**として台本に出す。同一話者の連続ターンは
        # butt-join されるので音声は連続した一発話になり、かつ合成 manifest から
        # 各部の onset が真値として取れる。以前は 1 ターンに結合していたため境界を
        # forced alignment で推定する必要があり、長い無音を含むチャネルで破綻した。
        idx = {}
        for key in ("lead", "body", "tail"):
            if pending[key]:
                turns.append({"speaker": "B", "text": pending[key]})
                idx[key] = len(turns) - 1
        if "lead" in idx and "body" in idx:
            struct.append({
                "lead_turn_index": idx["lead"],
                "body_turn_index": idx["body"],
                "tail_turn_index": idx.get("tail"),
                "lead": pending["lead"],
                "body": pending["body"],
                "tail": pending["tail"],
                "reference": pending["reference"],
            })
        pending = None

    for line in record:
        line = line.rstrip("\n")
        if line == "(augmented)":
            flush()
            pending = {"augmented": True, "lead": "", "body": "", "tail": "", "reference": ""}
        elif line == "(unaugmented)":
            flush()
            pending = {"augmented": False, "lead": "", "body": "", "tail": "", "reference": ""}
        elif line.startswith("Human:"):
            flush()
            turns.append({"speaker": "A", "text": line[len("Human:"):].strip()})
        elif line.startswith("moshi (lead):"):
            if pending is not None:
                pending["lead"] = line.split(":", 1)[1].strip()
        elif line.startswith("Reference:"):
            if pending is not None:
                pending["reference"] = line.split(":", 1)[1].strip()
        elif line.startswith("moshi (body):"):
            if pending is not None:
                pending["body"] = line.split(":", 1)[1].strip()
        elif line.startswith("moshi (tail):"):
            if pending is not None:
                t = line.split(":", 1)[1].strip()
                pending["tail"] = "" if t == "[empty]" else t
        elif line.startswith("moshi:"):
            if pending is not None:
                pending["body"] = line.split(":", 1)[1].strip()
    flush()
    return turns, struct


def main(args: argparse.Namespace) -> None:
    src = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    n_ok = n_aug = 0
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        turns, struct = parse_record(d["record"])
        if not turns:
            continue
        stem = f.stem

        # FireRedTTS2 の台本。A=[S1]（人間）, B=[S2]（moshi）
        script = "\n".join(
            f"[S1]{t['text']}" if t["speaker"] == "A" else f"[S2]{t['text']}" for t in turns
        )
        (out / f"{stem}.txt").write_text(script + "\n", encoding="utf-8")

        # 04_word_alignment.py 用
        (out / f"{stem}.json").write_text(
            json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        # <ret> 配置に使う構造情報
        (out / f"{stem}.struct.json").write_text(
            json.dumps({
                "stem": stem,
                "seed_question": d.get("seed_question", ""),
                "topic": d.get("topic", ""),
                "chunks": d.get("chunks", []),
                "greeting_lines": d.get("greeting_lines"),
                "code_commit": d.get("code_commit", ""),
                "n_turns": len(turns),
                "augmented": struct,
            }, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        n_ok += 1
        n_aug += len(struct)

    print(f"{n_ok} 会話を変換 -> {out}")
    print(f"  augmented ターン合計 {n_aug}（1 会話あたり {n_aug / max(n_ok, 1):.1f}）")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, help="pilot5k などのレコード JSON があるディレクトリ")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--limit", type=int, default=0, help="先頭 N 件だけ変換する（0=全件）")
    main(p.parse_args())
