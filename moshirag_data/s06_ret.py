"""合成 manifest から <ret> のフレーム位置と d_lead を算出する。

論文 §4.2：

    we replace the text token before the first text token in the lead portion of
    an RAG-enabled Moshi turn with the <ret> token.

<ret> は lead 部分の最初のテキストトークンの直前に置く。d_lead は lead の継続時間で、
学習時の遅延サンプリングに使われる。

    d' = U(0, d_lead)            if d_lead < 2 or p < 0.2
         U(1.0, d_lead - 1.0)    otherwise

**タイミングは合成 manifest（真値）から取る。** 元実装は多チャネル TTS が forced alignment を
返すためそれを使っているが、我々の TTS はターン単位の onset / duration を manifest で返すので、
lead / body を別ターンとして合成すれば境界が誤差なく求まる。後段 ASR によるアライメントに
依存させない（依存させた版は長い無音を含むチャネルで 34 秒ずれた）。

入力
  --manifest_dir  合成（+ 後段処理）の *.manifest.json
  --struct_dir    07_prepare_tts_inputs.py の *.struct.json

出力
  {"stem", "frame_rate", "ret_events": [...]}
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

FRAME_RATE = 12.5   # Mimi


def load_turns(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["turns"] if isinstance(d, dict) else d


def main(args: argparse.Namespace) -> None:
    md, sd = Path(args.manifest_dir), Path(args.struct_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_conv = n_ev = n_skip = 0
    d_leads: list[float] = []
    for sf in sorted(sd.glob("*.struct.json")):
        stem = sf.name[: -len(".struct.json")]
        mf = next((p for p in (md / f"{stem}.manifest.json", md / f"{stem}.annot.json",
                               md / f"{stem}.json") if p.exists()), None)
        if mf is None:
            continue
        struct = json.loads(sf.read_text(encoding="utf-8"))
        turns = load_turns(mf)

        events = []
        for aug in struct.get("augmented", []):
            li, bi = aug["lead_turn_index"], aug["body_turn_index"]
            if li >= len(turns) or bi >= len(turns):
                n_skip += 1
                continue
            lead_onset = float(turns[li].get("onset", 0.0))
            body_onset = float(turns[bi].get("onset", 0.0))
            d_lead = body_onset - lead_onset
            if d_lead <= 0:
                n_skip += 1
                continue
            events.append({
                "lead_turn_index": li,
                "body_turn_index": bi,
                "ret_frame": max(0, int(lead_onset * FRAME_RATE) - 1),
                "lead_onset": round(lead_onset, 3),
                "body_onset": round(body_onset, 3),
                "d_lead": round(d_lead, 3),
                "d_lead_frames": int(d_lead * FRAME_RATE),
                "reference": aug["reference"],
                "lead": aug["lead"],
                "body": aug["body"],
            })
            d_leads.append(d_lead)
        (out / f"{stem}.json").write_text(json.dumps({
            "stem": stem,
            "frame_rate": FRAME_RATE,
            "code_commit": struct.get("code_commit", ""),
            "ret_events": events,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        n_conv += 1
        n_ev += len(events)

    print(f"{n_conv} 会話 / <ret> イベント {n_ev} 件 / скип {n_skip} 件".replace("скип", "スキップ"))
    if d_leads:
        ok2 = sum(1 for d in d_leads if d >= 2.0)
        ok24 = sum(1 for d in d_leads if 2.0 <= d <= 4.0)
        print(f"d_lead: 平均 {st.mean(d_leads):.2f}s / 中央 {st.median(d_leads):.2f}s / "
              f"範囲 {min(d_leads):.2f}-{max(d_leads):.2f}s")
        print(f"  2 秒以上（主分岐に入る）: {ok2}/{len(d_leads)} = {100 * ok2 / len(d_leads):.0f}%")
        print(f"  2〜4 秒（設計目標）     : {ok24}/{len(d_leads)} = {100 * ok24 / len(d_leads):.0f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest_dir", required=True)
    p.add_argument("--struct_dir", required=True)
    p.add_argument("--output_dir", required=True)
    main(p.parse_args())
