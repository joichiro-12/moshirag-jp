"""会話記録の Reference 行を、判定役が挙げた資料すべてで組み直す（LLM 不要）。

旧 s02 は判定役が複数の資料を挙げたとき 2 本目以降を捨てていた。JSON に残る
judge_raw と chunks から Reference 行だけを差し替える。音声は作り直さなくてよい。
"""
import argparse, json, re, sys
from pathlib import Path


def parse_all(raw: str, n_chunks: int) -> dict[int, list[int]]:
    out = {}
    for line in raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[:：]\s*(.+)", line)
        if not m:
            continue
        idx, val = int(m.group(1)), m.group(2).strip()
        if "なし" in val:
            out[idx] = []
            continue
        nums = [int(x) for x in re.findall(r"\d+", val)]
        out[idx] = [n for n in dict.fromkeys(nums) if 1 <= n <= n_chunks]
    return out


def rebuild(d: dict) -> tuple[list[str], dict]:
    chunks = d["chunks"]
    used = parse_all(d["judge_raw"], len(chunks))
    rec, out, a_no = d["record"], [], 0
    st = {"ref": 0, "widened": 0, "kept_unaug": 0}
    for line in rec:
        if line in ("(unaugmented)", "(augmented)"):
            a_no += 1
            # 旧実装では unaugmented になったが、新実装なら有効な資料があるターン
            if line == "(unaugmented)" and a_no != 1 and used.get(a_no):
                st["kept_unaug"] += 1
            out.append(line)
        elif line.startswith("Reference:"):
            st["ref"] += 1
            nums = used.get(a_no) or []
            if len(nums) >= 2:
                st["widened"] += 1
            new = " ".join(chunks[c - 1] for c in nums) if nums else line[len("Reference:"):].strip()
            out.append(f"Reference: {new}")
        else:
            out.append(line)
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--show", type=int, default=2, help="差分を表示する件数")
    a = ap.parse_args()

    ind, outd = Path(a.in_dir), Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    tot = {"conv": 0, "ref": 0, "widened": 0, "kept_unaug": 0, "conv_changed": 0}
    shown = 0
    for f in sorted(ind.glob("*.json")):
        d = json.load(open(f, encoding="utf-8"))
        if "judge_raw" not in d or "chunks" not in d:
            continue
        new_rec, st = rebuild(d)
        tot["conv"] += 1
        for k in ("ref", "widened", "kept_unaug"):
            tot[k] += st[k]
        changed = new_rec != d["record"]
        tot["conv_changed"] += changed
        if changed and shown < a.show:
            shown += 1
            print(f"--- {f.stem} ---")
            for old, new in zip(d["record"], new_rec):
                if old != new:
                    print(f"  旧: {old[:120]}")
                    print(f"  新: {new[:240]}")
        d["record"] = new_rec
        d["chunk_used"] = {str(k): v for k, v in parse_all(d["judge_raw"], len(d["chunks"])).items()}
        d["reference_fix"] = "2026-09-26 multi-chunk"
        (outd / f.name).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        (outd / f"{f.stem}.txt").write_text("\n".join(new_rec) + "\n", encoding="utf-8")

    n = max(tot["ref"], 1)
    print(f"\n会話 {tot['conv']:,} 件 / 書き換えた会話 {tot['conv_changed']:,} 件"
          f"（{100*tot['conv_changed']/max(tot['conv'],1):.1f}%）")
    print(f"Reference 行 {tot['ref']:,} 本 / 複数の資料に広げた {tot['widened']:,} 本（{100*tot['widened']/n:.1f}%）")
    print(f"旧実装で unaugmented のまま残したターン（新実装なら augmented）: {tot['kept_unaug']:,}")
    print(f"出力: {outd}")


if __name__ == "__main__":
    main()
