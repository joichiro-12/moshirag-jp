"""既存の会話記録の Reference 行を、判定役が挙げた資料すべてで組み直す。

## なぜ必要か

02e の判定結果の読み取りに不具合があり、判定役が「1,2」のように複数の資料を挙げた
ターンで 2 本目以降が捨てられていた（augmented ターンの 16〜18%）。body は 2 本目の資料の
内容も話しているのに、Reference 行には 1 本目しか載っていない。

## LLM を呼ばずに直せる理由

JSON に `judge_raw`（判定役の生出力）と `chunks`（参照文 3 本）が残っている。
augmented になるかどうかは「有効な番号が 1 つ以上あるか」で決まり、旧実装では
「最初の番号が有効か」で決めていた。**最初の番号が有効なら新実装でも augmented** なので、
旧実装で augmented だったターンには必ずリードが存在する。Reference 行の中身だけを差し替えればよい。

旧実装で「最初の番号が範囲外で 2 本目以降が有効」だったターンは unaugmented のまま残す
（新たにリードを作らないと augmented にできないため）。件数は数えて報告する。

## 音声への影響

Reference 行は読み上げられないので、既存の音声は作り直さなくてよい。
影響を受けるのは `<ret>` 位置の算出（工程⑤）以降のリファレンス埋め込みと parquet である。
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
