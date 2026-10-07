"""TTS の manifest から MFA で語アライメントを取り、会話単位の語 JSON にする。

揃わないターンは広いビームで 1 回だけやり直し、それでも残った会話は <words_dir>_excluded に書く。
書き起こしは分かち書きせず素のテキストで渡す（分かち書きすると全音素が spn になる）。

Output: [{"speaker": "A", "word": "こんにちは", "start": 0.46, "end": 1.02, "turn_index": 1}, ...]
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time
from collections import defaultdict
from pathlib import Path


def build_corpus(audio_dir: Path, stems: list[str], corpus: Path, only: set[str] | None = None) -> dict:
    """話者ごとのフォルダに「wav（シンボリックリンク）と同名の .lab」を並べる。戻り値は base → (stem, 番号)。"""
    shutil.rmtree(corpus, ignore_errors=True)
    bases: dict = {}
    for stem in stems:
        for t in json.load(open(audio_dir / f"{stem}.manifest.json", encoding="utf-8"))["turns"]:
            base = f"{stem}__turn{t['index']:03d}"
            if only is not None and base not in only:
                continue
            src = audio_dir / t["wav"]
            if not src.exists():
                continue
            d = corpus / ("A" if t["channel"] == 0 else "B")
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{base}.wav").symlink_to(src.resolve())
            txt = re.sub(r"[…「」『』（）]+", "", t["text"]).strip()
            (d / f"{base}.lab").write_text(txt + "\n", encoding="utf-8")
            bases[base] = (stem, t["index"])
    return bases


def aligned(out: Path) -> set[str]:
    return {p.stem for p in out.glob("*/*.json")}


def run_mfa(mfa: str, corpus: Path, out: Path, jobs: int, config: str | None, log: Path) -> float:
    cmd = [mfa, "align", "--clean", "-j", str(jobs), "--output_format", "json"]
    if config:
        cmd += ["-c", config]
    cmd += [str(corpus), "japanese_mfa", "japanese_mfa", str(out)]
    print("  $ " + " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    el = time.time() - t0
    if rc != 0:
        print(f"  mfa align が終了コード {rc} で終わった。ログ: {log}", flush=True)
        sys.exit(rc)
    return el


def to_words(audio_dir: Path, stems: list[str], outs: list[Path], words_dir: Path) -> dict:
    """ターン内の相対時刻に manifest の onset を足し、会話全体の時間軸に直す（mfa_to_words.py と同じ）。"""
    files: dict = {}
    for out in outs:  # 2 回目は 1 回目で取りこぼしたターンだけなので、同じターンが両方に出ることはない
        for p in out.glob("*/*.json"):
            files.setdefault(p.stem, p)
    excl = Path(str(words_dir) + "_excluded")
    words_dir.mkdir(parents=True, exist_ok=True)
    excl.mkdir(parents=True, exist_ok=True)
    st = defaultdict(int)
    for stem in stems:
        turns = json.load(open(audio_dir / f"{stem}.manifest.json", encoding="utf-8"))["turns"]
        words, missing = [], 0
        for t in sorted(turns, key=lambda x: x["index"]):
            f = files.get(f"{stem}__turn{t['index']:03d}")
            if f is None:
                missing += 1
                continue
            spk = "A" if t["channel"] == 0 else "B"
            for a, b, w in json.load(open(f, encoding="utf-8"))["tiers"]["words"]["entries"]:
                if not w or w in ("<eps>", "spn"):
                    st["spn"] += w == "spn"
                    continue
                words.append({"speaker": spk, "word": w, "start": round(t["onset"] + a, 4),
                              "end": round(t["onset"] + b, 4), "turn_index": t["index"]})
        words.sort(key=lambda x: x["start"])
        dst = (excl if missing else words_dir) / f"{stem}.json"
        json.dump(words, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        st["excluded" if missing else "ok"] += 1
        st["words"] += len(words)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="manifest.json とターンごとの wav がある場所")
    ap.add_argument("--stems", help="対象の会話名を 1 行 1 件で書いたファイル。省略時は manifest があるもの全部")
    ap.add_argument("--work", required=True, help="コーパスと MFA の出力を置く作業場所（ノードのローカルディスクを推奨）")
    ap.add_argument("--words_dir", required=True)
    ap.add_argument("--mfa", required=True, help="mfa コマンドのパス")
    ap.add_argument("--beam_config", required=True, help="beam: 100 / retry_beam: 400 を書いた yaml")
    ap.add_argument("-j", "--jobs", type=int, default=8)
    a = ap.parse_args()

    audio_dir, work = Path(a.audio_dir), Path(a.work)
    if a.stems:
        stems = [l.strip() for l in open(a.stems) if l.strip()]
    else:
        stems = sorted(p.name[: -len(".manifest.json")] for p in audio_dir.glob("*.manifest.json"))
    print(f"  対象 {len(stems):,} 会話", flush=True)

    bases = build_corpus(audio_dir, stems, work / "corpus")
    print(f"  コーパス {len(bases):,} ターン", flush=True)
    t1 = run_mfa(a.mfa, work / "corpus", work / "out", a.jobs, None, work / "mfa_pass1.log")
    ok1 = aligned(work / "out") & set(bases)
    miss = set(bases) - ok1
    print(f"  1 回目（既定ビーム）: {len(ok1):,} / {len(bases):,} ターン / {t1:.0f} 秒"
          f"（{len(bases) / t1 * 60:,.0f} ターン/分）", flush=True)

    outs = [work / "out"]
    if miss:
        build_corpus(audio_dir, stems, work / "corpus_retry", only=miss)
        t2 = run_mfa(a.mfa, work / "corpus_retry", work / "out_retry", a.jobs, a.beam_config,
                     work / "mfa_pass2.log")
        ok2 = aligned(work / "out_retry") & miss
        print(f"  2 回目（広いビーム）: {len(ok2):,} / {len(miss):,} ターン / {t2:.0f} 秒", flush=True)
        outs.append(work / "out_retry")

    st = to_words(audio_dir, stems, outs, Path(a.words_dir))
    print(f"  語 JSON: 採用 {st['ok']:,} 会話 / 除外 {st['excluded']:,} 会話 / {st['words']:,} 語"
          f"（spn として落とした語 {st['spn']:,}）", flush=True)


if __name__ == "__main__":
    main()
