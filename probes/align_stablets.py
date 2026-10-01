"""stable-ts でターン別 wav に強制アライメントを掛ける（案 B-1）。

ABCI の ffmpeg は libvmaf.so.1 を欠いて起動しないため、soundfile で読んだ
配列を直接渡す。語の時刻はターン内の相対時刻で返るので、manifest の onset を
足して会話全体の時間軸に直す。
出力は KAME の 04_word_alignment.py と同じ形式（tools/tokenize_text.py が食える）。
"""
import argparse, json, time
from pathlib import Path
import numpy as np, soundfile as sf, stable_whisper

B = "/groups/gcg51557/experiments/0374_japanese_kame"
CH2SPK = {0: "A", 1: "B"}   # channel 0=左=S1=人間役, 1=右=S2=Moshi 役


def load_16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        n = int(round(len(x) * 16000 / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="large-v3")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = stable_whisper.load_model(a.model, device="cuda")
    stems = [l.strip() for l in open(a.stems) if l.strip()]
    t0 = time.time(); nturn = 0; nfail = 0
    for k, stem in enumerate(stems):
        if (out / f"{stem}.json").exists():
            continue
        man = json.load(open(f"{B}/pilot5k_audio/{stem}.manifest.json", encoding="utf-8"))
        words = []
        for t in man["turns"]:
            nturn += 1
            wav = f"{B}/pilot5k_audio/{t['wav']}"
            try:
                r = model.align(load_16k(wav), t["text"], language="ja")
                ws = [w for s in r.segments for w in s.words]
                if not ws:
                    raise ValueError("語が取れず")
            except Exception as e:
                nfail += 1
                print(f"  失敗 {stem} turn{t['index']}: {e}", flush=True)
                continue
            spk = CH2SPK[t["channel"]]
            for w in ws:
                words.append({"speaker": spk, "word": w.word,
                              "start": round(t["onset"] + w.start, 4),
                              "end":   round(t["onset"] + w.end, 4),
                              "turn_index": t["index"]})
        words.sort(key=lambda x: x["start"])
        (out / f"{stem}.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        if (k + 1) % 5 == 0:
            print(f"  {k+1}/{len(stems)} 会話 / {nturn} ターン / {time.time()-t0:.0f}s", flush=True)
    print(f"完了 {len(stems)} 会話 / {nturn} ターン / 失敗 {nfail} / {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
