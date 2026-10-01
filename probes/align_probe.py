"""ターン別 wav への強制アライメントが実用になるかを測る。

会話全体に掛けたときは長い無音で破綻した（真値から 34 秒ずれ）。ターン別 wav は
短く無音が無いので条件が違う。判定基準は次の 3 つ。
  1. 語の時刻が単調増加か
  2. 最後の語の終端がターンの長さに収まるか
  3. 均等割り（アライメント無しの代替案）との差がどれだけあるか
"""
import json, glob, sys, time
import numpy as np
import soundfile as sf
import stable_whisper

# ABCI の ffmpeg は libvmaf.so.1 を欠いていて起動できない。stable-ts は既定で
# ffmpeg 経由で読むため、soundfile で読んだ配列を直接渡して迂回する。
def load_16k_mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        n = int(round(len(x) * 16000 / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
    return x

B = "/groups/gcg51557/experiments/0374_japanese_kame"
model = stable_whisper.load_model("large-v3", device="cuda")

man = json.load(open(f"{B}/pilot5k_audio/jaquad_000000.manifest.json", encoding="utf-8"))
ok = bad = 0
t0 = time.time()
for t in man["turns"][:6]:
    wav = f"{B}/pilot5k_audio/{t['wav']}"
    dur = sf.info(wav).duration
    try:
        r = model.align(load_16k_mono(wav), t["text"], language="ja")
    except Exception as e:
        print(f"  [{t['index']}] 失敗: {e}"); bad += 1; continue
    words = [w for s in r.segments for w in s.words]
    if not words:
        print(f"  [{t['index']}] 語が取れず"); bad += 1; continue
    mono = all(words[i].start <= words[i+1].start for i in range(len(words)-1))
    fits = words[-1].end <= dur + 0.25
    # 均等割りとの最大差
    n = sum(len(w.word) for w in words)
    pos = 0.0; maxdiff = 0.0
    for w in words:
        even = dur * pos / max(n, 1)
        maxdiff = max(maxdiff, abs(w.start - even))
        pos += len(w.word)
    print(f"  [{t['index']}] {dur:5.2f}s 語{len(words):3d} 単調={mono} 収まる={fits} "
          f"均等割りとの最大差={maxdiff:5.2f}s  「{t['text'][:24]}」")
    ok += mono and fits
print(f"\n健全 {ok}/6 件 / 失敗 {bad} 件 / 所要 {time.time()-t0:.1f}s")
