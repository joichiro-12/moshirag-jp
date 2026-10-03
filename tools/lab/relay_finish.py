"""中継で展開した会話を確かめて、完全なものだけを audio/ に移す（g21 で実行。NAS 上の移動なので名前の付け替えで済む）。

    python3 relay_finish.py <展開用フォルダ> [<展開用フォルダ> ...]

会話ごとに次のどれかに振り分ける。消さずに、移した先を報告する。

  移した   ：audio/ に無く、manifest・ターンの wav・会話の wav がそろっていて、wav が書きかけでない
  複製     ：audio/ に同じ会話があり、会話の wav と manifest の大きさが一致する → relay_leftover/dupes/
  書きかけ ：中継が途中で切れて欠けている（wav のヘッダが示す長さに本体が足りない、ターンが欠けている など）
             → relay_leftover/broken/。次の中継で送り直される（audio/ に無いので）
  食い違い ：audio/ に同じ会話があるのに大きさが違う → 動かさずに報告する

wav が書きかけかどうかは、ヘッダの標本数から決まる大きさと実際の大きさを比べて判定する
（tar が途中で切れると、ヘッダは元のままで本体だけが短くなる）。
"""
import json, os, sys, shutil, wave

R = "/mnt/iot-qnap5/jsato/moshirag_tts"
A = f"{R}/audio"
DUP, BRK = f"{R}/relay_leftover/dupes", f"{R}/relay_leftover/broken"


def wav_ok(p):
    try:
        with wave.open(p) as w:
            need = w.getnframes() * w.getnchannels() * w.getsampwidth()
        return os.path.getsize(p) >= need + 78   # この TTS の wav のヘッダは 78 バイト（実測）
    except Exception:
        return False


def complete(d, s):
    w, m, t = f"{d}/{s}.wav", f"{d}/{s}.manifest.json", f"{d}/{s}_turns"
    if not (os.path.isfile(w) and os.path.isfile(m) and os.path.isdir(t)):
        return False
    try:
        turns = json.load(open(m, encoding="utf-8"))["turns"]
    except Exception:
        return False
    return wav_ok(w) and all(wav_ok(f"{d}/{x['wav']}") for x in turns)


def move_all(d, s, dst):
    os.makedirs(dst, exist_ok=True)
    for name in (f"{s}.wav", f"{s}.manifest.json", f"{s}_turns"):
        src = f"{d}/{name}"
        if os.path.exists(src):
            target = f"{dst}/{name}"
            if os.path.exists(target):          # 退避先に同名がある（前の回の残り）→ 名前をずらす
                target = f"{target}.{os.path.basename(d)}"
            shutil.move(src, target)


for d in sys.argv[1:]:
    stems = sorted({n[:-4] for n in os.listdir(d) if n.endswith(".wav")} |
                   {n[:-14] for n in os.listdir(d) if n.endswith(".manifest.json")} |
                   {n[:-6] for n in os.listdir(d) if n.endswith("_turns")})
    c = {"移した": 0, "複製": 0, "書きかけ": 0, "食い違い": []}
    for s in stems:
        if os.path.exists(f"{A}/{s}.wav"):
            same = all(os.path.exists(f"{d}/{n}") and os.path.getsize(f"{d}/{n}") == os.path.getsize(f"{A}/{n}")
                       for n in (f"{s}.wav", f"{s}.manifest.json"))
            if same:
                move_all(d, s, DUP); c["複製"] += 1
            elif not complete(d, s):
                move_all(d, s, BRK); c["書きかけ"] += 1
            else:
                c["食い違い"].append(s)
        elif complete(d, s) and not os.path.exists(f"{A}/{s}_turns"):
            for name in (f"{s}_turns", f"{s}.manifest.json", f"{s}.wav"):   # wav を最後に（後段は wav の有無で拾うため）
                os.rename(f"{d}/{name}", f"{A}/{name}")
            c["移した"] += 1
        else:
            move_all(d, s, BRK); c["書きかけ"] += 1
    left = os.listdir(d)
    if not left:
        os.rmdir(d)
    print(f"{os.path.basename(d)}: 会話 {len(stems)} / 移した {c['移した']} / 複製 {c['複製']} / 書きかけ {c['書きかけ']} / "
          f"食い違い {len(c['食い違い'])} {c['食い違い'][:5]} / 残った項目 {len(left)}")
