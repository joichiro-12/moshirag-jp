#!/bin/bash
# 研究室サーバで音声化の worker を起動する。
#
#   launch.sh NSHARD シャード:GPU [シャード:GPU ...]
#   例） launch.sh 9 5:0 6:0 7:1 8:1   ← GPU 0 に 2 つ、GPU 1 に 2 つ
#
# 全台で NSHARD を揃えること。各 worker は「添字 % NSHARD == シャード」の会話だけを処理する。
# 途中で NSHARD を変えると担当がずれ、同じ会話を 2 つの worker が同時に作ることがある。
#
# ssh を切っても止まらないよう setsid で切り離す。止まった場合は同じコマンドで再起動すれば、
# 既に音声がある会話は飛ばして続きから進む。
set -u
N=$1; shift
IN=${IN:-/mnt/iot-qnap5/jsato/moshirag_tts/in}   # g23 は NAS が見えないので環境変数で差し替える
OUT=${OUT:-/mnt/iot-qnap5/jsato/moshirag_tts/audio}
cd "$HOME/moshirag_tts/src/zoom1-tts"
export TTS_DIR=$HOME/moshirag_tts/src/zoom1-tts HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for sg in "$@"; do
    s=${sg%%:*}; g=${sg##*:}
    if pgrep -f "synth_worker.py.*--shard $s --nshard $N" >/dev/null; then
        echo "  シャード $s は既に動いている。飛ばす"; continue
    fi
    CUDA_VISIBLE_DEVICES=$g setsid nohup .venv/bin/python "$HOME/moshirag_tts/synth_worker.py" \
        --in_dir "$IN" --out_dir "$OUT" --shard "$s" --nshard "$N" \
        > "$HOME/moshirag_tts/logs/synth_shard${s}.log" 2>&1 < /dev/null &
    echo "  $(hostname): シャード $s → GPU $g（pid $!）"
done
