#!/bin/bash
# 研究室サーバで、共有の置き場（pool）から台本を取り出して音声化する worker を起動する（s04 の --pool_dir）。
#
#   launch_pool.sh [--after PID] GPU [GPU ...]
#   例） launch_pool.sh 0 0          ← GPU 0 に 2 本
#        launch_pool.sh --after 885362 0   ← プロセス 885362 が終わってから GPU 0 に 1 本
#
# 担当を固定で割る launch.sh と違い、空いた worker が残りを取りに行くので、サーバの速さが違っても
# 最後までそろって終わる。台本は NAS の pool/ から pool_claimed/<worker>/ に移してから作る。
# worker を止めたときは、pool_claimed/ に残った台本のうち音声の無いものを pool/ に戻す（s04 の claim_from_pool を参照）。
#
# 研究室サーバには s04 を ~/moshirag_tts/synth_worker.py という名前で写してある。
set -u
AFTER=""
if [ "${1:-}" = "--after" ]; then AFTER=$2; shift 2; fi
R=/mnt/iot-qnap5/jsato/moshirag_tts
POOL=${POOL:-$R/pool}
OUT=${OUT:-$R/audio}
cd "$HOME/moshirag_tts/src/zoom1-tts"
export TTS_DIR=$HOME/moshirag_tts/src/zoom1-tts HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
k=0
for g in "$@"; do
    k=$((k + 1))
    tag="$(hostname)_gpu${g}_$(date +%m%d%H%M%S)_$k"
    setsid nohup bash -c "
        if [ -n '$AFTER' ]; then while kill -0 $AFTER 2>/dev/null; do sleep 60; done; fi
        CUDA_VISIBLE_DEVICES=$g exec .venv/bin/python $HOME/moshirag_tts/synth_worker.py \
            --pool_dir $POOL --claim_dir $R/pool_claimed/$tag --out_dir $OUT
    " > "$HOME/moshirag_tts/logs/pool_$tag.log" 2>&1 < /dev/null &
    echo "  $tag → GPU $g（待つプロセス: ${AFTER:-なし}）"
done
