#!/bin/bash
# GPU の空きメモリが 17 GB（音声化の worker 1 本は約 16 GB）以上になるまで待ち、取り出し方式の worker を 1 本起動する。
#   wait_free_then_pool.sh GPU
# 他の利用者のプロセスは調べず、空きメモリの量だけを見る。
g=$1
while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g | tr -d " ")
    [ "${free:-0}" -ge 17000 ] && break
    sleep 120
done
echo "$(date) GPU $g の空き ${free} MiB。起動する"
cd $HOME/moshirag_tts && bash launch_pool.sh $g
