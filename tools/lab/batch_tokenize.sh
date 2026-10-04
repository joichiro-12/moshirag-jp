#!/bin/bash
# 研究室サーバで音声になった会話を、Mimi でトークナイズする（工程 ⑦）。
#
#   bash ~/moshirag_tok/batch_tokenize.sh [ワーカー数（既定 2。GPU0 から順に使う）]
#
# 何度流してもよい。トークン（npz）がある会話は --resume で飛ばす。
# 音声化の途中のファイルを拾わないよう、wav を書き終わってから 10 分以上経った会話だけを対象にする。
#
# 研究室側のトークナイズは A5000 に揃える。同じ音声でも GPU の種類でトークンが 1〜2% 変わるため
# （A5000 同士なら 100% 一致。進行管理表 2.12 節）。設定は ABCI と同じ既定のまま（TF32 も既定）。
set -eu
N=${1:-2}
A=/mnt/iot-qnap5/jsato/moshirag_tts/audio
O=/mnt/iot-qnap5/jsato/moshirag_tts/tokens_audio
T=$HOME/moshirag_tok
S=$T/stage_in   # 対象の wav へのシンボリックリンクを置く（tokenize_audio はフォルダ内の wav を全部読むため）

if nvidia-smi --query-gpu=name --format=csv,noheader | head -n "$N" | grep -qv "RTX A5000"; then
    echo "A5000 以外の GPU が含まれている。止める"; exit 1
fi
mkdir -p "$S" "$O"
cd "$A"
find . -maxdepth 1 -name "*.wav" -mmin +10 -printf "%f\n" | while read -r f; do
    [ -e "$S/$f" ] || ln -s "$A/$f" "$S/$f"
done
echo "===== $(date) 対象の wav $(ls "$S" | wc -l) 件 / トークン済み $(ls "$O" | wc -l) 件"

cd "$T/moshirag-jp"
.venv/bin/python -m moshirag_data.s07_tok_audio --audio_dir "$S" --output_dir "$O" --num_workers "$N" --resume
echo "===== $(date) 終了 / トークン $(ls "$O" | wc -l) 件"
