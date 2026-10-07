#!/bin/bash
# 研究室サーバで音声になった会話のうち、まだ語 JSON が無いものを語アライメントにかける（工程 ⑥）。
#
#   bash ~/moshirag_mfa/batch_align.sh [並列数（既定 20）]
#
# 何度流してもよい。語 JSON（words/ か words_excluded/）がある会話は飛ばす。
# 音声化の途中のファイルを拾わないよう、wav を書き終わってから 10 分以上経った会話だけを対象にする。
# 手順の中身は mfa_align.py（ABCI の moshirag_data/speech_dialogue/s05_align.py と同じもの）を参照。
set -eu
J=${1:-20}
A=/mnt/iot-qnap5/jsato/moshirag_tts/audio
WD=/mnt/iot-qnap5/jsato/moshirag_tts/words
M=$HOME/moshirag_mfa
E=$HOME/miniforge3/envs/mfa

# MFA の作業場所はローカルディスクに置く。モデルと設定は ~/Documents/MFA から写す
export MFA_ROOT_DIR=$M/mfa_root
if [ ! -d "$MFA_ROOT_DIR/pretrained_models" ]; then
    mkdir -p "$MFA_ROOT_DIR"
    cp -r "$HOME/Documents/MFA/pretrained_models" "$MFA_ROOT_DIR/"
    sed "s|^temporary_directory:.*|temporary_directory: $MFA_ROOT_DIR|" \
        "$HOME/Documents/MFA/global_config.yaml" > "$MFA_ROOT_DIR/global_config.yaml"
fi

B=$(date +%m%d_%H%M)
L=$M/lists/batch_$B.txt
mkdir -p "$WD" "${WD}_excluded" "$M/lists"
cd "$A"
find . -maxdepth 1 -name "*.wav" -mmin +10 -printf "%f\n" | sed 's/\.wav$//' | LC_ALL=C sort \
  | while read -r s; do
        { [ -e "$WD/$s.json" ] || [ -e "${WD}_excluded/$s.json" ]; } && continue
        [ -e "$s.manifest.json" ] && [ -d "${s}_turns" ] && echo "$s"
    done > "$L"
echo "===== $(date) 対象 $(wc -l < "$L") 会話 → $L"
[ -s "$L" ] || exit 0

PATH=$E/bin:$PATH "$E/bin/python" "$M/mfa_align.py" --audio_dir "$A" --stems "$L" \
    --work "$M/work_$B" --words_dir "$WD" --mfa "$E/bin/mfa" \
    --beam_config "$HOME/mfa_beam.yaml" -j "$J"
echo "===== $(date) 終了"
