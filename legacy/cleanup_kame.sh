#!/bin/bash
# KAME 時代の学習チェックポイントを整理する。
#   第 1 引数に "apply" を渡したときだけ実際に削除する。既定は列挙のみ。
#
# 方針：
#   - 各 run の「最終 step」に一致するもの（step_N / step_N_cleaned / step_N_fp32）は残す
#   - config.json は残す
#   - それ以外の step_* を消す
#   - smoke 3 本と j-moshi-kame-finetuned はディレクトリごと消す（12〜24 step の動作確認用）
set -u
MODE="${1:-dryrun}"
O=/groups/gcg51557/experiments/0374_japanese_kame/moshi_kame_finetune/output
cd "$O" || exit 1

DROP_WHOLE="0374_kame-spkB-smoke 0374_kame-spkB-drop-smoke 0374_kame-spkB-wandb-smoke j-moshi-kame-finetuned"
KEEP_RUNS="0374_kame-0610 0374_kame-0610-e5 0374_kame-0610-e5-spkB 0374_kame-0610-e5-spkB-drop 0374_kame-0610-e5-spkB-wandb"

echo "########## ディレクトリごと削除する ##########"
for d in $DROP_WHOLE; do
    [ -d "$d" ] || { echo "  (無し) $d"; continue; }
    echo "  削除: $d  $(du -sh "$d" 2>/dev/null | cut -f1)"
    [ "$MODE" = apply ] && rm -rf "$d"
done

echo
echo "########## 各 run：最終 step を残し中間を削除 ##########"
for d in $KEEP_RUNS; do
    [ -d "$d" ] || { echo "  (無し) $d"; continue; }
    # 最終 step 番号（step_12345 / step_12345_cleaned / step_12345_fp32 から数字を取る）
    N=$(ls "$d" | sed -n 's/^step_\([0-9]\+\).*/\1/p' | sort -n | tail -1)
    [ -n "$N" ] || { echo "  !! $d: step_* が見つからない。触らない"; continue; }
    echo "=== $d （最終 step=$N）"
    for s in "$d"/step_*; do
        b=$(basename "$s")
        case "$b" in
            step_${N}|step_${N}_*) echo "    残す  : $b  $(du -sh "$s" 2>/dev/null | cut -f1)" ;;
            *)                     echo "    削除  : $b"
                                   [ "$MODE" = apply ] && rm -rf "$s" ;;
        esac
    done
done

echo
echo "########## 結果 ##########"
du -sh "$O" 2>/dev/null
[ "$MODE" = apply ] || echo "（列挙のみ。実行するには apply を渡す）"
