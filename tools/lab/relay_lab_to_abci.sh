#!/bin/bash
# 研究室で作った後段の入力（manifest・語 JSON・音声トークン）を、Mac 経由で ABCI に送る（Mac で実行する）。
#
#   bash relay_lab_to_abci.sh
#
# 研究室の音声そのもの（1 件 18 MB）は Mac の回線では ABCI に運べない（9/27 夜の実測 毎秒約 0.9 MB）。
# そのため研究室の中で語アライメント（⑥）と音声トークナイズ（⑦）まで済ませ、小さな結果だけを送る。
# ⑤ 以降は ABCI で、ABCI で作った分と同じ手順で通す。
#
# 何度流してもよい。ABCI にすでにあるファイルは送らない。
set -eu
R=/mnt/iot-qnap5/jsato/moshirag_tts
B=/groups/gcg51557/experiments/0374_japanese_kame/lab
for d in manifests words words_excluded tokens_audio; do
    case $d in
        manifests) src=audio; pat='\.manifest\.json$' ;;
        *)         src=$d;    pat='' ;;
    esac
    have=$(ssh abci "mkdir -p $B/$d && ls $B/$d" 2>/dev/null)
    n=$(echo "$have" | ssh g21 "cat > /tmp/relay_abci_have_$d.txt; cd $R/$src && ls | grep '${pat:-.}' | LC_ALL=C sort \
          | comm -23 - <(LC_ALL=C sort /tmp/relay_abci_have_$d.txt) > /tmp/relay_abci_$d.txt; wc -l < /tmp/relay_abci_$d.txt" 2>/dev/null)
    echo "$d: 送る $n 件"
    [ "$n" -gt 0 ] || continue
    ssh g21 "cd $R/$src && tar -cf - -T /tmp/relay_abci_$d.txt" 2>/dev/null \
      | ssh abci "tar -xf - -C $B/$d && echo \"  ABCI の $d: \$(ls $B/$d | wc -l) 件\"" 2>/dev/null
done
