#!/bin/bash
# g23 のローカルに溜まった音声を、Mac 経由で NAS の audio/ に移す（Mac で実行する）。
#
#   bash relay_g23.sh [1 回に送る会話数（既定 100）]
#
# g23 は NAS（iot-qnap5）をマウントしておらず、g21 から g23 へは直接 ssh できないため、Mac で中継する。
# 何度流してもよい。NAS の audio/ にすでにある会話は送らない。
#
# 10/2 の回は 1 本の tar で 2,511 会話（約 45 GB）を送り、1 時間 44 分で接続が切れて全体を移せなかった。
# そのため、100 会話ずつ送り、送るたびに g21 の relay_finish.py で確かめて audio/ に移す。
# 切れても失うのはその回の 100 会話までで、次に流せば audio/ に無いものだけ送り直す。
# 書きかけ・複製は消さずに relay_leftover/ に分ける（relay_finish.py の冒頭を参照）。
set -u
N=${1:-100}
R=/mnt/iot-qnap5/jsato/moshirag_tts
F=/home/jsato/moshirag_tts_relay_finish.py   # g21 の上の確認スクリプト（tools/lab/relay_finish.py の写し）
Q='grep -v "post-quantum\|store now\|pq.html"'

have=$(ssh g21 "ls $R/audio | grep '^wiki_qa_0.*\.wav$' | sed 's/\.wav$//'" 2>/dev/null)
todo=$(echo "$have" | ssh g23 'cat > /tmp/relay_have.txt
cd $HOME/moshirag_tts/audio
find . -maxdepth 1 -name "wiki_qa_*.wav" -mmin +10 -printf "%f\n" | sed "s/\.wav$//" | LC_ALL=C sort \
  | comm -23 - <(LC_ALL=C sort /tmp/relay_have.txt)' 2>/dev/null)
total=$(echo "$todo" | grep -c . || true)
echo "送る会話: $total 件（$N 件ずつ）"
[ "$total" -gt 0 ] || exit 0

i=0
D=$(mktemp -d /tmp/relay_chunks.XXXX)   # 会話の一覧を N 件ずつに分けた置き場（Mac 側）
echo "$todo" | split -l "$N" - "$D/c_"
for c in "$D"/c_*; do
    i=$((i + 1))
    I=$R/audio_g23_incoming_$(date +%m%d_%H%M%S)
    cat "$c" | ssh g23 'cd $HOME/moshirag_tts/audio && awk "{print \$0\".wav\"; print \$0\".manifest.json\"; print \$0\"_turns\"}" | tar -cf - -T -' 2>/dev/null \
      | ssh g21 "mkdir -p $I && tar -xf - -C $I && python3 $F $I" 2>&1 | eval "$Q" | sed "s/^/  [$i] /"
done
echo "NAS の audio/ にある g23 由来: $(ssh g21 "ls $R/audio | grep -c '^wiki_qa_0\(0[89]\|1[0-4]\)[0-9]\{3\}\.wav$'" 2>/dev/null)"
