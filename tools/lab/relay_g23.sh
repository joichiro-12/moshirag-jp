#!/bin/bash
# g23 のローカルに溜まった音声を、Mac 経由で NAS の audio/ に移す（Mac で実行する）。
#
#   bash relay_g23.sh
#
# g23 は NAS（iot-qnap5）をマウントしておらず、g21 から g23 へは直接 ssh できないため、Mac で中継する
# （9/29 の実測で毎秒約 7 MB）。何度流してもよい。NAS にすでにある会話は送らない。
# 書きかけを audio/ に入れないよう、いったん audio_g23_incoming/ に展開し、展開が済んでから移す。
set -eu
R=/mnt/iot-qnap5/jsato/moshirag_tts
have=$(ssh g21 "ls $R/audio | grep '^wiki_qa_0.*\.wav$' | sed 's/\.wav$//'" 2>/dev/null)
echo "$have" | ssh g23 'cat > /tmp/relay_have.txt
cd $HOME/moshirag_tts/audio
find . -maxdepth 1 -name "wiki_qa_*.wav" -mmin +10 -printf "%f\n" | sed "s/\.wav$//" | LC_ALL=C sort \
  | comm -23 - <(LC_ALL=C sort /tmp/relay_have.txt) > /tmp/relay_send.txt
echo "送る会話: $(wc -l < /tmp/relay_send.txt) 件" >&2
awk "{print \$0\".wav\"; print \$0\".manifest.json\"; print \$0\"_turns\"}" /tmp/relay_send.txt | tar -cf - -T -' \
  2> >(grep -v "post-quantum\|store now\|pq.html" >&2) \
  | ssh g21 "set -e; mkdir -p $R/audio_g23_incoming; tar -xf - -C $R/audio_g23_incoming
n=\$(ls $R/audio_g23_incoming | grep -c '\.wav$' || true)
find $R/audio_g23_incoming -mindepth 1 -maxdepth 1 -exec mv -t $R/audio {} +
echo \"NAS に移した会話: \$n 件\"" 2>&1 | grep -v "post-quantum\|store now\|pq.html"
