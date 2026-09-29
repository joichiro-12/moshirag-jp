#!/bin/bash
# g23 のローカルに溜まった音声を、Mac 経由で NAS の audio/ に移す（Mac で実行する）。
#
#   bash relay_g23.sh
#
# g23 は NAS（iot-qnap5）をマウントしておらず、g21 から g23 へは直接 ssh できないため、Mac で中継する
# （9/29 の実測で毎秒約 7 MB）。何度流してもよい。NAS にすでにある会話は送らない。
# 書きかけを audio/ に入れないよう、いったん展開用のフォルダに展開し、展開が済んでから移す。
# 展開用のフォルダは回ごとに新しく作る。9/29 の 1 回目では、移したはずの 50 会話が展開用のフォルダにも
# 残った（audio/ 側と中身は同一。NAS 上の移動が「複写してから元を消す」形で、元の削除が遅れたか失敗した
# 可能性がある。未確認）。残りが前回の分と混ざらないようにするため。
set -eu
R=/mnt/iot-qnap5/jsato/moshirag_tts
I=$R/audio_g23_incoming_$(date +%m%d_%H%M)
have=$(ssh g21 "ls $R/audio | grep '^wiki_qa_0.*\.wav$' | sed 's/\.wav$//'" 2>/dev/null)
echo "$have" | ssh g23 'cat > /tmp/relay_have.txt
cd $HOME/moshirag_tts/audio
find . -maxdepth 1 -name "wiki_qa_*.wav" -mmin +10 -printf "%f\n" | sed "s/\.wav$//" | LC_ALL=C sort \
  | comm -23 - <(LC_ALL=C sort /tmp/relay_have.txt) > /tmp/relay_send.txt
echo "送る会話: $(wc -l < /tmp/relay_send.txt) 件" >&2
awk "{print \$0\".wav\"; print \$0\".manifest.json\"; print \$0\"_turns\"}" /tmp/relay_send.txt | tar -cf - -T -' \
  2> >(grep -v "post-quantum\|store now\|pq.html" >&2) \
  | ssh g21 "set -e; mkdir -p $I; tar -xf - -C $I
n=\$(ls $I | grep -c '\.wav$' || true)
find $I -mindepth 1 -maxdepth 1 -exec mv -t $R/audio {} + 2>/dev/null || true
echo \"NAS に移した会話: \$n 件 / 展開用のフォルダ（$I）に残った項目: \$(ls $I | wc -l) 件\"" 2>&1 | grep -v "post-quantum\|store now\|pq.html"
