# 元実装への変更（patch）

`kyutai-labs/moshi-rag` を clone して使っているが、学習のために最小限の変更を入れている。
変更はここに patch として保存し、リポジトリ本体は素のまま保つ。

適用方法

    cd <moshi-rag の clone 先>
    git apply <このディレクトリ>/*.patch

## moshirag_lm_training_streaming_sum.patch

`moshi/moshi/models/lm.py` の `forward_text` にある表明を緩める。

    変更前  assert condition_len == 1; assert S == 1
    変更後  assert condition_len in (1, S)

理由：公開されている moshi-rag には**推論時の参照注入しか入っていない**。
学習は系列全体を一度に通すため、1 ステップ専用の表明では通らない。
推論時の意味論（1 フレームに 1 ベクトルを加算）と同一で、加算を S 個まとめて行うだけ。
詳細は設計判断ログ D-31。
