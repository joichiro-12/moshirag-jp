# moshirag/：日本語 MoshiRAG の学習・評価の Python

`jobs/` の PBS から `uv run --extra data python moshirag/<名前>.py` で呼ぶ。
`train_moshirag.py`・`eval_body_tokens.py`・`eval_ref_contribution.py` は同じフォルダの `moshirag_bridge.py` を import するので、
この 4 本は同じ場所に置くこと。
