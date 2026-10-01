# probes/：使い終わった調査・検証

手法の選定や不具合の切り分けに使ったもの。結果は docs/ と研究ノートに記録済みで、本番の工程からは使っていない。
記録をたどれるよう、消さずに置いている。

ルートで実行する（例：`qsub probes/b0.pbs`）。`d1_verify.py` と `c_verify.py` は `moshirag/` のモジュールを import するので、
単体で動かすときは `PYTHONPATH=moshirag` を付ける。
