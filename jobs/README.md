# jobs/：日本語 MoshiRAG の PBS（いま使っているもの）

**リポジトリのルートで `qsub jobs/<工程>/<名前>.pbs` と投げる。** PBS の中の `tools/…`・`moshirag/…`・
`scripts/…` はルートからの相対パスで書いてあり、ルートで投げたときの `PBS_O_WORKDIR` を前提にしている。

| フォルダ | 工程 |
| --- | --- |
| `qa_gen/` | Wikipedia の QA 生成（`wikiqa_*`）と会話生成（`gen_*`） |
| `synth/` | 音声化（`synth_hg` が本番、`synth_verify`・`synth_test` は検証用） |
| `post/` | 語アライメント・トークナイズ・リファレンス埋め込み・parquet 化（`stage5*`・`stage6_prep`・`align_test`・`bench_stage7_10`） |
| `train_eval/` | 学習と評価（`d3`・`d3b`・`d4`・`eval*`・`sweep`） |

研究室サーバで回すものは `tools/lab/` にある。ABCI の利用ルール（アレイ禁止など）は LLM-jp の資料に従う。
