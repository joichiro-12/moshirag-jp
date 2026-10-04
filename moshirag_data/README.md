# moshirag_data/：日本語 MoshiRAG の学習データを作る工程

## この文書は何か

日本語 MoshiRAG の学習データ（parquet）を作る工程を、**1 つの番号体系（s01〜s10）で 1 か所にまとめた**ものの案内。
各工程のファイル・入出力・実行環境・投げ方と、各工程がいま「同じ約束」をどこまで守っているかを示す。

## 前提

- 2026-10-04 の整理（ブランチ `pipeline-reorg`）で、`scripts/japanese_kame/`・`tools/`・`moshirag/` に散っていた
  スクリプトをここへ移した。**処理の中身は変えていない**（変えたのは s01c が s02 を import する 1 行だけ）。
  旧パス → 新パスは `docs/file_moves_20261004.tsv`
- s07・s08 の実装は `tools/` に残っている（KAME から引き継ぎ、KAME 側のテストと README からも使われているため）。
  ここにあるのは、引数をそのまま渡す入口だけである
- PBS はリポジトリのルートで `qsub jobs/…/x.pbs` と投げる。`$B` は `/groups/gcg51557/experiments/0374_japanese_kame`
- 工程ごとに Python の環境が違うので、全工程をまとめて呼ぶ入口は作っていない

## 現在の状態

ファイルの置き場所と番号をそろえた段階。各工程の約束（下の「約束の守り方」）は**まだそろえていない**。

---

## 工程のつながり

| 工程 | 使う出力 |
| --- | --- |
| s02 会話 | s01（QA） |
| s03 台本・構造 JSON | s02 |
| s04 音声 | s03 の台本 |
| s05 語の時刻 | s04 の manifest とターンの wav |
| s06 `<ret>` の位置 | s03 の構造 JSON と s04 の manifest |
| s07 音声トークン | s04 の会話の wav |
| s08 テキストトークン | s05 |
| s09 参照の埋め込み | s06 |
| s10 parquet | s06・s07・s08・s09 |

## 工程の一覧

| 工程 | ファイル | 何をするか | 入力 → 出力 | 環境 | 投げ方 |
| --- | --- | --- | --- | --- | --- |
| s01a | `s01a_jaquad_qa.py` | 既存の QA データセットを 1 行 1 問にそろえる（`--sources jaquad` で JaQuAD だけを取る。MoshiRAG で使うのは JaQuAD だけ） | Hugging Face → `data/japanese_kame/qa_pairs/*.jsonl` | uv（`--extra data`） | 手で実行 |
| s01b | `s01b_wiki_candidates.py` | 日本語 Wikipedia から、題材にできる記事を機械的に絞る（本文 500 字以上・タイトル） | ja_wiki（パスは固定）→ `wiki_candidates.jsonl` | 標準ライブラリだけ | 手で実行 |
| s01c | `s01c_wiki_qa.py` | 記事の冒頭から LLM で問いと答えを 1 組作る | `wiki_candidates.jsonl` → `wiki_qa.sNNNofNNN.jsonl` | vLLM 用 venv＋vLLM サーバ | `jobs/qa_gen/wikiqa_main.pbs` |
| s02 | `s02_dialogue.py` | 問いと答えから会話を作る（lead / body / tail、参照、判定役による検索の要否） | QA の jsonl → `data/moshirag_jp/<名前>/<会話>.{json,txt}` | vLLM 用 venv＋vLLM サーバ | `jobs/qa_gen/gen_wiki_main.pbs`・`gen_stage1.pbs` |
| s02b | `s02b_rebuild_references.py` | 判定結果の読み取りの不具合（複数の資料の 2 本目以降が捨てられた）を、LLM を呼ばずに直す | s02 の出力 → 別のフォルダ（例：`pilot5k_fixed`） | 標準ライブラリだけ | 手で実行（JaQuAD 分に 1 回適用済み） |
| s03 | `s03_tts_input.py` | 会話を TTS の台本と構造 JSON に変える | s02 の出力 → `$B/<名前>_in/<会話>.{txt,json,struct.json}` | 標準ライブラリだけ | 手で実行 |
| s04 | `s04_synth.py` | 台本を 2 話者の音声にする（モデルを 1 回だけ読み、担当分をまとめて合成） | s03 の `.txt` → `<会話>.wav`・`.manifest.json`・`_turns/` | TTS の venv（zoom1-tts） | ABCI：`jobs/synth/synth_hg.pbs`／研究室：`tools/lab/launch.sh` |
| s05 | `s05_align.py` | ターンごとに MFA で語の時刻を出し、会話単位の語 JSON にまとめる | s04 の manifest とターンの wav → `words/<会話>.json` | MFA の conda 環境 | 研究室：`tools/lab/batch_align.sh`／ABCI：`jobs/post/align_test.pbs`（試験用） |
| s06 | `s06_ret.py` | `<ret>` を置くフレームと lead の長さを、合成の manifest から求める | s04 の manifest＋s03 の構造 JSON → `<会話>.json` | 標準ライブラリだけ | 手で実行 |
| s07 | `s07_tok_audio.py` | 会話の wav を Mimi のトークンにする | s04 の wav → `<会話>.npz` | uv（`--extra data`）＋GPU | ABCI：`jobs/post/stage5a_audio.pbs`／研究室：`tools/lab/batch_tokenize.sh` |
| s08 | `s08_tok_text.py` | 語 JSON を 12.5 Hz のテキストストリームにする | s05 の出力 → `<会話>.npz` | uv（`--extra data`） | `jobs/post/stage5b_text.pbs` |
| s09 | `s09_ref_embed.py` | `<ret>` ごとの参照文を ARC-Encoder で符号化する | s06 の出力 → `<会話>.npz` | uv（`--extra data`）＋GPU（約 14 GB） | `jobs/post/stage5c.pbs` |
| s10 | `s10_pack.py` | s06〜s09 をまとめて学習用の parquet にする | s06〜s09 → `*.parquet`（分割） | uv（`--extra data`） | `jobs/post/stage6_prep.pbs` |

研究室サーバには、s04 を `~/moshirag_tts/synth_worker.py`、s05 を `~/moshirag_mfa/mfa_align.py` という名前で写してある
（`tools/lab/` の起動スクリプトはこの名前で呼ぶ）。

## 約束の守り方（2026-10-04 時点）

目標とする約束は次の 4 つ。**表はいまの実装をそのまま読んだ結果で、そろえる作業はまだしていない。**

1. 入力フォルダと出力フォルダを引数で受け取る
2. 会話 1 件ごとに出力する
3. 出力済みの会話は飛ばす（何度流してもよい）
4. 使ったコードのコミットと設定を記録する

| 工程 | 2. 会話ごとの出力 | 3. 出力済みを飛ばす | 4. コミットの記録 |
| --- | --- | --- | --- |
| s01a | ×（データセットごとの jsonl） | ×（毎回作り直す） | × |
| s01b | ×（1 本の jsonl） | ×（毎回作り直す） | × |
| s01c | ×（シャードごとの jsonl に追記） | ○（出力済みの記事 ID を飛ばす） | PBS の出力にだけ残る |
| s02 | ○ | ○（`--resume`） | ○（各会話の JSON に `code_commit`） |
| s02b | ○ | ×（毎回作り直す） | × |
| s03 | ○ | ×（毎回作り直す） | s02 のコミットを写すだけ |
| s04 | ○ | ○（wav があれば飛ばす） | ABCI の PBS の出力にだけ残る |
| s05 | ○ | 本体は×。`batch_align.sh` が語 JSON の無い会話だけを渡す | ABCI の PBS の出力にだけ残る |
| s06 | ○ | ×（毎回作り直す） | s02 のコミットを写すだけ |
| s07 | ○ | ○（`--resume`） | × |
| s08 | ○ | ○（`--resume`） | × |
| s09 | ○ | ○（npz があれば飛ばす） | × |
| s10 | ×（parquet の分割） | ×（毎回作り直す） | × |

1. の入出力は、s01b（入力のパスが固定）を除いて引数で受け取っている。
