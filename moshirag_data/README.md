# moshirag_data/：日本語 MoshiRAG の学習データを作るスクリプト

## このリポジトリと、このフォルダの役割

このリポジトリは、**日本語の MoshiRAG を作るための作業リポジトリ**である（LLM-jp 対話グループ、実験番号 0374）。
音声対話モデル KAME の学習コードを土台にしているので、ルートの README・`finetune.py`・`kame_jp/` などは KAME のものである。

**MoshiRAG**（Chien ら, arXiv:2604.12928）は、全二重の音声対話モデル Moshi に検索を組み込んだ構成である。
モデルは、知識が要る質問を受けると、まず「えーと、〇〇ですね」のような前置き（lead）を話し始める。
同時に特別なトークン `<ret>` を出して検索を起動し、検索で得た文書（参照）を、圧縮したベクトルとして受け取る。
前置きを話しているあいだに参照が届くので、そのあとの本題（body）を参照に基づいて答えられる。

このフォルダは、その学習に要るデータを作る。1 件のデータは、次の 3 つがそろった 2 話者の会話である。

- 会話の音声（左が利用者役、右が Moshi 役のステレオ）
- どの時刻に `<ret>` を出すか
- そのとき読む参照の文

これを、日本語 Wikipedia の記事から 10 の工程（s01〜s10）を経て作る。
どんな話題を聞かれても答えられるモデルにするため、題材は Wikipedia の記事だけとし、記事を事前に絞り込まない。

## 用語

| 用語 | 意味 |
| --- | --- |
| 会話 | 利用者役と Moshi 役の 1 回の対話。長さは利用者役が EOC を出すまで（上限 8 往復）。データの単位で、`wiki_scenario_012345` のような ID を持つ（2026-10-06 より前の QA 方式の会話は `wiki_qa_012345`） |
| lead / body | 検索が要る応答の 2 つの部分。前置き・本題 |
| `<ret>` | 検索を起動する特別なトークン。lead の最初のテキストトークンの直前に置く |
| シナリオ | 会話の種。利用者役のペルソナ・ユーザの目的（user goal）・動機（状況・最初の関心・派生しうる関心・なぜ関心があるのか）。s01 で Wikipedia の記事から作る。答えも質問の一覧も書かない |
| 参照 | 検索で得たという想定の文書。学習データでは、LLM が Wikipedia の記事から、利用者役の直前の発話に合わせて作る（記事の本文そのものではない） |
| EOC | 利用者役が会話を終えるときに出す印。いつ出すかはユーザの目的（user goal。会話全体の目的と終了条件。`text_dialogue/constants/user_goals.json`）で決まる |
| augmented / unaugmented | 参照を使って答えるターン／使わないターン |
| 台本 | TTS に渡す、話者つきの発話の並び（`[S1]…` `[S2]…`） |
| manifest | 音声化の結果。各ターンが何秒から何秒までかを記録した JSON |
| 語アライメント | 音声のどこでどの語が話されたかを求めること（MFA を使う） |

## 1 つの会話がどう変わっていくか

```
s01  シナリオ       Wikipedia の記事と、乱択した利用者役のペルソナ・ユーザの目的から、LLM が動機を書く
s02  会話の記録     利用者役の発話 → 検索の要否 →（要るときだけ lead と参照）→ Moshi の応答 を EOC まで繰り返し、
                   最後に台本全体を検査する（不合格は rejected/ に置き、先へ進めない）
s03  台本           会話の記録を TTS の台本と構造 JSON（lead・body の位置と参照の文）に分ける
s04  音声           台本を 2 話者のステレオ音声にする（ターンごとの音声と manifest も残る）
s05  語の時刻       ターンごとの音声で語アライメントをして、会話全体の時間軸に並べる
s06  <ret> の位置   manifest と構造 JSON から、<ret> を置くフレームと lead の長さを求める
s07  音声トークン    会話の音声を Mimi で離散トークンにする（12.5 Hz、8 コードブック）
s08  テキストトークン 語の時刻から、12.5 Hz のテキストの並びを作る
s09  参照のベクトル  参照の文を ARC-Encoder で符号化する
s10  parquet        s06〜s09 をまとめて学習用の parquet にする
```

## 工程の一覧

工程は 3 つのディレクトリに分けてある。

| ディレクトリ | 中身 | 工程 |
| --- | --- | --- |
| `text_dialogue/` | テキスト対話データ作成 | s01〜s03 |
| `speech_dialogue/` | 音声対話データ作成 | s04・s05 |
| `postprocess/` | 後処理（`<ret>` の位置、トークン化、参照のベクトル、parquet） | s06〜s10 |

| 工程 | ファイル | 使う出力 | 出力 | Python の環境 | 投げ方 |
| --- | --- | --- | --- | --- | --- |
| s01 | `text_dialogue/s01_scenario.py` | （日本語 Wikipedia。`--wiki_dir` で渡す） | `wiki_scenario.sNNNofNNN.jsonl` | vLLM 用 venv＋vLLM サーバ | `jobs/qa_gen/wikiqa_main.pbs` |
| s02 | `text_dialogue/s02_dialogue.py` | s01・日本語 Wikipedia（`--wiki_dir`） | `data/moshirag_jp/<名前>/<会話>.{json,txt}`（不合格は `rejected/`） | vLLM 用 venv＋vLLM サーバ | `jobs/qa_gen/gen_wiki_main.pbs` |
| s03 | `text_dialogue/s03_tts_input.py` | s02 | `<会話>.{txt,json,struct.json}` | 標準ライブラリだけ | 手で実行 |
| s04 | `speech_dialogue/s04_synth.py` | s03 の台本 | `<会話>.wav`・`.manifest.json`・`_turns/` | TTS（zoom1-tts）の venv | ABCI：`jobs/synth/synth_resv.pbs`／研究室：`tools/lab/launch_pool.sh` |
| s05 | `speech_dialogue/s05_align.py` | s04 | `words/<会話>.json` | MFA の conda 環境 | 研究室：`tools/lab/batch_align.sh` |
| s06 | `postprocess/s06_ret.py` | s03 の構造 JSON・s04 の manifest | `<会話>.json` | 標準ライブラリだけ | 手で実行 |
| s07 | `postprocess/s07_tok_audio.py` | s04 の会話の音声 | `<会話>.npz` | uv＋GPU | `jobs/post/stage5a_audio.pbs`／研究室：`tools/lab/batch_tokenize.sh` |
| s08 | `postprocess/s08_tok_text.py` | s05 | `<会話>.npz` | uv | `jobs/post/stage5b_text.pbs` |
| s09 | `postprocess/s09_ref_embed.py` | s06 | `<会話>.npz` | uv＋GPU（約 14 GB） | `jobs/post/stage5c.pbs` |
| s10 | `postprocess/s10_pack.py` | s06〜s09 | `*.parquet`（分割） | uv | `jobs/post/stage6_prep.pbs` |

- s01・s02 が共有する部品は `text_dialogue/libs/` の `llm.py`（LLM の呼び出しと思考の除去）・`wiki.py`（記事の読み込み）・`persona.py`（定数とプロンプトの読み込みと整形）にある。Moshi とユーザのペルソナ、ユーザの目的、発話の書き方、フィラー、話し始めのパターン、Moshi の応じ方の決まりは `text_dialogue/constants/` の JSON に置く。工程ごとのプロンプトの本文は `text_dialogue/prompts/` に、工程の番号を付けたテキストファイル（`01_user.txt`・`03_body.txt` など）で置く
- 2026-10-06 に移したファイルの旧パスと新パスは `docs/file_moves_20261006.tsv` にある
- s07・s08 の実装は `tools/tokenize_audio.py`・`tools/tokenize_text.py` にある（KAME と共有しているため）。ここにあるのは入口だけ
- 工程ごとに Python の環境が違うので、本番（ABCI）用に全工程をまとめて呼ぶ入口は作っていない
- 手元（g21）で記事の束を s01〜s10 まで通すときは `run_local.py` を使う（walkthrough.ipynb と同じ処理を全件に流す。`--dry_run` で LLM の呼び出し回数の見積もり、`--steps s01-s03` で工程を選べる）
- PBS はリポジトリのルートで `qsub jobs/…/x.pbs` と投げる。出力は `logs/<ジョブ番号>.pbs1.OU` に出る
- 研究室サーバには s04 を `~/moshirag_tts/synth_worker.py`、s05 を `~/moshirag_mfa/mfa_align.py` という名前で写してある

## もう一度流したときの振る舞い（2026-10-04 時点）

会話は何万件もあり、何日にも分けて作る。そのため、途中で止まっても続きから流せることと、
どのコードで作ったかを後からたどれることが要る。いまの実装がどうなっているかを表にした。**そろえるのはこれからである。**

| 工程 | 出力の単位 | もう一度流したとき | どのコードで作ったかの記録 |
| --- | --- | --- | --- |
| s01 | シャードごとの jsonl に追記 | 出力済みの記事は飛ばす | PBS の出力にだけ残る |
| s02 | 会話ごと | `--resume` で出力済み（不合格の `rejected/` も含む）を飛ばす | 各会話の JSON に `code_commit` |
| s03・s06 | 会話ごと | 全部作り直す | s02 の `code_commit` を写すだけ |
| s04 | 会話ごと | 音声がある会話は飛ばす | ABCI の PBS の出力にだけ残る |
| s05 | 会話ごと | 本体は全部作り直す。`batch_align.sh` が語の時刻の無い会話だけを渡す | ABCI の PBS の出力にだけ残る |
| s07・s08・s09 | 会話ごと | 出力済みを飛ばす（s07・s08 は `--resume`） | 無い |
| s10 | parquet の分割 | 全部作り直す | 無い |
