# moshirag_data/

日本語 MoshiRAG の学習データを、日本語 Wikipedia の記事から作るスクリプト。

## 工程

```
text_dialogue/     テキストの会話を作る
  s01  記事 → シナリオ（ユーザのペルソナ・目的・動機）
  s02  シナリオ → 会話（ユーザ発話 → 検索の要否 → lead・参照チャンク → 応答 を繰り返し、最後に検査）
  s03  会話 → TTS の台本と、lead・body の位置

speech_dialogue/   音声にする
  s04  台本 → 2 話者のステレオ音声
  s05  音声 → 語ごとの時刻（MFA）

postprocess/       学習用の形にする
  s06  <ret> を置くフレームと lead の長さ
  s07  音声トークン（Mimi）
  s08  テキストトークン
  s09  参照チャンクのベクトル（ARC-Encoder）
  s10  parquet にまとめる
```

## 用語

| 用語 | 意味 |
| --- | --- |
| lead / body | 検索する応答の前置き／本題 |
| `<ret>` | 検索を起動するトークン。lead の直前に置く |
| 参照チャンク | 検索で得た想定の文書。s02 で LLM が記事・知識・推論から作る |
| シナリオ | ユーザ役のペルソナ・目的（user goal）・会話の動機 |
| EOC | ユーザ役が会話を終えるときの印 |

## 動かし方

- **1 件ずつ中身を見る**：`walkthrough.ipynb`
- **記事の束をまとめて処理**：`run_local.py`

```
python moshirag_data/run_local.py --input <記事の JSON> --work <作業場所> --dry_run        # 見積もりだけ
python moshirag_data/run_local.py --input <記事の JSON> --work <作業場所> --steps s01-s03  # テキストまで
python moshirag_data/run_local.py --input <記事の JSON> --work <作業場所>                  # 全工程
```

工程ごとの時間と LLM の使用量は `<作業場所>/step_log.csv`・`llm_usage.csv` に、集計は `summary.json` に出る。

## 必要な環境

| 工程 | 環境 |
| --- | --- |
| s01・s02 | OpenAI 互換の LLM（API か vLLM）。API のときは `moshirag_data/.env` に `OPENAI_API_KEY` |
| s04 | FireRedTTS2 |
| s05 | Montreal Forced Aligner（日本語モデル） |
| s07・s09 | PyTorch と GPU |
| s03・s06・s08・s10 | リポジトリの Python 環境 |

外部環境の置き場所は `run_local.py` の冒頭で指定する。

## 会話の作り方を変えるとき

コードではなく、次のファイルを編集する（`text_dialogue/` の下）。

| 場所 | 中身 |
| --- | --- |
| `prompts/` | 各工程のプロンプト（`0_scenario.txt`・`01_user.txt`・`02_need.txt`・`02-1_lead.txt`・`02-2_ref.txt`・`03_body.txt`・`04_check.txt`） |
| `constants/moshi_persona.json` | Moshi のペルソナ |
| `constants/user_persona.json` | ユーザのペルソナの候補と、年齢層の重み |
| `constants/user_goals.json` | ユーザの目的と終了条件 |
| `constants/openings.json` | 話し始めのパターンと、Moshi の最初の応じ方 |
| `constants/fillers.json` | lead で使うフィラー |
| `constants/spoken_style.json` | 発話の書き方 |
| `constants/moshi_reply_rules.json` | 聞き返しなどの応じ方 |
| `constants/llm_prices.json` | LLM の単価（料金の集計用） |

プロンプトの `{persona}` などは穴埋めの場所なので消さない。
