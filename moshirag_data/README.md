# moshirag_data/

日本語 MoshiRAG の学習データを、日本語 Wikipedia の記事から作るスクリプト。

## 工程

```
text_dialogue/     テキストの会話を作る
  s01  記事 → シナリオ（ユーザのペルソナ・目的・動機）
  s02  シナリオ → 会話（ユーザ発話 → 検索の要否 → lead・参照チャンク → 応答 を繰り返す）
  s02s 質問 → 1 ターンの会話（QA データセットの質問 1 つ → lead・参照チャンク → 応答）
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

dev/               開発用
  run_local.py       記事の束を s01〜s10 まで手元で通す
  check_dialogue.py  s02 の会話を LLM で検査し、不合格を rejected/ に移す（事後検査）
  walkthrough.ipynb  1 件ずつ中身を見ながら通す
```

## 1 ターンの会話（s02s）

MoshiRAG 論文（arXiv:2604.12928）の single-turn subset に合わせた工程。論文は、QA データセットの質問をそのまま
ユーザの発話にし、参照とモシの応答を LLM で作って、複数ターンの会話と一緒に学習データに入れた（§4.1.2・付録 A）。
複数ターンの会話は挨拶や前置きから始まることがあるが、この工程の会話は挨拶なしで質問から始まる。

```
python moshirag_data/text_dialogue/s02s_single_turn.py --input_file <質問の jsonl> --output_dir <出力先> \
    --model "$NAME" --llm_base_url "http://localhost:${PORT}/v1" --num_samples 1000 --resume
```

- 入力は 1 行 1 問の jsonl：`{"id": ..., "question": ..., "answers": [...], "source": ..., "wiki_id": ..., "title": ...}`。`question` のほかは任意
- 02-1（lead）・02-2（参照チャンク）・03（body）・04（相づち）は s02 と同じプロンプトを使う。話し始めのパターンは `質問` に固定
- `--need always`（既定）はすべての質問で検索する（論文の single-turn subset は、どの会話にも参照がある）。`--need judge` は s02 と同じ 02 の判定に従う
- `--answer_hint` は、`answers` をデータセットの正解として 02-2 に渡す。`--wiki_dir` があれば、`wiki_id` の記事も 02-2 に渡す
- 出力は s02 と同じ形（`<stem>.json` と `<stem>.txt`）なので、s03 以降はそのまま使える。ユーザの声の性別は、質問ごとに乱択したペルソナから取る
- 評価に使うデータセットの質問は入れない（学習とテストで同じ質問が重ならないようにする）

## 用語

| 用語 | 意味 |
| --- | --- |
| lead / body | 検索する応答の前置き／本題 |
| `<ret>` | 検索を起動するトークン。lead の直前に置く |
| 参照チャンク | 検索で得た想定の文書。s02 で LLM が記事・知識・推論から作る |
| シナリオ | ユーザ役のペルソナ・目的（user goal）・会話の動機 |
| EOC | ユーザ役が会話を終えるときの印 |

## 動かし方

- **1 件ずつ中身を見る**：`dev/walkthrough.ipynb`
- **記事の束をまとめて処理**：`dev/run_local.py`

```
python moshirag_data/dev/run_local.py --input <記事の JSON> --work <作業場所> --steps s01-s03  # テキストまで
python moshirag_data/dev/run_local.py --input <記事の JSON> --work <作業場所>                  # 全工程
```

## 必要な環境

| 工程 | 環境 |
| --- | --- |
| s01・s02 | OpenAI 互換の LLM（API か vLLM）。API のときは `moshirag_data/.env` に `OPENAI_API_KEY` |
| s04 | FireRedTTS2 |
| s05 | Montreal Forced Aligner（日本語モデル） |
| s07・s09 | PyTorch と GPU |
| s03・s06・s08・s10 | リポジトリの Python 環境 |

外部環境の置き場所は `dev/run_local.py` の冒頭で指定する。

s04 の声は `--voices speech_dialogue/sample_audio/voices.json` で指定する（書き方は `speech_dialogue/s04_synth.py` の `load_voices`）。
Moshi は固定の声。ユーザはシナリオの性別（s03 が `struct.json` の `user_gender` に書く）の候補から
会話ごとに 1 つ選び、選んだ声を `s04_audio/<stem>.voice.json` に残す。
省略するとモデル任せの声（会話ごとに変わる）。

## 会話の作り方を変えるとき

コードではなく、次のファイルを編集する（`text_dialogue/` の下）。

| 場所 | 中身 |
| --- | --- |
| `prompts/` | 各工程のプロンプト（`0_scenario.txt`・`01_user.txt`・`02_need.txt`・`02-1_lead.txt`・`02-2_ref.txt`・`03_body.txt`・`04_backchannel.txt`） |
| `constants/moshi_persona.json` | Moshi のペルソナ |
| `constants/user_persona.json` | ユーザのペルソナの候補と、年齢層の重み |
| `constants/user_goals.json` | ユーザの目的と終了条件。目的によっては、その目的のときだけ渡す指示（`scenario_rule`・`user_rule`）と、記事の一節を読み上げるか（`passage`） |
| `constants/openings.json` | 話し始めのパターンと、Moshi の最初の応じ方 |
| `constants/followup_forms.json` | 2 発話目以降にときどき指定する発話の形（短い追加の質問）と、その確率 |
| `constants/fillers.json` | lead で使うフィラー |
| `constants/spoken_style.json` | 発話の書き方 |
| `constants/moshi_reply_rules.json` | 聞き返しなどの応じ方 |

事後検査の基準は `dev/prompts/04_check.txt` にある。

プロンプトの `{persona}` などは穴埋めの場所なので消さない。
