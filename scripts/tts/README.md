# 工程 3（音声化）関連スクリプト

TTS は **FireRedTTS2**（Apache 2.0、日本語 FT 版 `finetuned_drop_v2`）。D-14 で決定。
本体コードはグループ領域の `0386_dialogue_model/FireRedTTS2` にあり、
実行用のコピーと venv は `0374_japanese_kame/tts/fireredtts2-code`。

| ファイル | 用途 |
| --- | --- |
| `tts_probe.pbs` | ◆1 の測定。lead を合成して秒数を測る |
| `tts_conv.pbs` | 1 会話まるごとの合成 |
| `annotate_manifest.py` | 行指向レコードの lead/body 構造を manifest に写し、重なりの可否を注釈する |
| `postprocess_structured_overlap.patch` | FireRedTTS2 の `dialogue_tools/postprocess.py` への差分 |
| `lead_probe.txt` / `lead_probe_meta.json` / `conv_probe.txt` | 測定に使った入力 |

## postprocess のパッチについて

元の `add_overlap` は話者交代を一律に扱い、`gap_prob` だけで間/重なりを決めていた。
2026-08-17 の聴取で「フィラーへの重なりは自然だが、body への重なりは違和感がある」
という指摘があり、manifest の `allow_overlap_in` / `allow_overlap_out` を見て
重なりの可否を判定するように変更した。フィールドが無い場合は従来の挙動になる。

MoshiRAG 固有の理由もある。body には検索で得た核心語が入るので、そこに重なると
(1) 検索した情報が踏まれる例を学習してしまい、(2) Keyword Delay の測定が汚れる。

適用:
    cd <FireRedTTS2 のコピー>
    cp dialogue_tools/postprocess.py dialogue_tools/postprocess.py.orig
    patch -p0 < scripts/tts/postprocess_structured_overlap.patch

実測（1 会話・10 交代、`gap_prob 0.15`）:

| | 構造を見ない | 構造を見る |
| --- | --- | --- |
| 抑止 | 0 件 | 6 件 |
| 間 : 重なり | 0 : 10 | 6 : 4 |
| 両ch同時発話 | 3.1% | 0.7% |
