"""工程 s07：音声（会話の wav）を Mimi のトークンに変える。

実装は `tools/tokenize_audio.py` にある（KAME から引き継いだもので、KAME 側のテストと README からも使われているため
moshirag_data/ には移していない）。このファイルは、工程の番号でたどれるようにするための入口で、
引数をそのまま `tools.tokenize_audio` に渡す。

    uv run --extra data python -m moshirag_data.s07_tok_audio --help

- 入力：会話ごとのステレオ wav（左が話者 A、右が話者 B）が並んだフォルダ
- 出力：会話ごとの npz（A・B それぞれ 8 コードブック × フレーム）
- 注意：A5000 同士なら完全に一致し、GPU の種類や TF32 の設定を変えると 1〜2% のトークンが入れ替わる（2026-09-28・29 の実測）
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("tools.tokenize_audio", run_name="__main__", alter_sys=True)
