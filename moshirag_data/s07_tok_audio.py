"""工程 s07：会話のステレオ wav（L=A, R=B）を Mimi のトークン（npz）に変える。

実装は tools/tokenize_audio.py で、引数をそのまま渡す入口。
GPU の種類や TF32 の設定が違うと 1〜2% のトークンが入れ替わる。

    uv run --extra data python -m moshirag_data.s07_tok_audio --help
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("tools.tokenize_audio", run_name="__main__", alter_sys=True)
