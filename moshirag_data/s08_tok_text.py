"""工程 s08：語のタイムスタンプ（工程 s05 の出力）を 12.5 Hz のテキストストリームに変える。

実装は `tools/tokenize_text.py` にある（KAME から引き継いだもので、KAME 側のテストと README からも使われているため
moshirag_data/ には移していない）。このファイルは、工程の番号でたどれるようにするための入口で、
引数をそのまま `tools.tokenize_text` に渡す。

    uv run --extra data python -m moshirag_data.s08_tok_text --help

- 入力：s05 の語 JSON（会話ごと）
- 出力：会話ごとの npz（A・B それぞれのテキストトークン列。12.5 Hz のフレームに並べたもの）
- 注意：MoshiRAG では text_padding_id 3・end_of_text_padding_id 0・--no_whitespace_before_word を使う（jobs/post/stage5b_text.pbs）
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("tools.tokenize_text", run_name="__main__", alter_sys=True)
