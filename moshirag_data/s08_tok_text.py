"""工程 s08：s05 の語 JSON を 12.5 Hz のテキストトークン（npz）に変える。

実装は tools/tokenize_text.py で、引数をそのまま渡す入口。MoshiRAG では
text_padding_id 3・end_of_text_padding_id 0・--no_whitespace_before_word を使う。

    uv run --extra data python -m moshirag_data.s08_tok_text --help
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("tools.tokenize_text", run_name="__main__", alter_sys=True)
