"""日本語 Wikipedia（ja_wiki の *.jsonl.gz）の記事を読む。s01・s02 が使う。"""

from __future__ import annotations

import glob
import gzip
import json
import os


def iter_wiki(wiki_dir: str):
    """ja_wiki の *.jsonl.gz を 1 記事ずつ返す: {"text": ..., "meta": {"id", "title", "url"}}"""
    files = sorted(glob.glob(os.path.join(wiki_dir, "*.jsonl.gz")))
    if not files:
        raise SystemExit(f"{wiki_dir} に *.jsonl.gz が無い")
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def read_articles(wiki_dir: str, ids: set, chars: int | None = None) -> dict:
    """指定した記事の本文だけを、先頭 chars 文字まで持つ（全記事の本文を持つとメモリが足りなくなる）。"""
    out = {}
    for d in iter_wiki(wiki_dir):
        i = d["meta"].get("id")
        if i in ids:
            out[i] = d["text"][:chars]
            if len(out) == len(ids):
                break
    return out
