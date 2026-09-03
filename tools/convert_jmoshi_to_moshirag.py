"""llm-jp-moshi-v1 の重みを moshi-rag の LMModel が読める形に変換する。

MoshiRAG を真とする方針（2026-09-03）に従い、モデル定義は元実装 kyutai-labs/moshi-rag を
そのまま使う。両者はアーキテクチャが同一で、**self-attention の格納形式だけが違う**。

    ckpt（moshi<=0.2.2 系）        moshi-rag
    self_attn.in_proj_weight   →  self_attn.in_projs.{i}.weight   （weights_per_step 分割）
    self_attn.out_proj.weight  →  self_attn.out_projs.{i}.weight

分割数は depformer が dep_q=8（weights_per_step=True のため）、transformer が 1。
設定の差は 2 つだけで、depformer_causal を落とし（元実装は causal に一本化。接頭辞を
剥がされて causal と衝突する）、rag_token_id=4 を足す。
"""
from __future__ import annotations
import argparse, json, re, torch
from safetensors.torch import load_file, save_file


def split_key(k: str, v: torch.Tensor, nsplit: int):
    """in_proj_weight / out_proj.weight を nsplit 個に分けて (新キー, テンソル) を返す。"""
    if k.endswith("self_attn.in_proj_weight"):
        base = k[: -len("in_proj_weight")]
        return [(f"{base}in_projs.{i}.weight", t.contiguous())
                for i, t in enumerate(v.chunk(nsplit, dim=0))]
    if k.endswith("self_attn.out_proj.weight"):
        base = k[: -len("out_proj.weight")]
        return [(f"{base}out_projs.{i}.weight", t.contiguous())
                for i, t in enumerate(v.chunk(nsplit, dim=0))]
    return [(k, v)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="llm-jp-moshi-v1 の step_*_cleaned")
    p.add_argument("--out", required=True)
    p.add_argument("--rag_token_id", type=int, default=4)
    a = p.parse_args()

    kw = json.load(open(f"{a.src}/moshi_lm_kwargs.json"))
    kw.pop("depformer_causal", None)
    kw["rag_token_id"] = a.rag_token_id
    dep_q = kw["dep_q"]

    csd = load_file(f"{a.src}/model.safetensors", device="cpu")
    out = {}
    for k, v in csd.items():
        n = dep_q if k.startswith("depformer.") else 1
        for nk, nv in split_key(k, v, n):
            out[nk] = nv
    save_file(out, f"{a.out}/model.safetensors")
    json.dump(kw, open(f"{a.out}/moshi_lm_kwargs.json", "w"), indent=2)
    print(f"  {len(csd)} → {len(out)} テンソルに変換して書き出した", flush=True)


if __name__ == "__main__":
    main()
