"""ARC-Encoder が日本語の参照文書を 4 倍圧縮しても情報を保てるかを測る。

論文（arXiv:2510.20535）Table 1 の枠組みを日本語で再現する。3 条件を比較し、
圧縮で失われた分だけを取り出す。

  closed-book : 参照を渡さない        → decoder のパラメトリック知識だけの下限
  open-book   : 参照を生テキストで渡す → 上限
  compressed  : 参照を 4 倍圧縮して渡す → 検証対象

材料は pilot5k の参照文書 + seed_question / seed_answer。参照に答えが含まれることが
生成時点で保証されている（body がその参照に根拠づいている）。

圧縮表現は ArcEncoderTransformer.forward_embedder の出力（3072 次元、Llama 空間）を
使う。bridge_module（Moshi 空間 4096 次元への変換）は通さない。decoder が読むのは
Llama 空間だからである。

decoder への入力は few-shot テンプレートを使わず、圧縮埋め込みを prompt の埋め込み列に
直接差し込む（inputs_embeds）。論文は 5-shot で評価しているが、ここでは圧縮の情報保持を
見たいので shot を入れず、3 条件で同じ形式を使って差だけを見る。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch


def load_arc(src: Path, hf_repo: str, tokenizer_name: str, device: str):
    """moshi-rag の arc_encoder.py を単体で読み込む。

    元の module は moshi パッケージ内の相対 import を持つので、必要な依存だけを
    偽装して読み込む。
    """
    import types

    src_text = src.read_text(encoding="utf-8")
    # 相対 import を落とし、必要なものだけ自前で与える
    src_text = re.sub(r"^from \.\.?[\w.]* import .*$", "", src_text, flags=re.M)
    src_text = re.sub(r"^from xformers.*$", "", src_text, flags=re.M)
    # Conditioner 系のクラスは Moshi への接続用で、圧縮の測定には不要。削除した相対
    # import（_BaseTextConditioner / ConditionType / TokenizedText）に依存するので、
    # クラス定義ごと落とす。使うのは ArcEncoderTransformer と ArcEncoderTokenizer だけ。
    cut = src_text.find("class ArcEncoderConditioner")
    if cut > 0:
        src_text = src_text[:cut]

    mod = types.ModuleType("arc_standalone")
    # xformers の attention を torch の SDPA で代替する
    helper = '''
import torch
from torch.nn.functional import scaled_dot_product_attention

class BlockDiagonalCausalMask:
    @staticmethod
    def from_seqlens(seqlens, kv_seqlen=None): return ("causal", seqlens)

class BlockDiagonalMask:
    @staticmethod
    def from_seqlens(seqlens, kv_seqlen=None): return ("full", seqlens)

def memory_efficient_attention(q, k, v, attn_bias=None):
    # q,k,v: (seqlen, heads, head_dim) を (1, heads, seqlen, head_dim) に直して SDPA
    causal = isinstance(attn_bias, tuple) and attn_bias[0] == "causal"
    qq = q.unsqueeze(0).transpose(1, 2)
    kk = k.unsqueeze(0).transpose(1, 2)
    vv = v.unsqueeze(0).transpose(1, 2)
    o = scaled_dot_product_attention(qq, kk, vv, is_causal=causal)
    return o.transpose(1, 2).squeeze(0)

def length_to_mask(lengths, max_len=None):
    if max_len is None: max_len = int(lengths.max().item()) if len(lengths) else 0
    return torch.arange(max_len)[None, :] < lengths[:, None]

class TorchAutocast:
    def __init__(self, enabled=True, **kw): self.enabled = enabled; self.kw = kw
    def __enter__(self):
        if self.enabled:
            self.ctx = torch.autocast(**self.kw); self.ctx.__enter__()
        return self
    def __exit__(self, *a):
        if self.enabled: self.ctx.__exit__(*a)
'''
    exec(compile(helper + "\n" + src_text, str(src), "exec"), mod.__dict__)

    enc = mod.ArcEncoderTransformer(compression_rate=-4).to(device).eval()
    tok = mod.ArcEncoderTokenizer(tokenizer_name)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    state = load_file(hf_hub_download(hf_repo, "model.safetensors"), device=device)
    enc_state = {k[len("embedder."):]: v for k, v in state.items() if k.startswith("embedder.")}
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    print(f"[arc] embedder 重み: 未読み込み {len(missing)} / 余り {len(unexpected)}", flush=True)
    return mod, enc, tok


@torch.no_grad()
def compress(enc, tok, text: str, device: str):
    ids = tok.encode(text, bos=False, eos=False)
    t = torch.tensor(ids, device=device)
    emb, seqlens = enc.forward_embedder(input_ids=t, seqlens=[t.shape[0]])
    return emb, len(ids), int(seqlens[0])


PROMPT = "参考情報にもとづいて質問に短く答えてください。\n質問: {q}\n答え:"


@torch.no_grad()
def ask(dec, dtok, device, question: str, ref_text: str | None = None,
        ref_emb: torch.Tensor | None = None, max_new: int = 24) -> str:
    """3 条件を同じ形式で走らせる。参照は埋め込み列の先頭に置く。"""
    emb_layer = dec.get_input_embeddings()
    parts = []
    if ref_text is not None:
        ids = dtok("参考情報: " + ref_text + "\n", return_tensors="pt",
                   add_special_tokens=False).input_ids.to(device)
        parts.append(emb_layer(ids))
    elif ref_emb is not None:
        head = dtok("参考情報: ", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        tailids = dtok("\n", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        parts += [emb_layer(head), ref_emb.unsqueeze(0).to(emb_layer.weight.dtype), emb_layer(tailids)]
    q = dtok(PROMPT.format(q=question), return_tensors="pt",
             add_special_tokens=False).input_ids.to(device)
    parts.append(emb_layer(q))
    inputs_embeds = torch.cat(parts, dim=1)
    out = dec.generate(inputs_embeds=inputs_embeds, max_new_tokens=max_new,
                       do_sample=False, pad_token_id=dtok.eos_token_id)
    return dtok.decode(out[0], skip_special_tokens=True).strip()


def hit(pred: str, answer: str) -> bool:
    """正解語が出力に含まれるか（論文の EM ではなく包含。日本語の表記揺れに寛容にする）。"""
    def norm(s):
        return re.sub(r"[\s、。・「」『』（）()]", "", s)
    a, p = norm(answer), norm(pred)
    return bool(a) and a in p


def main(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mod, enc, atok = load_arc(Path(args.arc_src), args.hf_repo, args.arc_tokenizer, device)

    print(f"[dec] {args.decoder} を読み込み", flush=True)
    dtok = AutoTokenizer.from_pretrained(args.decoder)
    dec = AutoModelForCausalLM.from_pretrained(
        args.decoder, dtype=torch.bfloat16, device_map=device).eval()

    # 材料：参照文書に答えが含まれている組を集める
    rng = random.Random(0)
    files = sorted(Path(args.pilot_dir).glob("*.json"))
    rng.shuffle(files)
    items = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        ans = str(d.get("seed_answer", "")).strip()
        q = str(d.get("seed_question", "")).strip()
        if not ans or not q or len(ans) < 2:
            continue
        for ch in d.get("chunks", []):
            if hit(ch, ans):           # 参照に答えが含まれているものだけ使う
                items.append({"q": q, "a": ans, "ref": ch})
                break
        if len(items) >= args.n:
            break
    print(f"[data] {len(items)} 件（参照に答えが含まれる組）", flush=True)

    res = {"closed": 0, "open": 0, "comp": 0}
    ratios = []
    samples = []
    for i, it in enumerate(items, 1):
        emb, ntok, ncomp = compress(enc, atok, it["ref"], device)
        ratios.append((ntok, ncomp))
        o_closed = ask(dec, dtok, device, it["q"])
        o_open = ask(dec, dtok, device, it["q"], ref_text=it["ref"])
        o_comp = ask(dec, dtok, device, it["q"], ref_emb=emb)
        for k, o in (("closed", o_closed), ("open", o_open), ("comp", o_comp)):
            res[k] += hit(o, it["a"])
        if len(samples) < 5:
            samples.append((it, o_closed, o_open, o_comp))
        if i % 20 == 0:
            print(f"  {i}/{len(items)}  closed {res['closed']} / open {res['open']} / comp {res['comp']}", flush=True)

    n = len(items)
    print("\n===== 結果 =====")
    print(f"n = {n}")
    tt = sum(a for a, _ in ratios); cc = sum(b for _, b in ratios)
    print(f"圧縮率（実測）: 入力 {tt} トークン -> 圧縮後 {cc}  = {tt / max(cc,1):.2f} 倍")
    for k, label in (("closed", "closed-book（参照なし）"), ("open", "open-book（生テキスト）"),
                     ("comp", "compressed（4 倍圧縮）")):
        print(f"  {label:<28} {res[k]}/{n} = {100*res[k]/max(n,1):.1f}%")
    if res["open"] > res["closed"]:
        keep = (res["comp"] - res["closed"]) / (res["open"] - res["closed"])
        print(f"\n情報保持率 =(comp-closed)/(open-closed) = {100*keep:.0f}%")
    print("\n===== 例 =====")
    for it, c, o, p in samples:
        print(f"Q: {it['q'][:56]}")
        print(f"A: {it['a']}")
        print(f"  参照: {it['ref'][:70]}")
        print(f"  closed: {c[:70]}")
        print(f"  open  : {o[:70]}")
        print(f"  comp  : {p[:70]}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arc_src", required=True, help="moshi-rag の arc_encoder.py")
    p.add_argument("--hf_repo", default="kyutai/ARC4_Encoder_Llama")
    p.add_argument("--arc_tokenizer", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--decoder", default="/groups/gcg51557/models/released/meta-llama/Llama-3.1-8B")
    p.add_argument("--pilot_dir", required=True)
    p.add_argument("--n", type=int, default=100)
    main(p.parse_args())
