"""工程 5c：<ret> ごとの参照文を ARC-Encoder で符号化する。

**保存するのは bridge の手前、encoder 出力の 3072 次元である。**

初版は bridge を通した 4096 次元を保存していたが誤りだった。論文 §4.2 は
    all parameters trainable except for the reference text encoder
    projected via a one-layer trainable linear layer
と述べており、**ARC-Encoder は凍結だが射影層は学習される**。bridge 出力を焼き込むと
射影層が学習されないまま固定値になる（2026-09-03 に arXiv:2604.12928v2 §4.2 で確認）。

したがって役割を分ける。
    ここ（5c、事前計算）  embedder.forward_embedder → (n/4, 3072)   … 凍結部分
    学習ループ            EmbProjector で 3072 → 4096              … 学習対象

なお論文は射影を "one-layer" と書くが、実装の EmbProjector は 2 層
（3072→2048→4096、bias なし）。配布重み ARC4_Encoder_Llama に layer1 / layer2 の
両方が入っているため、実装（2 層）に合わせ、同梱重みを初期値に使う。

注意：ARC4 の重みは Llama-3.1-8B から読み出せない（2026-08-27 の実測。同条件で ARC8 は
英語復元 token F1 0.347、ARC4 は 0.015）。原因は未特定。ここでは元実装準拠で ARC4 を
使い、Moshi 側の学習で読めるかを検証対象に残す。

出力：会話ごとに .npz（ret_frame と埋め込みの対）
"""
import argparse, json, os, sys, urllib.request
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ARC_URL = "https://raw.githubusercontent.com/kyutai-labs/moshi-rag/main/moshi/moshi/conditioners/arc_encoder.py"


def load_arc_module(cache: Path):
    src = cache / "arc_encoder_src.py"
    if not src.exists():
        src.write_text(urllib.request.urlopen(ARC_URL).read().decode(), encoding="utf-8")
    s = src.read_text(encoding="utf-8")
    s = s[:s.index("class ArcEncoderConditioner")]
    s = "\n".join(l for l in s.splitlines()
                  if not l.startswith("from .base") and not l.startswith("from ..")) + "\n"
    (cache / "arc_min.py").write_text(s, encoding="utf-8")
    sys.path.insert(0, str(cache))
    import arc_min
    return arc_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ret_dir", required=True)
    ap.add_argument("--stems", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arc_repo", default="kyutai/ARC4_Encoder_Llama")
    ap.add_argument("--arc_tokenizer", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--pf", type=int, default=-4)
    ap.add_argument("--cache", default="./_arc_cache")
    ap.add_argument("--device", default="cuda",
                    help="cuda（本番）か cpu。cpu は工程をつないだ試験用で、autocast を切って fp32 で計算する")
    a = ap.parse_args()

    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    arc = load_arc_module(cache)
    dev = a.device
    enc = arc.ArcEncoderTransformer(compression_rate=a.pf).to(dev).eval()
    st = load_file(hf_hub_download(a.arc_repo, "model.safetensors"), device="cpu")
    enc.load_state_dict({k[9:]: v for k, v in st.items() if k.startswith("embedder.")}, strict=False)
    # bridge_module の重みは学習ループ側の初期値として別ファイルに出す（ここでは通さない）
    br = {k[len("bridge_module."):]: v for k, v in st.items() if k.startswith("bridge_module.")}
    assert set(br) == {"layer1.weight", "layer2.weight"}, sorted(br)
    from safetensors.torch import save_file
    save_file(br, os.path.join(os.path.dirname(a.out.rstrip("/")) or ".", "arc_bridge_init.safetensors"))
    print(f"  bridge 初期値を書き出した: {sorted((k, tuple(v.shape)) for k, v in br.items())}", flush=True)
    del st; torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(a.arc_tokenizer)
    print("[arc] 読み込み完了", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stems = [l.strip() for l in open(a.stems) if l.strip()]
    nconv = nref = 0
    for i, stem in enumerate(stems):
        p = out / f"{stem}.npz"
        if p.exists():
            continue
        rf = Path(a.ret_dir) / f"{stem}.json"
        if not rf.exists():
            continue
        ev = json.load(open(rf, encoding="utf-8")).get("ret_events", [])
        if not ev:
            np.savez_compressed(p, ret_frames=np.zeros((0,), np.int32),
                                lengths=np.zeros((0,), np.int32),
                                embeds=np.zeros((0, 3072), np.float16))
            nconv += 1
            continue
        frames, lens, embs = [], [], []
        with torch.no_grad():
            for e in ev:
                ids = tok.encode(e["reference"], add_special_tokens=False)
                t = torch.tensor(ids, device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                    h, sl = enc.forward_embedder(input_ids=t, seqlens=[t.shape[0]])
                frames.append(e["ret_frame"]); lens.append(int(sl[0]))
                embs.append(h.float().cpu().numpy().astype(np.float16))
                nref += 1
        np.savez_compressed(p, ret_frames=np.array(frames, np.int32),
                            lengths=np.array(lens, np.int32),
                            embeds=np.concatenate(embs, 0))
        nconv += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(stems)} 会話 / 参照 {nref} 件", flush=True)
    print(f"完了 {nconv} 会話 / 参照 {nref} 件", flush=True)


if __name__ == "__main__":
    main()
