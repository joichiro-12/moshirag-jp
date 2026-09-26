"""TTS のモデルを 1 回だけ読み込み、担当分の会話をまとめて音声化する。

## なぜ必要か

既存の `synth_shard.pbs` は会話 1 本ごとに `uv run zoom1-dialogue-tts` を起動する。
`synthesize()` は内部で毎回 `FireRedTTS2(...)` を作るので、**会話ごとにモデル読み込みが
走っている**。実測は 100 秒/会話で、うちどれだけが読み込みかは未測定である。

音声化は工程全体の律速で、会話生成の 10.4 倍の GPU 時間を使う。ここを詰めると
全体の見通しが変わる。

## やり方

TTS パッケージは git 管理外なので**本体は書き換えない**。
`fireredtts2.fireredtts2.FireRedTTS2` をキャッシュ付きに差し替えることで、
`synthesize()` を無改造のまま呼び、2 回目以降の読み込みを省く。
`synthesis.py` は関数内で import しているため、モジュール属性の差し替えが効く。

## 確認していないこと

**モデルを使い回したときに会話間で状態が漏れないか。** 漏れると出力が変わる。
`--verify` は既存の wav と再合成結果を突き合わせてこれを確かめる。
乱数種は `synthesize()` 内で会話ごとに再設定される（`torch.manual_seed(timing.seed)`）。
"""
from __future__ import annotations
import argparse, hashlib, os, sys, time
from pathlib import Path

TTS_DIR = Path("/groups/gcg51557/experiments/0374_japanese_kame/tts/zoom1-tts")
sys.path.insert(0, str(TTS_DIR))


def install_model_cache() -> dict:
    """FireRedTTS2 をキャッシュ付きに差し替える。戻り値はキャッシュ辞書（件数確認用）。"""
    import fireredtts2.fireredtts2 as f2
    orig = f2.FireRedTTS2
    cache: dict = {}

    def cached(*args, **kw):
        key = (kw.get("pretrained_dir"), kw.get("gen_type"), kw.get("device"))
        if key not in cache:
            t0 = time.time()
            cache[key] = orig(*args, **kw)
            print(f"  [モデル読み込み] {time.time()-t0:.1f} 秒", flush=True)
        return cache[key]

    f2.FireRedTTS2 = cached
    return cache


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="台本（.txt）のディレクトリ")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 で全件")
    ap.add_argument("--max_turn_ms", type=float, default=45000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", type=int, default=0,
                    help="既存 wav と一致するかを確かめる件数。合成結果は捨てる")
    a = ap.parse_args()

    cache = install_model_cache()
    from zoom1_dialogue_tts.model import resolve_model
    from zoom1_dialogue_tts.synthesis import synthesize
    from zoom1_dialogue_tts.timing import TimingConfig
    from zoom1_dialogue_tts.script import load_script

    model_dir = resolve_model("llm-jp/zoom1-dialogue-tts", "drop", None, None)
    in_dir, out_dir = Path(a.in_dir), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.txt"))
    # 添字は分割前のまま保つ（会話生成と同じ方針）
    targets = [f for i, f in enumerate(files) if i % a.nshard == a.shard]
    print(f"  台本 {len(files):,} 件 / シャード {a.shard}/{a.nshard} 担当 {len(targets):,} 件",
          flush=True)

    def run(script: Path, out: Path):
        return synthesize(
            model_dir=model_dir, turns=load_script(str(script)), output_path=out,
            prompts=[], timing=TimingConfig(seed=a.seed),
            turn_timing="stat", turn_vap_json=None,
            vap_python=".venv-vap/bin/python", vap_device="cpu",
            backchannels="stat", vap_json=None, bc_per_minute=3.1,
            temperature=0.9, topk=20, max_turn_ms=a.max_turn_ms,
        )

    if a.verify:
        # 既存 wav と一致するかを見る。1 プロセス内で連続合成し、状態の漏れを検出する。
        done = [f for f in targets if (out_dir / f"{f.stem}.wav").exists()][: a.verify]
        print(f"\n=== 検証: 既存 {len(done)} 件を同一プロセスで再合成して突き合わせる ===",
              flush=True)
        tmp = out_dir / "_verify"
        tmp.mkdir(exist_ok=True)
        ok = ng = 0
        for k, f in enumerate(done, 1):
            t0 = time.time()
            run(f, tmp / f"{f.stem}.wav")
            same = sha(out_dir / f"{f.stem}.wav") == sha(tmp / f"{f.stem}.wav")
            ok, ng = (ok + 1, ng) if same else (ok, ng + 1)
            print(f"  {k}/{len(done)} {f.stem}: "
                  f"{'一致' if same else '**不一致**'}  {time.time()-t0:.1f} 秒", flush=True)
        print(f"\n  一致 {ok} / 不一致 {ng}")
        print("  判定:", "モデルの使い回しは安全" if ng == 0
              else "**状態が漏れている。使い回してはいけない**")
        return

    n = skip = fail = 0
    t0 = time.time()
    for f in targets:
        if a.limit and n >= a.limit:
            break
        out = out_dir / f"{f.stem}.wav"
        if out.exists() and out.stat().st_size > 0:
            skip += 1
            continue
        try:
            run(f, out)
            n += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"  失敗 {f.stem}: {e}", flush=True)
        if n and n % 10 == 0:
            el = time.time() - t0
            print(f"  {n} 件 / {el:.0f}s（{el/n:.1f} 秒/会話）", flush=True)

    el = time.time() - t0
    print(f"\n  合成 {n} / スキップ {skip} / 失敗 {fail} / {el:.0f} 秒")
    if n:
        print(f"  {el/n:.1f} 秒/会話（モデル読み込みを含む合計時間を件数で割った値）")
    print(f"  モデルの読み込み回数: {len(cache)}")


if __name__ == "__main__":
    main()
