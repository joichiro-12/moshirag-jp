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

TTS_DIR = Path(os.environ.get("TTS_DIR", "/groups/gcg51557/experiments/0374_japanese_kame/tts/zoom1-tts"))
sys.path.insert(0, str(TTS_DIR))


def install_net_audit() -> list:
    """AUDIT_NET=1 のとき、外部への接続の試みをすべて記録する。

    9/20・9/26 に音声化ジョブが管理者に削除された。9/20 は huggingface_hub が
    HF_HUB_OFFLINE 無しで外部へ接続を試みていたことが分かっている。9/26 は設定済みだったが
    消されたため、見落としている接続が無いかを確かめる。

    名前の問い合わせ（getaddrinfo）と接続（connect）の宛先を標準エラーに書く。
    ループバック（127.0.0.1 / ::1 / localhost）と UNIX ドメインソケットは記録しない。
    戻り値は外部宛ての記録の一覧（終了時に件数を報告するため）。
    """
    import socket
    hits: list = []
    local = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}

    orig_gai = socket.getaddrinfo
    def gai(host, *a, **k):
        if str(host) not in local:
            hits.append(("getaddrinfo", str(host)))
            print(f"  [外部接続の試み] 名前の問い合わせ: {host}", file=sys.stderr, flush=True)
        return orig_gai(host, *a, **k)
    socket.getaddrinfo = gai

    orig_connect = socket.socket.connect
    def connect(self, address):
        if isinstance(address, tuple) and address and str(address[0]) not in local:
            hits.append(("connect", str(address)))
            print(f"  [外部接続の試み] 接続: {address}", file=sys.stderr, flush=True)
        return orig_connect(self, address)
    socket.socket.connect = connect
    return hits


def with_retry(fn, what: str, tries: int = 6):
    """起動が重なったジョブ同士の競合を、間を空けてやり直すことで避ける。

    zoom1_dialogue_tts の resolve_model は、呼ばれるたびに
    assembled/…/llm_posttrain.pt のシンボリックリンクを消して作り直す（model.py 78〜81 行）。
    別のジョブがその瞬間にリンクを消す・読むと FileNotFoundError / FileExistsError になる
    （9/27 の 2451945 で発生。同じ秒に起動した 2451944 と競合した）。
    """
    import random
    for k in range(tries):
        try:
            return fn()
        except (FileNotFoundError, FileExistsError) as e:
            if k == tries - 1:
                raise
            w = random.uniform(5, 30)
            print(f"  [{what}] 起動の重なりによる競合と考えられる失敗。{w:.0f} 秒後にやり直す: {e}",
                  flush=True)
            time.sleep(w)


def install_model_cache() -> dict:
    """FireRedTTS2 をキャッシュ付きに差し替える。戻り値はキャッシュ辞書（件数確認用）。"""
    import fireredtts2.fireredtts2 as f2
    orig = f2.FireRedTTS2
    cache: dict = {}

    def cached(*args, **kw):
        key = (kw.get("pretrained_dir"), kw.get("gen_type"), kw.get("device"))
        if key not in cache:
            t0 = time.time()
            cache[key] = with_retry(lambda: orig(*args, **kw), "モデルの読み込み")
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
    ap.add_argument("--max_seconds", type=float, default=0,
                    help="起動からこの秒数を過ぎたら新しい会話を始めない（0 で無制限）。"
                         "walltime で合成の途中に殺されると、書きかけの wav が次の回で"
                         "完成品として飛ばされるため、walltime より手前で止める")
    ap.add_argument("--oom_retries", type=int, default=12,
                    help="GPU のメモリ不足で失敗したとき、同じ会話をやり直す回数。"
                         "共用の GPU で他の利用者が後からメモリを取ると、飛ばした会話が次々に"
                         "失敗扱いになるため（9/29 の g23 で 6 会話）、飛ばさずに待つ")
    ap.add_argument("--oom_wait", type=float, default=300, help="やり直すまでに待つ秒数")
    a = ap.parse_args()
    t_start = time.time()

    # 外部接続の監査は、モデルやライブラリを読み込む前に仕掛ける
    net_hits = install_net_audit() if os.environ.get("AUDIT_NET") == "1" else None
    if net_hits is not None:
        print("  外部接続の監査: 有効", flush=True)

    cache = install_model_cache()
    from zoom1_dialogue_tts.model import resolve_model
    from zoom1_dialogue_tts.synthesis import synthesize
    from zoom1_dialogue_tts.timing import TimingConfig
    from zoom1_dialogue_tts.script import load_script

    model_dir = with_retry(lambda: resolve_model("llm-jp/zoom1-dialogue-tts", "drop", None, None),
                           "モデルの組み立て")
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
        # 状態の漏れと GPU の非決定性を区別するため、同一プロセス内で 2 種の比較をする。
        #   A: 同じ会話を連続して 2 回合成する（間に何も挟まない）
        #   B: 同じ会話を、間に別の会話を挟んで 2 回合成する
        # A が一致して B が一致しなければ状態の漏れ。A から一致しなければ非決定性であり、
        # ハッシュ比較では漏れを判定できない（その場合は長さで見る）。
        # 既存 wav との比較は参考値として残す（プロセスも GPU も違うので一致を期待しない）。
        import soundfile as sf
        pool = [f for f in targets if (out_dir / f"{f.stem}.wav").exists()]
        x, others = pool[0], pool[1: 1 + max(a.verify - 1, 2)]
        tmp = out_dir / "_verify"
        tmp.mkdir(exist_ok=True)

        def synth(f, tag):
            p = tmp / f"{f.stem}.{tag}.wav"
            t0 = time.time()
            run(f, p)
            return p, time.time() - t0

        def nsamp(p):
            return sf.info(str(p)).frames

        print(f"\n=== 検証（対象 {x.stem}、挟む会話 {len(others)} 件）===", flush=True)
        p1, t1 = synth(x, "a1")
        p2, t2 = synth(x, "a2")
        for k, o in enumerate(others):
            synth(o, f"mid{k}")
        p3, t3 = synth(x, "b")

        h1, h2, h3 = sha(p1), sha(p2), sha(p3)
        n1, n2, n3 = nsamp(p1), nsamp(p2), nsamp(p3)
        n0 = nsamp(out_dir / f"{x.stem}.wav")
        print(f"  1 回目（読み込み直後） {t1:5.1f} 秒  {n1:,} サンプル")
        print(f"  2 回目（連続）         {t2:5.1f} 秒  {n2:,} サンプル  "
              f"{'一致' if h1 == h2 else '不一致'}")
        print(f"  3 回目（{len(others)} 件を挟む）   {t3:5.1f} 秒  {n3:,} サンプル  "
              f"{'一致' if h1 == h3 else '不一致'}")
        print(f"  参考：既存 wav（別プロセス）{n0:,} サンプル  "
              f"{'一致' if sha(out_dir / f'{x.stem}.wav') == h1 else '不一致'}")

        if h1 == h2 == h3:
            print("\n  判定: **モデルの使い回しは安全**（同一プロセス内で決定的、挟んでも変わらない）")
        elif h1 == h2 and h1 != h3:
            print("\n  判定: **状態が漏れている。使い回してはいけない**")
        else:
            print("\n  判定: **同一プロセス内でも非決定的。ハッシュでは漏れを判定できない**")
            print(f"        長さの差 A={abs(n1-n2):,} / B={abs(n1-n3):,} サンプル。"
                  "B が A より桁違いに大きければ漏れを疑う")
        print(f"\n  読み込みを除いた 1 会話の合成時間の目安: {min(t2, t3):.1f} 秒"
              f"（読み込み込みの 1 回目は {t1:.1f} 秒）")
        return

    n = skip = fail = 0
    t0 = time.time()
    for f in targets:
        if a.limit and n >= a.limit:
            break
        if a.max_seconds and time.time() - t_start > a.max_seconds:
            print(f"  起動から {a.max_seconds:.0f} 秒を過ぎたので、新しい会話を始めずに止める", flush=True)
            break
        out = out_dir / f"{f.stem}.wav"
        if out.exists() and out.stat().st_size > 0:
            skip += 1
            continue
        for attempt in range(a.oom_retries + 1):
            try:
                run(f, out)
                n += 1
                break
            except Exception as e:  # noqa: BLE001
                if "out of memory" in str(e).lower() and attempt < a.oom_retries:
                    import torch
                    torch.cuda.empty_cache()
                    print(f"  GPU のメモリ不足 {f.stem}（{attempt + 1} 回目）。"
                          f"{a.oom_wait:.0f} 秒待ってやり直す", flush=True)
                    time.sleep(a.oom_wait)
                    continue
                fail += 1
                print(f"  失敗 {f.stem}: {e}", flush=True)
                break
        if n and n % 10 == 0:
            el = time.time() - t0
            print(f"  {n} 件 / {el:.0f}s（{el/n:.1f} 秒/会話）", flush=True)

    el = time.time() - t0
    print(f"\n  合成 {n} / スキップ {skip} / 失敗 {fail} / {el:.0f} 秒")
    if n:
        print(f"  {el/n:.1f} 秒/会話（モデル読み込みを含む合計時間を件数で割った値）")
    print(f"  モデルの読み込み回数: {len(cache)}")
    if net_hits is not None:
        print(f"  外部接続の試み: {len(net_hits)} 件"
              + ("（無し）" if not net_hits else f" → 宛先 {sorted(set(h[1] for h in net_hits))}"))


if __name__ == "__main__":
    main()
