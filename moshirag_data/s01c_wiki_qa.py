"""s01b の候補記事から、LLM で問いと答えの組を作る。

記事の適否判定と QA 生成を 1 回の呼び出しで行い、題材にならない記事はスキップさせる。
出力は jaquad.jsonl と同じ形で、そのまま s02 の --input_file に渡せる。

Each line: {"question": "...", "answer": "...", "source": "...", "title": "...", "url": "...", "index": ...}
"""
from __future__ import annotations
import argparse, json, os, random, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# API の呼び出しと思考混入の検出は s02（旧 02e）の実装をそのまま使う。
# reasoning_effort の渡し方はサーバ実装に依存し、s02 は 3 通り試して当たったものを
# 覚える作りになっている。複製すると片方だけ直す事故が起きるので import する。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
_v5 = __import__("s02_dialogue")
call = _v5.call

QA_ROLE = """\
Wikipedia の記事の冒頭を渡します。その内容から、**会話の中で自然に出てきそうな問い**と、
その答えを 1 組作ってください。

- 答えは**記事に書かれている事実**にする。記事に無いことを書かない
- 答えは短くする。固有名詞・数値・短い語句
- 問いは、記事を読んでいない人が口に出しそうなものにする
- 問いに答えを含めない

**次のような記事は題材にならないので、問いを作らず「スキップ」とだけ書いてください。**

- 自治体の外郭団体、学校、駅など、内容が定型的で話題にならないもの
- 統計や年表だけで構成され、人が尋ねる内容が無いもの
- 個別の試合結果・大会の回次など、その年その回にしか意味がないもの
- 記事が短すぎて事実を 1 つも取り出せないもの

出力は次の 2 行だけ。スキップの場合は「スキップ」の 1 行だけ。

問い: ...
答え: ...

例：

記事タイトル: 手塚治虫
記事冒頭: 手塚 治虫（てづか おさむ、1928年11月3日 - 1989年2月9日）は、日本の漫画家、
アニメーター、アニメーション監督。医師免許取得者であり、医学博士。戦後日本において
ストーリー漫画の第一人者として...

問い: 漫画家で医師免許も持っていた人っていましたよね、誰でしたっけ
答え: 手塚治虫

記事タイトル: 岡山県交通安全協会
記事冒頭: 一般財団法人岡山県交通安全協会は、岡山県内の地区交通安全協会を統括する...

スキップ"""


# 既定より厳しい判定。対象そのものの知名度を見る。
# 既定は 1.6% しか落とさず、無名の人物・地方の施設が残った（パイロット 1,000 件で確認）。
QA_ROLE_STRICT = """\
Wikipedia の記事の冒頭を渡します。その内容から、**会話の中で自然に出てきそうな問い**と、
その答えを 1 組作ってください。

まず、**その記事の対象を、人が雑談や相談の中で自発的に持ち出すか**を判断してください。
持ち出さないものは、問いを作らず「スキップ」とだけ書いてください。

**スキップするもの（対象そのものが知られていない）**

- 専門家の間でしか知られていない人物。地方の実業家、地域史の人物、脇役の研究者など
- 外国のスポーツ選手・芸能人のうち、日本でほとんど報じられない人
- 地方の施設、団体の支部、道路、バスの営業所、個別の建物
- 作品に付随するもの（サウンドトラック、単発のイベント、個別の話数）
- 個別の艦船・車両・機体のうち、事件や事故で知られていないもの
- 統計や年表だけで構成され、人が尋ねる内容が無いもの

**残すもの（対象が知られている、または誰でも関心を持ちうる）**

- 歴史上の出来事、よく知られた人物、国や地域、制度や法律
- 食べ物、動植物、自然現象、病気、技術、学問の概念
- 広く知られた作品・企業・製品・スポーツチーム
- 日常で話題になる物事（天気、交通、健康、料理、旅行など）

残す場合は次の条件で問いと答えを作ってください。

- 答えは**記事に書かれている事実**にする。記事に無いことを書かない
- 答えは短くする。固有名詞・数値・短い語句
- 問いは、記事を読んでいない人が口に出しそうなものにする
- 問いに答えを含めない

出力は次の 2 行だけ。スキップの場合は「スキップ」の 1 行だけ。

問い: ...
答え: ...

例：

記事タイトル: 手塚治虫
記事冒頭: 手塚 治虫（てづか おさむ、1928年11月3日 - 1989年2月9日）は、日本の漫画家、
アニメーター、アニメーション監督。医師免許取得者であり、医学博士...

問い: 漫画家で医師免許も持っていた人っていましたよね、誰でしたっけ
答え: 手塚治虫

記事タイトル: 黒田茂助
記事冒頭: 黒田 茂助（くろだ もすけ）は、明治期の漆器商。石川県に生まれ...

スキップ

記事タイトル: 東急バス高津営業所
記事冒頭: 東急バス高津営業所は、神奈川県川崎市高津区に所在する東急バスの営業所...

スキップ"""

ROLES = {"default": QA_ROLE, "strict": QA_ROLE_STRICT}


def parse(out: str) -> tuple[str, str] | None:
    """「問い: … / 答え: …」を取り出す。スキップまたは解釈できない形なら None。"""
    if "スキップ" in out[:20]:
        return None
    q = a = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(("問い:", "問い：")):
            q = line.split(":", 1)[-1].split("：", 1)[-1].strip()
        elif line.startswith(("答え:", "答え：")):
            a = line.split(":", 1)[-1].split("：", 1)[-1].strip()
    if not q or not a or len(a) > 60:
        return None
    return q, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True, help="s01b の出力")
    ap.add_argument("--output_file", required=True)
    ap.add_argument("--num_samples", type=int, default=0, help="0 で全件")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--role", default="default", choices=["default", "strict"],
                    help="判定プロンプト。strict は対象の知名度を見る")
    ap.add_argument("--model", required=True)
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--api_key", default=None)
    a = ap.parse_args()

    client = OpenAI(api_key=a.api_key or os.getenv("OPENAI_API_KEY") or "dummy",
                    base_url=a.llm_base_url or None)

    rows = [json.loads(l) for l in open(a.input_file, encoding="utf-8") if l.strip()]
    # s02 と同じ方針：乱数種のみでシャッフルし前方を切る。件数を増やしても既存分が活きる
    random.Random(a.seed).shuffle(rows)
    if a.num_samples:
        rows = rows[: a.num_samples]
    # 添字は分割前のまま保つ（再開時の同一性のため）
    targets = [(i, r) for i, r in enumerate(rows) if i % a.nshard == a.shard]
    print(f"  判定プロンプト: {a.role} / 候補 {len(rows):,} 件 / "
          f"シャード {a.shard}/{a.nshard} 担当 {len(targets):,} 件", flush=True)

    # 出力はシャードごとに分ける。共有ファイルへの並行追記はノードをまたぐと
    # 保護できないため（スレッドロックはプロセス内にしか効かない）。
    out_path = Path(a.output_file)
    if a.nshard > 1:
        out_path = out_path.with_name(
            f"{out_path.stem}.s{a.shard:03d}of{a.nshard:03d}{out_path.suffix}")
        print(f"  出力: {out_path.name}", flush=True)

    # 既に書いた記事は飛ばす（--resume 相当）
    done = set()
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            try:
                done.add(json.loads(l)["wiki_id"])
            except Exception:
                pass
        print(f"  既存 {len(done):,} 件をスキップ", flush=True)

    def work(idx_row):
        i, r = idx_row
        if r["wiki_id"] in done:
            return "skip_done", None
        try:
            out = call(client, a.model, ROLES[a.role],
                       f"記事タイトル: {r['title']}\n記事冒頭: {r['lead']}",
                       a.effort)
        except Exception as e:  # noqa: BLE001
            return f"FAIL {e}", None
        qa = parse(out)
        if qa is None:
            return "skip_llm", None
        q, ans = qa
        return "ok", {"question": q, "answer": ans, "source": "ja_wiki",
                      "category": "wiki_generated", "wiki_id": r["wiki_id"],
                      "title": r["title"], "url": r["url"], "index": i}

    n_ok = n_skip_llm = n_skip_done = n_fail = 0
    import threading
    lock = threading.Lock()
    with out_path.open("a", encoding="utf-8") as w, \
         ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, t) for t in targets]
        for k, fu in enumerate(as_completed(futs), 1):
            status, rec = fu.result()
            if status == "ok":
                n_ok += 1
                with lock:
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    w.flush()
            elif status == "skip_llm":
                n_skip_llm += 1
            elif status == "skip_done":
                n_skip_done += 1
            else:
                n_fail += 1
                if n_fail <= 3:
                    print(f"    {status}", flush=True)
            if k % 200 == 0:
                print(f"    {k:,}/{len(targets):,}  採用 {n_ok:,} / "
                      f"スキップ {n_skip_llm:,} / 失敗 {n_fail:,}", flush=True)

    tried = n_ok + n_skip_llm + n_fail
    print(f"\n  採用 {n_ok:,} / LLM スキップ {n_skip_llm:,} / 既存 {n_skip_done:,} / 失敗 {n_fail:,}")
    if tried:
        print(f"  採用率 {100*n_ok/tried:.1f}%（既存を除く {tried:,} 件に対して）")


if __name__ == "__main__":
    main()
