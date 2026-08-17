"""行指向レコードの lead/body 構造を manifest に写し、重なりの可否を注釈する。

allow_overlap_in  : このターンの冒頭に、相手が被せてよいか
                    → 冒頭がフィラー・応答詞（= lead）なら可
allow_overlap_out : このターンの末尾に、相手が被せてよいか
                    → 末尾が body（実質的な内容）なら不可
"""
import json, re, sys

FILLER_HEAD = re.compile(r"^(えーと|えっと|えー|あのー|あの|うーん|うん|あー|そのー|なんか|まあ|まー"
                         r"|はい|そうですね|そうだね|なるほど|ふーん|へえ)")

manifest_path, record_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
man = json.load(open(manifest_path))
turns = man["turns"] if isinstance(man, dict) else man

# 行指向レコードから、moshi 各ターンが lead で始まるか / body で終わるかを拾う
rec = [l.rstrip("\n") for l in open(record_path, encoding="utf-8")]
moshi_flags = []          # (starts_with_lead, ends_with_body)
i = 0
while i < len(rec):
    l = rec[i]
    if l == "(augmented)":
        # lead → Reference → body → tail
        tail = next((x for x in rec[i:i+6] if x.startswith("moshi (tail):")), "")
        ends_body = tail.endswith("[empty]")     # tail が空なら末尾は body
        moshi_flags.append((True, ends_body))
        i += 1
    elif l == "(unaugmented)":
        nxt = rec[i+1] if i+1 < len(rec) else ""
        t = nxt.split(":", 1)[1].strip() if ":" in nxt else ""
        moshi_flags.append((bool(FILLER_HEAD.match(t)), False))   # 相槌系なら被せ可、末尾も軽い
        i += 1
    else:
        i += 1

mi = 0
for t in turns:
    txt = str(t.get("text", "")).lstrip("[]S12 ").strip()
    is_moshi = str(t.get("speaker", "")).upper().find("S2") >= 0
    if is_moshi and mi < len(moshi_flags):
        starts_lead, ends_body = moshi_flags[mi]; mi += 1
        t["allow_overlap_in"] = bool(starts_lead)
        t["allow_overlap_out"] = not ends_body
    else:
        # 人間側は冒頭がフィラーなら被せ可。末尾は質問なので被せ可とする
        t["allow_overlap_in"] = bool(FILLER_HEAD.match(txt))
        t["allow_overlap_out"] = True

json.dump(man, open(out_path, "w"), ensure_ascii=False, indent=1)
n_in = sum(1 for t in turns if t.get("allow_overlap_in"))
n_out = sum(1 for t in turns if t.get("allow_overlap_out"))
print(f"{len(turns)} ターン: 冒頭に被せ可 {n_in} / 末尾に被せられ可 {n_out}")
for t in turns:
    print(f"  {t.get('speaker')} in={str(t.get('allow_overlap_in')):5} out={str(t.get('allow_overlap_out')):5} "
          f"{str(t.get('text',''))[:38]}")
