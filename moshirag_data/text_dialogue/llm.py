"""LLM（OpenAI 互換 API）の呼び出しと、出力の後始末。s01・s02 が使う。"""

from __future__ import annotations

import re

_HARMONY_FINAL = re.compile(r"assistant\s*final\s*", re.IGNORECASE)


# Harmony 形式は analysis チャネルで思考し、assistantfinal 以降が答えになる。
# 生成が途中で打ち切られると assistantfinal が出ず、思考がそのまま返る。
# v5 の 5,000 会話では 81 ターン（1.58% の会話）でこれが起き、英語の独白が
# 台本に入って音声化まで通ってしまった。素通りさせず、検出して生成し直す。
_LOOKS_LIKE_THINKING = re.compile(
    r"^\s*analysis\b"
    r"|^\s*<think>"
    r"|\bNeed to produce\b"
    r"|\bLet's craft\b"
    r"|\bThe user (?:wants|is asking|gave)\b"
    r"|\bWe need to\b"
    r"|\bCount moras?\b",
    re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    parts = _HARMONY_FINAL.split(text)
    out = parts[-1] if len(parts) > 1 else text
    return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()


def looks_like_thinking(text: str) -> bool:
    """思考が剥がれずに残っているか。残っていれば生成をやり直す。"""
    return bool(_LOOKS_LIKE_THINKING.search(text))


def first_line(text: str) -> str:
    for raw in text.splitlines():
        if raw.strip():
            return raw.strip()
    return ""


def strip_label(text: str, *labels: str) -> str:
    """Drop a leading 'label:' the model may have echoed, so we never double-prefix."""
    out = text.strip()
    for _ in range(3):
        for lab in labels:
            if out.startswith(lab):
                out = out[len(lab):].lstrip("：: ").strip()
                break
        else:
            break
    return out


_EFFORT_SHAPE: dict = {}


def call(client, model: str, system: str, user: str, effort: str, tries: int = 3) -> str:
    """思考が混ざった出力は捨てて引き直す。tries 回とも駄目なら例外にする
    （黙って思考混じりを返すより、その会話を落とす方が学習データとして安全）。"""
    last = ""
    for attempt in range(tries):
        out = _call_once(client, model, system, user, effort)
        if not looks_like_thinking(out):
            return out
        last = out
        print(f"    [警告] 思考の混入を検出、引き直す（{attempt + 1}/{tries}）: {out[:60]}",
              flush=True)
    raise RuntimeError(f"思考の混入が {tries} 回続いた: {last[:120]}")


def _call_once(client, model: str, system: str, user: str, effort: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    shapes = [
        {"reasoning_effort": effort},
        {"chat_template_kwargs": {"reasoning_effort": effort}},
        {},
    ]
    if effort in _EFFORT_SHAPE:
        shapes = [shapes[_EFFORT_SHAPE[effort]]] + shapes
    err = None
    for i, body in enumerate(shapes):
        try:
            r = client.chat.completions.create(
                model=model, messages=msgs, stream=False, extra_body=body
            )
            _EFFORT_SHAPE.setdefault(effort, i)
            return strip_thinking(r.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"reasoning_effort rejected: {err}")
