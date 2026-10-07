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

# 思考の切り替え方はモデルの系列で違う。reasoning_effort（low/medium/high）が効くのは
# Harmony 系（llm-jp-4・gpt-oss）だけで、Qwen3.6・Gemma 4 はチャットテンプレートの
# enable_thinking で入れるか切るかしか選べない。low は切り、medium・high は入れる。
# 系列はモデル名（vLLM の --served-model-name）の小文字に次の文字列が含まれるかで決め、
# 含まれなければ従来どおり reasoning_effort を試す。
# vLLM は --reasoning-parser（qwen3 / gemma4）付きで起動し、思考を本文から分けて返させること
_ENABLE_THINKING_MODELS = ("qwen3", "gemma-4")

# Qwen3.6 のモデルカードの推奨値（思考あり＝一般タスク向け、思考なし＝instruct）。
# 同梱の generation_config.json は思考ありの値しか持たないので、ここで渡す。
# 思考ありの presence_penalty だけがモデルで違う（27B は 0.0、35B-A3B は 1.5）
_QWEN36_NO_THINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                    "presence_penalty": 1.5, "repetition_penalty": 1.0}
_QWEN36_THINK = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                 "repetition_penalty": 1.0}
_SAMPLING = {
    "qwen3.6-27b": {True: {**_QWEN36_THINK, "presence_penalty": 0.0}, False: _QWEN36_NO_THINK},
    "qwen3.6-35b-a3b": {True: {**_QWEN36_THINK, "presence_penalty": 1.5}, False: _QWEN36_NO_THINK},
}


def _thinking_body(model: str, effort: str) -> dict | None:
    name = model.lower()
    if not any(k in name for k in _ENABLE_THINKING_MODELS):
        return None
    think = effort != "low"
    sampling = next((v[think] for k, v in _SAMPLING.items() if k in name), {})
    return {**sampling, "chat_template_kwargs": {"enable_thinking": think}}


def call(client, model: str, system: str, user: str, effort: str, tries: int = 3) -> str:
    """思考が混ざった出力と空の出力は捨てて引き直す。tries 回とも駄目なら例外にする
    （黙って思考混じりを返すより、その会話を落とす方が学習データとして安全）。
    空の出力は、思考を本文から分けて返す設定で、思考の途中で長さの上限に達したときに出る。"""
    last = ""
    for attempt in range(tries):
        out = _call_once(client, model, system, user, effort)
        if out and not looks_like_thinking(out):
            return out
        last = out
        what = "思考の混入" if out else "空の出力"
        print(f"    [警告] {what}を検出、引き直す（{attempt + 1}/{tries}）: {out[:60]}",
              flush=True)
    raise RuntimeError(f"思考の混入か空の出力が {tries} 回続いた: {last[:120]}")


def _call_once(client, model: str, system: str, user: str, effort: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    body = _thinking_body(model, effort)
    if body is not None:
        # 引数を外して呼び直すと思考の有無が黙って変わるので、ここでは試し直さない
        r = client.chat.completions.create(
            model=model, messages=msgs, stream=False, extra_body=body
        )
        return strip_thinking(r.choices[0].message.content or "")
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
