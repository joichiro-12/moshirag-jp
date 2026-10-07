"""Moshi のペルソナ、ユーザのペルソナと目的（user goal）、発話の書き方。s01・s02 のプロンプトに埋め込む。

中身はすべて constants/ の JSON に置き、ここでは読み込みとプロンプト用の整形だけをする。
各工程のプロンプトの本文は prompts/ のテキストファイルに置き、load_prompt で読む。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# constants/ と prompts/ は libs/ と同じ階層（text_dialogue/ の直下）にある
CONSTANTS_DIR = Path(__file__).resolve().parent.parent / "constants"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_constant(name: str):
    """constants/<name> の JSON を読む。"""
    return json.loads((CONSTANTS_DIR / name).read_text(encoding="utf-8"))


def load_prompt(name: str) -> str:
    """prompts/<name> のプロンプトを読む。{persona} などの穴は呼ぶ側で format する。"""
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text   # ファイル末尾の改行はプロンプトに含めない


# ======================================================================================
# Moshi のペルソナ（固定）：constants/moshi_persona.json
# 工程ごとに見せる範囲が違うので 3 つに分ける
#   名前と役割 → 0・01 / ペルソナ → 02-1・03・04 / 知識の範囲 → 02
# ======================================================================================
_MOSHI = load_constant("moshi_persona.json")
MOSHI_NAME_ROLE: str = _MOSHI["名前と役割"]
MOSHI_PERSONA: str = "\n".join(f"- {k}：{v}" for k, v in _MOSHI["ペルソナ"].items())
MOSHI_KNOWLEDGE: str = "\n".join(f"- {x}" for x in _MOSHI["知識の範囲"])

# ======================================================================================
# ユーザのペルソナと目的（s01 で乱択し、s01 と 01 に見せる）
#   constants/user_persona.json・constants/user_goals.json
# ======================================================================================
_USER = load_constant("user_persona.json")
AGES: list = _USER["年齢層"]
GENDERS: list = _USER["性別"]
# 年齢層と矛盾しない職業だけを選ぶ。年齢層ごとの指定が無ければ「その他の年齢層」から選ぶ
JOBS: dict = _USER["職業"]["年齢層ごと"]
JOBS_DEFAULT: list = _USER["職業"]["その他の年齢層"]
KNOWLEDGE: list = _USER["前提知識"]

# ユーザの目的（user goal）：会話全体の目的 goal（0 に見せる）と終了条件 end_condition（01 に見せる）と出現の重み weight
USER_GOALS: dict = load_constant("user_goals.json")


def sample_persona(rng: random.Random) -> dict:
    age = rng.choice(AGES)
    return {"年齢層": age, "性別": rng.choice(GENDERS),
            "職業": rng.choice(JOBS.get(age, JOBS_DEFAULT)), "前提知識": rng.choice(KNOWLEDGE)}


def sample_user_goal(rng: random.Random) -> str:
    names = list(USER_GOALS)
    return rng.choices(names, weights=[USER_GOALS[n]["weight"] for n in names])[0]


def render_persona(p: dict) -> str:
    return " / ".join(f"{k}: {v}" for k, v in p.items())


def render_scenario(s: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in s.items())


# ======================================================================================
# 発話の書き方（音声で読み上げる文の決まり）：constants/spoken_style.json
# ======================================================================================
_STYLE = load_constant("spoken_style.json")
SPOKEN_STYLE: str = "\n".join([_STYLE["見出し"]] + [f"- {x}" for x in _STYLE["決まり"]])

# ======================================================================================
# フィラー（02-1 の lead に一覧で見せる）：constants/fillers.json
# ======================================================================================
# 語 → 働き。「フィラー」とだけ書くと LLM が「えーと」「うーん」ばかり使うので、列挙して選ばせる
FILLERS: dict = load_constant("fillers.json")


def render_fillers() -> str:
    return "\n".join(f"  - {k}：{v}" for k, v in FILLERS.items())
