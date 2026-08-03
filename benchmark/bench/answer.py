# -*- coding: utf-8 -*-
"""에이전트가 낸 답에서 값을 꺼낸다.

[왜 따로 있나]
answer 는 모델이 만든 JSON 이라 모양이 한 가지가 아니다. 물질 순위를 물으면
    {"top3": [{"spectrum_id": "PS_01", "material": "polystyrene", "score": 0.99}, ...]}
    {"ranking": ["PS_01", "PS_02", "PET_02"]}
    {"materials": ["polystyrene", ...], "scores": [...]}
가 전부 온다. 이걸 문항 파일마다 풀어 쓰면 같은 20 줄이 아홉 번 복사되고, 한 곳만
고치면 문항끼리 채점 기준이 달라진다. 모양 흡수는 여기 한 곳에서만 한다.

[관대함의 한계]
모양은 관대하게 받되 **값은 관대하게 받지 않는다**. 키를 못 찾으면 None 을 돌려주고,
판정은 chk 쪽에서 '보고 없음'으로 떨어진다. 여기서 추측해 채워 넣으면 답을 안 낸
실행이 답을 낸 것처럼 채점된다.
"""
from __future__ import annotations


def _first(d: dict, keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


def seq(run, *keys, field=None, cast=None):
    """answer[keys 중 하나] 를 목록으로. 원소가 dict 면 field 를 뽑는다.

        seq(run, "top3", "ranking", field="material")   → ["polystyrene", ...]
        seq(run, "explained_variance_ratio", cast=float) → [0.98, 0.001, 0.001]
    """
    v = _first(run.answer, keys)
    if v is None:
        return None
    if isinstance(v, dict):                       # {"1": "PS_01", "2": ...} 형태도 받는다
        v = [v[k] for k in sorted(v)]
    if not isinstance(v, list):
        v = [v]
    out = []
    for item in v:
        if isinstance(item, dict):
            item = _first(item, [field]) if field else next(iter(item.values()), None)
        if item is None:
            continue
        if cast is not None:
            try:
                item = cast(item)
            except (TypeError, ValueError):
                continue
        out.append(item)
    return out or None


def value(run, *keys, cast=None):
    """스칼라 하나. 없으면 None."""
    v = _first(run.answer, keys)
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    if v is None or cast is None:
        return v
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def flag(run, *keys):
    """참/거짓 답. 'yes'/'true'/'match' 같은 문자열도 받는다. 모르면 None."""
    v = _first(run.answer, keys)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1", "match", "matches", "accept", "above", "pass"):
            return True
        if s in ("false", "no", "n", "0", "mismatch", "reject", "below", "fail"):
            return False
    return None


def grid(run, *keys):
    """2차원 표. 평평한 목록이 오면 그대로 돌려준다(모양 판정은 문항이 한다)."""
    v = _first(run.answer, keys)
    if v is None:
        return None
    if isinstance(v, list) and v and isinstance(v[0], list):
        return [[_f(c) for c in row] for row in v]
    if isinstance(v, list):
        return [_f(c) for c in v]
    return None


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
