# -*- coding: utf-8 -*-
"""도구 인자 계약 — **함수 시그니처가 유일한 정본이다.**

인자의 이름·타입·범위·설명을 함수 시그니처에 직접 적고, OpenAI function 스키마는 여기서
그것을 읽어 만든다:

    def set_ccd_exposure(
        exposure_time: Annotated[float, Field(description="Exposure time [seconds].")],
    ) -> dict:

    RAMAN_TOOLS = [tool_schema(set_ccd_exposure, "When to use this tool...")]

필수/선택은 파이썬 기본값이 정한다 — 기본값이 없으면 required.

[선언을 하나로 두는 이유]
스키마와 시그니처를 따로 적으면 둘을 맞춰 줄 장치가 필요해지는데, **검사가 필요하다는 것
자체가 정본이 하나가 아니라는 뜻이다.** 인자를 Pydantic DTO 로 빼도 마찬가지다 — DTO 는
함수에서 파생되지 않으므로 'DTO ↔ 시그니처'라는 새 중복이 생긴다. 그래서 파생 방향을
한쪽으로 고정했다: 함수가 정본, 스키마는 결과물.

[모델에게 숨길 인자]
``Field(json_schema_extra=INTERNAL)`` 을 주면 스키마에서 빠진다. 내부 호출자만 쓰는 인자를
'실수로 빠뜨린 것'과 구분해 **명시적으로** 숨기는 방법이다.
(``Field(exclude=True)`` 는 model_dump 에만 걸리고 model_json_schema 에는 안 먹는다.)

[변환 규칙]
Pydantic 의 ``model_json_schema()`` 는 OpenAI function 포맷과 세 군데가 다르다.

    title           모든 필드에 붙는다 → 버린다(모델에게 의미 없는 잡음)
    anyOf[X, null]  Optional[X] 가 이렇게 나온다 → X 로 평탄화한다
    default         선언한 기본값이 실린다 → 버린다. 이 툴셋은 '인자를 생략하면 현재 장비
                    설정 유지'가 규약이라, 기본값을 실으면 모델이 그 값을 명시적으로 보낸다.

남기는 것: type · description · enum · items · minimum · maximum.

    python backend/tools/schema.py      자체 검사
"""
from __future__ import annotations

import inspect
from typing import Any, get_type_hints

from pydantic import create_model

# OpenAI function 스키마가 property 하나에 허용하는 키. 이 밖은 전부 버린다.
_KEEP = ("type", "description", "enum", "items", "minimum", "maximum")

#: 모델에게 보이지 않을 인자에 붙이는 표식.
#:     restore_guide_beam: Annotated[bool, Field(json_schema_extra=INTERNAL, ...)] = True
#: 내부 호출자(예: run_grid_scan)만 쓰는 인자를 '빠뜨린 것'과 구분해 준다.
INTERNAL = {"x-internal": True}


def _flatten(prop: dict) -> dict:
    """property 하나를 OpenAI 포맷으로 다듬는다 — Optional 의 anyOf 를 풀고 잡음을 턴다.

    Parameters
    ----------
    prop : dict
        Pydantic 이 만든 property 한 개.

    Returns
    -------
    dict
        _KEEP 에 있는 키만 남긴 property.
    """
    if "anyOf" in prop:
        # Optional[X] → anyOf: [X, {"type": "null"}]. null 이 아닌 갈래만 쓴다.
        real = next((b for b in prop["anyOf"] if b.get("type") != "null"), {})
        # 바깥에 있던 description 등은 살리고 anyOf 자체만 버린다.
        prop = {**real, **{k: v for k, v in prop.items() if k != "anyOf"}}
    return {k: prop[k] for k in _KEEP if k in prop}


def args_model(fn) -> type:
    """함수 시그니처를 Pydantic 모델로 바꾼다 — 스키마 생성과 인자 검증이 같은 것을 본다.

    Parameters
    ----------
    fn : callable
        도구 함수.

    Returns
    -------
    type
        인자마다 필드를 가진 Pydantic 모델 클래스.

    Notes
    -----
    어노테이션을 ``signature`` 가 아니라 ``get_type_hints`` 로 읽는다. 대상 모듈이
    ``from __future__ import annotations`` 를 쓰면 signature 는 어노테이션을 **문자열로**
    돌려주고, 그대로 create_model 에 넘기면 Annotated·Optional 을 못 찾아 "is not fully
    defined" 로 죽는다. get_type_hints 는 그 모듈의 네임스페이스에서 실제 타입으로 풀어
    준다(``include_extras=True`` 라야 Field(...) 가 살아남는다).
    """
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue                                   # *args/**kwargs 는 스키마로 못 옮긴다
        # 기본값이 없으면 ... (필수), 있으면 그 값 — required 목록이 여기서 결정된다.
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[pname] = (hints.get(pname, p.annotation), default)
    return create_model(f"{fn.__name__}_args", __module__=fn.__module__, **fields)


def tool_schema(fn, description: str) -> dict:
    """(도구 함수, 설명) 을 OpenAI function 스키마 한 개로 만든다.

    Parameters
    ----------
    fn : callable
        도구 함수. 스키마의 ``name`` 도 여기서 나온다.
    description : str
        '이 도구를 언제 부를지'. 모델의 판단 근거라 프롬프트의 일부이며, 시그니처에서
        뽑을 수 없는 유일한 것이라 따로 받는다 — 길게 쓰라고 있는 자리다.

    Returns
    -------
    dict
        ``{"type": "function", "function": {"name", "description", "parameters"}}``
    """
    raw = args_model(fn).model_json_schema()
    # INTERNAL 표식이 붙은 인자는 모델에게 보여주지 않는다.
    props = {n: _flatten(p) for n, p in raw.get("properties", {}).items()
             if not p.get("x-internal")}
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": list(raw.get("required", [])),
            },
        },
    }


def parse_args(fn, raw: dict) -> dict:
    """LLM 이 보낸 인자 dict 를 검증해 fn 에 넘길 kwargs 로 만든다.

    **도구를 검증과 함께 부르는 유일한 방법이다. 직접 model_dump() 하지 말 것.**

    Parameters
    ----------
    fn : callable
        도구 함수.
    raw : dict
        모델이 보낸 인자.

    Returns
    -------
    dict
        모델이 **실제로 보낸 인자만** 담긴 kwargs.

    Raises
    ------
    pydantic.ValidationError
        범위를 벗어난 값, 없는 enum, 타입 불일치. 호출부가 잡아서
        ``{"ok": False, "error": ...}`` 로 바꿔 모델에게 돌려준다.

    Notes
    -----
    이 툴셋의 규약은 "인자를 생략하면 현재 장비 설정을 유지한다"이다. 그래서 모델이 보내지
    않은 인자는 함수에 **아예 넘기지 않아야** 한다. 그냥 model_dump() 를 하면 선언된
    기본값이 전부 채워져 나가고, '생략'이 '명시적 None'이 되어 규약이 조용히 깨진다 —
    노출만 바꾸려던 호출이 파워까지 리셋하는 식이다.
    """
    return args_model(fn)(**raw).model_dump(exclude_unset=True)


def call_with(fn, raw: dict, **fixed) -> dict:
    """LLM 이 보낸 인자 dict 로 도구 함수를 부른다 — **선언된 인자만, 없으면 None 으로.**

    Parameters
    ----------
    fn : callable
        도구 함수.
    raw : dict
        모델이 보낸 인자. 선언에 없는 키는 버린다.
    **fixed
        모델이 주지 않는 호출자 쪽 인자(장기기억 도구의 ctx). 스키마에서는 이미
        ``Field(json_schema_extra=INTERNAL)`` 로 숨겨져 있고, 값은 여기서 채운다.

    Returns
    -------
    dict
        도구 결과.

    Notes
    -----
    ``fn(**raw)`` 를 직접 쓰지 않는 이유가 둘 있다.

    1. 모르는 키가 오면 ``fn(**raw)`` 는 TypeError 로 죽는다. 하드웨어 도구는 호출부
       (runtime._dispatch)가 try 로 감싸 에러 dict 로 바꿔 주지만, 파일·KB·장기기억 도구는
       그 감싸기 밖에 있어 예외가 그대로 위로 샌다.
    2. 필수 인자를 모델이 빠뜨려도 각 함수 머리의 ``str(x or "").strip()`` 같은 정규화가
       그대로 동작해야 한다. 선언된 인자를 전부 넘기되 없는 것을 None 으로 주면 그렇게 된다.

    즉 스키마는 필수를 선언해 모델에게 알리고, 실행은 관용을 유지한다.
    """
    names = [p for p in inspect.signature(fn).parameters if p not in fixed]
    return fn(**fixed, **{k: raw.get(k) for k in names})


def schema_of(tools: list[dict], name: str) -> dict[str, Any]:
    """스키마 목록에서 이름으로 하나 꺼낸다(디버깅용).

    Raises
    ------
    KeyError
        그 이름의 도구가 목록에 없을 때.
    """
    for t in tools:
        if t["function"]["name"] == name:
            return t
    raise KeyError(name)


# ──────────────────────────────────────────────────────────────────────────────
# 자체 점검:  python backend/tools/schema.py
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from typing import Annotated, Literal, Optional

    from pydantic import Field

    def probe(
        x: Annotated[float, Field(ge=0, le=75.7, description="Required float.")],
        n: Annotated[Optional[int], Field(ge=1, description="Optional integer.")] = None,
        mode: Annotated[Optional[Literal["single", "kinetic"]],
                        Field(description="Optional enum.")] = None,
        names: Annotated[Optional[list[str]], Field(description="Array of strings.")] = None,
        flag: Annotated[bool, Field(description="Boolean.")] = True,
        hidden: Annotated[bool, Field(json_schema_extra=INTERNAL,
                                      description="Hidden from the model.")] = True,
    ) -> dict:
        return {}

    s = tool_schema(probe, "Probe tool.")["function"]
    p = s["parameters"]["properties"]

    assert s["name"] == "probe", s["name"]            # 이름도 시그니처에서 온다
    assert s["parameters"]["required"] == ["x"], s["parameters"]["required"]
    assert p["x"] == {"type": "number", "description": "Required float.",
                      "minimum": 0, "maximum": 75.7}, p["x"]
    assert p["n"] == {"type": "integer", "description": "Optional integer.",
                      "minimum": 1}, p["n"]           # Optional 의 anyOf 가 풀렸는가
    assert p["mode"] == {"type": "string", "description": "Optional enum.",
                         "enum": ["single", "kinetic"]}, p["mode"]
    assert p["names"] == {"type": "array", "description": "Array of strings.",
                          "items": {"type": "string"}}, p["names"]
    assert p["flag"] == {"type": "boolean", "description": "Boolean."}, p["flag"]
    assert "hidden" not in p, "INTERNAL 인데 스키마에 남았다"
    assert not any("title" in v or "default" in v for v in p.values()), "잡음이 남았다"

    # parse_args: 보낸 것만 넘어가고 안 보낸 것은 빠진다 (= '생략 = 현재 설정 유지')
    assert parse_args(probe, {"x": 1.0}) == {"x": 1.0}
    assert parse_args(probe, {"x": 1.0, "n": 3}) == {"x": 1.0, "n": 3}
    for bad in ({"x": 999.0}, {"x": 1.0, "mode": "no_such_mode"}, {}):
        try:
            parse_args(probe, bad)
        except Exception:
            continue
        raise AssertionError(f"거부했어야 하는 인자가 통과했다: {bad}")

    # call_with: 모르는 키는 버리고, 빠뜨린 필수 인자는 None 으로 들어간다.
    def probe2(a, b=None, *, ctx=None):
        return {"a": a, "b": b, "ctx": ctx}

    assert call_with(probe2, {"a": 1, "b": 2}) == {"a": 1, "b": 2, "ctx": None}
    assert call_with(probe2, {"a": 1, "zzz": 9}) == {"a": 1, "b": None, "ctx": None}
    assert call_with(probe2, {}) == {"a": None, "b": None, "ctx": None}, "필수 누락에 죽으면 안 된다"
    assert call_with(probe2, {"a": 1}, ctx={"s": 1}) == {"a": 1, "b": None, "ctx": {"s": 1}}

    print(f"통과: 시그니처에서 스키마 생성 (property {len(p)}개) "
          f"· INTERNAL 숨김 · parse_args 생략 보존 + 범위/enum/필수 거부 "
          f"· call_with 미지 키 무시/누락 관용/ctx 주입")
