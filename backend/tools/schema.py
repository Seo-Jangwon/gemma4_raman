# -*- coding: utf-8 -*-
"""OpenAI function 스키마 — **함수 시그니처가 유일한 정본이다.**

[왜 이렇게 됐나 — 2026-08-12]
처음에는 손으로 쓴 JSON 1,460줄이었다. 같은 정보가 함수 시그니처에도 있는데 둘을 맞춰
주는 장치가 없었고, 실제로 5건이 어긋나 있었다 — 함수엔 있는데 스키마엔 없어서 모델이
영원히 못 쓰는 인자들이었다. 에러가 안 나므로 아무도 몰랐다.

그래서 인자를 Pydantic DTO 로 옮겼는데 그것도 답이 아니었다. DTO 는 함수에서 파생되지
않으므로 'DTO ↔ 시그니처' 라는 새 중복이 생겼고, 그걸 지키려면 또 검사가 필요했다.
**검사가 필요하다는 것 자체가 정본이 하나가 아니라는 증거다.**

지금은 선언이 하나다. 인자의 타입·범위·설명을 함수 시그니처에 직접 적고, 스키마는
여기서 그것을 읽어 만든다:

    def set_ccd_exposure(
        exposure_time: Annotated[float, Field(description="Exposure time [seconds].")],
    ) -> dict:

    RAMAN_TOOLS = [tool_schema(set_ccd_exposure, "When to use this tool...")]

필수/선택은 파이썬 기본값이 결정한다 — 기본값이 없으면 required.

[모델에게 숨길 인자]
Field(json_schema_extra=INTERNAL) 을 주면 스키마에서 빠진다. 내부 호출자만 쓰는 인자를
'실수로 빠진 것'과 구분해 **명시적으로** 숨기는 방법이다.
(Field(exclude=True) 는 model_dump 에만 걸리고 model_json_schema 에는 안 먹는다.)

[변환 규칙]
Pydantic 의 model_json_schema() 는 OpenAI function 포맷과 세 군데가 다르다:
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
    """property 하나를 OpenAI 포맷으로. Optional 의 anyOf 를 풀고 잡음을 턴다."""
    if "anyOf" in prop:
        # Optional[X] → anyOf: [X, {"type": "null"}]. null 이 아닌 갈래만 쓴다.
        real = next((b for b in prop["anyOf"] if b.get("type") != "null"), {})
        prop = {**real, **{k: v for k, v in prop.items() if k != "anyOf"}}
    return {k: prop[k] for k in _KEEP if k in prop}


def args_model(fn) -> type:
    """함수 시그니처 → Pydantic 모델. 스키마 생성과 인자 검증이 같은 것을 본다.

    어노테이션을 signature 가 아니라 get_type_hints 로 읽는 이유: 대상 모듈이
    `from __future__ import annotations` 를 쓰면 signature 는 어노테이션을 **문자열로**
    돌려준다. 그 상태로 create_model 에 넘기면 Annotated·Optional 을 못 찾아
    "is not fully defined" 로 죽는다. get_type_hints 는 그 모듈의 네임스페이스에서
    실제 타입으로 풀어 준다(include_extras=True 라야 Field(...) 가 살아남는다).
    """
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[pname] = (hints.get(pname, p.annotation), default)
    return create_model(f"{fn.__name__}_args", __module__=fn.__module__, **fields)


def tool_schema(fn, description: str) -> dict:
    """(도구 함수, 설명) → OpenAI function 스키마 한 개.

    description 은 모델이 '이 도구를 언제 부를지' 판단하는 근거라 프롬프트의 일부다.
    시그니처에서 못 뽑는 유일한 것이라 여기서 따로 받는다 — 길게 쓰라고 있는 자리다.
    """
    raw = args_model(fn).model_json_schema()
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
    """LLM 이 보낸 인자 dict 를 검증해서 fn 에 넘길 kwargs 로 만든다.

    **도구를 검증과 함께 부르는 유일한 방법이다. 직접 model_dump() 하지 말 것.**

    이 툴셋의 규약은 "인자를 생략하면 현재 장비 설정을 유지한다"이다. 그래서 모델이
    보내지 않은 인자는 함수에 **아예 넘기지 않아야** 한다. 그냥 model_dump() 를 하면
    선언된 기본값이 전부 채워져 나가고, 그러면 '생략'이 '명시적 None'이 되어 규약이
    조용히 깨진다 — 예를 들어 노출만 바꾸려던 호출이 파워까지 리셋한다.

    Raises
    ------
    pydantic.ValidationError — 범위를 벗어난 값, 없는 enum, 타입 불일치.
        호출부가 잡아서 {"ok": False, "error": ...} 로 바꿔 모델에게 돌려준다.
    """
    return args_model(fn)(**raw).model_dump(exclude_unset=True)


def schema_of(tools: list[dict], name: str) -> dict[str, Any]:
    """스키마 목록에서 이름으로 하나 꺼낸다(디버깅용)."""
    for t in tools:
        if t["function"]["name"] == name:
            return t
    raise KeyError(name)


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
                      "minimum": 1}, p["n"]
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

    print(f"통과: 시그니처에서 스키마 생성 (property {len(p)}개) "
          f"· INTERNAL 숨김 · parse_args 생략 보존 + 범위/enum/필수 거부")
