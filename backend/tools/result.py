# -*- coding: utf-8 -*-
"""도구 응답 형식 — **모든 도구는 같은 모양으로 답한다.**

    {"ok": True,  **payload}                  성공. payload 는 도구마다 다르다.
    {"ok": False, "error": "...", **extra}    실패. error 는 모델이 읽고 다음 수를 정한다.

schema.py 가 '도구에 무엇이 들어가는가'를 정하고, 이 파일이 '도구에서 무엇이 나오는가'를
정한다. 둘 다 아무것도 import 하지 않으므로 어느 계층에서 불러도 순환이 생기지 않는다.

[도구마다 결과 클래스를 두지 않는 이유]
인자는 선언을 한곳에 모을 이유가 분명하다 — 모델에게 보낼 JSON 스키마를 거기서 만든다.
결과는 다르다. 성공 페이로드는 도구 45개가 35가지 모양이고, 그것을 읽는 소비자는 **LLM
하나뿐**이다. 결과는 slim() 을 지나 JSON 문자열이 되어 tool 메시지에 실리므로 타입을
검사할 코드가 애초에 없다. 도구마다 결과 모델을 두면 아무도 읽지 않는 선언만 45개 늘어난다.
45개가 실제로 공유하는 것은 응답 형식뿐이라, 공통 형식만 여기 둔다.

[이 형식이 보장하는 것]
· 실패에는 error 가 반드시 있다. 없으면 화면에 "⚠️ move_stage failed: " 가 이유 없이 뜨고
  (runtime.describe_tool), 모델은 무엇을 고칠지 모르는 채 같은 호출을 되풀이한다.
  fail() 은 error 를 위치인자로 강제해 그 상태를 만들 수 없게 한다.
· ok 는 반드시 bool 이다. 키가 빠지거나 정수 1 이면 두 곳이 **예외 없이** 오작동한다 —
  describe_tool 은 실패를 성공으로 보고하고, call_tool 은 레이저가 나간 호출을 조사량
  누계에서 빠뜨린다. 둘 다 조용해서, 모든 결과가 지나는 검사 지점(call_tool)에서 normalize()
  가 검사한다.

    python backend/tools/result.py      자체 검사
"""
from __future__ import annotations

from typing import Any

#: repr 을 잘라 넣는 길이. 잘린 원본이라도 있어야 어느 도구가 무엇을 뱉었는지 추적된다.
_REPR_LIMIT = 200


def ok(**payload: Any) -> dict:
    """성공 결과를 만든다. payload 는 그대로 모델에게 간다(slim() 이 긴 배열만 걷어낸다).

    Parameters
    ----------
    **payload
        도구별 결과 필드.

    Returns
    -------
    dict
        ``{"ok": True, **payload}``

    Notes
    -----
    두 경우는 이 함수를 쓰지 못하고 dict 리터럴을 그대로 만든다 — normalize() 가 어느
    쪽이든 똑같이 검사하므로 문제되지 않는다.

    · 키가 파이썬 식별자가 아닐 때. ``"raman_shift_cm-1"`` 은 kwargs 로 넘길 수 없다.
    · ok 가 조건식일 때. 부분 성공이 있는 도구들이다(reconnect_hardware 는 일부 장비만
      붙을 수 있고, run_grid_scan 은 중단되면 측정된 점이 있어도 성공이 아니다).
    """
    return {"ok": True, **payload}


def fail(error: Any, **extra: Any) -> dict:
    """실패 결과를 만든다.

    Parameters
    ----------
    error : Any
        **모델이 읽는 문장.** 무엇이 왜 막혔고 다음에 무엇을 하면 되는지까지 적는다.
        예외 객체를 그대로 넘겨도 되도록 문자열로 변환한다.
    **extra
        복구에 쓸 부가 정보(hint, busy_with 등).

    Returns
    -------
    dict
        ``{"ok": False, "error": "...", **extra}``

    Notes
    -----
    ``None`` 은 ``"None"`` 이라는 네 글자짜리 가짜 사유가 되지 않도록 빈 문자열로 둔다.
    아래 계층의 결과에서 사유를 꺼내 올리는 자리(``res.get("error")``)가 None 을 줄 수
    있는데, 그대로 str() 에 넣으면 normalize() 의 검사를 통과해 버린다.
    """
    return {"ok": False, "error": "" if error is None else str(error), **extra}


def is_ok(result: Any) -> bool:
    """성공인가. dict 가 아니거나 ok 가 정확히 ``True`` 가 아니면 전부 False."""
    return isinstance(result, dict) and result.get("ok") is True


def normalize(result: Any, tool: str) -> dict:
    """도구가 돌려준 것을 응답 형식 규약에 맞춘다 — 어기면 **모델에게 보이는 실패**로 바꾼다.

    Parameters
    ----------
    result : Any
        도구가 실제로 돌려준 값. dict 가 아닐 수도 있다.
    tool : str
        도구 이름. 규약 위반 메시지에 실려 어느 도구가 어겼는지 드러낸다.

    Returns
    -------
    dict
        규약을 지킨 결과는 **손대지 않고 그대로** 돌려준다(같은 객체).

    Notes
    -----
    call_tool 이 유일한 호출자다 — 모든 도구 결과가 그 한 지점을 지나므로, 규약 위반이
    조용히 성공으로 흘러가는 대신 반드시 여기서 드러난다.

    사유 없는 실패는 **덧붙이고, 갈아치우지 않는다.** 앞의 두 경우는 돌려줄 페이로드가
    없지만(dict 도 아니거나 성공/실패조차 모른다), 세 번째는 다르다. 도구가 '실패했다'
    까지는 정확히 말했고 빠뜨린 것은 사유 한 줄뿐인데, 그 dict 안에 대개 가장 값진 진단이
    들어 있다. 응답을 새로 만들어 갈아치우면 어느 장비가 왜 죽었는지가 통째로 사라진다.
    """
    if not isinstance(result, dict):
        # dict 조차 아니다 — 원본을 잘라서라도 실어야 어느 도구가 무엇을 뱉었는지 추적된다.
        return fail(f"{tool} returned {type(result).__name__}, expected a dict: "
                    f"{repr(result)[:_REPR_LIMIT]}")

    # ok 가 bool 이 아니면 성공/실패를 판정할 수 없다(정수 1 도 여기서 걸린다).
    if not isinstance(result.get("ok"), bool):
        return fail(f"{tool} returned a malformed result (no boolean 'ok' field): "
                    f"{repr(result)[:_REPR_LIMIT]}")

    # 실패인데 사유가 비었을 때. fail() 은 이미 None 을 막고 있지만, dict 리터럴로 답하는
    # 도구는 그 문을 지나지 않는다.
    err = result.get("error")
    if not result["ok"] and (err is None or not str(err).strip()):
        # 원본 필드를 한 개도 잃지 않도록 새로 만들지 않고 펼쳐 담는다.
        return {**result,
                "error": f"{tool} reported failure without an error message. "
                         f"Look at the other fields of this result for the reason."}
    return result


__all__ = ["ok", "fail", "is_ok", "normalize"]


# ──────────────────────────────────────────────────────────────────────────────
# 자체 점검:  python backend/tools/result.py
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 응답 형식
    assert ok() == {"ok": True}
    assert ok(position={"x": 1}) == {"ok": True, "position": {"x": 1}}
    assert fail("nope") == {"ok": False, "error": "nope"}
    assert fail("nope", hint="try again") == {"ok": False, "error": "nope", "hint": "try again"}
    # error 는 모델이 읽는 문장이라 예외 객체를 넘겨도 문자열이어야 한다.
    assert fail(ValueError("bad")) == {"ok": False, "error": "bad"}

    # is_ok 는 ok=True 만 성공으로 본다.
    assert is_ok(ok(a=1)) and not is_ok(fail("x"))
    assert not is_ok({"a": 1}) and not is_ok(None) and not is_ok("ok")
    assert not is_ok({"ok": 1})                      # 정수 1 은 True 가 아니다

    # 규약을 지킨 결과는 손대지 않는다(같은 객체가 그대로 나온다).
    good = ok(count=3)
    assert normalize(good, "t") is good
    bad = fail("boom")
    assert normalize(bad, "t") is bad

    # 규약 위반 7종 — 전부 모델이 읽을 수 있는 실패가 된다.
    for broken in (None, "done", 42, ["a"], {"count": 3}, {"ok": 1}, {"ok": False}):
        r = normalize(broken, "move_stage")
        assert r["ok"] is False and "move_stage" in r["error"], (broken, r)

    # ok 키가 빠진 dict 가 성공으로 새면 조사량 누계에서 빠진다(레이저는 나갔는데).
    assert not is_ok(normalize({"position": {"x": 1}}, "acquire_spectrum"))

    # 사유 없는 실패: 사유를 덧붙이되 나머지는 한 필드도 잃지 않는다.
    partial = {"ok": False, "reconnected": ["stage", "camera"],
               "errors": {"ccd": "re-initialization failed"},
               "now_connected": {"ccd": False}}
    r = normalize(partial, "reconnect_hardware")
    assert r["reconnected"] == ["stage", "camera"], r
    assert r["errors"] == {"ccd": "re-initialization failed"}, r
    assert r["now_connected"] == {"ccd": False}, r
    assert "reconnect_hardware" in r["error"], r
    assert partial is not r and "error" not in partial, "원본을 건드리면 안 된다"

    # error 가 None/공백이어도 같은 처리 — 가짜 사유("None")를 만들지 않는다.
    for blank in (None, "", "   "):
        r = normalize({"ok": False, "error": blank, "kept": 1}, "t")
        assert r["kept"] == 1 and r["error"].startswith("t reported failure"), (blank, r)

    print("통과: ok/fail 응답 형식 · is_ok 엄격 판정 · normalize 규약 위반 7종 차단 · 페이로드 보존")
