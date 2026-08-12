# -*- coding: utf-8 -*-
"""도구 결과 — **모든 도구는 같은 모양으로 답한다.**

    {"ok": True,  **payload}                  성공. payload 는 도구마다 다르다.
    {"ok": False, "error": "...", **extra}    실패. error 는 모델이 읽고 다음 수를 정한다.

schema.py 가 '도구에 무엇이 들어가는가'를 정하고, 이 파일이 '도구에서 무엇이 나오는가'를
정한다. 둘 다 아무것도 import 하지 않으므로 어느 계층에서 불러도 안전하다.

[왜 도구마다 결과 클래스를 만들지 않는가]
인자(schema.py)는 선언을 하나로 모을 이유가 분명했다 — 모델에게 보낼 JSON 스키마를
거기서 만들어야 했다. 결과는 반대다. 성공 페이로드는 45개 도구가 35가지 모양이고,
그걸 읽는 소비자는 **LLM 하나뿐**이다: 결과는 slim() 을 지나 JSON 문자열이 되어 tool
메시지에 실린다. 타입을 검사할 코드가 애초에 없다. 도구마다 ResultModel 을 두면 아무도
읽지 않는 선언이 45개 생기고, 그게 정확히 1단계에서 걷어낸 그 중복이다.
도구 45개가 실제로 공유하는 것은 봉투뿐이라 봉투만 여기 둔다.

[봉투가 지키는 것 — 실패에는 error 가 반드시 있다]
없으면 화면에 "⚠️ move_stage failed: " 가 이유 없이 뜨고(runtime.describe_tool),
모델은 무엇을 고쳐야 할지 모르는 채로 같은 호출을 되풀이한다. fail() 은 error 를
위치인자로 강제해 그 상태를 만들 수 없게 한다.

[normalize() 가 막는 것 — 실제로 열려 있던 구멍]
ok 키가 빠진 dict 를 돌려주면 두 곳이 **예외 없이** 오작동한다:

    runtime.describe_tool   result.get("ok", True)   실패가 성공으로 사용자에게 보고된다
    runtime.call_tool       result.get("ok")         레이저는 나갔는데 조사량 누계에 안 잡힌다

둘 다 조용하다. 그래서 모든 도구 결과가 지나는 단 하나의 관문(call_tool)에서 검사한다.

    python backend/tools/result.py      자체 검사
"""
from __future__ import annotations

from typing import Any

#: repr 을 잘라 넣는 길이. 잘린 원본이라도 있어야 어느 도구가 무엇을 뱉었는지 추적된다.
_REPR_LIMIT = 200


def ok(**payload: Any) -> dict:
    """성공 결과. payload 가 그대로 모델에게 간다(slim() 이 원본 배열만 걷어낸다).

    다음 둘은 dict 리터럴을 그대로 쓴다. normalize() 가 어느 쪽이든 똑같이 검사한다:
      · 키가 식별자가 아닐 때 — "raman_shift_cm-1" 은 kwargs 로 못 쓴다
      · ok 가 조건식일 때 — 부분 성공(reconnect_hardware 는 일부 컴포넌트만 붙을 수 있고,
        run_grid_scan 은 중단되면 측정된 점이 있어도 성공이 아니다)
    """
    return {"ok": True, **payload}


def fail(error: Any, **extra: Any) -> dict:
    """실패 결과. error 는 **모델이 읽는 문장**이다 — 무엇이 왜 막혔고 다음에 무엇을
    하면 되는지까지 적는다. extra 는 모델이 복구에 쓸 부가 정보(hint, busy_with 등).

    예외 객체를 그대로 넘겨도 되게 문자열로 만든다(fail(e) 가 32곳이다).
    None 은 "None" 이라는 가짜 사유가 아니라 빈 문자열로 두어 normalize() 가 잡게 한다 —
    아래 계층 결과에서 error 를 꺼내 올리는 자리(res.get("error"))가 None 을 줄 수 있다.
    """
    return {"ok": False, "error": "" if error is None else str(error), **extra}


def is_ok(result: Any) -> bool:
    """성공인가. dict 가 아니거나 ok 가 True 가 아니면 전부 False."""
    return isinstance(result, dict) and result.get("ok") is True


def normalize(result: Any, tool: str) -> dict:
    """도구가 돌려준 것을 봉투 규약에 맞춘다 — 어기면 **모델에게 보이는 실패**로 바꾼다.

    call_tool 이 유일한 호출자다(모든 도구 결과가 지나는 지점). 위 머리말의 두 구멍을
    여기 한 곳에서 닫는다: 규약 위반이 조용히 성공으로 흘러가는 대신 바로 드러난다.

    [사유 없는 실패는 **덧붙이고**, 갈아치우지 않는다 — 2026-08-12]
    앞의 두 경우는 돌려줄 페이로드가 없다(dict 도 아니거나, 성공/실패조차 모른다).
    세 번째는 다르다. 도구가 '실패했다'까지는 정확히 말했고 빠뜨린 건 사유 한 줄뿐인데,
    그 dict 안에 대개 가장 값진 것이 들어 있다.

    실제 사고: reconnect_hardware 가 사유를 error(단수)가 아니라 errors(복수)에 담았다.
    옛 코드는 fail() 로 **새 봉투를 만들어** 돌려줬고, 그래서 어느 장비가 왜 죽었는지,
    나머지 셋은 붙었는지가 통째로 사라졌다. 모델은 그 사실을 알아채고("This doesn't
    explicitly say...") 추측으로 답을 지었고, DetailLog 에도 155초짜리 작업의 기록이
    이 한 줄만 남았다. 관문이 증거를 없애면 관문이 아니라 손실이다.
    """
    if not isinstance(result, dict):
        return fail(f"{tool} returned {type(result).__name__}, expected a dict: "
                    f"{repr(result)[:_REPR_LIMIT]}")
    if not isinstance(result.get("ok"), bool):
        return fail(f"{tool} returned a malformed result (no boolean 'ok' field): "
                    f"{repr(result)[:_REPR_LIMIT]}")
    # None 을 str() 에 넣으면 "None" 이라는 **네 글자짜리 가짜 사유**가 되어 이 검사를
    # 통과한다. 그러면 모델은 error: null 을 받고 아무것도 못 한다. fail() 은 같은 이유로
    # 이미 None 을 막고 있는데(위), 리터럴 dict 로 답하는 도구는 그 문을 지나지 않는다.
    err = result.get("error")
    if not result["ok"] and (err is None or not str(err).strip()):
        return {**result,
                "error": f"{tool} reported failure without an error message. "
                         f"Look at the other fields of this result for the reason."}
    return result


__all__ = ["ok", "fail", "is_ok", "normalize"]


if __name__ == "__main__":
    assert ok() == {"ok": True}
    assert ok(position={"x": 1}) == {"ok": True, "position": {"x": 1}}
    assert fail("nope") == {"ok": False, "error": "nope"}
    assert fail("nope", hint="try again") == {"ok": False, "error": "nope", "hint": "try again"}
    # error 는 모델이 읽는 문장이라 항상 문자열이어야 한다 — 예외 객체를 그대로 넘겨도.
    assert fail(ValueError("bad")) == {"ok": False, "error": "bad"}

    assert is_ok(ok(a=1)) and not is_ok(fail("x"))
    assert not is_ok({"a": 1}) and not is_ok(None) and not is_ok("ok")
    # ok=1 은 True 가 아니다 — 규약은 bool 이다.
    assert not is_ok({"ok": 1})

    # normalize: 규약을 지킨 결과는 손대지 않는다(같은 객체 그대로).
    good = ok(count=3)
    assert normalize(good, "t") is good
    bad = fail("boom")
    assert normalize(bad, "t") is bad

    # 규약 위반 7종 — 전부 모델이 읽을 수 있는 실패가 된다.
    for broken in (None, "done", 42, ["a"], {"count": 3}, {"ok": 1}, {"ok": False}):
        r = normalize(broken, "move_stage")
        assert r["ok"] is False and "move_stage" in r["error"], (broken, r)

    # ok 키가 빠진 dict 가 성공으로 새어 나가면 조사량 누계에서 빠진다(레이저는 나갔는데).
    assert not is_ok(normalize({"position": {"x": 1}}, "acquire_spectrum"))

    # 사유 없는 실패: 사유를 '덧붙이되' 나머지는 한 필드도 잃지 않는다.
    # (reconnect_hardware 가 errors 복수형에 진단을 담아 이 자리를 지나간다.)
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

    print("통과: ok/fail 봉투 · is_ok 엄격 판정 · normalize 규약 위반 7종 차단 · 페이로드 보존")
