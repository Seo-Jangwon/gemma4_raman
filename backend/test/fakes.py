# -*- coding: utf-8 -*-
"""가짜 LLM·가짜 장비 — 두 아키텍처를 같은 조건에서 돌리기 위한 대역.

[왜 대역이 필요한가]
공평성은 '같은 입력에 두 루프가 어떻게 달리 반응하는가'로만 잴 수 있다. 진짜 LLM 은
같은 프롬프트에도 매번 다르게 답하므로 그 비교가 성립하지 않는다. 대본대로만 답하는
LLM 을 쓰면 **모델의 변덕이 사라져 아키텍처 차이만 남는다.**

장비도 같은 이유로 가짜다. 여기서 재는 것은 '에이전트가 무엇을 부르기로 했는가'이지
스테이지가 실제로 움직였는가가 아니다. 덤으로 개발 PC 에서 돌아 회귀를 실험 전에 잡는다.
"""
from __future__ import annotations

import shutil

from langchain_core.messages import AIMessage

# ══════════════════════════════════════════════════════════════════════════════
# 가짜 LLM
# ══════════════════════════════════════════════════════════════════════════════


class FakeLLM:
    """대본을 순서대로 돌려주는 LLM 대역.

    대본 한 칸이 곧 응답 하나다. 칸의 타입으로 무엇을 낼지 정한다:

        [("acquire_spectrum", {"power": 1}), ...]   tool_calls 를 담은 AIMessage
        "최종 보고서 문장"                            텍스트만 담은 AIMessage(= 턴 종료)
        {"text": "...", "done_reason": "length"}    출력 상한에 잘린 응답
        Exception("...")                            invoke 가 그 예외를 올린다

    대본이 바닥나면 빈 텍스트를 낸다 — 루프가 '할 말 없음'으로 읽고 턴을 닫으므로,
    대본을 잘못 써도 무한 루프가 아니라 즉시 끝난다.

    Attributes
    ----------
    seen : 호출마다 받은 프롬프트 전문. '무엇이 실렸는가'를 사후에 검사할 때 쓴다.
    reasoning_seen : 호출마다 넘어온 reasoning 인자(think 를 켰는지/껐는지).
    """

    def __init__(self, script: list | None = None, carry_thinking: bool = False):
        self.script = list(script or [])
        self.seen: list[str] = []
        self.reasoning_seen: list = []
        self.calls = 0
        # 진짜 thinking 모델처럼 reasoning_content 를 실어 보낼지. 두 루프가 그것을
        # 히스토리에 남기는지 대조할 때 켠다(thinking 대칭 검사).
        self.carry_thinking = carry_thinking

    def invoke(self, messages, **kw):
        self.calls += 1
        self.reasoning_seen.append(kw.get("reasoning"))
        self.seen.append("\n".join(_text_of(m) for m in messages))

        item = self.script.pop(0) if self.script else ""
        extra = {"reasoning_content": f"thought-{self.calls}"} if self.carry_thinking else {}

        if isinstance(item, Exception):
            raise item
        if isinstance(item, list):
            return AIMessage(
                content="", additional_kwargs=extra,
                tool_calls=[{"name": n, "args": dict(a), "id": f"call_{self.calls}_{i}"}
                            for i, (n, a) in enumerate(item)])
        if isinstance(item, dict):
            return AIMessage(content=item.get("text", ""), additional_kwargs=extra,
                             response_metadata={"done_reason": item.get("done_reason", "stop")})
        return AIMessage(content=str(item), additional_kwargs=extra)


def _text_of(m) -> str:
    """메시지의 텍스트. 콘텐츠 블록 리스트로 오는 경우(이미지 주입)도 받는다."""
    c = getattr(m, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return str(c or "")


# ══════════════════════════════════════════════════════════════════════════════
# 가짜 장비
# ══════════════════════════════════════════════════════════════════════════════

#: 하드웨어 디스패치 대역. 도구 이름 → 그럴듯한 성공 결과.
#:
#: 실패를 내는 도구가 하나(`run_autofocus`) 섞여 있다 — '도구가 실패해도 루프가 죽지
#: 않는가'는 두 아키텍처 모두에서 확인해야 하는 성질이고, 실패 경로가 없는 대역으로는
#: 그것을 못 잰다.
#:
#: 파일·KB 도구는 여기 없다. runtime._dispatch 가 FILE_DISPATCH·RUNTIME_DISPATCH 를
#: 하드웨어 가드보다 **먼저** 보므로 그쪽은 진짜 구현이 그대로 탄다(그게 정상 동작이다).
FAKE_DISPATCH = {
    "start_camera_stream": lambda a: {"ok": True, "already_streaming": False},
    "analyze_microscope_image": lambda a: {"ok": True, "width": 1060, "height": 800,
                                           "sharpness_score": 96.4},
    "acquire_spectrum": lambda a: {"ok": True, "mode": "single", "length": 1024,
                                   "max_intensity": 4411.0,
                                   "laser_power_pct": a.get("power"),
                                   "exposure_time": a.get("exposure")},
    "move_stage": lambda a: {"ok": True, "position": {"x": a.get("x"), "y": a.get("y")}},
    "move_to_pixel": lambda a: {"ok": True, "position": {"x": 37.9, "y": 25.3}},
    "get_hardware_status": lambda a: {"ok": True, "summary": "connected: stage, laser, ccd, camera",
                                      "connected": {"stage": True, "laser": True,
                                                    "ccd": True, "camera": True}},
    "get_laser_status": lambda a: {"ok": True, "is_on": False, "power_armed": False},
    "get_ccd_info": lambda a: {"ok": True, "temperature": -40, "exposure_time": 1.0},
    "run_autofocus": lambda a: {"ok": False, "error": "Autofocus did not converge."},
    "set_laser_power": lambda a: {"ok": True, "power_percent": a.get("percent")},
    "laser_off": lambda a: {"ok": True, "status": "Laser OFF"},
    "reconnect_hardware": lambda a: {"ok": True, "reconnected": ["stage", "ccd"]},
}


# ══════════════════════════════════════════════════════════════════════════════
# 두 아키텍처를 같은 방식으로 돌리는 어댑터
# ══════════════════════════════════════════════════════════════════════════════

def install_fake_hardware() -> None:
    """runtime.get_tool_dispatch 를 가짜로 갈아 끼운다.

    에이전트가 턴 시작 때 이것을 불러 ctx["dispatch"] 를 채우므로, 여기만 바꾸면 두
    아키텍처가 같은 가짜 장비를 본다. 원본을 되돌리지 않는 이유: 이 모듈을 import 하는
    것은 테스트뿐이고, 테스트 프로세스에서 진짜 장비를 다시 쓸 일이 없다.
    """
    from backend.agents.runtime import runtime
    runtime.get_tool_dispatch = lambda: FAKE_DISPATCH        # noqa: E731


def drive(arch: str, script: list, question: str = "measure this sample",
          eval_script: list | None = None, session_id: str = "faketest",
          carry_thinking: bool = False) -> dict:
    """한 아키텍처를 대본대로 한 턴 돌리고 결과를 정리해 돌려준다.

    Parameters
    ----------
    arch : "AILA" | "CoALA"
    script : propose(제안) LLM 의 대본. FakeLLM 참고.
    eval_script : CoALA 평가 LLM 의 대본. AILA 에서는 무시된다(평가 단계가 없다).

    Returns
    -------
    dict
        ``{"events", "ctx", "messages", "final", "tools", "llm_calls", "eval_stats"}``
        두 아키텍처가 **같은 모양**으로 나오게 맞춘 것이 이 함수의 요점이다 —
        run_stream 의 시그니처가 서로 다르므로(CoALA 는 평가용 LLM 을 하나 더 받는다),
        호출부마다 분기하면 검사 코드가 아키텍처를 의식하게 되고 그러면 대조가 흐려진다.
    """
    from backend.service.store import run_store

    install_fake_hardware()
    run_store.begin_session(session_id, arch, isolated=True)

    plan_llm = FakeLLM(script, carry_thinking=carry_thinking)
    eval_llm = FakeLLM(eval_script or [], carry_thinking=carry_thinking)

    if arch == "AILA":
        from backend.agents.architectures import single_agent_AILA as mod
        stream = mod.run_stream(plan_llm, [], question)
    else:
        from backend.agents.architectures import single_agent_CoALA as mod
        stream = mod.run_stream(plan_llm, eval_llm, [], question, session_id=session_id)

    events = list(stream)
    final = next((e for e in events if e["type"] == "final"), None)
    err = next((e for e in events if e["type"] == "error"), None)
    ctx = (final or {}).get("ctx") or {}
    return {
        "arch": arch,
        "events": events,
        "ctx": ctx,
        "messages": (final or {}).get("messages") or [],
        "final": (final or {}).get("text"),
        "error": (err or {}).get("detail"),
        # 실행된 도구를 순서대로. 두 아키텍처를 비교하는 1차 지표다.
        "tools": [e["name"] for e in events if e["type"] == "tool"],
        "dose": ctx.get("dose", 0.0),
        "llm_calls": plan_llm.calls + eval_llm.calls,
        "plan_llm": plan_llm,
        "eval_llm": eval_llm,
        "eval_stats": ctx.get("eval_stats"),
    }


def remove_tree(path) -> bool:
    """디렉터리를 지우고 **실제로 사라졌는지** 돌려준다.

    [왜 ignore_errors 를 쓰지 않는가]
    이 저장소는 OneDrive 동기화 폴더 안에 있어서, 방금 만든 디렉터리를 지우려 하면
    동기화 프로세스가 잡고 있어 PermissionError(WinError 5)가 난다 — 비어 있는
    디렉터리에서도 난다. shutil.rmtree(..., ignore_errors=True) 는 그 실패를 삼키므로
    호출부가 "정리함"이라고 보고하면서 실제로는 남는다. 조용히 틀리는 종류라, 다음
    실행이 앞 실행의 파일을 보게 되어도 아무도 모른다(격리 검사가 그때 무의미해진다).

    한 번 짧게 재시도한다 — 동기화 잠금은 대개 그 사이에 풀린다. 그래도 남으면 남았다고
    **사실대로** 돌려주고, 판단은 호출부가 한다.
    """
    import time as _t
    from pathlib import Path as _P

    p = _P(path)
    for attempt in range(2):
        if not p.exists():
            return True
        try:
            shutil.rmtree(p)
        except OSError:
            if attempt == 0:
                _t.sleep(0.3)
    return not p.exists()


__all__ = ["FakeLLM", "FAKE_DISPATCH", "install_fake_hardware", "drive", "remove_tree"]
