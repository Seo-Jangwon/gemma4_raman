# -*- coding: utf-8 -*-
"""
AILA — 라만 분광기를 LLM 하나로 제어하는 **ReAct baseline** 에이전트.

이 파일에는 ReAct 루프 하나만 있다. 프롬프트는 backend.agents.prompts, 배관(LLM 연결·
도구 디스패치·조사량 가드·메시지 변환·세션)은 backend.agents.runtime 에 있다.

[왜 이 파일이 "깡통"이어야 하는가 — 의도적 설계]
이 에이전트는 다른 아키텍처와 비교하기 위한 baseline 이다. baseline 에 계획/검증 로직을
덧붙이면 "ReAct vs CoALA" 실험의 독립변수가 오케스트레이션이 아니라 "내가 추가로 넣은
부가기능"이 되어 비교가 무너진다. 참고로 AILA(Nature Comm., IIT Delhi 의 AFM 자동화
시스템)의 단일 에이전트 baseline 도 정확히 이 구조다 — LLM 하나 + 도구 전부 바인딩 +
ReAct 루프, 에이전트 로직은 사실상 2줄이다. 따라서 얇은 것은 결함이 아니라 baseline 의
정의다.

[ReAct 루프의 전부]
    모델에게 묻는다 → tool_calls 가 없으면 그게 최종 답변(턴 종료)
                    → 있으면 **emit 된 것을 전부 순서대로 실행**하고 결과를 붙여 다시 묻는다
계획 단계도, 후보 평가도, 장기기억도 없다. 그게 CoALA 와의 차이 전부다.

[LLM 계층 — CoALA 와 100% 동일]
양쪽 모두 ChatOllama(...).bind_tools(...) + 수동 루프다(runtime.get_chat_model).
즉 LLM 상호작용 방식은 같고 오케스트레이션만 다르다.
  ※ create_react_agent(LangGraph prebuilt)를 쓰지 않은 이유: prebuilt 의 ToolNode 는 아래
    이미지 주입 패턴(도구 결과와 별개로 user 메시지를 끼워 넣기)을 표현하지 못해 Command +
    InjectedToolCallId 우회가 필요하고, 히스토리 트리밍도 pre_model_hook 으로 다시 구현해야
    한다 — 얻는 것 없이 검증된 코드만 흔드는 셈이라 채택하지 않았다.

[공개 API — 서버가 의존하는 계약]
  ALL_TOOLS            : 바인딩된 도구 스키마 리스트 (/api/agents/health 가 len() 호출)
  stream_experiment()  : SSE용 이벤트 제너레이터 (/api/experiment/stream)
  run_experiment()     : 동기 1회 실행 (/api/experiment/run, 벤치마크용)
  run_stream()         : 저수준 루프 (벤치 러너가 원본 도구 이벤트를 보려고 직접 소비)
"""

from __future__ import annotations

import json
from typing import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from backend.agents import reason_log, runtime
from backend.service.store import run_store
from backend.agents.prompts import AUTONOMOUS, build_system_prompt
from backend.hw_tools.hw_tools.raman_tool_schemas import RAMAN_TOOLS
from backend.tools.file_tools import FILE_TOOLS

# /api/agents/health 가 mod.OLLAMA_MODEL / mod.OLLAMA_HOST 를 읽는다 — 실제로 바인딩된
# 모델을 그대로 보고하기 위한 재수출이다(어느 모델로 돌았는지 사후 확인용).
from backend.llm_config import OLLAMA_HOST, OLLAMA_MODEL   # noqa: F401

ARCH = "AILA"
SYSTEM_PROMPT = build_system_prompt("ReAct")

#: LLM 무한 루프 방지. 벽시계 가드는 아니다 — 응답이 아예 안 오는 경우는
#: agent_config.LLM_TIMEOUT_S 가 막는다.
MAX_AGENT_STEPS = 100


# ══════════════════════════════════════════════════════════════════════════════
# 액션 공간
# ══════════════════════════════════════════════════════════════════════════════

# KB 검색 도구. RAMAN_TOOLS 와 동일한 OpenAI function 포맷으로 직접 정의한다 — 하드웨어를
# 만지지 않으므로 raman_tool_schemas 가 아니라 여기에 둔다.
#
# [왜 baseline 에 이게 있는가 — 비교 공정성]
# CoALA 는 장기기억과 KB 를 조회할 수 있는데 baseline 이 그 지식에 접근할 방법이 아예
# 없으면, 그건 아키텍처 차이가 아니라 그냥 불공정한 능력 격차다. 같은 KB 를 같은
# 알고리즘으로 검색하는 도구를 준다(구현은 runtime.search_knowledge_base).
_KB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the Raman measurement protocol and recommended parameters (laser power %, "
            "exposure time in seconds, main peak positions and assignments) by sample type "
            "(graphene, cell, exosome, silicon, CNT, etc.). "
            "Call it before deciding measurement parameters - do not guess; base them on this result. "
            "It does not turn on the laser, so it is harmless to the sample, and may be called multiple times."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Sample/material keyword to search. e.g. 'graphene', 'exosome cell', 'silicon'. "
                        "English keywords match better (keyword substring matching)."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

#: 모델에 바인딩되는 도구 전체. FILE_TOOLS 는 CoALA 와 '같은 리스트 객체'를 import 한
#: 것이라 두 에이전트의 파일 분석 능력이 구조적으로 어긋날 수 없다.
ALL_TOOLS = RAMAN_TOOLS + FILE_TOOLS + [_KB_TOOL_SCHEMA]


def _get_llm():
    """도구가 바인딩된 ChatOllama Runnable (실패 시 None)."""
    return runtime.get_chat_model(ALL_TOOLS)


# ══════════════════════════════════════════════════════════════════════════════
# ReAct 루프 — 이 파일의 전부
# ══════════════════════════════════════════════════════════════════════════════

def run_stream(llm, history: list, user_message: str) -> Iterator[dict]:
    """ReAct 루프: 제안 → (도구 전부 실행 → 관측) → 반복, 도구가 없으면 종료.

    Parameters
    ----------
    llm          : _get_llm() 이 반환한, 도구가 바인딩된 Runnable (None 이면 error 이벤트)
    history      : 이전 턴의 LangChain 메시지 리스트
    user_message : 이번 턴 사용자 입력

    yield 이벤트:
      {"type": "tool",  "name": str, "args": dict, "result": dict}
      {"type": "error", "detail": str}
      {"type": "final", "text": str, "ctx": dict, "messages": list}
    """
    # 추론 로그(results/<run_id>/<문항>.log). 세션은 run_store 의 스레드로컬에서 읽으므로
    # 시그니처가 바뀌지 않는다 — 벤치 경로는 호출 직전에 begin_session 을 부른다.
    # 로깅이 꺼져 있으면 무동작 대역이다.
    rlog = reason_log.open_turn(ARCH, user_message)
    try:
        if llm is None:
            rlog.failed(runtime.LLM_NOT_CONNECTED)
            yield {"type": "error", "detail": runtime.LLM_NOT_CONNECTED}
            return

        ctx = {"dispatch": runtime.get_tool_dispatch(), "dose": 0.0, "tool_call_order": []}
        messages: list[BaseMessage] = list(history) + [HumanMessage(content=user_message)]

        for step in range(MAX_AGENT_STEPS):
            # ── 제안 ──────────────────────────────────────────────────────────
            try:
                # 시스템 프롬프트는 세션 히스토리에 남기지 않고 매 호출마다 새로 붙인다 —
                # history 에 넣으면 턴마다 중복 누적된다. 세션 요약(내 세션 라벨 + 지금까지
                # 저장한 산출물)도 마찬가지다: 히스토리에 넣으면 산출물이 늘 때마다 낡은
                # 목록이 쌓여 모델이 지난 상태를 현재로 오인한다. 매번 '현재'만 실어준다.
                session_note = run_store.summary_for_prompt()
                system_text = SYSTEM_PROMPT + (f"\n\n[This session]\n{session_note}\n"
                                               if session_note else "")
                # rlog.invoke 는 llm.invoke 를 그대로 부르고 프롬프트·생각·본문·토큰통계를
                # 남긴다. 로깅이 꺼져 있으면 llm.invoke(messages) 와 완전히 동일하다.
                ai_msg: AIMessage = rlog.invoke(
                    [SystemMessage(content=system_text)] + messages,
                    llm=llm, stage="ReAct propose", step=step + 1)
            except Exception as e:
                detail = runtime.llm_error_detail(e, "LLM call")
                rlog.failed(detail)
                yield {"type": "error", "detail": detail}
                return

            # ── 도구 호출이 없으면 = 모델이 할 말을 다 했다 = 이번 턴 종료 ──────
            if not ai_msg.tool_calls:
                final_text = runtime.text_of(ai_msg).strip() or runtime.EMPTY_REPLY
                messages.append(ai_msg)
                rlog.reasoning(step + 1, "No tool call -> writing the final report, turn ends")
                rlog.final(final_text, ctx)
                yield {"type": "final", "text": final_text, "ctx": ctx, "messages": messages}
                return

            messages.append(ai_msg)   # tool_calls 를 담은 assistant 메시지를 그대로 추가
            rlog.reasoning(step + 1,
                           f"Decided to run {len(ai_msg.tool_calls)} tool(s) -> "
                           + ", ".join(str(tc["name"]) for tc in ai_msg.tool_calls))

            # ── 실행 + 관측 — ★ ReAct 의 정의: emit 된 것을 전부, 그 순서대로 ────
            #    (CoALA 는 여기서 후보를 평가해 하나만 고른다. 그게 유일한 구조 차이다.)
            for tc in ai_msg.tool_calls:
                # LangChain 의 tool_call 은 dict: {"name":..., "args":..., "id":...}
                name = tc["name"]
                args = dict(tc.get("args") or {})

                rlog.acting(step + 1, name, args)
                ex = runtime.execute_tool(ctx, name, args, tc.get("id") or "")
                rlog.observation(step + 1, name, ex["result"], ex["elapsed_ms"])
                if ex["img_b64"]:
                    rlog.rec("ReAct OBSERVATION",
                             f"step {step + 1} · {name} · injected 1 image into the model "
                             f"(base64 {len(ex['img_b64'])} chars, not written to this log)")

                yield {"type": "tool", "name": name, "args": args, "result": ex["result"]}

                # ToolMessage 는 반드시 자신을 유발한 tool_call 의 id 를 가져야 한다 —
                # 없으면 다음 invoke 에서 assistant.tool_calls 와 짝이 안 맞아 거부된다.
                messages.append(ToolMessage(
                    content=json.dumps(ex["result"], ensure_ascii=False, default=str),
                    tool_call_id=ex["tool_call_id"],
                ))
                if ex["img_b64"]:
                    messages.append(runtime.image_message(ex["img_b64"], ex["question"]))

        capped = (f"Stopped after reaching the maximum number of steps ({MAX_AGENT_STEPS}). "
                  f"Please check the progress and request again.")
        rlog.final(capped, ctx)
        yield {"type": "final", "text": capped, "ctx": ctx, "messages": messages}
    finally:
        # 벤치 러너가 중단하면 서버가 gen.close() 를 부른다(GeneratorExit).
        # 그때도 로그 꼬리가 닫히도록 finally 에 둔다.
        rlog.close()


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API — 껍데기는 runtime 이 갖고 있고, 여기서는 루프만 꽂아 준다
# ══════════════════════════════════════════════════════════════════════════════

_SESSIONS: runtime.SessionStore = {}


def stream_experiment(user_message: str, session_id: str = "") -> Iterator[dict]:
    """이 에이전트를 프론트엔드 SSE 이벤트 제너레이터로 실행한다."""
    return runtime.stream_turn(
        ARCH, _SESSIONS,
        lambda history, sid: run_stream(_get_llm(), history, user_message),
        user_message, session_id,
        # 자율 모드(기본)에서는 대화 경로에서도 그리드 승인 게이트를 끈다: "그리드 스캔도
        # 승인 없이 스스로" 라는 자율 정책과 코드 인터록이 어긋나면, 모델은 프롬프트대로
        # 실행하려 하는데 도구가 거부해 턴이 헛돈다. RAMAN_AUTONOMOUS=0 이면 종전처럼
        # 미리보기 → 사람 승인 → 실행 2턴 흐름으로 복귀한다.
        interactive_grid_gate=not AUTONOMOUS)


def run_experiment(user_message: str, session_id: str = "") -> dict:
    """동기 1회 실행 — 벤치마크/레거시용 (세션 히스토리 없이 매번 새로 시작)."""
    return runtime.run_turn_once(
        ARCH,
        lambda history, sid: run_stream(_get_llm(), history, user_message),
        user_message, session_id)
