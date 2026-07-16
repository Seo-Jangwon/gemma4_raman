# -*- coding: utf-8 -*-
"""
SingleAgent — 라만 분광기 전체를 gemma4(Ollama) 하나로 제어하는 단일 파일 에이전트.

[설계 원칙 — 2026-07-15 사용자 지시로 확정]
계획 수립·측정 파라미터 결정·재시도 판단·스펙트럼 해석·보고서 작성까지
전부 gemma4의 chain-of-thought(반복적 tool-calling)에 맡긴다. 이 시스템에
"에이전트"는 이 파일 하나뿐이다 — 별도 계획러/분석러/검증러 모듈, 별도
"적응형 튜닝" 같은 파이썬 알고리즘, 별도 지식/경험 저장소를 두지 않는다.
raman_tool_schemas.RAMAN_TOOLS 전체를 그대로 바인딩해 모델이 스스로
무엇을 몇 번 어떻게 호출할지 판단하게 한다.

[왜 이 파일이 "깡통"이어야 하는가 — 의도적 설계]
이 에이전트는 다중 에이전트와 비교하기 위한 baseline이다. baseline에 계획/검증
로직을 덧붙이면 "단일 vs 다중" 실험의 독립변수가 오케스트레이션 방식이 아니라
"내가 추가로 넣은 부가기능"이 되어 비교가 무너진다. 참고로 AILA(Nature Comm.,
IIT Delhi의 AFM 자동화 시스템)의 단일 에이전트 baseline도 정확히 이 구조다 —
LLM 하나 + 도구 전부 바인딩 + ReAct 루프, 에이전트 로직은 사실상 2줄이다.
따라서 얇은 것은 결함이 아니라 baseline의 정의다.

[LLM 계층 — 2026-07-16 변경: raw ollama.chat() → LangChain ChatOllama]
과거에는 ollama.Client().chat()을 직접 호출했다. 문제는 다중 에이전트 쪽이
LangChain(_llm.bind_tools)을 쓴다는 것 — 같은 모델을 써도 tool-calling 직렬화와
프롬프트 조립 경로가 서로 달라, 성능 차이가 아키텍처 때문인지 LLM 어댑터 때문인지
분리할 수 없었다. 이제 양쪽 모두 `ChatOllama(...).bind_tools(...)` + 수동 루프를
쓴다(다중 쪽 hw_manager.py:854와 동일한 패턴). 즉 LLM 상호작용 방식은 동일하고
오케스트레이션만 다르다.

  ※ create_react_agent(LangGraph prebuilt)를 쓰지 않은 이유:
    다중 에이전트도 prebuilt가 아니라 bind_tools + 수동 루프다. 단일만 prebuilt로
    바꾸면 오히려 비대칭이 커진다. 게다가 prebuilt의 ToolNode는 아래 이미지 주입
    패턴(도구 결과와 별개로 user 메시지를 끼워 넣기)을 표현하지 못해 Command +
    InjectedToolCallId 우회가 필요하고, _trim_history도 pre_model_hook으로 다시
    구현해야 한다 — 얻는 것 없이 검증된 코드만 흔드는 셈이라 채택하지 않았다.

[비교 공정성을 위해 추가된 것 — search_knowledge_base]
다중 에이전트의 Planner는 knowledge_base.json을 자동 검색해 권장 파워/노출을
주입받는데 단일 에이전트는 그 지식에 접근할 방법이 아예 없었다. 이건 아키텍처
차이가 아니라 그냥 불공정한 능력 격차라서, 같은 KB를 같은 알고리즘으로 검색하는
도구를 추가했다(backend.agents.knowledge 참고). 단, Planner는 결과를 강제 주입받고
단일은 스스로 호출을 판단한다는 차이는 남겨뒀다 — 그게 바로 측정 대상이다.

[유일하게 남은 비-LLM "판단" 코드 — 왜 있는가]
`_call_tool`의 조사량(dose) 누계 가드 하나뿐이다. 이건 "판단"이 아니라
물리적 회로차단기다: 레이저가 시편에 조사되는 acquire_spectrum 호출마다
이번 대화 턴의 누적 조사량을 더하고, 절대 상한을 넘으면 그 호출 자체를
막는다. 별도 에이전트나 도구가 아니라 dispatch 경로에 낀 몇 줄의 방어
코드일 뿐이며, 모델의 판단 범위(무엇을 측정할지, 파워를 얼마로 할지)에는
전혀 개입하지 않는다 — 모델이 무엇을 요청하든, 그 총합이 물리적으로
위험한 수준에 도달했을 때만 차단한다.

[공개 API — server.py가 의존하는 계약]
  ALL_TOOLS            : 바인딩된 도구 스키마 리스트 (/api/agents/health가 len() 호출)
  stream_experiment()  : SSE용 이벤트 제너레이터 (/api/experiment/stream)
  run_experiment()     : 동기 1회 실행 (/api/experiment/run, 벤치마크용)
"""

from __future__ import annotations

import json
import uuid
from typing import Iterator

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from backend.agents.knowledge import search_kb
from backend.hw_tools.raman_tool_schemas import RAMAN_TOOLS

# hardware_manager는 장비 PC의 Config.ini를 읽으므로 개발 PC에서 import가 실패할 수
# 있다. 모델명/호스트는 실패해도 기본값으로 굴러가야 하므로 try로 감싼다.
try:
    from backend.hardware_manager import OLLAMA_HOST, OLLAMA_MODEL
except Exception:
    OLLAMA_HOST = "http://192.168.1.16:11434"
    OLLAMA_MODEL = "gemma4:31b"

_MAX_AGENT_STEPS = 40   # LLM 무한 루프 방지

# 조사량 하드 상한 (대화 한 턴 기준, mJ 단위 근사치 = power_pct * exposure_s * 0.01의 누계).
# 별도 위치별 추적이나 시료별 클램프 없이 "이번 턴에 쏜 총량"만 본다 —
# 유일한 목적은 폭주(무한 재시도로 계속 고출력 조사)를 막는 최후의 회로차단기.
_MAX_DOSE_MJ_PER_TURN = 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# 도구 스키마
# ══════════════════════════════════════════════════════════════════════════════

# KB 검색 도구. RAMAN_TOOLS와 동일한 OpenAI function 포맷으로 직접 정의한다 —
# 이건 하드웨어를 만지지 않으므로 raman_tool_schemas가 아니라 여기에 둔다.
_KB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "시편 종류(graphene, cell, exosome, silicon, CNT 등)로 라만 측정 프로토콜과 "
            "권장 파라미터(레이저 파워 %, 노출 시간 초, 주요 피크 위치와 귀속)를 검색한다. "
            "측정 파라미터를 정하기 전에 호출하라 — 추측하지 말고 이 결과를 근거로 삼는다. "
            "레이저를 켜지 않으므로 시편에 무해하고, 여러 번 호출해도 된다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "검색할 시편/재료 키워드. 예: 'graphene', 'exosome cell', 'silicon'. "
                        "영문 키워드가 더 잘 매칭된다(키워드 부분문자열 매칭 방식)."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

# 모델에 바인딩되는 도구 전체. RAMAN_TOOLS(하드웨어 41종) + KB 검색 1종.
# server.py의 /api/agents/health가 len(ALL_TOOLS)를 읽는다.
ALL_TOOLS = RAMAN_TOOLS + [_KB_TOOL_SCHEMA]


# ══════════════════════════════════════════════════════════════════════════════
# 시스템 프롬프트
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
당신은 라만 분광기를 처음부터 끝까지 직접 제어하는 단일 AI 에이전트입니다.
계획 수립, 측정, 파라미터 조정, 스펙트럼 해석, 보고서 작성까지 전부 스스로 판단하고
실행합니다. 당신을 검증해 줄 별도의 검증자나 보조자는 없습니다 — 당신의 판단이 곧
최종 판단입니다.

[대화 유형 판단 — 모든 메시지에서 가장 먼저 수행]
- 인사 / 잡담 / 시스템 능력 질문: 도구를 호출하지 말고 즉시 한국어로 답한다.
- 장비 상태 질문("화면 보여?", "스테이지 어디야?"): 관측 도구(get_stage_position,
  capture_camera_frame, analyze_microscope_image, get_ccd_info)로 실제 확인한 뒤에만
  답한다. 레이저는 켜지 않는다.
- 라만 측정 요청: 아래 측정 절차를 스스로 계획해 실행한다.

[측정 절차 — 단계별로 스스로 판단하며 진행]
1. 시편 종류, 기판, 목표 위치(좌표나 외형)를 모르면 도구를 호출하기 전에 사용자에게
   먼저 묻는다. 시편을 특정하지 못한 상태로 레이저를 켜지 않는다.
2. 시편 종류를 파악했으면 search_knowledge_base로 해당 시편의 측정 프로토콜과 권장
   파라미터(레이저 파워, 노출 시간, 주요 피크 위치)를 조회한다. 파라미터를 추측으로
   정하지 말고 조회 결과를 근거로 삼되, KB에 없는 시편이면 스스로 판단하고 그 사실을
   보고서에 밝힌다.
3. 목표 위치를 모르면 analyze_microscope_image로 현미경 화면을 보고(이미지가 제공된다)
   목표의 픽셀 좌표를 스스로 읽어 move_to_pixel로 이동한다. 필요하면 run_autofocus로
   초점을 맞춘다.
4. 목표 신호를 기판 배경과 구분해야 한다면, 목표와 "완전히 동일한" 파워와 노출로 빈
   영역을 한 번 측정해 배경 기준선으로 삼는다. 측정 후 원래 목표 위치로 되돌아가는 것을
   잊지 않는다.
5. 신호 대 배경비, 포화, SNR 등을 평가하고 필요하면 위치를 옮기거나 파라미터를 조정해
   재측정한다. 단 무한히 반복하지 않는다 — 1~2회 재시도해도 개선이 없으면 기존 데이터로
   진행하고 한계를 보고서에 명시한다.
6. 측정이 끝나면 더 이상 도구를 호출하지 말고 한국어로 최종 보고서를 작성한다:
   6.1. 실험 목적
   6.2. 측정 조건 (어떻게 조정했는지 포함)
   6.3. 측정 결과 요약 (목표 vs 배경)
   6.4. 스펙트럼의 물리적 분석 (주요 피크 위치와 귀속, SNR, 포화 여부. 배경과 겹치는
        피크는 기판 유래로 보고 제외)
   6.5. 시편 종류에 맞는 도메인 전문가 수준의 해석과 결론
   6.6. 진행 중 발생한 문제와 대처 방법
   6.7. 결론 및 권고사항

[안전 규칙 — 반드시 준수]
- 도구가 오류를 반환하거나 안전 차단이 걸리면 즉시 사용자에게 상황을 그대로 보고한다.
  우회하거나 강제로 재시도하지 않는다.
- 모르는 것을 추측하지 않는다 — 도구로 확인하거나 사용자에게 묻는다."""


# ══════════════════════════════════════════════════════════════════════════════
# LLM / 도구 dispatch 로딩
# ══════════════════════════════════════════════════════════════════════════════

# bind_tools까지 끝난 Runnable을 재사용하기 위한 캐시.
# 매 턴 새로 만들면 HTTP 커넥션 풀이 매번 새로 뜬다.
_llm_cache = None


def _get_llm():
    """ChatOllama에 ALL_TOOLS를 바인딩한 Runnable을 반환한다(실패 시 None).

    [왜 지연 생성인가]
    langchain_ollama 미설치나 Ollama 호스트 불통이 모듈 import 자체를 깨뜨리면
    server.py 기동이 통째로 실패한다. 여기서 지연 import + 예외 격리해
    "LLM 없음(None)"이라는 정상적인 실패 상태로 강등한다.

    [raw dict 스키마를 그대로 bind_tools에 넘기는 이유]
    RAMAN_TOOLS는 이미 OpenAI function 포맷({"type":"function","function":{...}})이고
    LangChain의 convert_to_openai_tool은 이 포맷 dict를 그대로 통과시킨다.
    게다가 이 파일은 수동 루프라 도구 "실행"을 LangChain에 맡기지 않고 TOOL_DISPATCH로
    직접 한다 — 즉 LangChain은 스키마를 모델에 알려주는 역할만 하면 되므로
    BaseTool/StructuredTool 객체로 감쌀 이유가 전혀 없다.
    """
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache
    try:
        from langchain_ollama import ChatOllama
        _llm_cache = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
        ).bind_tools(ALL_TOOLS)
    except Exception:
        return None
    return _llm_cache


def _get_dispatch():
    """raman_tools.TOOL_DISPATCH 로드. 하드웨어 모듈이 없으면 None.

    ImportError만이 아니라 Exception 전체를 잡는 이유: raman_tools가 import하는
    config.py는 장비 PC의 Config.ini를 읽는데, 파일이 없는 개발 PC에서는
    NoSectionError(ImportError 아님)가 발생한다.
    """
    try:
        from backend.hw_tools.raman_tools import TOOL_DISPATCH
        return TOOL_DISPATCH
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 도구 호출
# ══════════════════════════════════════════════════════════════════════════════

def _slim(result):
    """대용량 배열(스펙트럼 원본 등)은 컨텍스트에 그대로 싣지 않는다 —
    토큰 낭비 + 모델 혼란 방지(구 hw_manager의 동일 정책 계승).

    길이 32 초과 리스트를 통째로 버린다. KB 검색 결과는 최대 3개짜리 리스트라
    이 필터에 걸리지 않는다.
    """
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if not (isinstance(v, list) and len(v) > 32)}
    return result


def _search_knowledge_base(args: dict) -> dict:
    """search_knowledge_base 도구의 실제 구현 — backend.agents.knowledge에 위임.

    다중 에이전트의 Planner와 "같은 파일을 같은 알고리즘으로" 검색해야 비교가
    공정하므로, 검색 로직 자체는 절대 여기에 복제하지 않는다(knowledge.py 참고).
    """
    query = str(args.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "query가 비어 있습니다. 시편/재료 키워드를 주세요."}

    hits = search_kb(query, top_k=3)
    if not hits:
        # 빈 결과를 에러로 만들지 않는 이유: "KB에 없는 시편"은 정상적인 상황이고,
        # 모델은 이때 스스로 파라미터를 판단해야 한다(시스템 프롬프트 2번 참고).
        # 에러로 주면 모델이 재시도 루프에 빠지거나 측정을 포기할 수 있다.
        return {"ok": True, "results": [],
                "note": f"'{query}'에 해당하는 프로토콜이 지식베이스에 없습니다. "
                        "직접 판단해 파라미터를 정하고 보고서에 그 사실을 밝히세요."}
    return {"ok": True, "results": hits}


def _call_tool(ctx: dict, name: str, args: dict) -> dict:
    """단일 도구 호출 관문. 유일한 비-LLM 판단은 조사량 회로차단기뿐 —
    그 외에는 raman_tools.TOOL_DISPATCH를 그대로 호출한다."""

    # KB 검색은 하드웨어를 만지지 않으므로 dispatch 유무와 무관하게 먼저 처리한다.
    # (아래 dispatch is None 가드보다 뒤에 두면 하드웨어 미연결 상태에서 KB 조회까지
    #  "하드웨어가 연결되어 있지 않습니다"로 막혀버린다.)
    if name == "search_knowledge_base":
        return _search_knowledge_base(args)

    dispatch = ctx["dispatch"]
    if dispatch is None:
        return {"ok": False, "error": "하드웨어가 연결되어 있지 않습니다."}
    fn = dispatch.get(name)
    if fn is None:
        return {"ok": False, "error": f"알 수 없는 도구: {name}"}

    if name == "acquire_spectrum":
        power = float(args.get("power", 40.0))
        exposure = float(args.get("exposure", 0.2))
        dose_inc = power * exposure * 0.01
        if ctx["dose"] + dose_inc > _MAX_DOSE_MJ_PER_TURN:
            return {"ok": False,
                    "error": (f"안전 차단: 이번 턴 누적 조사량이 상한"
                              f"({_MAX_DOSE_MJ_PER_TURN}mJ)을 초과합니다. "
                              "측정을 마무리하거나 새 요청으로 다시 시작하세요.")}
        try:
            result = fn(dict(args))
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        # 실패한 조사는 누계에 넣지 않는다 — 레이저가 실제로 나가지 않았으므로.
        if isinstance(result, dict) and result.get("ok"):
            ctx["dose"] += dose_inc
        return result

    try:
        return fn(dict(args))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# 메시지 유틸
# ══════════════════════════════════════════════════════════════════════════════

# 이미지 주입용으로 끼워 넣은 HumanMessage를 표시하는 키.
# 이 메시지는 "사용자 턴"이 아니므로 _trim_history의 턴 계산에서 제외해야 한다.
_INJECTED_IMAGE = "_injected_image"


def _msg_text(msg) -> str:
    """AIMessage의 content에서 순수 텍스트를 뽑는다.

    [왜 msg.content를 그냥 쓰지 않는가]
    LangChain 메시지의 content는 str일 수도, 콘텐츠 블록 리스트
    ([{"type":"text","text":...}, ...])일 수도 있다. 어느 쪽이 오는지는
    langchain-core 버전과 모델 어댑터에 따라 달라지므로 둘 다 처리한다.
    (.text / .text() 프로퍼티는 버전에 따라 있고 없고가 갈려 의존하지 않는다.)
    """
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


# ══════════════════════════════════════════════════════════════════════════════
# 에이전트 루프
# ══════════════════════════════════════════════════════════════════════════════

def run_stream(llm, history: list, user_message: str) -> Iterator[dict]:
    """단일 에이전트 ReAct 루프 (ChatOllama + bind_tools, 수동 루프).

    다중 에이전트의 각 노드(hw_manager.py:854 `_llm.bind_tools(lc_tools)`)와
    동일한 패턴이다 — 두 아키텍처의 LLM 상호작용 방식을 일치시켜, 성능 차이가
    오케스트레이션에서만 오도록 만든다.

    Parameters
    ----------
    llm  : _get_llm()이 반환한, 도구가 바인딩된 Runnable (None이면 error 이벤트)
    history : 이전 턴의 LangChain 메시지 리스트
    user_message : 이번 턴 사용자 입력

    yield 이벤트:
      {"type": "tool",  "name": str, "args": dict, "result": dict}
      {"type": "error", "detail": str}
      {"type": "final", "text": str, "ctx": dict, "messages": list}
    """
    if llm is None:
        yield {"type": "error",
               "detail": "Ollama LLM이 연결되지 않았습니다. "
                         "(langchain-ollama 설치 및 Ollama 서버 상태를 확인하세요)"}
        return

    ctx = {"dispatch": _get_dispatch(), "dose": 0.0, "tool_call_order": []}
    messages: list[BaseMessage] = list(history) + [HumanMessage(content=user_message)]

    for _ in range(_MAX_AGENT_STEPS):
        try:
            # 시스템 프롬프트는 세션 히스토리에 남기지 않고 매 호출마다 새로 붙인다 —
            # history에 넣으면 턴마다 중복 누적된다.
            ai_msg: AIMessage = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
        except Exception as e:
            yield {"type": "error", "detail": f"LLM 호출 실패: {type(e).__name__}: {e}"}
            return

        # 도구 호출이 없으면 = 모델이 할 말을 다 했다 = 이번 턴 종료.
        if not ai_msg.tool_calls:
            final_text = _msg_text(ai_msg).strip() or "응답을 생성하지 못했습니다."
            messages.append(ai_msg)
            yield {"type": "final", "text": final_text, "ctx": ctx, "messages": messages}
            return

        messages.append(ai_msg)   # tool_calls를 담은 assistant 메시지를 그대로 추가

        for tc in ai_msg.tool_calls:
            # LangChain의 tool_call은 dict: {"name":..., "args":..., "id":...}
            # (raw ollama의 tc.function.name / tc.function.arguments 와 형태가 다르다)
            name = tc["name"]
            args = dict(tc.get("args") or {})
            tool_call_id = tc.get("id") or ""
            ctx["tool_call_order"].append(name)

            raw_result = _call_tool(ctx, name, args)
            result = _slim(raw_result) if isinstance(raw_result, dict) else raw_result

            # 이미지 반환 도구(analyze_microscope_image)는 base64를 tool 메시지에
            # 그대로 싣지 않고, 별도 user 메시지의 이미지 블록으로 전달한다 —
            # gemma4가 실제로 "보고" 판단하게 하면서도 tool 메시지 자체는 가볍게 유지한다.
            img_b64 = result.pop("image_base64", None) if isinstance(result, dict) else None
            question = result.pop("question", None) if isinstance(result, dict) else None

            yield {"type": "tool", "name": name, "args": args, "result": result}

            # ToolMessage는 반드시 자신을 유발한 tool_call의 id를 가져야 한다 —
            # 없으면 다음 invoke에서 assistant.tool_calls와 짝이 안 맞아 거부된다.
            messages.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                tool_call_id=tool_call_id,
            ))

            if img_b64:
                # ChatOllama는 image_url 콘텐츠 블록을 ollama API의 images 필드로
                # 변환해 준다. data URI의 base64 부분만 잘라 보내므로 접두사가 필요하다.
                messages.append(HumanMessage(
                    content=[
                        {"type": "text", "text": question or "현미경 카메라 이미지:"},
                        {"type": "image_url",
                         "image_url": f"data:image/png;base64,{img_b64}"},
                    ],
                    # 이 메시지는 사람이 친 게 아니라 시스템이 끼워 넣은 것이므로
                    # 히스토리 트리밍의 "사용자 턴" 계산에서 빠져야 한다.
                    additional_kwargs={_INJECTED_IMAGE: True},
                ))

    yield {"type": "final",
          "text": f"최대 처리 단계({_MAX_AGENT_STEPS}회)에 도달해 중단했습니다. 진행 상황을 확인하고 다시 요청해 주세요.",
          "ctx": ctx, "messages": messages}


# ══════════════════════════════════════════════════════════════════════════════
# 세션 관리 + SSE 진입점
# ══════════════════════════════════════════════════════════════════════════════

# 세션별 LangChain 메시지 히스토리. {session_id: [BaseMessage, ...]}
# 로컬 단일 사용자 도구라 in-memory dict로 충분하다(프로세스 종료 시 초기화).
_SESSIONS: dict[str, list] = {}
_HISTORY_MAX_TURNS = 20   # 세션 히스토리에 보존할 최대 사용자 턴 수


def _is_user_turn(msg) -> bool:
    """이 메시지가 '사람이 친 사용자 턴'인지 — 트리밍 경계 판정용.

    이미지 주입용으로 끼워 넣은 HumanMessage는 제외한다. 포함시키면 이미지 분석을
    여러 번 하는 세션에서 한 턴이 여러 턴으로 세어져 히스토리가 과하게 잘린다.
    """
    if not isinstance(msg, HumanMessage):
        return False
    return not msg.additional_kwargs.get(_INJECTED_IMAGE, False)


def _trim_history(messages: list) -> list:
    """마지막 _HISTORY_MAX_TURNS번째 사용자 메시지 지점부터 보존 —
    도구호출↔응답 쌍이 중간에서 끊기지 않도록 '사용자 턴' 단위로 자른다.

    [왜 사용자 메시지 경계에서만 자르는가]
    ToolMessage는 자신을 유발한 AIMessage(tool_calls) 뒤에 와야만 유효하다.
    임의 지점에서 자르면 짝 잃은 ToolMessage가 맨 앞에 남아 API가 거부한다.
    사용자 메시지 경계에서 자르면 그 앞의 AIMessage+ToolMessage 쌍이 통째로
    함께 사라지므로 항상 안전하다.
    """
    user_idx = [i for i, m in enumerate(messages) if _is_user_turn(m)]
    if len(user_idx) <= _HISTORY_MAX_TURNS:
        return messages
    start = user_idx[-_HISTORY_MAX_TURNS]
    return messages[start:]


def _describe_tool(name: str, args: dict, result: dict) -> str:
    """tool 호출 1건을 사람이 읽는 한 줄로 요약 — SSE "node" 이벤트 메시지."""
    ok = result.get("ok", True)
    if not ok:
        return f"⚠️ {name} 실패: {result.get('error', '')}"
    if name == "acquire_spectrum":
        return f"📈 스펙트럼 획득 (max {result.get('max_intensity', 0):.0f} ADU)"
    if name in ("move_stage", "move_stage_relative", "move_to_pixel"):
        pos = result.get("position", {})
        return f"🧭 이동 → ({pos.get('x', '?')}, {pos.get('y', '?')})"
    if name == "analyze_microscope_image":
        return "👁️ 현미경 이미지 확인"
    if name == "run_autofocus":
        return "🔬 오토포커스 완료"
    if name == "apply_background_subtraction":
        return "🧹 형광 배경 제거 적용"
    if name == "search_knowledge_base":
        hits = result.get("results", [])
        if not hits:
            return f"📚 지식베이스 조회 — '{args.get('query', '')}' 해당 없음"
        titles = ", ".join(h.get("title", "?") for h in hits)
        return f"📚 지식베이스 조회 → {titles}"
    return f"🔧 {name} 호출"


def stream_experiment(user_message: str, session_id: str = "") -> Iterator[dict]:
    """단일 에이전트를 이벤트 제너레이터로 실행한다 (프론트엔드 SSE용).

    yield하는 이벤트 — 모두 "type"과 "session_id"를 포함:
      {"type": "node",  "node": str, "message": str}   도구 호출 진행상황
      {"type": "chat",  "reply": str}                  도구 호출 없이 끝난 턴
      {"type": "done",  "final_report": str}           측정을 포함한 턴 완료
      {"type": "error", "detail": str}
    """
    sid = session_id or str(uuid.uuid4())

    def ev(d: dict) -> dict:
        d["session_id"] = sid
        return d

    try:
        llm = _get_llm()
        history = _SESSIONS.get(sid, [])

        final_text = None
        final_ctx = None
        final_messages = history

        for event in run_stream(llm, history, user_message):
            etype = event["type"]
            if etype == "tool":
                yield ev({"type": "node", "node": event["name"],
                          "message": _describe_tool(event["name"], event["args"], event["result"])})
            elif etype == "error":
                yield ev({"type": "error", "detail": event["detail"]})
                return
            elif etype == "final":
                final_text = event["text"]
                final_ctx = event["ctx"]
                final_messages = event["messages"]

        if final_ctx is None:
            yield ev({"type": "error", "detail": "에이전트가 응답을 생성하지 못했습니다."})
            return

        _SESSIONS[sid] = _trim_history(final_messages)

        # 측정(레이저 조사)이 실제로 있었는지로 "실험 보고서" vs "일반 대화"를 가른다.
        used_measurement = "acquire_spectrum" in final_ctx.get("tool_call_order", [])
        if used_measurement:
            yield ev({"type": "done", "final_report": final_text})
        else:
            yield ev({"type": "chat", "reply": final_text})

    except Exception as e:
        yield ev({"type": "error", "detail": str(e)})


def run_experiment(user_message: str, session_id: str = "") -> dict:
    """동기 1회 실행 — 벤치마크/레거시용 (세션 히스토리 없이 매번 새로 시작)."""
    llm = _get_llm()
    final_text = ""
    for event in run_stream(llm, [], user_message):
        if event["type"] == "final":
            final_text = event["text"]
        elif event["type"] == "error":
            final_text = f"[오류] {event['detail']}"
    return {"final_report": final_text}
