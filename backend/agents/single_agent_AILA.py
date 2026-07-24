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

from backend.agents.detail_log import new_turn
from backend.agents.file_tools import FILE_DISPATCH, FILE_TOOLS
from backend.agents.knowledge import search_kb
from backend.hw_tools.raman_tool_schemas import RAMAN_TOOLS
from backend.spectrum_store import spectrum_event

# hardware_manager는 장비 PC의 Config.ini를 읽으므로 개발 PC에서 import가 실패할 수
# 있다. 모델명/호스트는 실패해도 기본값으로 굴러가야 하므로 try로 감싼다.
try:
    from backend.hardware_manager import OLLAMA_HOST, OLLAMA_MODEL
except Exception:
    OLLAMA_HOST = "http://192.168.1.15:11434"
    OLLAMA_MODEL = "gemma4:31b"

_MAX_AGENT_STEPS = 100   # LLM 무한 루프 방지

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

# 모델에 바인딩되는 도구 전체. RAMAN_TOOLS(하드웨어) + 첨부파일 도구 + KB 검색 1종.
# FILE_TOOLS는 CoALA와 '같은 리스트 객체'를 import한 것이라 두 에이전트의 파일 분석
# 능력이 구조적으로 어긋날 수 없다(backend/agents/file_tools.py 머리말 참고).
# server.py의 /api/agents/health가 len(ALL_TOOLS)를 읽는다.
ALL_TOOLS = RAMAN_TOOLS + FILE_TOOLS + [_KB_TOOL_SCHEMA]


# ══════════════════════════════════════════════════════════════════════════════
# 시스템 프롬프트
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a single AI agent that directly controls a Raman spectrometer from start to finish.
You plan, measure, adjust parameters, interpret spectra, and write the report entirely by your
own judgment. There is no separate validator or assistant to check you - your judgment is the
final judgment.

[Conversation-type decision - do this first on every message]
- Greeting / small talk / questions about system capabilities: do not call tools; answer immediately in English.
- Instrument-status questions ("can you see the view?", "where is the stage?"): answer only after
  actually checking with observation tools (get_stage_position, capture_camera_frame,
  analyze_microscope_image, get_ccd_info). Do not turn the laser on.
- Raman measurement requests: plan and execute the measurement procedure below yourself.
- Requests about a file the user attached: follow the attached-data-files section below. Do not turn
  the laser on just because a file arrived.

[Attached data files - csv / excel / txt]
1. When the user attaches a data file or refers to one, call list_uploaded_files, then call
   inspect_file on each relevant file. inspect_file returns only the structure - row/column counts,
   column names, numeric-or-text per column, min/max/mean, and the first few rows.
2. Decide yourself what the columns mean. Nothing has been interpreted for you: judge which numeric
   column is a Raman shift axis in cm-1, which is intensity, which is a wavelength or a stage
   coordinate, and which columns are not spectra at all but metadata (sample name, laser power,
   exposure time, date, operator notes). Use the value ranges and column names as evidence, and say
   what you concluded and why.
3. Then call run_analysis with file_ids to compute on the full data - peak detection, baseline
   correction, normalization, plotting, or comparison against spectra you measured. Inside the code
   the file is available as files[i]["table"]["<column name>"].
4. Report both kinds of content separately: the spectral information you extracted (peak positions
   and assignments, SNR, etc.) and any other information the file carried (measurement conditions,
   sample identity, anything that changes how the spectrum should be read).
5. If the file turns out to hold no spectrum, say so plainly and report what it does hold instead -
   do not force a spectral interpretation onto it.

[Measurement procedure - proceed step by step, using your own judgment]
1. If you do not know the sample type, substrate, or target location (coordinates or appearance),
   ask the user first before calling any tool. Do not turn the laser on without identifying the sample.
2. Once you know the sample type, use search_knowledge_base to look up the measurement protocol and
   recommended parameters (laser power, exposure time, main peak positions) for that sample. Do not
   guess parameters - base them on the lookup result; if the sample is not in the KB, decide yourself
   and state that in the report.
3. If you do not know the target location, use analyze_microscope_image to view the microscope image
   (the image is provided), read the target's pixel coordinates yourself, and move with move_to_pixel.
   Focus with run_autofocus if needed.
4. If you must distinguish the target signal from the substrate background, measure a blank area once
   with the "exactly identical" power and exposure as the target to use as a background baseline. Do
   not forget to return to the original target location after measuring.
5. Evaluate signal-to-background ratio, saturation, SNR, etc., and if needed move the position or
   adjust parameters and re-measure. But do not repeat indefinitely - if 1-2 retries show no
   improvement, proceed with the existing data and state the limitation in the report.
6. When the measurement is done, stop calling tools and write the final report in English:
   6.1. Experiment objective
   6.2. Measurement conditions (including how they were adjusted)
   6.3. Summary of measurement results (target vs background)
   6.4. Physical analysis of the spectrum (main peak positions and assignments, SNR, saturation.
        Peaks overlapping the background are treated as substrate-derived and excluded)
   6.5. Domain-expert-level interpretation and conclusion appropriate to the sample type
   6.6. Problems encountered during the process and how they were handled
   6.7. Conclusion and recommendations

[Safety rules - must be followed]
- If a tool returns an error or a safety block is triggered, immediately report the situation to the
  user as is. Do not bypass it or force a retry.
- Do not guess what you do not know - verify with a tool or ask the user.
- Stage coordinate units: mm (X: 0-75.3, Y: 0-50.2, Z: -1.0-1.0; origin at x=37.8759, y=25.24805, z n/a)
"""


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
        return {"ok": False, "error": "query is empty. Provide a sample/material keyword."}

    hits = search_kb(query, top_k=3)
    if not hits:
        # 빈 결과를 에러로 만들지 않는 이유: "KB에 없는 시편"은 정상적인 상황이고,
        # 모델은 이때 스스로 파라미터를 판단해야 한다(시스템 프롬프트 2번 참고).
        # 에러로 주면 모델이 재시도 루프에 빠지거나 측정을 포기할 수 있다.
        return {"ok": True, "results": [],
                "note": f"No protocol matching '{query}' in the knowledge base. "
                        "Decide the parameters yourself and state that in the report."}
    return {"ok": True, "results": hits}


def _call_tool(ctx: dict, name: str, args: dict) -> dict:
    """단일 도구 호출 관문. 유일한 비-LLM 판단은 조사량 회로차단기뿐 —
    그 외에는 raman_tools.TOOL_DISPATCH를 그대로 호출한다."""

    # KB 검색은 하드웨어를 만지지 않으므로 dispatch 유무와 무관하게 먼저 처리한다.
    # (아래 dispatch is None 가드보다 뒤에 두면 하드웨어 미연결 상태에서 KB 조회까지
    #  "하드웨어가 연결되어 있지 않습니다"로 막혀버린다.)
    if name == "search_knowledge_base":
        return _search_knowledge_base(args)

    # 첨부 파일 조회/분석도 같은 이유로 하드웨어 가드보다 먼저. run_analysis도 여기 포함되어
    # 있어(file_tools.FILE_DISPATCH) 순수 계산인 분석이 장비 유무에 묶이지 않는다.
    if name in FILE_DISPATCH:
        return FILE_DISPATCH[name](args)

    dispatch = ctx["dispatch"]
    if dispatch is None:
        return {"ok": False, "error": "Hardware is not connected."}
    fn = dispatch.get(name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    if name == "acquire_spectrum":
        power = float(args.get("power", 40.0))
        exposure = float(args.get("exposure", 0.2))
        dose_inc = power * exposure * 0.01
        if ctx["dose"] + dose_inc > _MAX_DOSE_MJ_PER_TURN:
            return {"ok": False,
                    "error": (f"Safety block: this turn's cumulative dose would exceed the limit "
                              f"({_MAX_DOSE_MJ_PER_TURN} mJ). "
                              "Wrap up the measurement or start again with a new request.")}
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
               "detail": "Ollama LLM is not connected. "
                         "(Check that langchain-ollama is installed and the Ollama server is running)"}
        return

    ctx = {"dispatch": _get_dispatch(), "dose": 0.0, "tool_call_order": []}
    messages: list[BaseMessage] = list(history) + [HumanMessage(content=user_message)]

    for _ in range(_MAX_AGENT_STEPS):
        try:
            # 시스템 프롬프트는 세션 히스토리에 남기지 않고 매 호출마다 새로 붙인다 —
            # history에 넣으면 턴마다 중복 누적된다.
            ai_msg: AIMessage = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
        except Exception as e:
            yield {"type": "error", "detail": f"LLM call failed: {type(e).__name__}: {e}"}
            return

        # 도구 호출이 없으면 = 모델이 할 말을 다 했다 = 이번 턴 종료.
        if not ai_msg.tool_calls:
            final_text = _msg_text(ai_msg).strip() or "Failed to generate a response."
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
                        {"type": "text", "text": question or "Microscope camera image:"},
                        {"type": "image_url",
                         "image_url": f"data:image/png;base64,{img_b64}"},
                    ],
                    # 이 메시지는 사람이 친 게 아니라 시스템이 끼워 넣은 것이므로
                    # 히스토리 트리밍의 "사용자 턴" 계산에서 빠져야 한다.
                    additional_kwargs={_INJECTED_IMAGE: True},
                ))

    yield {"type": "final",
          "text": f"Stopped after reaching the maximum number of steps ({_MAX_AGENT_STEPS}). Please check the progress and request again.",
          "ctx": ctx, "messages": messages}


# ══════════════════════════════════════════════════════════════════════════════
# 세션 관리 + SSE 진입점
# ══════════════════════════════════════════════════════════════════════════════

# 세션별 LangChain 메시지 히스토리. {session_id: [BaseMessage, ...]}
# 로컬 단일 사용자 도구라 in-memory dict로 충분하다(프로세스 종료 시 초기화).
_SESSIONS: dict[str, list] = {}
_HISTORY_MAX_TURNS = 100   # 세션 히스토리에 보존할 최대 사용자 턴 수


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
        return f"⚠️ {name} failed: {result.get('error', '')}"
    if name == "acquire_spectrum":
        return f"📈 Spectrum acquired (max {result.get('max_intensity', 0):.0f} ADU)"
    if name in ("move_stage", "move_stage_relative", "move_to_pixel"):
        pos = result.get("position", {})
        return f"🧭 Moved → ({pos.get('x', '?')}, {pos.get('y', '?')})"
    if name == "analyze_microscope_image":
        return "👁️ Microscope image checked"
    if name == "run_autofocus":
        return "🔬 Autofocus complete"
    if name == "apply_background_subtraction":
        return "🧹 Fluorescence background subtraction applied"
    if name == "search_knowledge_base":
        hits = result.get("results", [])
        if not hits:
            return f"📚 KB lookup — no match for '{args.get('query', '')}'"
        titles = ", ".join(h.get("title", "?") for h in hits)
        return f"📚 KB lookup → {titles}"
    if name == "list_uploaded_files":
        files = result.get("files", [])
        if not files:
            return "📎 Attached files — none"
        return f"📎 Attached files → {', '.join(f.get('filename', '?') for f in files)}"
    if name == "inspect_file":
        return (f"🔍 Inspected {result.get('filename', '?')} "
                f"({result.get('n_rows', '?')} rows × {result.get('n_cols', '?')} cols)")
    if name == "run_analysis":
        return f"🧮 Analysis code executed ({result.get('image_count', 0)} figure(s))"
    return f"🔧 {name} called"


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

    # 벤치마크 로그: resolved sid(빈 값이면 위에서 uuid로 확정한 것)를 넘겨야 서로 다른
    # 세션이 'nosession' 파일 하나로 뭉치지 않는다. run_stream 소비 전에 만들어 첫
    # tool 이벤트부터 관측한다. 로깅 실패는 detail_log가 내부에서 삼키므로 여기선 그냥 먹인다.
    turn = new_turn("AILA", sid, user_message)

    try:
        llm = _get_llm()
        history = _SESSIONS.get(sid, [])

        final_text = None
        final_ctx = None
        final_messages = history

        for event in run_stream(llm, history, user_message):
            turn.observe(event)
            etype = event["type"]
            if etype == "tool":
                yield ev({"type": "node", "node": event["name"],
                          "message": _describe_tool(event["name"], event["args"], event["result"])})
                sp = spectrum_event(event["result"])   # 측정이면 스펙트럼 이미지도 전달
                if sp:
                    yield ev(sp)
            elif etype == "error":
                turn.fail(event["detail"])
                yield ev({"type": "error", "detail": event["detail"]})
                return
            elif etype == "final":
                final_text = event["text"]
                final_ctx = event["ctx"]
                final_messages = event["messages"]

        if final_ctx is None:
            turn.fail("The agent failed to generate a response.")
            yield ev({"type": "error", "detail": "The agent failed to generate a response."})
            return

        _SESSIONS[sid] = _trim_history(final_messages)

        # 측정(레이저 조사)이 실제로 있었는지로 "실험 보고서" vs "일반 대화"를 가른다.
        used_measurement = "acquire_spectrum" in final_ctx.get("tool_call_order", [])
        turn.complete("done" if used_measurement else "chat", final_text, final_ctx)
        if used_measurement:
            yield ev({"type": "done", "final_report": final_text})
        else:
            yield ev({"type": "chat", "reply": final_text})

    except Exception as e:
        turn.fail(str(e))
        yield ev({"type": "error", "detail": str(e)})


def run_experiment(user_message: str, session_id: str = "") -> dict:
    """동기 1회 실행 — 벤치마크/레거시용 (세션 히스토리 없이 매번 새로 시작)."""
    # 벤치마크 로그: 빈 session_id면 매 실행마다 uuid를 새로 만들어 실행 1회 = 파일 1개로
    # 분리한다(안 그러면 모든 벤치마크 질의가 'nosession' 한 파일에 뭉친다).
    sid = session_id or str(uuid.uuid4())
    turn = new_turn("AILA", sid, user_message)
    llm = _get_llm()
    final_text = ""
    final_ctx = None
    error_detail = None
    for event in run_stream(llm, [], user_message):
        turn.observe(event)
        if event["type"] == "final":
            final_text = event["text"]
            final_ctx = event["ctx"]
        elif event["type"] == "error":
            error_detail = event["detail"]
            final_text = f"[Error] {event['detail']}"
    if error_detail is not None:
        turn.fail(error_detail, final_ctx)
    else:
        used_measurement = "acquire_spectrum" in (final_ctx or {}).get("tool_call_order", [])
        turn.complete("done" if used_measurement else "chat", final_text, final_ctx)
    return {"final_report": final_text}
