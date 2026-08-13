# -*- coding: utf-8 -*-
"""
두 에이전트(AILA/ReAct · CoALA)가 공유하는 실행 기반.

여기 있는 것은 전부 **오케스트레이션과 무관한 배관**이다: LLM 연결, 도구 디스패치,
조사량 회로차단기, 메시지 변환, 히스토리 트리밍, 진행 문구, 세션 껍데기.
아키텍처(ReAct 루프 / CoALA 의사결정 사이클)는 각 에이전트 파일에만 있다.

[왜 공유하는가]
비교 실험의 독립변수는 오케스트레이션 **하나**여야 한다. 예전에는 이 배관이 두 파일에
그대로 복사돼 있었고(약 222줄), 사본이 갈라지면 성능 차이가 아키텍처 때문인지 배관
때문인지 분리할 수 없다. CoALA 파일 머리말이 이미 "두 에이전트가 어긋나면 비교 자체가
무너지는 값은 사본으로 두지 않고 공통 상위 모듈에 둔다"고 적고 있었다 — 이 모듈은 그
원칙을 배관 전체로 넓힌 것이다. 서로를 import 하는 것이 아니라 둘 다 이쪽을 본다.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, Iterator, Optional

from langchain_core.messages import HumanMessage

from backend.service.store import run_store
from backend.service.store.detail_log import new_turn
from backend.tools.non_hw_tools.file_tools import FILE_DISPATCH
# 도구 응답 형식의 단일 출처. normalize 가 여기(call_tool)에서만 불리는 이유는 result.py 참고.
from backend.tools.result import fail, is_ok, normalize, ok
# KB 검색 도구. 구현·스키마는 backend.tools.data_tools 한 곳에 있다 — 예전에는 스키마가
# 두 에이전트 파일에 복붙되고 구현은 여기 있었다.
from backend.tools.non_hw_tools.data_tools import search_knowledge_base
from backend.tools.schema import call_with
from backend.service.store.spectrum_store import spectrum_event
from backend.llm_config import (
    LLM_TIMEOUT_S, NUM_CTX, OLLAMA_HOST, OLLAMA_MODEL,
)
# 조사량 상한·계산식은 backend.util.safety_limits 단일 출처. 예전에는 같은 1000.0 과 같은
# 식이 AILA·CoALA·도구 계층 세 곳에 박혀 있어 한 곳만 고치면 나머지가 조용히 갈라졌다 —
# 하필 갈라지는 대상이 '레이저를 얼마나 쏘게 둘 것인가'였다.
from backend.service.safety.safety_limits import MAX_DOSE_MJ_PER_TURN, estimate_dose_mj
# 관측 축약(도구 결과 중 무엇을 모델 컨텍스트에 싣는가)도 단일 출처. 옛 규칙은 최상위 한
# 겹만 봐서 배열이 dict 안에 든 kinetic 측정을 통과시켰고(5프레임 하나가 224,078자),
# 그게 num_ctx 를 넘겨 Ollama 가 프롬프트를 조용히 잘라낸 것이 빈 응답의 실제 원인이었다.
from backend.service.agents.tool_slim import slim


# ══════════════════════════════════════════════════════════════════════════════
# LLM 연결
# ══════════════════════════════════════════════════════════════════════════════

_llm_cache: dict[str, object] = {}


def get_chat_model(tools: Optional[list] = None):
    """ChatOllama Runnable 을 반환한다(실패 시 None). tools 를 주면 bind_tools 까지.

    [왜 지연 생성인가]
    langchain_ollama 미설치나 Ollama 호스트 불통이 모듈 import 자체를 깨뜨리면 서버 기동이
    통째로 실패한다. 여기서 지연 import + 예외 격리해 "LLM 없음(None)"이라는 정상적인
    실패 상태로 강등한다.

    [왜 캐시하는가]
    매 턴 새로 만들면 HTTP 커넥션 풀이 매번 새로 뜬다.

    [raw dict 스키마를 그대로 bind_tools 에 넘기는 이유]
    RAMAN_TOOLS 는 이미 OpenAI function 포맷이고 LangChain 의 convert_to_openai_tool 은 이
    포맷 dict 를 그대로 통과시킨다. 게다가 두 에이전트 모두 수동 루프라 도구 '실행'을
    LangChain 에 맡기지 않고 TOOL_DISPATCH 로 직접 한다 — LangChain 은 스키마를 모델에
    알려주는 역할만 하면 되므로 BaseTool 로 감쌀 이유가 없다.

    tools=None 은 CoALA 의 evaluate 단계용이다: 후보를 '점수화'하는 순수 추론이라
    tool 바인딩이 없는 편이 JSON 출력을 방해받지 않아 안정적이다. 모델·호스트는 동일하다.
    """
    key = "plain" if tools is None else "tools"
    if key in _llm_cache:
        return _llm_cache[key]
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, num_ctx=NUM_CTX,
                         client_kwargs={"timeout": LLM_TIMEOUT_S})
        _llm_cache[key] = llm.bind_tools(tools) if tools else llm
    except Exception:
        return None
    return _llm_cache[key]


def get_tool_dispatch():
    """tools.TOOL_DISPATCH 로드. 하드웨어 모듈이 없으면 None.

    ImportError 만이 아니라 Exception 전체를 잡는 이유: 하드웨어 도구 모듈이 import 하는
    config.py 는 장비 PC 의 Config.ini 를 읽는데, 파일이 없는 개발 PC 에서는
    NoSectionError(ImportError 아님)가 발생한다.
    """
    try:
        from backend.tools.tools import TOOL_DISPATCH
        return TOOL_DISPATCH
    except Exception:
        return None


def grid_gate_begin_turn(interactive: bool) -> None:
    """그리드 사람-승인 게이트에 턴 시작을 알린다(대화=ON, 벤치마크=OFF).

    하드웨어 모듈 import 가 실패하는 개발 PC 에서는 조용히 무시한다 — 그 경우 grid scan
    자체가 'Hardware not connected' 로 막히므로 게이트는 무의미하다.
    """
    try:
        from backend.tools.hw_tools.hw_tools.acquire_tools import grid_gate_begin_turn as _begin
        _begin(interactive=interactive)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 도구 호출 — 유일한 비-LLM 판단은 조사량 회로차단기뿐
# ══════════════════════════════════════════════════════════════════════════════

#: 하드웨어와 무관한데 TOOL_DISPATCH 에 실을 수 없는 도구 — 그 표는 하드웨어 모듈을
#: import 하므로 Config.ini 가 없으면 통째로 None 이 된다. 여기 있는 것은 그 가드보다
#: 먼저 처리되어 장비 유무와 무관하게 동작한다(FILE_DISPATCH 와 같은 이유).
#: 이름을 _dispatch 안에 문자열로 박아 두면 tools.py 자체 점검이 '구현 없는 도구'로
#: 오판하므로, 목록으로 노출해 둔다.
RUNTIME_DISPATCH = {
    "search_knowledge_base": lambda a: call_with(search_knowledge_base, a),
}


# 레이저가 시편에 조사되는 도구와, 그 도구의 조사 횟수를 args 에서 읽는 방법.
# acquire_spectrum 은 1회, run_grid_scan 은 내부에서 rows*cols 회 조사한다.
_IRRADIATING_TOOLS = {
    "acquire_spectrum": lambda a: 1,
    "run_grid_scan":    lambda a: int(a.get("rows", 0) or 0) * int(a.get("cols", 0) or 0),
}


def call_tool(ctx: dict, name: str, args: dict,
              extra_handlers: Optional[dict] = None) -> dict:
    """단일 도구 호출 지점 — **모든 도구 결과가 지나는 단 하나의 지점**.

    그래서 응답 형식 규약(result.py)을 여기서 강제한다. 도구가 규약을 어긴 것을 돌려주면
    모델이 읽을 수 있는 실패로 바꾼다. 예전에는 ok 키가 빠진 dict 가 성공으로 새어
    나가 사용자 화면(describe_tool)에도, 조사량 누계에도 조용히 잘못 반영됐다.

    실제 디스패치는 _dispatch 가 한다 — 분리해 둬야 나중에 반환 경로가 하나 늘어도
    검사를 우회할 수 없다.
    """
    return normalize(_dispatch(ctx, name, args, extra_handlers), name)


def _dispatch(ctx: dict, name: str, args: dict,
              extra_handlers: Optional[dict] = None) -> dict:
    """도구를 골라 부른다. 유일한 비-LLM '판단'은 조사량 회로차단기뿐.

    회로차단기는 판단이 아니라 물리적 차단이다: 이번 턴의 누적 조사량이 절대 상한을
    넘으면 그 호출 자체를 막는다. 모델의 판단 범위(무엇을 측정할지, 파워를 얼마로 할지)
    에는 전혀 개입하지 않는다 — 총합이 물리적으로 위험한 수준에 닿았을 때만 차단한다.

    extra_handlers 는 그 아키텍처에만 있는 내부 액션이다(CoALA 의 장기기억 4종).
    하드웨어 가드보다 먼저 처리되므로 장비 미연결 상태에서도 동작한다.
    """
    # 내부 액션 → KB → 첨부파일 순으로, 전부 하드웨어 가드보다 먼저. 이 순서가 아니면
    # 하드웨어 미연결 상태에서 KB 조회나 파일 분석까지 "하드웨어 미연결"로 막힌다.
    # run_analysis 도 FILE_DISPATCH 라 순수 계산인 분석이 장비 유무에 묶이지 않는다.
    if extra_handlers and name in extra_handlers:
        return extra_handlers[name](ctx, args)
    if name in RUNTIME_DISPATCH:
        return RUNTIME_DISPATCH[name](args)
    if name in FILE_DISPATCH:
        return FILE_DISPATCH[name](args)

    dispatch = ctx["dispatch"]
    if dispatch is None:
        return fail("Hardware is not connected.")
    fn = dispatch.get(name)
    if fn is None:
        return fail(f"Unknown tool: {name}")

    shots = _IRRADIATING_TOOLS.get(name)
    if shots is None:
        try:
            return fn(dict(args))
        except Exception as e:
            return fail(f"{type(e).__name__}: {e}")

    # power/exposure 를 생략하면 도구가 '현재 장비 설정을 유지'한다. 여기서는 그 값을 알 수
    # 없으므로 기본값으로 근사한다 — 누계가 실제보다 작게 잡힐 수 있지만, 도구 계층
    # (run_grid_scan)과 시료 자체의 상한이 따로 있다.
    power = float(args.get("power") or 40.0)
    exposure = float(args.get("exposure") or 0.2)
    dose_inc = estimate_dose_mj(power, exposure, shots(args))
    if ctx["dose"] + dose_inc > MAX_DOSE_MJ_PER_TURN:
        tail = (" after this grid scan. Reduce the grid size, power, or exposure, or start a new request."
                if name == "run_grid_scan" else
                ". Wrap up the measurement or start again with a new request.")
        return fail(f"Safety block: this turn's cumulative dose would exceed the limit "
                    f"({MAX_DOSE_MJ_PER_TURN} mJ){tail}")
    try:
        result = fn(dict(args))
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")
    # 실패한 조사는 누계에 넣지 않는다(레이저가 안 나갔다). is_ok 는 ok=True 만 성공으로
    # 보므로, 형식을 어긴 결과는 여기서 성공으로 세어지지 않고 call_tool 이 실패로 바꾼다.
    if is_ok(result):
        ctx["dose"] += dose_inc
    return result


def execute_tool(ctx: dict, name: str, args: dict, tool_call_id: str = "",
                 extra_handlers: Optional[dict] = None) -> dict:
    """도구 하나를 실행하고 결과를 축약해 돌려준다. 메시지·이벤트 처리는 호출자 몫.

    반환: {name, args, result, img_b64, question, tool_call_id, elapsed_ms}
    이미지 반환 도구(analyze_microscope_image)의 base64 는 결과에서 떼어낸다 — tool
    메시지에 그대로 실으면 무거워지므로, 호출자가 image_message() 로 별도 전달한다.
    """
    ctx["tool_call_order"].append(name)
    t0 = time.time()
    raw = call_tool(ctx, name, args, extra_handlers)
    elapsed_ms = (time.time() - t0) * 1000.0
    # call_tool 이 응답 형식 규약을 보장하므로 여기부터는 dict 임이 확실하다(isinstance 가드 불필요).
    result = slim(raw)
    img_b64 = result.pop("image_base64", None)
    question = result.pop("question", None)
    return {"name": name, "args": args, "result": result, "img_b64": img_b64,
            "question": question, "tool_call_id": tool_call_id, "elapsed_ms": elapsed_ms}


# ══════════════════════════════════════════════════════════════════════════════
# 메시지 / 히스토리
# ══════════════════════════════════════════════════════════════════════════════

#: 이미지 주입용으로 끼워 넣은 HumanMessage 표시. 이 메시지는 '사용자 턴'이 아니므로
#: 히스토리 트리밍의 턴 계산에서 제외해야 한다.
INJECTED_IMAGE = "_injected_image"

#: 세션 히스토리에 보존할 최대 사용자 턴 수.
#: [100 → 30] 이건 서버 RAM 이 아니라 '매 호출 프롬프트에 실리는 토큰 수'(=num_ctx 예산)를
#: 좌우한다. 100 은 34번째 문항에서 앞 문항 맥락이 통째로 누적돼 컨텍스트가 폭주, 무응답을
#: 냈다. 30 이면 대부분 소형인 문항 30턴이 예산 안에 들어오고, 되묻기(원 요청 + 확인응답
#: 여러 회)에 필요한 직전 맥락도 넉넉히 유지된다.
HISTORY_MAX_TURNS = 30


def text_of(msg) -> str:
    """AIMessage 의 content 에서 순수 텍스트를 뽑는다.

    [왜 msg.content 를 그냥 쓰지 않는가]
    LangChain 메시지의 content 는 str 일 수도, 콘텐츠 블록 리스트
    ([{"type":"text","text":...}, ...])일 수도 있다. 어느 쪽이 오는지는 langchain-core
    버전과 모델 어댑터에 따라 달라지므로 둘 다 처리한다.
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


def image_message(img_b64: str, question: str = "") -> HumanMessage:
    """모델에게 이미지를 '보여주는' user 메시지.

    도구 결과(ToolMessage)가 아니라 별도 user 메시지로 보내는 이유: 모델이 실제로 보고
    판단하게 하면서도 tool 메시지 자체는 가볍게 유지하기 위해서다. ChatOllama 는
    image_url 콘텐츠 블록을 ollama API 의 images 필드로 변환하며 data URI 의 base64
    부분만 잘라 보내므로 접두사가 필요하다.
    """
    return HumanMessage(
        content=[{"type": "text", "text": question or "Microscope camera image:"},
                 {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"}],
        # 사람이 친 게 아니라 시스템이 끼워 넣은 메시지 — 트리밍의 턴 계산에서 뺀다.
        additional_kwargs={INJECTED_IMAGE: True},
    )


def trim_history(messages: list) -> list:
    """다음 턴으로 넘길 히스토리를 만든다 — 오래된 턴을 자르고, 주입 이미지는 전부 걷어낸다.

    턴이 끝날 때 한 번만 불린다(stream_turn). 따라서 **턴 안에서는 이미지가 몇 장이든 그대로
    살아 있고**, 다음 턴으로는 한 장도 넘어가지 않는다.

    [왜 사용자 메시지 경계에서만 자르는가]
    ToolMessage 는 자신을 유발한 AIMessage(tool_calls) 뒤에 와야만 유효하다. 임의 지점에서
    자르면 짝 잃은 ToolMessage 가 맨 앞에 남아 API 가 거부한다. 사용자 메시지 경계에서
    자르면 그 앞의 AIMessage+ToolMessage 쌍이 통째로 함께 사라지므로 항상 안전하다.

    이미지 주입용 HumanMessage 는 사용자 턴이 아니다 — 포함시키면 이미지 분석을 여러 번
    하는 세션에서 한 턴이 여러 턴으로 세어져 히스토리가 과하게 잘린다.

    [왜 이미지를 한 장도 안 남기는가 — 2026-08-12]
    현미경 캡처 1장이 base64 2,543,820자다. 히스토리에 남으면 **세션이 끝날 때까지 매
    LLM 호출마다 통째로 재전송된다**(실측: 한 세션에서 8회 ≈ 20MB). 한 장만 남겨도 그 비용은
    세션 내내 고정으로 붙는다. 컨텍스트 창 문제는 아니다 — Ollama 비전 인코더가 이미지를
    ~300 토큰으로 바꾸므로 num_ctx 소모는 미미하다(실측: 주입 전후 18,505 → 19,030 토큰).
    아까운 건 네트워크와 RAM 이다.

    [왜 base64 문자열만 자리표시로 바꾸지 않는가]
    ChatOllama 는 image_url 의 쉼표 뒷부분을 그대로 ollama API 의 images 필드에 넣는다
    (langchain-ollama 1.1.0 에서 확인). 자리표시 문자열을 남기면 그게 **진짜 이미지
    데이터로 취급되어** 서버가 디코드를 시도한다. 메시지를 통째로 버리면 images 필드
    자체가 안 생겨 그 위험이 없다.

    [버려도 정보가 사라지지 않는 이유 — 히스토리는 사본이 아니라 포인터를 나른다]
    이미지를 낸 도구가 결과에 image_file 을 실어 두고, 그 ToolMessage 는 히스토리에 그대로
    남는다(가볍다 — 164자). 모델은 open_file(image_file) 로 언제든 되돌아갈 수 있다.
    그래서 여기서 버리는 것은 '기억'이 아니라 '사본'이다.
    """
    def is_user_turn(m) -> bool:
        return isinstance(m, HumanMessage) and not m.additional_kwargs.get(INJECTED_IMAGE, False)

    def is_injected_image(m) -> bool:
        return isinstance(m, HumanMessage) and bool(m.additional_kwargs.get(INJECTED_IMAGE, False))

    user_idx = [i for i, m in enumerate(messages) if is_user_turn(m)]
    if len(user_idx) > HISTORY_MAX_TURNS:
        messages = messages[user_idx[-HISTORY_MAX_TURNS]:]

    # 주입 이미지는 tool_call 짝이 없는 독립 HumanMessage 라, 빼도 짝이 깨지지 않는다
    # (위 '사용자 메시지 경계' 규칙이 지키려는 그 불변식과 무관하다).
    return [m for m in messages if not is_injected_image(m)]


# ══════════════════════════════════════════════════════════════════════════════
# 진행 문구 — SSE "node" 이벤트에 실리는 사람용 한 줄
# ══════════════════════════════════════════════════════════════════════════════

def describe_tool(name: str, args: dict, result: dict) -> str:
    """도구 호출 1건을 사람이 읽는 한 줄로 요약한다.

    장기기억 도구(recall_*/record_*)는 CoALA 에서만 호출되지만 분기를 함께 둔다 —
    ReAct 에서는 그냥 닿지 않는 가지일 뿐이고, 표현이 두 벌로 갈라지는 것보다 낫다.
    """
    if not isinstance(result, dict):
        return f"🔧 {name} called"
    if not result.get("ok", True):
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
    if name == "preview_grid_scan":
        return (f"🔲 Grid preview {result.get('rows', '?')}×{result.get('cols', '?')} "
                f"({result.get('n_in_view', '?')}/{result.get('n_points', '?')} in view)")
    if name == "run_grid_scan":
        return (f"🗺️ Grid scan done "
                f"({result.get('n_measured', '?')}/{result.get('n_points', '?')} points)")
    if name == "apply_background_subtraction":
        return "🧹 Fluorescence background subtraction applied"
    if name == "search_knowledge_base":
        hits = result.get("results", [])
        titles = ", ".join(h.get("title", "?") for h in hits) if hits else \
            f"no match for '{args.get('query', '')}'"
        return f"📚 Knowledge (semantic) lookup → {titles}"
    if name == "recall_experiences":
        return f"🧠 Experience (episodic) lookup → {len(result.get('results', []))} item(s)"
    if name == "recall_insights":
        return f"🔎 Insight (semantic) lookup → {len(result.get('results', []))} item(s)"
    if name == "record_experience":
        return f"💾 Experience recorded (episodic) → {result.get('sample', '')}"
    if name == "record_insight":
        return f"💡 Insight recorded (semantic) → {result.get('topic', '')}"
    if name == "list_uploaded_files":
        files = result.get("files", [])
        if not files:
            return "📎 Attached files — none"
        return f"📎 Attached files → {', '.join(f.get('filename', '?') for f in files)}"
    if name == "open_file":
        # 한 도구가 종류마다 다른 payload 를 내므로 kind 로 갈라 적는다 — 안 그러면
        # 표를 열든 그림을 열든 화면에 "open_file called" 한 줄만 뜬다.
        kind = result.get("kind")
        if kind == "image":
            return f"🖼️ Opened image {result.get('file_id', '?')}"
        if kind == "table":
            return (f"🔍 Opened {result.get('filename', '?')} "
                    f"({result.get('n_rows', '?')} rows × {result.get('n_cols', '?')} cols)")
        return f"📄 Opened {result.get('file_id', '?')}"
    if name == "run_analysis":
        return f"🧮 Analysis code executed ({result.get('image_count', 0)} figure(s))"
    return f"🔧 {name} called"


# ══════════════════════════════════════════════════════════════════════════════
# LLM 호출 실패를 사용자에게 설명하는 문구
# ══════════════════════════════════════════════════════════════════════════════

LLM_NOT_CONNECTED = ("Ollama LLM is not connected. "
                     "(Check that langchain-ollama is installed and the Ollama server is running)")

#: 모델이 텍스트도 도구 호출도 내지 않고 턴을 끝냈을 때 사용자에게 보이는 문구.
#: [왜 상수인가] AILA·CoALA 두 곳에서 같은 문장을 냈다. 두 에이전트는 서로를 import 하지
#: 않으므로(비교 실험의 독립변수는 오케스트레이션 하나여야 한다) 공용 자리는 여기다.
#: [왜 이렇게 긴가] 옛 문구는 "Failed to generate a response." 뿐이라, 받은 사람이 할 수
#: 있는 일이 없었다. 원인은 로그의 Gemma EMPTY-REPLY 항목에 원문째로 남으므로
#: (reason_log._empty_reply_dump) 어디를 봐야 하는지 알려 준다.
EMPTY_REPLY = ("The model ended its turn without producing an answer or a tool call. "
               "This is usually a malformed tool call that could not be parsed - it is not "
               "something you did wrong. Try asking again, and if it repeats, check the "
               "'Gemma EMPTY-REPLY' entry in DetailLog/reasoning/ for the raw response.")


def llm_error_detail(e: Exception, stage: str = "LLM call") -> str:
    """LLM 호출 예외를 사람이 읽는 한 줄로.

    타임아웃은 "모델이 틀렸다"가 아니라 "응답이 아예 안 왔다"이므로 구분해서 알린다 —
    로그만 보고 Ollama/네트워크 쪽을 봐야 한다는 걸 알 수 있게.
    """
    if "timeout" in type(e).__name__.lower() or "timeout" in str(e).lower():
        return (f"No response from the LLM within {LLM_TIMEOUT_S:.0f}s ({type(e).__name__}). "
                f"The Ollama server may have dropped the request or be overloaded - check that "
                f"{OLLAMA_HOST} is healthy and retry.")
    return f"{stage} failed: {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 세션 껍데기 — 공개 API(stream_experiment / run_experiment)의 공통 부분
# ══════════════════════════════════════════════════════════════════════════════
#
# 두 에이전트의 진입점은 '어떤 run_stream 을 부르는가' 하나만 달랐다. 나머지(세션 id 확정,
# 로그 턴 열기, 산출물 폴더 열기, 그리드 게이트, 이벤트 번역, 히스토리 저장, 측정 여부
# 판정)는 한 글자도 다르지 않았으므로 여기 한 벌만 둔다.

#: 세션별 LangChain 메시지 히스토리. {session_id: [BaseMessage, ...]}
#: 로컬 단일 사용자 도구라 in-memory dict 로 충분하다(프로세스 종료 시 초기화).
#: 에이전트마다 별도 dict 를 갖도록 각 파일이 자기 것을 만들어 넘긴다.
SessionStore = dict


def _used_measurement(ctx: Optional[dict]) -> bool:
    """이번 턴에 레이저 조사가 실제로 있었는지 — '실험 보고서' vs '일반 대화'를 가른다."""
    return bool(set(_IRRADIATING_TOOLS) & set((ctx or {}).get("tool_call_order", [])))


def stream_turn(arch: str, sessions: SessionStore, run: Callable[[list, str], Iterator[dict]],
                user_message: str, session_id: str = "",
                interactive_grid_gate: bool = False) -> Iterator[dict]:
    """에이전트 한 턴을 프론트엔드 SSE 이벤트로 흘린다.

    run(history, sid) 가 아키텍처별 루프다(ReAct 루프 또는 CoALA 사이클). 그 루프가 내는
    저수준 이벤트(tool/phase/final/error)를 프론트 계약 이벤트로 번역한다:
      {"type": "node",  "node": str, "message": str}   진행상황
      {"type": "chat",  "reply": str}                  측정 없이 끝난 턴
      {"type": "done",  "final_report": str}           측정을 포함한 턴 완료
      {"type": "error", "detail": str}
    모든 이벤트에 session_id 가 실린다.
    """
    sid = session_id or str(uuid.uuid4())

    def ev(d: dict) -> dict:
        d["session_id"] = sid
        return d

    # 벤치마크 로그: resolved sid 를 넘겨야 서로 다른 세션이 'nosession' 파일 하나로 뭉치지
    # 않는다. run 소비 전에 열어 첫 이벤트부터 관측한다(로깅 실패는 detail_log 가 삼킨다).
    turn = new_turn(arch, sid, user_message)
    # 이 턴의 산출물이 갈 세션 폴더를 연다(data/runs/<sid>/). 같은 sid 로 다시 불러도 같은
    # 폴더를 이어 쓰므로 멀티턴 세션의 산출물이 한곳에 모인다.
    # isolated=False — 대화에서는 지난 세션 결과를 물어보는 것이 정상적인 사용이다.
    run_store.begin_session(sid, arch, isolated=False)

    try:
        history = sessions.get(sid, [])
        grid_gate_begin_turn(interactive=interactive_grid_gate)

        final_text, final_ctx, final_messages = None, None, history
        for event in run(history, sid):
            turn.observe(event)
            etype = event["type"]
            if etype == "phase":                      # CoALA 사이클 단계 (ReAct 는 안 낸다)
                yield ev({"type": "node", "node": f"cycle:{event['phase']}",
                          "message": event["message"]})
            elif etype == "tool":
                yield ev({"type": "node", "node": event["name"],
                          "message": describe_tool(event["name"], event["args"], event["result"])})
                sp = spectrum_event(event["result"])   # 측정이면 스펙트럼 이미지도 전달
                if sp:
                    yield ev(sp)
            elif etype == "error":
                turn.fail(event["detail"])
                yield ev({"type": "error", "detail": event["detail"]})
                return
            elif etype == "final":
                final_text, final_ctx = event["text"], event["ctx"]
                final_messages = event["messages"]

        if final_ctx is None:
            turn.fail("The agent failed to generate a response.")
            yield ev({"type": "error", "detail": "The agent failed to generate a response."})
            return

        sessions[sid] = trim_history(final_messages)
        measured = _used_measurement(final_ctx)
        turn.complete("done" if measured else "chat", final_text, final_ctx)
        yield ev({"type": "done", "final_report": final_text} if measured
                 else {"type": "chat", "reply": final_text})

    except Exception as e:
        turn.fail(str(e))
        yield ev({"type": "error", "detail": str(e)})


def run_turn_once(arch: str, run: Callable[[list, str], Iterator[dict]],
                  user_message: str, session_id: str = "") -> dict:
    """동기 1회 실행 — 벤치마크/레거시용 (세션 히스토리 없이 매번 새로 시작).

    빈 session_id 면 매 실행마다 uuid 를 새로 만들어 실행 1회 = 로그 파일 1개로 분리한다
    (안 그러면 모든 벤치마크 질의가 'nosession' 한 파일에 뭉친다).
    """
    sid = session_id or str(uuid.uuid4())
    turn = new_turn(arch, sid, user_message)
    # isolated=True — 벤치마크는 문항마다 새 세션이고, 앞 문항이 남긴 파일을 읽으면 점수가
    # 문항 순서에 의존한다(run_store.isolated_label 참고). 이 경로는 벤치 전용이다.
    run_store.begin_session(sid, arch, isolated=True)
    # 벤치마크는 사람이 없는 자율 평가 — 그리드 승인 게이트를 끈다(안 끄면 모든 격자
    # 스캔이 승인 없이 거부된다).
    grid_gate_begin_turn(interactive=False)

    final_text, final_ctx, error_detail = "", None, None
    for event in run([], sid):
        turn.observe(event)
        if event["type"] == "final":
            final_text, final_ctx = event["text"], event["ctx"]
        elif event["type"] == "error":
            error_detail = event["detail"]
            final_text = f"[Error] {event['detail']}"

    if error_detail is not None:
        turn.fail(error_detail, final_ctx)
    else:
        turn.complete("done" if _used_measurement(final_ctx) else "chat", final_text, final_ctx)
    return {"final_report": final_text}


__all__ = [
    # LLM · 도구
    "get_chat_model", "get_tool_dispatch", "grid_gate_begin_turn",
    "search_knowledge_base", "call_tool", "execute_tool", "MAX_DOSE_MJ_PER_TURN",
    # 메시지 · 히스토리
    "INJECTED_IMAGE", "HISTORY_MAX_TURNS", "text_of", "image_message", "trim_history",
    # 진행 문구 · 오류 문구
    "describe_tool", "LLM_NOT_CONNECTED", "EMPTY_REPLY", "llm_error_detail",
    # 세션 껍데기
    "SessionStore", "stream_turn", "run_turn_once",
]
