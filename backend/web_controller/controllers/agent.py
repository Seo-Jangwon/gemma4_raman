"""
에이전트 라우터 — 프론트 채팅창이 실제로 부르는 곳.

  POST /api/experiment/run     동기 1회 실행 (세션 히스토리 없음, 레거시/디버깅용)
  POST /api/experiment/stream  SSE 스트리밍 — 프론트 채팅의 기본 경로
  GET  /api/agents/health      어떤 모델·도구 수로 바인딩됐는지

AILA / CoALA 선택은 core/agent_module.py 한 곳에서 한다.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import backend.llm_config as llm_config
from backend.web_controller.setups.agent_module import select_agent_module
from backend.web_controller.setups.state import StateDep, run_in_worker
from backend.web_controller.setups.streaming import sse_response
from backend.web_controller.schemas.requests import ExperimentRequest

router = APIRouter(prefix="/api", tags=["agent"])


@router.post("/experiment/run")
async def experiment_run(body: ExperimentRequest, state: StateDep):
    """단일 에이전트 동기 실행 (세션 히스토리 없이 1회, 벤치마크/레거시용)."""
    mod, _ = select_agent_module(body.agent)
    try:
        return await run_in_worker(
            state, lambda: mod.run_experiment(body.message, body.session_id or ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiment/stream")
async def experiment_stream(body: ExperimentRequest, state: StateDep) -> StreamingResponse:
    """단일 에이전트 — SSE 스트리밍(도구 호출 진행상황 + 세션 대화 기억).

    프론트엔드는 fetch 로 이 엔드포인트를 열고 ReadableStream 을 파싱한다.
    이벤트 형식은 표준 SSE 이고, type ∈ {chat, node, spectrum, clarification, done, error}
    이다(single_agent_*.stream_experiment 참고).
    """
    mod, _ = select_agent_module(body.agent)

    def produce(emit):
        # 워커 스레드: 동기 제너레이터를 소비해 이벤트를 밀어넣는다.
        try:
            for event in mod.stream_experiment(body.message, body.session_id or ""):
                emit(event)
        except Exception as e:  # 방어적 — stream_experiment 가 자체적으로 error 이벤트를 내지만
            emit({"type": "error", "detail": str(e)})

    return sse_response(state, produce)


@router.get("/agents/health")
async def agents_health(agent: str = "AILA"):
    """단일 에이전트 시스템 상태 확인. agent 쿼리로 AILA/CoALA 중 선택(기본 AILA)."""
    mod, name = select_agent_module(agent)
    return {
        "status": "ok",
        "agents": [f"single_agent_{name}"],
        # 실제로 바인딩된 모델을 그대로 보고한다. 예전에는 "gemma4 (ollama)" 라는
        # 고정 문자열이라, 모델을 바꿔 돌려도 health 는 늘 같은 답을 했다 —
        # 어느 모델로 벤치를 돌렸는지 사후에 확인할 방법이 없었다.
        "model": mod.OLLAMA_MODEL,
        "host": mod.OLLAMA_HOST,
        "num_ctx": llm_config.NUM_CTX,
        "tools": len(mod.ALL_TOOLS),
    }
