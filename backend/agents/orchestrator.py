"""
ExperimentOrchestrator — LangGraph 그래프 실행 entry point.

서버 시작 시 build_graph()를 1회 compile해 _graph에 저장.
run_experiment()는 ThreadPoolExecutor에서 동기 실행된다.
"""

from __future__ import annotations

from backend.agents.graph import build_graph
from backend.agents.state import ExperimentState, initial_state

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_experiment(user_message: str, session_id: str = "") -> dict:
    """
    멀티에이전트 실험 파이프라인 실행.
    반환값: ExperimentState 최종 상태 dict
    """
    state = initial_state(user_message=user_message, session_id=session_id)
    graph = _get_graph()
    result: ExperimentState = graph.invoke(state)
    return dict(result)
