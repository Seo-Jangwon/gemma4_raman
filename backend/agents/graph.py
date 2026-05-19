"""
LangGraph StateGraph 정의 — Hub-and-spoke 토폴로지.

Planner가 허브: 모든 스포크 에이전트는 완료 후 Planner로 복귀.
Critic의 Tier-A (HARD VETO) ABORT만 즉시 END로 라우팅.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from backend.agents.state import ExperimentState

# 에이전트 노드 함수들은 지연 import (순환 방지 + 서버 시작 시간 단축)
def _import_nodes():
    from backend.agents.translator       import translator_node
    from backend.agents.planner          import planner_node
    from backend.agents.critic           import critic_node
    from backend.agents.hw_manager       import hw_manager_node
    from backend.agents.spectrum_specialist import spectrum_specialist_node
    from backend.agents.domain_specialist   import domain_specialist_node
    from backend.agents.rag_searcher     import rag_searcher_node
    from backend.agents.roi_detector     import roi_detector_node
    from backend.agents.debate           import debate_node
    from backend.agents.planner          import report_generator_node
    return {
        "translator":          translator_node,
        "planner":             planner_node,
        "critic":              critic_node,
        "hw_manager":          hw_manager_node,
        "spectrum_specialist": spectrum_specialist_node,
        "domain_specialist":   domain_specialist_node,
        "rag_searcher":        rag_searcher_node,
        "roi_detector":        roi_detector_node,
        "debate":              debate_node,
        "report_generator":    report_generator_node,
    }


# ── 라우팅 함수 ───────────────────────────────────────────────────────────────

def _route_from_planner(state: ExperimentState) -> str:
    if state.get("abort_reason"):
        return END
    nxt = state.get("next_node", "")
    valid = {
        "rag_searcher", "hw_manager", "critic",
        "spectrum_specialist", "domain_specialist",
        "roi_detector", "debate", "report_generator",
    }
    if nxt not in valid:
        return END
    return nxt


def _route_after_critic(state: ExperimentState) -> str:
    log = state.get("critic_log", [])
    if log:
        last = log[-1]
        if last["verdict"] == "ABORT" and last["tier"] == "A":
            return END
    return "planner"


# ── 그래프 빌드 ───────────────────────────────────────────────────────────────

def build_graph():
    """StateGraph를 조립하고 compile된 Runnable을 반환한다."""
    nodes = _import_nodes()

    g = StateGraph(ExperimentState)

    for name, fn in nodes.items():
        g.add_node(name, fn)

    # 진입점
    g.set_entry_point("translator")
    g.add_edge("translator", "planner")

    # Planner → 스포크 (조건부)
    g.add_conditional_edges(
        "planner",
        _route_from_planner,
        {
            "rag_searcher":        "rag_searcher",
            "hw_manager":          "hw_manager",
            "critic":              "critic",
            "spectrum_specialist": "spectrum_specialist",
            "domain_specialist":   "domain_specialist",
            "roi_detector":        "roi_detector",
            "debate":              "debate",
            "report_generator":    "report_generator",
            END:                   END,
        },
    )

    # 스포크 → Planner 복귀 (Critic 제외)
    for spoke in [
        "rag_searcher", "hw_manager",
        "spectrum_specialist", "domain_specialist",
        "roi_detector", "debate", "report_generator",
    ]:
        g.add_edge(spoke, "planner")

    # Critic → 조건부 (HARD VETO → END, 나머지 → Planner)
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"planner": "planner", END: END},
    )

    return g.compile()
