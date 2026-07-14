"""
LangGraph StateGraph 정의 — Hub-and-spoke 토폴로지.

Planner가 허브: 모든 스포크 에이전트는 완료 후 Planner로 복귀.
Critic의 Tier-A (HARD VETO) ABORT만 즉시 END로 라우팅.

[토폴로지를 hub-and-spoke로 유지하는 이유]
스포크 간 직접 edge(예: hw_manager → spectrum_specialist)를 만들면 그래프는
짧아지지만, 실패 처리·품질 게이트·재계획 같은 제어 로직이 각 스포크에 흩어진다.
모든 제어 판단을 Planner 한 곳에 모으면:
  - step 실패 → on_fail 정책(retry/replan/skip/abort)을 Planner가 일괄 적용
  - 스펙트럼 획득 step 완료 → Planner가 C3 품질 게이트를 강제 삽입
  - 흐름 전체가 state.plan 하나로 관측 가능 (디버깅/벤치마크 용이)
왕복 비용은 Planner 노드가 대부분 LLM 없는 결정적 분기라 무시할 수준이다.

[Critic 라우팅 정책]
  - Tier-A ABORT (C2 하드웨어 안전): 즉시 END — 어떤 복구도 시도하지 않는다.
  - 그 외 (C1 계획 거부 포함): Planner로 복귀 — Planner가 재계획/재시도/수용을
    결정한다. C1 ABORT는 tier-B로 내려 재계획 기회를 준다 (critic.py 참고).
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
