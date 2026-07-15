# -*- coding: utf-8 -*-
"""
LangGraph StateGraph 정의 — Hub-and-spoke 토폴로지 (4노드).

    planner (허브) ⇄ { hw_manager, analyst, critic }

Planner가 허브: 모든 스포크 에이전트는 완료 후 Planner로 복귀.
Critic의 Tier-A (HARD VETO) ABORT만 즉시 END로 라우팅.

[에이전트 통합 노트 — 왜 노드가 4개인가]
과거 10노드(translator/planner/critic/hw_manager/spectrum_specialist/
domain_specialist/rag_searcher/roi_detector/debate/report_generator)를
"계획 → 실행 → 해석 → 검증" 역할 하나당 노드 하나로 재편했다:
  - planner  : 계획 + 제어 + KB 검색(구 rag_searcher) + 보고서(구 report_generator)
  - hw_manager: 하드웨어를 만지는 모든 것 — 위치 탐색(구 roi_detector) + 측정 + 배경
  - analyst  : 해석 전부 — 물리 분석 + 도메인 해석 + 교차검증
               (구 spectrum_specialist + domain_specialist + debate)
  - critic   : 독립 검증자 — C1(계획)~C5(보고서). 해석자와 검증자는 분리 유지.
translator는 그래프 밖으로: intent는 orchestrator가 항상 사전 번역해 주입하므로
그래프 안에서 재번역할 일이 없다 (clarification 게이트도 그래프 밖에서 실행).
노드 수가 줄어도 기능은 그대로다 — 흡수된 역할은 task 플래그/내부 함수가 되어
ablation 시 "에이전트 제거"가 아니라 "플래그 끄기"로 통제할 수 있다.

[토폴로지를 hub-and-spoke로 유지하는 이유]
스포크 간 직접 edge(예: hw_manager → analyst)를 만들면 그래프는 짧아지지만,
실패 처리·품질 게이트·재계획 같은 제어 로직이 각 스포크에 흩어진다.
모든 제어 판단을 Planner 한 곳에 모으면:
  - step 실패 → on_fail 정책(retry/replan/skip/abort)을 Planner가 일괄 적용
  - 스펙트럼 획득 step 완료 → Planner가 C3 품질 게이트를 강제 삽입
  - 흐름 전체가 state.plan 하나로 관측 가능 (디버깅/벤치마크 용이)
왕복 비용은 Planner 노드가 대부분 LLM 없는 결정적 분기라 무시할 수준이다.

[planner self-edge]
planner → planner 라우팅이 두 곳에서 쓰인다:
  - C1이 계획을 거부했을 때 (실행 전): plan을 비우고 재진입해 재생성
  - 마지막 step 전진 후 C4까지 끝났을 때: 재진입해 보고서 생성(내장) 시작
LangGraph에서 conditional edge의 self-loop는 정상 동작이며, 두 경우 모두
다음 재진입에서 반드시 다른 분기(critic 라우팅)로 빠지므로 무한 루프가 없다.

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
    from backend.agents.planner    import planner_node
    from backend.agents.critic     import critic_node
    from backend.agents.hw_manager import hw_manager_node
    from backend.agents.analyst    import analyst_node
    return {
        "planner":    planner_node,
        "critic":     critic_node,
        "hw_manager": hw_manager_node,
        "analyst":    analyst_node,
    }


# ── 라우팅 함수 ───────────────────────────────────────────────────────────────

def _route_from_planner(state: ExperimentState) -> str:
    if state.get("abort_reason"):
        return END
    nxt = state.get("next_node", "")
    valid = {"hw_manager", "analyst", "critic", "planner"}
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

    # 진입점 — intent는 orchestrator가 사전 주입하므로 곧바로 계획부터 시작
    g.set_entry_point("planner")

    # Planner → 스포크 (조건부, self-edge 포함)
    g.add_conditional_edges(
        "planner",
        _route_from_planner,
        {
            "hw_manager": "hw_manager",
            "analyst":    "analyst",
            "critic":     "critic",
            "planner":    "planner",
            END:          END,
        },
    )

    # 스포크 → Planner 복귀 (Critic 제외)
    g.add_edge("hw_manager", "planner")
    g.add_edge("analyst", "planner")

    # Critic → 조건부 (HARD VETO → END, 나머지 → Planner)
    g.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"planner": "planner", END: END},
    )

    return g.compile()
