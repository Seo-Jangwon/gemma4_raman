"""
ExperimentState — LangGraph 공유 상태 정의.

모든 에이전트 노드는 이 TypedDict의 부분 집합을 반환해 상태를 갱신한다.
append-only 필드(observations, critic_log)는 Annotated[list, operator.add] reducer를 사용.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class ClarifiedIntent(TypedDict):
    primary_objective: str
    sample_type: str           # persona binding 트리거 (domain_specialist)
    success_criteria: list[str]
    constraints: dict          # max_laser_power_pct, max_exposure_s, x, y, z 등
    user_preferences: dict
    raw_user_message: str


class PlanStep(TypedDict):
    step_id: str
    agent: str                 # "hw_manager"|"spectrum_specialist"|"domain_specialist"|...
    action: str
    params: dict
    status: str                # "pending"|"running"|"done"|"failed"|"skipped"
    result: Optional[dict]


class CriticLogEntry(TypedDict):
    checkpoint: str            # "C1"~"C5"
    verdict: str               # "APPROVE"|"WARNING"|"ABORT"
    reason: str
    timestamp: float
    tier: str                  # "A" (HARD VETO) | "B" (SOFT) | "none"


class ExperimentState(TypedDict):
    # ── 입력 ──────────────────────────────────────────────────────────────────
    user_message: str
    session_id: str

    # ── Translator 출력 ───────────────────────────────────────────────────────
    intent: Optional[ClarifiedIntent]

    # ── Planner 제어 ─────────────────────────────────────────────────────────
    plan: list[PlanStep]
    current_step_idx: int      # 현재 실행 중인 plan 인덱스
    replan_count: int
    next_node: str             # Planner가 다음 라우팅 목적지를 여기에 씀
    critic_checkpoint: str     # "C1"~"C5", Planner가 critic 라우팅 시 설정

    # ── 실행 결과 (append-only reducer) ──────────────────────────────────────
    observations: Annotated[list[dict], operator.add]
    critic_log: Annotated[list[CriticLogEntry], operator.add]

    # ── 하드웨어 상태 ─────────────────────────────────────────────────────────
    cumulative_dose_mj: float
    stage_position: Optional[dict]

    # ── Specialist 출력 ───────────────────────────────────────────────────────
    spectrum_analysis: Optional[str]
    domain_interpretation: Optional[str]
    debate_result: Optional[str]

    # ── RAG / ROI ─────────────────────────────────────────────────────────────
    rag_results: list[dict]
    next_roi: Optional[dict]

    # ── 종료 제어 ─────────────────────────────────────────────────────────────
    abort_reason: Optional[str]   # Critic Tier-A HARD VETO 시 설정
    final_report: Optional[str]


def initial_state(user_message: str, session_id: str = "") -> ExperimentState:
    """서버에서 그래프를 invoke할 때 사용할 초기 상태를 반환한다."""
    import uuid
    return ExperimentState(
        user_message=user_message,
        session_id=session_id or str(uuid.uuid4()),
        intent=None,
        plan=[],
        current_step_idx=0,
        replan_count=0,
        next_node="",
        critic_checkpoint="",
        observations=[],
        critic_log=[],
        cumulative_dose_mj=0.0,
        stage_position=None,
        spectrum_analysis=None,
        domain_interpretation=None,
        debate_result=None,
        rag_results=[],
        next_roi=None,
        abort_reason=None,
        final_report=None,
    )
