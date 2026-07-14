"""
ExperimentState — LangGraph 공유 상태 정의.

모든 에이전트 노드는 이 TypedDict의 부분 집합을 반환해 상태를 갱신한다.
append-only 필드(observations, critic_log, failure_log)는
Annotated[list, operator.add] reducer를 사용한다.

[설계 의도]
- 타겟/기판 복잡성 대응을 위해 다음 필드를 추가했다:
  * acquisition_params  : 물질별 최적 레이저 파워·노출을 "측정하면서" 찾아가는
                          적응형 튜닝 결과를 보관. 이후 step(배경 측정 등)이
                          동일 조건을 재사용해 스펙트럼 간 비교 가능성을 보장한다.
  * background_reference: 기판(substrate) 배경 스펙트럼 참조. 타겟 스펙트럼에서
                          기판 유래 피크를 분리할 때 spectrum_specialist가 사용.
  * next_roi            : roi_detector가 결정한 타겟 위치 + 기판 배경 측정 위치.
- 안정적 failure 처리를 위해 다음 필드를 추가했다:
  * failure_log         : 각 step의 실패 원인을 구조화 기록. Planner가 재계획 시
                          LLM 프롬프트에 넣어 "같은 실패를 반복하지 않는" 계획을
                          만들도록 한다. (append-only — 노드가 동시에 써도 안전)
  * retry_map           : step_id별 재시도 횟수. 무한 재시도 루프 방지의 유일한
                          근거이므로 plan step 내부가 아닌 상태 최상위에 둔다.
                          (plan은 재계획 시 통째로 교체될 수 있어 step 내부 카운터는
                          유실될 위험이 있다.)
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class ClarifiedIntent(TypedDict):
    primary_objective: str
    sample_type: str           # persona binding 트리거 (domain_specialist)
    substrate: str             # 기판 종류 (예: "glass", "sio2 wafer", "gold film").
                               # — 기판마다 배경 신호가 다르므로 경험 저장소의
                               #   에피소드 컨텍스트 키로 사용된다 (experience.py).
    target_description: str    # 시각 탐색용 타겟 외형 설명 (예: "둥근 어두운 세포 덩어리")
                               # — 타겟 위치를 모르는 경우 roi_detector visual_search가
                               #   vision LLM에게 "무엇을 찾을지" 알려주는 근거가 된다.
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
    on_fail: str               # 실패 정책: "retry"|"replan"|"skip"|"abort"
                               # — step마다 실패의 의미가 다르기 때문에 정책을 계획
                               #   시점에 명시한다. 예: 하드웨어 일시 오류는 retry,
                               #   타겟 탐색 실패는 계획 자체를 바꿔야 하므로 replan,
                               #   보조 분석 실패는 skip해도 실험이 유효하다.


class CriticLogEntry(TypedDict):
    checkpoint: str            # "C1"~"C5"
    verdict: str               # "APPROVE"|"WARNING"|"ABORT"
    reason: str
    timestamp: float
    tier: str                  # "A" (HARD VETO) | "B" (SOFT) | "none"
    suggestion: dict           # 구조화된 보정 제안 (예: {"power_scale": 0.5}).
                               # — reason은 사람이 읽는 문장이라 Planner가 기계적으로
                               #   파라미터를 보정할 수 없다. Critic이 "무엇을 얼마나
                               #   바꿔야 하는지"를 숫자로 주면 Planner는 LLM 호출 없이
                               #   결정적으로(deterministic) 재측정 파라미터를 만든다.


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

    # ── Failure 처리 ──────────────────────────────────────────────────────────
    # 실패 기록은 append-only: 재계획 후에도 과거 실패 이력이 보존되어야
    # "같은 방식으로 또 실패"하는 것을 LLM 재계획 프롬프트에서 막을 수 있다.
    failure_log: Annotated[list[dict], operator.add]
    retry_map: dict            # {step_id: 재시도 횟수} — Planner만 갱신 (단일 작성자)

    # ── 하드웨어 상태 ─────────────────────────────────────────────────────────
    cumulative_dose_mj: float
    cumulative_dose_map: dict       # {"x.x_y.y": float_mJ} 위치별 누적 dose
    stage_position: Optional[dict]

    # ── 적응형 측정 파라미터 ──────────────────────────────────────────────────
    # hw_manager의 적응형 튜닝(acquire_target)이 확정한 파라미터.
    # {"power_pct": float, "exposure_s": float, "tuned": bool, "history": [...]}
    # 배경 측정(acquire_background)이 이 값을 그대로 사용해 타겟↔기판 스펙트럼의
    # 측정 조건을 일치시킨다 (조건이 다르면 강도 비교가 무의미).
    acquisition_params: dict

    # ── 기판 배경 참조 ────────────────────────────────────────────────────────
    # {"position": {...}, "max_intensity": float, "summary": "wn:val, ...",
    #  "power_pct": float, "exposure_s": float}
    # spectrum_specialist가 타겟 스펙트럼과 나란히 놓고 기판 유래 피크를 배제한다.
    background_reference: Optional[dict]

    # ── Specialist 출력 ───────────────────────────────────────────────────────
    spectrum_analysis: Optional[str]
    domain_interpretation: Optional[str]
    debate_result: Optional[str]

    # ── RAG / ROI ─────────────────────────────────────────────────────────────
    rag_results: list[dict]
    # next_roi: roi_detector 출력.
    # {"x", "y", "z", "mode": "manual"|"visual_search"|"current_position",
    #  "confidence": float, "background_position": {"x","y"} | None}
    next_roi: Optional[dict]

    # ── 종료 제어 ─────────────────────────────────────────────────────────────
    abort_reason: Optional[str]   # Critic Tier-A HARD VETO 시 설정
    final_report: Optional[str]


def initial_state(user_message: str, session_id: str = "",
                  intent: Optional[ClarifiedIntent] = None) -> ExperimentState:
    """서버에서 그래프를 invoke할 때 사용할 초기 상태를 반환한다.

    intent 인자: orchestrator가 clarification 게이트를 통과시키며 "사전 번역"한
    intent를 여기에 넣는다. 그러면 그래프의 translator_node가 재번역 없이 통과한다
    (LLM 중복 호출 방지). None이면 그래프 안에서 translator가 번역한다.
    """
    import uuid
    return ExperimentState(
        user_message=user_message,
        session_id=session_id or str(uuid.uuid4()),
        intent=intent,
        plan=[],
        current_step_idx=0,
        replan_count=0,
        next_node="",
        critic_checkpoint="",
        observations=[],
        critic_log=[],
        failure_log=[],
        retry_map={},
        cumulative_dose_mj=0.0,
        cumulative_dose_map={},
        stage_position=None,
        acquisition_params={},
        background_reference=None,
        spectrum_analysis=None,
        domain_interpretation=None,
        debate_result=None,
        rag_results=[],
        next_roi=None,
        abort_reason=None,
        final_report=None,
    )
