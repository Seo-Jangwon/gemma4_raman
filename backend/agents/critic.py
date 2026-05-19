"""
CriticNode — 5개 체크포인트에서 실험 안전성/품질 감시.

MVP: C2 (H/W Safety) rule-based만 구현. 나머지는 APPROVE stub.
C2는 Tier-A HARD VETO → abort_reason 설정 → graph가 즉시 END 라우팅.

진입 방법: state["critic_checkpoint"] 필드로 호출할 checkpoint 지정.
  Planner가 critic 노드로 라우팅할 때 상태에 "critic_checkpoint" 키를 추가한다.
"""

from __future__ import annotations

import time

from backend.agents.state import CriticLogEntry, ExperimentState

_BIO_KEYWORDS = {"exosome", "cell", "lipid", "tissue", "bacteria", "protein", "membrane"}
_MAX_DOSE_MJ   = 500.0
_MAX_POWER_BIO = 40.0   # bio 샘플 레이저 출력 상한 (%)


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _entry(checkpoint: str, verdict: str, reason: str, tier: str = "none") -> CriticLogEntry:
    return CriticLogEntry(
        checkpoint=checkpoint,
        verdict=verdict,
        reason=reason,
        timestamp=time.time(),
        tier=tier,
    )


# ── 체크포인트별 함수 ─────────────────────────────────────────────────────────

def check_c1_plan_sanity(state: ExperimentState) -> CriticLogEntry:
    plan = state.get("plan", [])
    if not plan:
        return _entry("C1", "WARNING", "plan이 비어 있음 — default plan 적용 필요", "B")
    intent = state.get("intent") or {}
    sample = intent.get("sample_type", "").lower()
    constraints = intent.get("constraints", {})
    power = constraints.get("max_laser_power_pct")
    if any(kw in sample for kw in _BIO_KEYWORDS) and power and power > _MAX_POWER_BIO:
        return _entry("C1", "WARNING", f"Bio 샘플에 고출력({power}%) 계획 감지 — 재검토 권고", "B")
    return _entry("C1", "APPROVE", "", "none")


def check_c2_hardware_safety(
    state: ExperimentState,
    power_pct: float = 0.0,
    exposure_s: float = 0.0,
) -> CriticLogEntry:
    """Tier-A HARD VETO. LLM 없음, rule-based 전용."""
    intent = state.get("intent") or {}
    sample = intent.get("sample_type", "").lower()

    if any(kw in sample for kw in _BIO_KEYWORDS) and power_pct > _MAX_POWER_BIO:
        return _entry(
            "C2", "ABORT",
            f"Bio 샘플 광손상 위험: 레이저 출력 {power_pct}% > 한도 {_MAX_POWER_BIO}%",
            "A",
        )

    dose_inc = power_pct * exposure_s * 0.01
    current_dose = state.get("cumulative_dose_mj", 0.0)
    if current_dose + dose_inc > _MAX_DOSE_MJ:
        return _entry(
            "C2", "ABORT",
            f"누적 조사량 한도 초과: {current_dose + dose_inc:.1f} mJ > {_MAX_DOSE_MJ} mJ",
            "A",
        )

    return _entry("C2", "APPROVE", "", "none")


def check_c3_spectrum_quality(state: ExperimentState) -> CriticLogEntry:
    """MVP stub — V1에서 포화도/SNR 기반 rule 추가."""
    obs = state.get("observations", [])
    if obs:
        last = obs[-1]
        data = last.get("result", {}).get("spectrum_data", [])
        if data and max(data) > 60000:
            return _entry("C3", "WARNING", "스펙트럼 포화 의심 (max > 60000 ADU)", "B")
    return _entry("C3", "APPROVE", "", "none")


def check_c4_interpretation(state: ExperimentState) -> CriticLogEntry:
    """MVP stub — V1에서 LLM 기반 hallucination 검출."""
    return _entry("C4", "APPROVE", "C4 stub (V1에서 구현)", "none")


def check_c5_report_coherence(state: ExperimentState) -> CriticLogEntry:
    """MVP stub — V1에서 LLM 기반 내부 모순 검출."""
    return _entry("C5", "APPROVE", "C5 stub (V1에서 구현)", "none")


# ── 노드 함수 ─────────────────────────────────────────────────────────────────

_CHECKPOINT_MAP = {
    "C1": check_c1_plan_sanity,
    "C2": check_c2_hardware_safety,
    "C3": check_c3_spectrum_quality,
    "C4": check_c4_interpretation,
    "C5": check_c5_report_coherence,
}


def critic_node(state: ExperimentState) -> dict:
    """
    Planner가 state["critic_checkpoint"]에 "C1"~"C5" 중 하나를 설정하고 호출.
    C2는 추가로 state["critic_c2_params"] = {"power_pct": ..., "exposure_s": ...} 필요.
    """
    checkpoint = state.get("critic_checkpoint", "C1")
    fn = _CHECKPOINT_MAP.get(checkpoint, check_c1_plan_sanity)

    if checkpoint == "C2":
        params = state.get("critic_c2_params", {})
        entry = fn(state, **params)
    else:
        entry = fn(state)

    result = {"critic_log": [entry]}
    if entry["verdict"] == "ABORT" and entry["tier"] == "A":
        result["abort_reason"] = entry["reason"]

    return result
