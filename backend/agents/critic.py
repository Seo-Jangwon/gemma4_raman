"""
CriticNode — 5개 체크포인트에서 실험 안전성/품질 감시.

C1: plan sanity (Tier-A ABORT: bio + 80%↑, Tier-B WARNING: bio + 40%↑)
C2: H/W safety — 위치별 dose 기반 Tier-A HARD VETO
C3: spectrum quality — 포화도 rule
C4: interpretation hallucination — LLM 기반 모순 검출 (Tier-B WARNING)
C5: report coherence — LLM 기반 일관성 검증 (Tier-B WARNING)

진입 방법: state["critic_checkpoint"] 필드로 호출할 checkpoint 지정.
"""

from __future__ import annotations

import json
import re
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.state import CriticLogEntry, ExperimentState

_BIO_KEYWORDS = {"exosome", "cell", "lipid", "tissue", "bacteria", "protein", "membrane"}
_MAX_DOSE_MJ_PER_SPOT = 500.0
_MAX_POWER_BIO = 40.0
_MAX_POWER_BIO_HARD = 80.0

_critic_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _entry(checkpoint: str, verdict: str, reason: str, tier: str = "none") -> CriticLogEntry:
    return CriticLogEntry(
        checkpoint=checkpoint,
        verdict=verdict,
        reason=reason,
        timestamp=time.time(),
        tier=tier,
    )


def _parse_json(text: str) -> dict:
    """LLM 응답에서 JSON 파싱. 실패 시 빈 dict."""
    try:
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        return json.loads(cleaned)
    except Exception:
        return {}


# ── 체크포인트별 함수 ─────────────────────────────────────────────────────────

def check_c1_plan_sanity(state: ExperimentState) -> CriticLogEntry:
    plan = state.get("plan", [])
    if not plan:
        return _entry("C1", "WARNING", "plan이 비어 있음 — default plan 적용 필요", "B")

    intent = state.get("intent") or {}
    sample = intent.get("sample_type", "").lower()
    is_bio = any(kw in sample for kw in _BIO_KEYWORDS)

    for step in plan:
        params = step.get("params", {})
        power = float(params.get("power_pct", 0))

        if is_bio and power > _MAX_POWER_BIO_HARD:
            return _entry(
                "C1", "ABORT",
                f"Bio 샘플에 위험 출력({power}%) 계획 감지 — 즉시 재계획 필요",
                "A",
            )
        if is_bio and power > _MAX_POWER_BIO:
            return _entry(
                "C1", "WARNING",
                f"Bio 샘플에 고출력({power}%) 계획 감지 — 재검토 권고",
                "B",
            )

    return _entry("C1", "APPROVE", "", "none")


def check_c2_hardware_safety(
    state: ExperimentState,
    power_pct: float = 0.0,
    exposure_s: float = 0.0,
) -> CriticLogEntry:
    """Tier-A HARD VETO. rule-based 전용."""
    intent = state.get("intent") or {}
    sample = intent.get("sample_type", "").lower()

    if any(kw in sample for kw in _BIO_KEYWORDS) and power_pct > _MAX_POWER_BIO:
        return _entry(
            "C2", "ABORT",
            f"Bio 샘플 광손상 위험: 레이저 출력 {power_pct}% > 한도 {_MAX_POWER_BIO}%",
            "A",
        )

    # 위치별 누적 dose 체크
    pos = state.get("stage_position") or {}
    pos_key = f"{pos.get('x', 0):.1f}_{pos.get('y', 0):.1f}"
    dose_map = state.get("cumulative_dose_map", {})
    pos_dose = dose_map.get(pos_key, 0.0)
    dose_inc = power_pct * exposure_s * 0.01

    if pos_dose + dose_inc > _MAX_DOSE_MJ_PER_SPOT:
        return _entry(
            "C2", "ABORT",
            f"spot ({pos_key}) 누적 조사량 한도 초과: {pos_dose + dose_inc:.1f} mJ > {_MAX_DOSE_MJ_PER_SPOT} mJ",
            "A",
        )

    return _entry("C2", "APPROVE", "", "none")


def check_c3_spectrum_quality(state: ExperimentState) -> CriticLogEntry:
    obs = state.get("observations", [])
    if obs:
        last = obs[-1]
        data = last.get("result", {}).get("spectrum_data", [])
        if data and max(data) > 60000:
            return _entry("C3", "WARNING", "스펙트럼 포화 의심 (max > 60000 ADU)", "B")
    return _entry("C3", "APPROVE", "", "none")


_C4_SYSTEM = """\
당신은 과학적 해석의 내부 모순을 검출하는 검증자입니다.
물리 분석 결과와 도메인 해석이 서로 명백하게 모순되는지 판단하세요.
모순의 기준: 동일 피크를 완전히 다른 화학종으로 귀속하거나, 결정성/비정질 판정이 반대이거나,
측정 대상 물질이 완전히 다른 경우입니다. 단순한 강조점 차이나 표현 방식 차이는 모순이 아닙니다.
JSON으로만 답변하세요: {"has_contradiction": true/false, "reason": "구체적 근거 또는 없음"}"""


def check_c4_interpretation(state: ExperimentState) -> CriticLogEntry:
    """LLM 기반 hallucination/모순 검출."""
    spec = state.get("spectrum_analysis", "")
    domain = state.get("domain_interpretation", "")

    if not spec or not domain or spec.startswith("[SKIP]"):
        return _entry("C4", "APPROVE", "분석 결과 없음 — 스킵", "none")

    prompt = (
        f"물리 분석:\n{spec[:800]}\n\n"
        f"도메인 해석:\n{domain[:800]}\n\n"
        "두 해석 간 명백한 모순이 있습니까?"
    )
    try:
        resp = _critic_llm.invoke([
            SystemMessage(content=_C4_SYSTEM),
            HumanMessage(content=prompt),
        ])
        parsed = _parse_json(resp.content)
        if parsed.get("has_contradiction"):
            return _entry(
                "C4", "WARNING",
                f"해석 모순 감지: {parsed.get('reason', '')}",
                "B",
            )
    except Exception:
        pass

    return _entry("C4", "APPROVE", "", "none")


_C5_SYSTEM = """\
당신은 실험 보고서의 논리적 일관성을 검증합니다.
보고서 내에서 측정 조건, 데이터, 결론이 서로 일치하는지 확인하세요.
불일치의 기준: 측정하지 않은 항목에 대한 결론, 데이터와 정반대되는 결론,
실험 목적과 무관한 결론 등입니다.
JSON으로만 답변하세요: {"is_coherent": true/false, "issue": "문제점 설명 또는 없음"}"""


def check_c5_report_coherence(state: ExperimentState) -> CriticLogEntry:
    """LLM 기반 보고서 내부 일관성 검증."""
    report = state.get("final_report", "")
    if not report:
        return _entry("C5", "APPROVE", "보고서 없음 — 스킵", "none")

    try:
        resp = _critic_llm.invoke([
            SystemMessage(content=_C5_SYSTEM),
            HumanMessage(
                content=f"보고서:\n{report[:1500]}\n\n이 보고서의 논리적 일관성을 검증하세요."
            ),
        ])
        parsed = _parse_json(resp.content)
        if not parsed.get("is_coherent", True):
            return _entry(
                "C5", "WARNING",
                f"보고서 일관성 문제: {parsed.get('issue', '')}",
                "B",
            )
    except Exception:
        pass

    return _entry("C5", "APPROVE", "", "none")


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
