"""
CriticNode — 5개 체크포인트에서 실험 안전성/품질 감시.

C1: plan sanity (ABORT[Tier-B]: bio + 80%↑ → Planner가 재계획, WARNING: bio + 40%↑)
C2: H/W safety — 위치별 dose 기반 Tier-A HARD VETO
C3: spectrum quality — 포화/신호부족/배경우세 rule + 구조화된 보정 제안
C4: interpretation hallucination — LLM 기반 모순 검출 (Tier-B WARNING)
C5: report coherence — LLM 기반 일관성 검증 (Tier-B WARNING)

진입 방법: state["critic_checkpoint"] 필드로 호출할 checkpoint 지정.

[설계 원칙]
- C2(안전)는 rule-based 전용: LLM이 흔들려도 레이저 안전은 결정적으로 지킨다.
- C3(품질)도 rule-based: 포화/신호부족은 숫자로 판정 가능하므로 LLM을 쓰지 않는다.
  대신 verdict 문장만 주면 Planner가 "얼마나 고칠지"를 알 수 없으므로,
  suggestion 필드에 {"power_scale": 0.5} 같은 기계가 읽는 보정 계수를 담는다.
  → Planner는 이 계수를 곱해 재측정 파라미터를 만들 뿐, 추론하지 않는다.
- C4/C5(해석/보고서)는 텍스트 의미 비교가 필요하므로 LLM을 쓰되,
  실패(파싱 오류·API 오류)는 항상 APPROVE로 강등한다 — 검증기 자체의 장애가
  실험을 중단시키면 안 되기 때문(검증은 부가 안전망이지 필수 경로가 아님).
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

# ── C3 품질 판정 임계값 ────────────────────────────────────────────────────────
# CCD는 16-bit(최대 65535 ADU). 60000 이상이면 포화 직전/포화로 간주 —
# 포화된 피크는 상대 강도 비교가 불가능해 분석 가치가 없다.
_SATURATION_ADU = 60000.0
# 피크(최대값) - 베이스라인(중앙값)이 이 값보다 작으면 "신호 부족"으로 판정.
# 노이즈 표준편차의 수 배 수준을 경험적으로 잡은 값 — 이보다 작은 피크는
# 피크 귀속(assignment)의 신뢰도가 낮다.
_MIN_PEAK_OVER_BASELINE = 300.0
# 타겟 신호가 기판 배경의 1.15배 이하면 "배경 우세" — 측정 위치가 타겟을
# 벗어났거나 초점이 나갔을 가능성이 높다.
_BG_DOMINANCE_RATIO = 1.15

_critic_llm = ChatAnthropic(model="claude-opus-4-8", temperature=0)


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _entry(
    checkpoint: str,
    verdict: str,
    reason: str,
    tier: str = "none",
    suggestion: dict | None = None,
) -> CriticLogEntry:
    return CriticLogEntry(
        checkpoint=checkpoint,
        verdict=verdict,
        reason=reason,
        timestamp=time.time(),
        tier=tier,
        suggestion=suggestion or {},
    )


def _parse_json(text: str) -> dict:
    """LLM 응답에서 JSON 파싱. 실패 시 빈 dict."""
    try:
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        return json.loads(cleaned)
    except Exception:
        return {}


def _last_spectrum_stats(observations: list[dict]) -> dict | None:
    """
    observations에서 가장 최근 스펙트럼의 통계를 추출.
    acquire_spectrum 결과는 {"data": [...], "max_intensity": ...} 형태이고,
    IPBSA 결과는 {"corrected_data": [...]} 형태 — 둘 다 지원한다.
    C3는 '원본' 신호의 포화를 봐야 하므로 acquire 원본을 우선 찾는다.
    """
    for obs in reversed(observations):
        result = obs.get("result", {}) or {}
        if not result.get("ok"):
            continue
        data = result.get("data")
        if not data and obs.get("tool") == "acquire_spectrum":
            data = (result.get("result") or {}).get("data")
        if data:
            sorted_d = sorted(data)
            return {
                "max": float(max(data)),
                "median": float(sorted_d[len(sorted_d) // 2]),  # 베이스라인 근사
                "n": len(data),
            }
        # data 배열 없이 max_intensity만 있는 경우 (요약 결과)
        if "max_intensity" in result:
            return {"max": float(result["max_intensity"]), "median": 0.0, "n": 0}
    return None


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
            # tier를 "B"로 두는 이유: C1은 "계획" 단계라 아직 아무것도 실행되지
            # 않았다. Tier-A로 두면 그래프가 즉시 END로 가서 Planner의 재계획
            # 기회 자체가 사라진다 (기존 코드의 재계획 로직이 죽은 코드였던 원인).
            # 위험한 계획은 '거부(ABORT verdict)'하되, 재계획으로 복구할 수 있게
            # Planner로 돌려보낸다. 재계획 한도 초과 시 Planner가 최종 중단한다.
            # 실제 조사 직전의 하드 안전망은 C2(Tier-A)가 담당한다.
            return _entry(
                "C1", "ABORT",
                f"Bio 샘플에 위험 출력({power}%) 계획 감지 — 재계획 필요",
                "B",
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
    """Tier-A HARD VETO. rule-based 전용 — 어떤 LLM 판단도 개입하지 않는다."""
    intent = state.get("intent") or {}
    sample = intent.get("sample_type", "").lower()

    if any(kw in sample for kw in _BIO_KEYWORDS) and power_pct > _MAX_POWER_BIO:
        return _entry(
            "C2", "ABORT",
            f"Bio 샘플 광손상 위험: 레이저 출력 {power_pct}% > 한도 {_MAX_POWER_BIO}%",
            "A",
        )

    # 위치별 누적 dose 체크 — 적응형 튜닝이 같은 자리를 여러 번 프로브 측정하므로
    # "위치별" 누적이 특히 중요하다 (전체 합산만 보면 한 spot 과다 조사를 놓친다)
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
    """
    스펙트럼 품질 게이트 (rule-based).

    Planner는 acquire 계열 step이 done 될 때마다 여기로 라우팅한다.
    WARNING이면 suggestion의 보정 계수(power_scale/exposure_scale)를 읽어
    Planner가 해당 step을 재실행(파라미터 보정)할 수 있다 — 재시도 한도는
    Planner의 retry_map이 관리하므로 여기서는 순수하게 판정만 한다.

    판정 순서가 중요하다: 포화 > 배경우세 > 신호부족.
    포화된 스펙트럼은 max가 크므로 배경우세/신호부족 판정이 왜곡되기 때문에
    포화를 가장 먼저 확인한다.
    """
    stats = _last_spectrum_stats(state.get("observations", []))
    if stats is None:
        return _entry("C3", "APPROVE", "스펙트럼 데이터 없음 — 스킵", "none")

    # 1. 포화: 파워/노출을 함께 절반으로 — 포화 시 실제 광량을 알 수 없어
    #    (검출기가 clip) 선형 역산이 불가능하므로 보수적으로 크게 줄인다.
    if stats["max"] >= _SATURATION_ADU:
        return _entry(
            "C3", "WARNING",
            f"스펙트럼 포화 (max {stats['max']:.0f} ≥ {_SATURATION_ADU:.0f} ADU) — 파워/노출 감소 필요",
            "B",
            suggestion={"issue": "saturation", "power_scale": 0.5, "exposure_scale": 0.5},
        )

    # 2. 배경 우세: 기판 참조가 있을 때만 판정 가능.
    #    타겟 신호가 기판과 비슷하면 "타겟 위에 있지 않다"는 신호 —
    #    파라미터 보정으로 해결되지 않으므로 reposition을 제안한다.
    bg = state.get("background_reference")
    if bg and bg.get("max_intensity"):
        if stats["max"] <= float(bg["max_intensity"]) * _BG_DOMINANCE_RATIO:
            return _entry(
                "C3", "WARNING",
                f"타겟 신호({stats['max']:.0f})가 기판 배경({bg['max_intensity']:.0f})과 유사 — "
                "측정 위치가 타겟을 벗어났거나 초점 불량 의심",
                "B",
                suggestion={"issue": "background_dominant", "reposition": True},
            )

    # 3. 신호 부족: 피크-베이스라인 차이가 작으면 노출을 늘린다.
    #    (파워보다 노출을 먼저 늘리는 이유: 노출 증가는 광손상 관점에서
    #     피크 파워 증가보다 안전하고, 신호는 노출에 선형으로 비례한다)
    peak_over_baseline = stats["max"] - stats["median"]
    if stats["n"] > 0 and peak_over_baseline < _MIN_PEAK_OVER_BASELINE:
        return _entry(
            "C3", "WARNING",
            f"신호 부족 (피크-베이스라인 {peak_over_baseline:.0f} < {_MIN_PEAK_OVER_BASELINE:.0f} ADU) — 노출 증가 필요",
            "B",
            suggestion={"issue": "weak_signal", "power_scale": 1.5, "exposure_scale": 2.0},
        )

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
        # 검증기 장애가 실험을 막으면 안 됨 — APPROVE로 강등 (모듈 docstring 참고)
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
