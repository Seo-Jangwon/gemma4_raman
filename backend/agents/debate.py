"""
SpecialistDebateSession — Spectrum-specialist ↔ Domain-specialist 3-round 토론.

MVP: debate_node는 graph에 등록되지만 Planner가 라우팅하지 않음.
V1: Planner가 debate 노드를 plan step으로 포함.

토론 프로토콜:
  Round 1: spectrum_specialist가 스펙트럼 분석
  Round 2: domain_specialist가 이의 제기 (challenge)
  Round 3: spectrum_specialist가 반론 (rebuttal)
  수렴: token overlap ratio > 0.5 → commit
  미수렴: domain_specialist 결론 채택 + uncertainty_flag=True
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.state import ExperimentState


def _advance_plan(state: ExperimentState) -> dict:
    """현재 step을 done으로 표시하고 idx를 올린다."""
    idx = state.get("current_step_idx", 0)
    plan = list(state.get("plan", []))
    if idx < len(plan):
        plan[idx] = {**plan[idx], "status": "done"}
    return {"plan": plan, "current_step_idx": idx + 1}


_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2)

_CONVERGENCE_THRESHOLD = 0.5


def _token_overlap(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def _run_debate(
    spectrum_analysis: str,
    domain_interpretation: str,
    sample_type: str,
) -> tuple[str, bool]:
    """3-round 토론 실행. (debate_result, uncertainty_flag) 반환."""

    # Round 2: domain_specialist가 spectrum 분석에 이의 제기
    challenge_prompt = (
        f"스펙트럼 물리 분석 결과:\n{spectrum_analysis}\n\n"
        f"샘플 종류: {sample_type}\n"
        "위 물리 분석에서 도메인 관점으로 동의하기 어렵거나 보완이 필요한 부분을 "
        "구체적으로 지적하고, 도메인 전문 지식을 근거로 이의를 제기하세요. 한국어로."
    )
    r2 = _llm.invoke([
        SystemMessage(content="당신은 도메인 전문가로서 스펙트럼 물리 분석에 이의를 제기합니다."),
        HumanMessage(content=challenge_prompt),
    ])
    challenge = r2.content

    # Round 3: spectrum_specialist가 반론
    rebuttal_prompt = (
        f"원래 스펙트럼 분석:\n{spectrum_analysis}\n\n"
        f"도메인 전문가의 이의 제기:\n{challenge}\n\n"
        "물리적 증거를 근거로 이의 제기에 반론하거나, 타당한 지적은 수용하고 "
        "수정된 결론을 제시하세요. 한국어로."
    )
    r3 = _llm.invoke([
        SystemMessage(content="당신은 라만 스펙트럼 물리 전문가입니다."),
        HumanMessage(content=rebuttal_prompt),
    ])
    rebuttal = r3.content

    overlap = _token_overlap(challenge, rebuttal)
    uncertainty = overlap < _CONVERGENCE_THRESHOLD

    if uncertainty:
        # 미수렴 → domain_specialist 결론 우선
        result = (
            f"[토론 결과 — 불일치 (overlap={overlap:.2f}, uncertainty_flag=True)]\n\n"
            f"## Domain-specialist 결론 (채택)\n{domain_interpretation}\n\n"
            f"## 이의 제기\n{challenge}\n\n"
            f"## Spectrum-specialist 반론\n{rebuttal}"
        )
    else:
        # 수렴 → rebuttal(spectrum) 결론 채택
        result = (
            f"[토론 결과 — 수렴 (overlap={overlap:.2f})]\n\n"
            f"## 최종 결론\n{rebuttal}\n\n"
            f"## 도메인 관점 보완\n{challenge}"
        )

    return result, uncertainty


def debate_node(state: ExperimentState) -> dict:
    """V1에서 Planner가 라우팅. MVP에서는 호출되지 않음."""
    spectrum_analysis   = state.get("spectrum_analysis", "")
    domain_interpretation = state.get("domain_interpretation", "")
    intent = state.get("intent") or {}
    sample_type = intent.get("sample_type", "unknown")

    if not spectrum_analysis:
        return {"debate_result": "debate 스킵: 스펙트럼 분석 없음", **_advance_plan(state)}

    debate_result, _ = _run_debate(spectrum_analysis, domain_interpretation, sample_type)
    return {"debate_result": debate_result, **_advance_plan(state)}
