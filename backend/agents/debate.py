"""
SpecialistDebateSession — Spectrum-specialist ↔ Domain-specialist 3-round 토론.

V1: Planner가 debate 노드를 plan step으로 포함해 실제 호출.

토론 프로토콜:
  Round 1: spectrum_specialist 분석 (기존 state.spectrum_analysis 재사용)
  Round 2: domain_specialist가 이의 제기 (challenge)
  Round 3: spectrum_specialist가 반론 (rebuttal)
  수렴 판정: LLM judge가 rebuttal이 challenge에 합의했는지 판단
  미수렴: domain_specialist 결론 채택 + uncertainty_flag=True
"""

from __future__ import annotations

import json
import re

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


_llm = ChatAnthropic(model="claude-opus-4-8")

_JUDGE_SYSTEM = """\
두 전문가의 토론 결과를 보고 최종 합의 여부를 판단하세요.
합의(agreed=true)의 기준: 반론이 이의 제기를 수용하거나, 양측 결론이 실질적으로 같은 내용을 지지하는 경우.
불일치(agreed=false)의 기준: 반론이 이의를 명확히 반박하고, 두 결론이 서로 충돌하는 경우.
JSON으로만 답변하세요: {"agreed": true/false, "reason": "판단 근거"}"""


def _check_convergence(challenge: str, rebuttal: str) -> bool:
    """LLM judge로 합의 여부 판정. True = 수렴, False = 불일치."""
    try:
        resp = _llm.invoke([
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=(
                f"도메인 전문가 이의 제기:\n{challenge[:600]}\n\n"
                f"스펙트럼 전문가 반론:\n{rebuttal[:600]}\n\n"
                "반론이 이의를 수용해 합의에 이르렀습니까, 아니면 반박했습니까?"
            )),
        ])
        cleaned = re.sub(r"```(?:json)?", "", resp.content).strip().rstrip("`").strip()
        parsed = json.loads(cleaned)
        return bool(parsed.get("agreed", False))
    except Exception:
        return False  # 파싱 실패 → 불일치로 처리 (보수적)


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

    # LLM judge로 수렴 판정
    converged = _check_convergence(challenge, rebuttal)
    uncertainty = not converged

    if uncertainty:
        result = (
            f"[토론 결과 — 불일치 (uncertainty_flag=True)]\n\n"
            f"## Domain-specialist 결론 (채택)\n{domain_interpretation}\n\n"
            f"## 이의 제기\n{challenge}\n\n"
            f"## Spectrum-specialist 반론\n{rebuttal}"
        )
    else:
        result = (
            f"[토론 결과 — 수렴]\n\n"
            f"## 최종 결론\n{rebuttal}\n\n"
            f"## 도메인 관점 보완\n{challenge}"
        )

    return result, uncertainty


def debate_node(state: ExperimentState) -> dict:
    """Planner가 plan step으로 포함해 호출."""
    spectrum_analysis   = state.get("spectrum_analysis", "")
    domain_interpretation = state.get("domain_interpretation", "")
    intent = state.get("intent") or {}
    sample_type = intent.get("sample_type", "unknown")

    if not spectrum_analysis:
        return {"debate_result": "debate 스킵: 스펙트럼 분석 없음", **_advance_plan(state)}

    try:
        debate_result, _ = _run_debate(spectrum_analysis, domain_interpretation, sample_type)
    except Exception as e:
        # 토론은 교차 검증용 보조 절차 — LLM 장애 시 실험을 막지 않고 스킵한다.
        # (_fail_step 대신 즉시 스킵하는 이유: debate의 기본 on_fail 정책도 skip이라
        #  Planner 왕복만 한 번 늘어날 뿐 결과가 같다 — 여기서 단락(short-circuit))
        return {"debate_result": f"debate 스킵: LLM 호출 실패 ({e})", **_advance_plan(state)}

    return {"debate_result": debate_result, **_advance_plan(state)}
