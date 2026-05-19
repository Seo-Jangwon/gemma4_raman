"""
DomainSpecialistNode — 동적 persona 도메인 전문가.

sample_type 기반으로 PERSONA_REGISTRY에서 persona를 자동 선택.
spectrum_analysis를 입력으로 받아 도메인 관점 해석을 제공.

LLM: Claude claude-sonnet-4-6 (교체 포인트: _llm 변수)
"""

from __future__ import annotations

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

# ── LLM 설정 (교체 포인트) ────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# ── Persona Registry ──────────────────────────────────────────────────────────
# key: persona 이름, value: 해당 sample_type 키워드 튜플
PERSONA_REGISTRY: dict[str, tuple[str, ...]] = {
    "biologist":          ("exosome", "cell", "lipid", "tissue", "bacteria", "protein", "membrane"),
    "materials_engineer": ("graphene", "cnt", "carbon nanotube", "2d material", "mxene", "silicon"),
    "electrochemist":     ("battery", "electrode", "electrolyte", "lithium", "cathode", "anode"),
    "pharma_chemist":     ("drug", "api", "tablet", "crystalline", "pharmaceutical", "excipient"),
    "polymer_scientist":  ("polymer", "plastic", "pvdf", "nylon", "polyethylene", "polystyrene"),
    "food_scientist":     ("food", "adulteration", "olive oil", "milk", "spice", "grain"),
    "forensic_chemist":   ("explosive", "narcotics", "forensic", "gunshot", "residue"),
    "general_analytical_chemist": (),  # fallback
}

_PERSONA_PROMPTS: dict[str, str] = {
    "biologist": """\
당신은 세포생물학/분자생물학 전문가로서 라만 스펙트럼을 생물학적 관점에서 해석합니다.
광손상(photodamage)에 민감하며, 생체분자(단백질, 지질, DNA)의 라만 특성을 잘 압니다.
한국어로 답변하세요.""",

    "materials_engineer": """\
당신은 신소재 공학자로서 라만 스펙트럼을 재료 특성 평가에 활용합니다.
그래핀, CNT, 2D 재료의 D, G, 2D 밴드와 결함 분석에 전문성이 있습니다.
한국어로 답변하세요.""",

    "electrochemist": """\
당신은 전기화학 전문가로서 배터리/전극 소재의 라만 분석을 수행합니다.
Li-ion 배터리 구성 요소의 상 변화와 구조적 변화를 스펙트럼으로 추적합니다.
한국어로 답변하세요.""",

    "pharma_chemist": """\
당신은 제약 화학자로서 의약품 원료 및 제형의 라만 분석을 전문으로 합니다.
다형체(polymorph) 식별과 API 정량 분석에 강점이 있습니다.
한국어로 답변하세요.""",

    "polymer_scientist": """\
당신은 고분자 과학자로서 플라스틱 및 고분자 재료의 라만 분석을 수행합니다.
결정화도, 분자 배향, 중합도 평가에 전문성이 있습니다.
한국어로 답변하세요.""",

    "food_scientist": """\
당신은 식품 과학자로서 식품 품질 및 이물질 검출에 라만 분광을 활용합니다.
식품 위변조 검출과 성분 분석에 전문성이 있습니다.
한국어로 답변하세요.""",

    "forensic_chemist": """\
당신은 법과학 전문가로서 폭발물, 마약류, 법의학 증거물의 라만 분석을 수행합니다.
미량 물질의 동정과 혼합물 분석에 전문성이 있습니다.
한국어로 답변하세요.""",

    "general_analytical_chemist": """\
당신은 분석화학 전문가로서 라만 스펙트럼을 통해 화학적 조성과 구조를 해석합니다.
한국어로 답변하세요.""",
}


def _select_persona(sample_type: str) -> str:
    sample_lower = sample_type.lower()
    tokens = set(re.split(r'[\s_\-,;]+', sample_lower))
    for persona, keywords in PERSONA_REGISTRY.items():
        if persona == "general_analytical_chemist":
            continue
        for kw in keywords:
            kw_parts = kw.split()
            if len(kw_parts) == 1:
                if kw in tokens:       # 단일 키워드: 정확한 토큰 매칭
                    return persona
            else:
                if kw in sample_lower: # 복합 키워드: 문자열 포함
                    return persona
    return "general_analytical_chemist"


def domain_specialist_node(state: ExperimentState) -> dict:
    intent = state.get("intent") or {}
    sample_type = intent.get("sample_type", "unknown")
    spectrum_analysis = state.get("spectrum_analysis") or ""

    if not spectrum_analysis or spectrum_analysis.startswith("[SKIP]"):
        return {
            "domain_interpretation": "[SKIP] 분석할 스펙트럼 없음",
            **_advance_plan(state),
        }

    persona = _select_persona(sample_type)
    system_prompt = _PERSONA_PROMPTS[persona]

    prompt = (
        f"샘플 종류: {sample_type}\n\n"
        f"스펙트럼 분석 결과 (물리 분석가):\n{spectrum_analysis}\n\n"
        f"실험 목적: {intent.get('primary_objective', '불명')}\n\n"
        "위 스펙트럼 분석 결과를 도메인 관점에서 해석하고, "
        "실험 목적과 연관된 결론을 제시하세요."
    )

    response = _llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ])

    return {
        "domain_interpretation": f"[{persona}]\n{response.content}",
        **_advance_plan(state),
    }
