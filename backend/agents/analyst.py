# -*- coding: utf-8 -*-
"""
AnalystNode — 스펙트럼 해석 통합 에이전트 (2-phase).

[에이전트 통합 노트 — spectrum_specialist + domain_specialist + debate → analyst]
과거에는 해석이 3개 노드로 나뉘어 있었다:
  spectrum_specialist(물리 분석) → domain_specialist(도메인 해석)
  → debate(3-round 토론 + LLM judge, 총 3~4회 추가 LLM 호출)
셋은 항상 이 순서로 붙어 실행됐고, 사이에 하드웨어 동작·품질 게이트 같은
제어 개입 지점이 전혀 없다 — 즉 "그래프 노드로 분리해서 얻는 것"이 없었다.
노드가 3개면 ablation 단위도 3개가 되어 "왜 이렇게 복잡하냐"는 질문만 낳는다.

지금은 한 노드 안의 2-phase로 통합한다:
  Phase 1 — 물리 분석: 피크 귀속, SNR/포화 평가, 기판 배경 대조 (구 spectrum_specialist 그대로)
  Phase 2 — 도메인 해석 + 교차검증: sample_type에 맞는 전문가 페르소나가
            물리 분석을 "검토하면서" 해석한다. 동의하지 않는 귀속은 근거를 들어
            지적하고 수정된 결론을 제시하도록 프롬프트에 명시 —
            구 debate의 challenge(이의 제기) 역할을 phase 2에 흡수한 것이다.
LLM 호출: 기존 최대 6회(분석1+해석1+토론3+judge1) → 2회.
교차검증의 최종 안전망은 critic C4가 그대로 담당한다(두 phase 산출물의
모순을 독립 LLM이 검증) — 검증자를 해석자와 분리하는 원칙은 유지된다.

[기판 배경 분리 설계 — 구 spectrum_specialist의 설계 그대로]
  1. 스펙트럼 소스 우선순위 — IPBSA 배경 제거본(corrected_data, version="target")이
     있으면 그것을 쓴다. 형광 hump가 제거된 스펙트럼이 피크 식별에 훨씬 유리하다.
     없으면 원본으로 fallback — 분석 자체는 항상 가능해야 한다.
  2. 기판 대조 — state.background_reference(acquire_background 산출물)가 있으면
     타겟 스펙트럼과 나란히 프롬프트에 넣고 "양쪽에 모두 나타나는 피크는
     기판 유래로 배제하라"고 명시적으로 지시한다.

[Failure 처리]
- Phase 1 실패: step을 "failed"로 표시하고 idx를 전진시키지 않는다 —
  Planner의 on_fail 정책(기본 skip)이 다음 행동을 결정한다.
- Phase 2 실패: phase 1 결과는 이미 확보됐으므로 버리지 않는다.
  도메인 해석만 [SKIP] 표기하고 step은 done으로 전진 —
  "물리 분석만 있는 보고서"가 "아무 분석도 없는 보고서"보다 낫다.

LLM: Claude claude-opus-4-8 (교체 포인트: _llm 변수)
"""

from __future__ import annotations

import re
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.state import ExperimentState

# ── LLM 설정 (교체 포인트) ────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-opus-4-8")


# ══════════════════════════════════════════════════════════════════════════════
# plan 전진/실패 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _advance_plan(state: ExperimentState) -> dict:
    """현재 step을 done으로 표시하고 idx를 올린다.
    (해석 step은 C3 게이트가 필요 없으므로 스스로 전진 — Planner 왕복 절약)"""
    idx = state.get("current_step_idx", 0)
    plan = list(state.get("plan", []))
    if idx < len(plan):
        plan[idx] = {**plan[idx], "status": "done"}
    return {"plan": plan, "current_step_idx": idx + 1}


def _fail_step(state: ExperimentState, error: str) -> dict:
    """LLM 실패 시: step failed + failure_log. 전진하지 않음 (Planner가 정책 결정)."""
    idx = state.get("current_step_idx", 0)
    plan = list(state.get("plan", []))
    step = plan[idx] if idx < len(plan) else {}
    if idx < len(plan):
        plan[idx] = {**plan[idx], "status": "failed", "result": {"error": error}}
    return {
        "plan": plan,
        "failure_log": [{
            "step_id": step.get("step_id", "?"),
            "agent": "analyst",
            "action": step.get("action", ""),
            "error": error,
            "timestamp": time.time(),
        }],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — 물리 분석 (구 spectrum_specialist)
# ══════════════════════════════════════════════════════════════════════════════

_PHYSICS_SYSTEM = """\
당신은 라만 분광 물리 전문가입니다. 장비 종류에 무관하게 스펙트럼 데이터를 분석합니다.

분석 시 반드시 포함할 항목:
1. 주요 피크 위치 (cm⁻¹) 및 귀속 (peak assignment)
2. 스펙트럼 품질 평가 (SNR, 포화 여부, 배경 형광)
3. 결정성/비정질 특성 추정
4. 전체 스펙트럼 특성 요약 (overall_signature)

기판 배경 스펙트럼이 함께 제공되는 경우:
- 타겟과 배경 양쪽에 모두 나타나는 피크는 '기판 유래'로 명시하고 타겟 귀속에서 제외하세요
- 타겟에만 나타나거나 배경 대비 유의하게 강한 피크만 '타겟 고유 피크'로 귀속하세요
- 타겟 신호가 배경과 구분되지 않으면 그 사실을 솔직하게 보고하세요

알 수 없는 피크는 "미귀속"으로 표시하고 추측하지 마세요.
한국어로 답변하세요."""


def _sample_pairs(data: list, wavenumbers: list, max_points: int = 100) -> str:
    """긴 배열을 프롬프트에 넣기 위한 다운샘플 (LLM 컨텍스트/비용 절약).
    피크 위치 파악에는 ~100포인트면 충분하다."""
    if not data:
        return ""
    step = max(1, len(data) // max_points)
    sampled_d = data[::step]
    sampled_w = wavenumbers[::step] if wavenumbers else list(range(len(sampled_d)))
    return ", ".join(f"{w:.0f}:{v:.1f}" for w, v in zip(sampled_w, sampled_d))


def _extract_spectrum_data(observations: list[dict]) -> str:
    """
    observations에서 분석할 스펙트럼을 추출.
    우선순위: ① IPBSA 배경 제거본(version_label='target') → ② acquire 원본.
    (① 이유는 모듈 docstring 참고 — 형광 제거본이 피크 식별에 유리)
    """
    # ① 배경 제거본 탐색 (최근 것 우선)
    for obs in reversed(observations):
        result = obs.get("result", {}) or {}
        if (obs.get("tool") == "apply_background_subtraction"
                and result.get("ok")
                and result.get("version_label") == "target"
                and result.get("corrected_data")):
            data = result["corrected_data"]
            wn = result.get("raman_shift_cm-1") or []
            return (
                "[IPBSA 형광 배경 제거 적용본]\n"
                f"스펙트럼 포인트 수: {len(data)}\n"
                f"최대 강도(보정 후): {max(data):.1f}\n"
                f"샘플 데이터 (웨이브넘버: 강도): {_sample_pairs(data, wn)}"
            )

    # ② acquire 원본 fallback — 다양한 결과 스키마(data/spectrum_data/intensities) 지원
    for obs in reversed(observations):
        result = obs.get("result", {}) or {}
        if not result.get("ok"):
            continue
        inner = result.get("result", result)
        data = (inner.get("data") or inner.get("spectrum_data")
                or inner.get("intensities") or [])
        if data:
            wn = (inner.get("raman_shift_cm-1") or inner.get("wavenumbers")
                  or inner.get("wavelengths") or [])
            wn_range = f"{min(wn):.0f} ~ {max(wn):.0f} cm⁻¹" if wn else "미교정 (픽셀 인덱스)"
            return (
                "[원본 스펙트럼 — 배경 제거 미적용]\n"
                f"스펙트럼 포인트 수: {len(data)}\n"
                f"웨이브넘버 범위: {wn_range}\n"
                f"최대 강도: {max(data):.1f}\n"
                f"샘플 데이터 (웨이브넘버: 강도): {_sample_pairs(data, wn)}"
            )

    return "스펙트럼 데이터 없음 — 측정 결과를 먼저 확인하세요."


def _run_physics_analysis(state: ExperimentState, spectrum_text: str) -> str:
    """Phase 1 LLM 호출. 예외는 호출부(analyst_node)가 처리한다."""
    intent = state.get("intent") or {}
    sample_type = intent.get("sample_type", "unknown")

    # ── 기판 배경 대조 블록 구성 ──────────────────────────────────────────────
    # background_reference는 hw_manager(acquire_background)가 "타겟과 동일 조건"으로
    # 측정해 저장한 것 — 조건이 같으므로 강도 비교가 유효하다.
    bg = state.get("background_reference") or {}
    bg_block = ""
    if bg.get("summary"):
        bg_block = (
            f"\n\n기판 배경 스펙트럼 (타겟과 동일 측정 조건: "
            f"레이저 {bg.get('power_pct')}%, 노출 {bg.get('exposure_s')}s, "
            f"max {bg.get('max_intensity', 0):.0f} ADU):\n{bg['summary']}\n"
            "→ 위 배경과 타겟 스펙트럼을 비교하여 기판 유래 피크를 식별·배제하세요."
        )

    # 측정 조건 컨텍스트 — 적응형 튜닝 결과를 알려주면 SNR/포화 평가가 정확해진다
    acq = state.get("acquisition_params") or {}
    cond_block = ""
    if acq:
        cond_block = (
            f"\n측정 조건: 레이저 {acq.get('power_pct')}%, 노출 {acq.get('exposure_s')}s "
            f"(적응형 튜닝 {'수렴' if acq.get('tuned') else '미수렴 — 신호 한계 도달'})"
        )

    prompt = (
        f"샘플 종류: {sample_type}{cond_block}\n\n"
        f"타겟 스펙트럼 데이터:\n{spectrum_text}"
        f"{bg_block}\n\n"
        "위 라만 스펙트럼을 물리적으로 분석하세요."
    )
    response = _llm.invoke([
        SystemMessage(content=_PHYSICS_SYSTEM),
        HumanMessage(content=prompt),
    ])
    return response.content


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — 도메인 해석 + 교차검증 (구 domain_specialist + debate의 challenge 흡수)
# ══════════════════════════════════════════════════════════════════════════════

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


def _run_domain_interpretation(state: ExperimentState, persona: str,
                               spectrum_analysis: str) -> str:
    """Phase 2 LLM 호출. 예외는 호출부(analyst_node)가 처리한다.

    구 debate의 교차검증(challenge)을 프롬프트에 흡수: 페르소나가 물리 분석을
    "수용하고 해석"만 하는 게 아니라 "검토하고, 틀렸으면 근거를 들어 고치도록"
    지시한다. 별도 토론 라운드 없이도 두 관점의 충돌이 표면화되며,
    남은 모순은 critic C4(독립 검증자)가 잡는다.
    """
    intent = state.get("intent") or {}
    sample_type = intent.get("sample_type", "unknown")

    prompt = (
        f"샘플 종류: {sample_type}\n\n"
        f"스펙트럼 물리 분석 결과 (물리 분석가):\n{spectrum_analysis}\n\n"
        f"실험 목적: {intent.get('primary_objective', '불명')}\n\n"
        "위 물리 분석을 도메인 전문가 관점에서 다음 순서로 처리하세요:\n"
        "1. [교차검증] 물리 분석의 피크 귀속·품질 평가 중 도메인 지식과 어긋나거나 "
        "보완이 필요한 부분이 있으면 구체적 근거와 함께 지적하세요. 없으면 '이견 없음'.\n"
        "2. [도메인 해석] (1을 반영한) 스펙트럼의 도메인 관점 해석을 제시하세요.\n"
        "3. [결론] 실험 목적과 연관된 최종 결론을 제시하세요. "
        "확신이 낮은 부분은 낮다고 명시하세요."
    )
    response = _llm.invoke([
        SystemMessage(content=_PERSONA_PROMPTS[persona]),
        HumanMessage(content=prompt),
    ])
    return response.content


# ══════════════════════════════════════════════════════════════════════════════
# 노드 진입점
# ══════════════════════════════════════════════════════════════════════════════

def analyst_node(state: ExperimentState) -> dict:
    observations = state.get("observations", [])
    intent = state.get("intent") or {}
    sample_type = intent.get("sample_type", "unknown")

    spectrum_text = _extract_spectrum_data(observations)

    if "스펙트럼 데이터 없음" in spectrum_text:
        # 데이터 부재는 '분석 실패'가 아니라 '분석 불가' — skip 마킹으로 전진.
        # (실패 처리하면 Planner가 재시도하지만 데이터가 없는 건 재시도로 안 바뀜)
        return {
            "spectrum_analysis": "[SKIP] 스펙트럼 데이터 없음 — acquire_spectrum 먼저 실행 필요",
            "domain_interpretation": "[SKIP] 분석할 스펙트럼 없음",
            **_advance_plan(state),
        }

    # ── Phase 1: 물리 분석 ────────────────────────────────────────────────────
    try:
        spectrum_analysis = _run_physics_analysis(state, spectrum_text)
    except Exception as e:
        # phase 1이 없으면 phase 2도 의미 없다 → step 전체를 실패로 보고,
        # 처리(기본 skip — 해석 실패는 측정 데이터를 해치지 않음)는 Planner에 위임
        return _fail_step(state, f"analyst 물리 분석 LLM 호출 실패: {e}")

    # ── Phase 2: 도메인 해석 + 교차검증 ───────────────────────────────────────
    # plan step params에 persona가 명시된 경우 우선 사용 (Planner 제어)
    plan = state.get("plan", [])
    idx = state.get("current_step_idx", 0)
    step_params = plan[idx].get("params", {}) if idx < len(plan) else {}
    forced_persona = step_params.get("persona", "")

    if forced_persona and forced_persona in _PERSONA_PROMPTS:
        persona = forced_persona
    else:
        persona = _select_persona(sample_type)

    try:
        interpretation = _run_domain_interpretation(state, persona, spectrum_analysis)
        domain_interpretation = f"[{persona}]\n{interpretation}"
    except Exception as e:
        # phase 1 결과는 확보됐으므로 버리지 않는다 (모듈 docstring의 Failure 처리)
        domain_interpretation = f"[SKIP] 도메인 해석 LLM 호출 실패 ({e}) — 물리 분석만 사용"

    return {
        "spectrum_analysis": spectrum_analysis,
        "domain_interpretation": domain_interpretation,
        **_advance_plan(state),
    }
