"""
SpectrumSpecialistNode — 물리 기반 라만 스펙트럼 분석 전문가.

static persona: 항상 동일한 시스템 프롬프트 (instrument-agnostic 물리 분석가).
state.observations 마지막 스펙트럼 데이터를 읽어 LLM에 주입.

LLM: Claude claude-sonnet-4-6 (교체 포인트: _llm 변수)
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


# ── LLM 설정 (교체 포인트) ────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

_SYSTEM = """\
당신은 라만 분광 물리 전문가입니다. 장비 종류에 무관하게 스펙트럼 데이터를 분석합니다.

분석 시 반드시 포함할 항목:
1. 주요 피크 위치 (cm⁻¹) 및 귀속 (peak assignment)
2. 스펙트럼 품질 평가 (SNR, 포화 여부, 배경 형광)
3. 결정성/비정질 특성 추정
4. 전체 스펙트럼 특성 요약 (overall_signature)

알 수 없는 피크는 "미귀속"으로 표시하고 추측하지 마세요.
한국어로 답변하세요."""


def _extract_spectrum_data(observations: list[dict]) -> str:
    """observations에서 가장 최근 acquire_spectrum 결과를 추출."""
    for obs in reversed(observations):
        result = obs.get("result", {})
        if not result.get("ok"):
            continue
        # acquire_spectrum 결과 구조: result.spectrum_data, result.wavelengths
        inner = result.get("result", result)
        if "spectrum_data" in inner or "intensities" in inner:
            data = inner.get("spectrum_data") or inner.get("intensities", [])
            wavenumbers = inner.get("wavenumbers") or inner.get("wavelengths", [])
            if data:
                # 최대 100 포인트 요약 (긴 배열 truncate)
                step = max(1, len(data) // 100)
                sampled_data = data[::step]
                sampled_wn   = wavenumbers[::step] if wavenumbers else list(range(len(sampled_data)))
                return (
                    f"스펙트럼 포인트 수: {len(data)}\n"
                    f"웨이브넘버 범위: {min(sampled_wn):.0f} ~ {max(sampled_wn):.0f} cm⁻¹\n"
                    f"최대 강도: {max(data):.1f}\n"
                    f"샘플 데이터 (웨이브넘버: 강도): "
                    + ", ".join(f"{wn:.0f}:{v:.1f}" for wn, v in zip(sampled_wn, sampled_data))
                )
        # 텍스트 형태로도 포함되는 경우
        if obs.get("tool") == "acquire_spectrum" and isinstance(inner, dict):
            return str(inner)[:1000]
    return "스펙트럼 데이터 없음 — 측정 결과를 먼저 확인하세요."


def spectrum_specialist_node(state: ExperimentState) -> dict:
    observations = state.get("observations", [])
    intent = state.get("intent") or {}
    sample_type = intent.get("sample_type", "unknown")

    spectrum_text = _extract_spectrum_data(observations)

    if "스펙트럼 데이터 없음" in spectrum_text:
        return {
            "spectrum_analysis": "[SKIP] 스펙트럼 데이터 없음 — acquire_spectrum 먼저 실행 필요",
            **_advance_plan(state),
        }

    prompt = (
        f"샘플 종류: {sample_type}\n\n"
        f"스펙트럼 데이터:\n{spectrum_text}\n\n"
        "위 라만 스펙트럼을 물리적으로 분석하세요."
    )

    response = _llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ])

    return {"spectrum_analysis": response.content, **_advance_plan(state)}
