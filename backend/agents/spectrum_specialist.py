"""
SpectrumSpecialistNode — 물리 기반 라만 스펙트럼 분석 전문가.

static persona: 항상 동일한 시스템 프롬프트 (instrument-agnostic 물리 분석가).

[기판 배경 분리 설계]
기판 background 신호와 타겟 신호의 구분이 어렵다는 문제에 대한 분석 측 대응:
  1. 스펙트럼 소스 우선순위 — IPBSA 배경 제거본(corrected_data, version="target")이
     있으면 그것을 쓴다. 형광 hump가 제거된 스펙트럼이 피크 식별에 훨씬 유리하다.
     없으면 원본(acquire_spectrum)으로 fallback — 분석 자체는 항상 가능해야 한다.
  2. 기판 대조 — state.background_reference(hw_manager의 acquire_background 산출물)가
     있으면 타겟 스펙트럼과 나란히 프롬프트에 넣고, "양쪽에 모두 나타나는 피크는
     기판 유래로 배제하라"고 명시적으로 지시한다. LLM이 두 스펙트럼을 비교하는 것이
     사람이 하는 배경 판별 작업과 같은 원리다.

[Failure 처리]
LLM 호출 실패 시 step을 "failed"로 표시하고 idx를 전진시키지 않는다 —
Planner의 on_fail 정책(기본 skip)이 다음 행동을 결정한다.

LLM: Claude claude-opus-4-8 (교체 포인트: _llm 변수)
"""

from __future__ import annotations

import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.state import ExperimentState


def _advance_plan(state: ExperimentState) -> dict:
    """현재 step을 done으로 표시하고 idx를 올린다.
    (분석 step은 C3 게이트가 필요 없으므로 스스로 전진 — Planner 왕복 절약)"""
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
            "agent": "spectrum_specialist",
            "action": step.get("action", ""),
            "error": error,
            "timestamp": time.time(),
        }],
    }


# ── LLM 설정 (교체 포인트) ────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-opus-4-8")

_SYSTEM = """\
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


def spectrum_specialist_node(state: ExperimentState) -> dict:
    observations = state.get("observations", [])
    intent = state.get("intent") or {}
    sample_type = intent.get("sample_type", "unknown")

    spectrum_text = _extract_spectrum_data(observations)

    if "스펙트럼 데이터 없음" in spectrum_text:
        # 데이터 부재는 '분석 실패'가 아니라 '분석 불가' — skip 마킹으로 전진.
        # (실패 처리하면 Planner가 재시도하지만 데이터가 없는 건 재시도로 안 바뀜)
        return {
            "spectrum_analysis": "[SKIP] 스펙트럼 데이터 없음 — acquire_spectrum 먼저 실행 필요",
            **_advance_plan(state),
        }

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

    try:
        response = _llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
    except Exception as e:
        # LLM 장애 → 실패로 보고, 처리(기본 skip)는 Planner에 위임
        return _fail_step(state, f"spectrum_specialist LLM 호출 실패: {e}")

    return {"spectrum_analysis": response.content, **_advance_plan(state)}
