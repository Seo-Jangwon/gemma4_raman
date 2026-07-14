# -*- coding: utf-8 -*-
"""
ClarifyGate — 실험을 시작하기 전에 "필수 정보가 다 모였는가"를 판정하는 결정적 게이트.

[왜 그래프 밖에서, 규칙 기반으로 하는가]
1. 안전이 걸린 판단이라 재현 가능해야 한다.
   시료 종류를 모르면 hw_manager의 bio 파워 클램프(40%)·dose 한도 같은 안전 규칙이
   적용될지 판단할 근거가 없다. 이런 결정을 LLM의 그때그때 판단에 맡기면
   "물어봐야 할 때 안 물어보는" 사고가 날 수 있다. 규칙으로 못박는다.
   (critic의 C2 하드 비토를 규칙으로 둔 것과 같은 철학.)
2. 그래프 중간에서 멈췄다 사용자 입력을 받는 것(LangGraph interrupt/checkpoint)은
   구현이 무겁다. 반면 필수 정보는 "어떤 하드웨어 동작보다도 먼저" 필요하므로,
   그래프를 아예 시작하기 전에 되묻는 편이 단순하고 안전하다.
   → orchestrator.stream_experiment가 translate() 직후 이 게이트를 호출한다.

[무엇을 필수로 보는가 — 사용자가 제시한 3대 난제와 1:1 대응]
- 타겟 물질별 파워/노출 조절 어려움  → sample_type (안전·파워 결정의 근거)
- 타겟 위치 식별 어려움              → 좌표(constraints.x/y) 또는 target_description
                                       (roi_detector가 "어디를/무엇을" 찾을지)
- 기판 배경 vs 타겟 신호 구분 어려움 → substrate (배경 분리·경험 매칭의 키)

[무한 되묻기 방지]
사용자가 끝내 답을 못/안 주는 경우가 있다(예: "시료가 뭔지 모른다").
orchestrator가 라운드 수를 세어 _MAX_CLARIFY_ROUNDS를 넘으면 이 게이트를 무시하고
가진 정보로 진행한다. 그래도 안전한 이유: sample_type을 모르면 hw_manager의 적응형
획득이 5% 저출력 프로브부터 시작하므로(결정적 안전장치), 미지 시료라도 광손상 위험이
최소화된다. 즉 "물어는 보되, 끝내 모르면 저출력으로 조심스럽게 진행"이 최종 방어선.
"""

from __future__ import annotations

from backend.agents.state import ClarifiedIntent

# sample_type이 이 값이면 "모른다"로 간주하고 되묻는다.
_UNKNOWN_SAMPLE = {"", "unknown", "미지", "모름"}


def _has_coords(intent: ClarifiedIntent) -> bool:
    """사용자가 타겟 좌표를 직접 준 경우 — roi_detector 없이도 위치를 안다."""
    c = intent.get("constraints") or {}
    return c.get("x") is not None and c.get("y") is not None


def check_intent(intent: ClarifiedIntent) -> dict:
    """
    intent의 완성도를 판정한다.

    반환: {
      "ok": bool,                 # True면 그대로 실험 진행 가능
      "missing": list[str],       # 빠진 항목 키 ("sample_type"|"target"|"substrate")
      "question": str,            # 사용자에게 보낼 통합 질문 (ok면 "")
    }
    """
    missing: list[str] = []
    prompts: list[str] = []

    # ── 1. 시료 종류 (안전·파워 결정의 근거) ──
    sample = (intent.get("sample_type") or "").strip().lower()
    if sample in _UNKNOWN_SAMPLE:
        missing.append("sample_type")
        prompts.append(
            "• 측정할 **시료(물질) 종류**는 무엇인가요? "
            "(예: 그래핀, 엑소좀, 배터리 전극 등 — 레이저 출력·안전 한도 결정에 필요합니다)"
        )

    # ── 2. 타겟 위치: 좌표 OR 외형 설명 중 하나는 있어야 한다 ──
    desc = (intent.get("target_description") or "").strip()
    if not _has_coords(intent) and not desc:
        missing.append("target")
        prompts.append(
            "• 타겟이 **어디에 있나요? 혹은 어떻게 생겼나요?** "
            "(스테이지 좌표를 알면 좌표로, 모르면 현미경으로 찾을 수 있도록 외형을 알려주세요 "
            "— 예: '기판 위 어두운 원형 입자')"
        )

    # ── 3. 기판 종류 (배경 분리·경험 매칭의 키) ──
    substrate = (intent.get("substrate") or "").strip()
    if not substrate:
        missing.append("substrate")
        prompts.append(
            "• **기판(substrate)**은 무엇인가요? "
            "(예: 유리, SiO2 웨이퍼, 금박 — 기판 배경 신호를 타겟 신호와 분리하는 데 필요합니다)"
        )

    if not missing:
        return {"ok": True, "missing": [], "question": ""}

    question = (
        "실험을 안전하고 정확하게 진행하려면 아래 정보가 더 필요합니다:\n\n"
        + "\n".join(prompts)
    )
    return {"ok": False, "missing": missing, "question": question}
