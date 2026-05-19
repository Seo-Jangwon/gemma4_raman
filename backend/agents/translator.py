"""
TranslatorNode — 자연어 사용자 메시지를 ClarifiedIntent JSON으로 변환.

LLM: Claude claude-sonnet-4-6 (파일 상단 _llm 교체로 모델 변경 가능)
MVP: 단일 LLM 호출, clarification round 없음.
파싱 실패 시 raw_user_message 기반으로 fallback intent 생성.
"""

from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.state import ClarifiedIntent, ExperimentState

# ── LLM 설정 (교체 포인트) ────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

_SYSTEM = """\
당신은 라만 분광 실험 요청을 구조화된 JSON으로 변환하는 전문가입니다.
사용자의 자연어 요청을 분석하여 반드시 아래 JSON 스키마만 출력하세요.

출력 JSON 스키마:
{
  "primary_objective": "측정 목적 한 문장",
  "sample_type": "샘플 종류 (예: graphene, exosome, battery_electrode, unknown)",
  "success_criteria": ["성공 기준 1", "성공 기준 2"],
  "constraints": {
    "max_laser_power_pct": 숫자 또는 null,
    "max_exposure_s": 숫자 또는 null,
    "x": 숫자 또는 null,
    "y": 숫자 또는 null,
    "z": 숫자 또는 null
  },
  "user_preferences": {}
}

주의: JSON 코드블록 없이 순수 JSON만 출력하세요."""


def translator_node(state: ExperimentState) -> dict:
    user_msg = state["user_message"]

    try:
        response = _llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        raw = response.content.strip()
        # 코드블록 제거
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(raw)
        intent = ClarifiedIntent(
            primary_objective=parsed.get("primary_objective", user_msg),
            sample_type=parsed.get("sample_type", "unknown"),
            success_criteria=parsed.get("success_criteria", []),
            constraints=parsed.get("constraints", {}),
            user_preferences=parsed.get("user_preferences", {}),
            raw_user_message=user_msg,
        )
    except Exception as e:
        print(f"[Translator] 파싱 실패, fallback 사용: {e}")
        intent = ClarifiedIntent(
            primary_objective=user_msg,
            sample_type="unknown",
            success_criteria=[],
            constraints={},
            user_preferences={},
            raw_user_message=user_msg,
        )

    return {"intent": intent}
