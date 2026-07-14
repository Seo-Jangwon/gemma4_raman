"""
TranslatorNode — 자연어 사용자 메시지를 ClarifiedIntent JSON으로 변환.

LLM: Claude claude-opus-4-8 (파일 상단 _llm 교체로 모델 변경 가능)
파싱 실패 시 raw_user_message 기반으로 fallback intent 생성.

[clarification 루프와의 관계]
과거에는 이 노드가 "단일 호출, clarification 없음"이었다. 이제는 orchestrator가
그래프 실행 전에 translate()를 직접 호출해 intent를 만들고, 필수 정보가 빠지면
사용자에게 되묻는다(backend/agents/clarify.py). 되물어 얻은 답을 누적한 메시지로
translate()를 다시 부르는 방식이라, 핵심 파싱 로직을 translate() 함수로 분리해
노드와 orchestrator가 공유한다.

그래서 translator_node는 state에 intent가 "이미" 채워져 있으면(=orchestrator가
사전 번역해 넣었으면) 재번역하지 않고 통과한다. LLM 중복 호출을 막기 위함이다.
"""

from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.state import ClarifiedIntent, ExperimentState

# ── LLM 설정 (교체 포인트) ────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-opus-4-8", temperature=0)

_SYSTEM = """\
당신은 라만 분광 실험 요청을 구조화된 JSON으로 변환하는 전문가입니다.
사용자의 자연어 요청을 분석하여 반드시 아래 JSON 스키마만 출력하세요.

출력 JSON 스키마:
{
  "primary_objective": "측정 목적 한 문장",
  "sample_type": "샘플 종류 (예: graphene, exosome, battery_electrode, unknown)",
  "substrate": "기판 종류. 사용자 요청에서 유추 (예: glass, sio2 wafer, gold film). 언급 없으면 빈 문자열",
  "target_description": "현미경으로 찾아야 할 타겟의 외형 설명. 사용자가 위치(좌표)를 주지 않았다면 요청에서 유추해 작성 (예: '기판 위 어두운 원형 입자'). 유추 불가하면 빈 문자열",
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


def translate(user_msg: str) -> ClarifiedIntent:
    """자연어 메시지 → ClarifiedIntent. (노드와 orchestrator가 공유하는 순수 함수)

    orchestrator는 clarification 루프에서 "누적된" 대화 문자열을 넘긴다.
    (예: "그래핀 측정해줘\n[추가 정보] 기판은 SiO2 웨이퍼, 타겟은 어두운 육각 플레이크")
    LLM이 이 누적 문맥을 하나의 intent로 병합한다.
    """
    try:
        response = _llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        raw = response.content.strip()
        # 코드블록 제거
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(raw)
        return ClarifiedIntent(
            primary_objective=parsed.get("primary_objective", user_msg),
            sample_type=parsed.get("sample_type", "unknown"),
            # 기판 종류 — 경험 저장소의 에피소드 컨텍스트 키.
            # 같은 시료라도 기판(유리/실리콘/금박)에 따라 배경 신호가 달라
            # 과거 경험 매칭 시 기판 일치 여부가 유사도에 반영된다.
            substrate=parsed.get("substrate", "") or "",
            # 타겟 외형 설명 — roi_detector의 visual_search가 vision LLM에게
            # "무엇을 찾을지" 전달하는 근거. 좌표가 없는 요청에서 특히 중요.
            target_description=parsed.get("target_description", "") or "",
            success_criteria=parsed.get("success_criteria", []),
            constraints=parsed.get("constraints", {}),
            user_preferences=parsed.get("user_preferences", {}),
            raw_user_message=user_msg,
        )
    except Exception as e:
        print(f"[Translator] 파싱 실패, fallback 사용: {e}")
        return ClarifiedIntent(
            primary_objective=user_msg,
            sample_type="unknown",
            substrate="",
            target_description="",
            success_criteria=[],
            constraints={},
            user_preferences={},
            raw_user_message=user_msg,
        )


def translator_node(state: ExperimentState) -> dict:
    # intent가 이미 있으면(orchestrator가 clarification 게이트를 통과시키며
    # 사전 번역해 넣었으면) 재번역하지 않는다 — LLM 중복 호출 방지.
    if state.get("intent"):
        return {}
    return {"intent": translate(state["user_message"])}
