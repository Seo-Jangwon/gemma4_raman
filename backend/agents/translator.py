# -*- coding: utf-8 -*-
"""
Translator — 자연어 → ClarifiedIntent 변환(LLM) + 필수 정보 게이트(규칙).

[에이전트 통합 노트 — 왜 이 모듈은 '그래프 밖'인가]
translate()와 check_intent()는 모두 "그래프를 시작하기 전"에 필요한 전처리다:
  - translate(): 사용자의 자연어를 구조화된 intent로 파싱 (LLM 1회)
  - check_intent(): 실험에 필수인 3대 정보(시료/타겟 위치/기판)가 모였는지
    판정하는 결정적 게이트 (구 clarify.py를 이 파일로 흡수 — 파싱과 게이트는
    항상 붙어 다니는 한 몸이라 모듈을 나눌 이유가 없었다)
orchestrator.stream_experiment가 이 둘을 순서대로 호출해, 정보가 부족하면
그래프를 아예 실행하지 않고 사용자에게 되묻는다. 그래프 안에 translator 노드를
두지 않는 이유: intent는 항상 orchestrator가 사전 주입하므로 그래프 안에서
재번역할 일이 없다 (LLM 중복 호출 방지 + 그래프 노드 수 최소화).

[check_intent를 규칙 기반으로 두는 이유 — 구 clarify.py의 설계 그대로]
1. 안전이 걸린 판단이라 재현 가능해야 한다. 시료 종류를 모르면 hw_manager의
   bio 파워 클램프(40%)·dose 한도 같은 안전 규칙이 적용될지 판단할 근거가 없다.
   이런 결정을 LLM의 그때그때 판단에 맡기면 "물어봐야 할 때 안 물어보는" 사고가
   난다. 규칙으로 못박는다 (critic C2 하드 비토와 같은 철학).
2. 그래프 중간 interrupt보다 "그래프 시작 전 되묻기"가 단순하고 안전하다.

[무한 되묻기 방지]
orchestrator가 라운드 수를 세어 한도(_MAX_CLARIFY_ROUNDS)를 넘으면 게이트를
무시하고 진행한다. 그래도 안전한 이유: 시료를 모르면 hw_manager 적응형 획득이
5% 저출력 프로브부터 시작하므로 미지 시료라도 광손상 위험이 최소화된다.

LLM: Claude claude-opus-4-8 (파일 상단 _llm 교체로 모델 변경 가능)
파싱 실패 시 raw_user_message 기반으로 fallback intent 생성.
"""

from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.agents.state import ClarifiedIntent

# ── LLM 설정 (교체 포인트) ────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-opus-4-8")

_SYSTEM = """\
당신은 라만 분광 실험 요청을 구조화된 JSON으로 변환하는 전문가입니다.
사용자의 자연어 요청을 분석하여 반드시 아래 JSON 스키마만 출력하세요.

출력 JSON 스키마:
{
  "is_experiment_request": true 또는 false,
  "direct_reply": "is_experiment_request가 false일 때만 채우는 필드. 사용자에게 바로 보여줄
                    자연스러운 한국어 답변(인사 응대, 기능 소개 등). true면 빈 문자열.",
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

is_experiment_request 판정 기준:
- 라만 분광 측정을 요청하거나, 이전 요청에 대한 추가 정보(시료/기판/위치 등)를 주는 메시지 → true
- 인사말, 잡담, "너 뭐 할 수 있어?" 같은 메타 질문, 실험과 무관한 질문 → false
  (false인 경우 direct_reply에 이 시스템이 라만 실험 설계·측정을 돕는 도구임을 안내하고,
   primary_objective 등 나머지 실험 필드는 빈 값/기본값으로 채우세요)

주의: 항상 위 JSON 스키마 하나만 출력하세요. 코드블록(```)이나 설명 문장을 앞뒤에 절대 붙이지 마세요."""


def _invoke_and_parse(user_msg: str, history: list[dict] | None = None) -> dict:
    """LLM 호출 1회 + JSON 파싱. 실패 시 예외를 그대로 던진다(재시도는 호출부 책임).

    history: [{"role": "user"|"assistant", "text": str}, ...] — 이번 세션의 과거 대화.
    "내가 이전에 뭐라고 했지?" 같은 메타 질문에 답하거나, 예전 실험에서 언급된
    시료/기판을 다시 참조할 수 있도록 실제 멀티턴 메시지로 LLM에 넘긴다.
    (과거엔 이 문맥이 통째로 사라져 clarification 라운드가 끝나면 매번 "새 대화"였다.)
    """
    messages = [SystemMessage(content=_SYSTEM)]
    for turn in history or []:
        cls = HumanMessage if turn.get("role") == "user" else AIMessage
        messages.append(cls(content=turn.get("text", "")))
    messages.append(HumanMessage(content=user_msg))

    response = _llm.invoke(messages)
    raw = (response.content or "").strip()
    # 코드블록 제거
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # raw_decode: 맨 앞의 JSON 값 하나만 파싱하고 뒤따르는 텍스트는 무시한다.
    # (모델이 JSON 뒤에 부연 설명을 덧붙이는 경우 json.loads는 "Extra data"로 실패한다)
    return json.JSONDecoder().raw_decode(raw)[0]


def translate(user_msg: str, history: list[dict] | None = None) -> ClarifiedIntent:
    """자연어 메시지 → ClarifiedIntent. (orchestrator가 그래프 실행 전에 호출)

    orchestrator는 clarification 루프에서 "누적된" 대화 문자열을 user_msg로 넘긴다.
    (예: "그래핀 측정해줘\\n[추가 정보] 기판은 SiO2 웨이퍼, 타겟은 어두운 육각 플레이크")
    LLM이 이 누적 문맥을 하나의 intent로 병합한다.
    history는 그와 별개로 "이번 세션에서 이미 끝난 지난 턴들"을 전달한다(위 docstring 참고).

    드물게 모델이 빈 응답/비JSON 응답을 줄 때가 있어(일시적) 1회 재시도한다.
    """
    parsed = None
    last_err = None
    for attempt in range(2):
        try:
            parsed = _invoke_and_parse(user_msg, history)
            break
        except Exception as e:
            last_err = e
            if attempt == 0:
                print(f"[Translator] 파싱 실패, 재시도: {e}")

    if parsed is None:
        print(f"[Translator] 파싱 실패, fallback 사용: {last_err}")
        # is_experiment_request는 판정 불가 → True(안전한 기본값)로 두어
        # 기존처럼 check_intent 게이트가 부족한 정보를 되묻도록 한다.
        return ClarifiedIntent(
            is_experiment_request=True,
            direct_reply="",
            primary_objective=user_msg,
            sample_type="unknown",
            substrate="",
            target_description="",
            success_criteria=[],
            constraints={},
            user_preferences={},
            raw_user_message=user_msg,
        )

    return ClarifiedIntent(
        is_experiment_request=bool(parsed.get("is_experiment_request", True)),
        direct_reply=parsed.get("direct_reply", "") or "",
        primary_objective=parsed.get("primary_objective", user_msg),
        sample_type=parsed.get("sample_type", "unknown"),
        # 기판 종류 — 경험 저장소의 에피소드 컨텍스트 키.
        # 같은 시료라도 기판(유리/실리콘/금박)에 따라 배경 신호가 달라
        # 과거 경험 매칭 시 기판 일치 여부가 유사도에 반영된다.
        substrate=parsed.get("substrate", "") or "",
        # 타겟 외형 설명 — hw_manager locate_target의 visual_search가
        # vision LLM에게 "무엇을 찾을지" 전달하는 근거. 좌표 없는 요청에서 특히 중요.
        target_description=parsed.get("target_description", "") or "",
        success_criteria=parsed.get("success_criteria", []),
        constraints=parsed.get("constraints", {}),
        user_preferences=parsed.get("user_preferences", {}),
        raw_user_message=user_msg,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 필수 정보 게이트 (구 clarify.py — 결정적 규칙, LLM 미사용)
# ══════════════════════════════════════════════════════════════════════════════

# sample_type이 이 값이면 "모른다"로 간주하고 되묻는다.
_UNKNOWN_SAMPLE = {"", "unknown", "미지", "모름"}


def _has_coords(intent: ClarifiedIntent) -> bool:
    """사용자가 타겟 좌표를 직접 준 경우 — 시각 탐색 없이도 위치를 안다."""
    c = intent.get("constraints") or {}
    return c.get("x") is not None and c.get("y") is not None


def check_intent(intent: ClarifiedIntent) -> dict:
    """
    intent의 완성도를 판정한다. (사용자가 제시한 3대 난제와 1:1 대응하는 필수 정보)
      - 물질별 파워/노출 조절 어려움  → sample_type (안전·파워 결정의 근거)
      - 타겟 위치 식별 어려움         → 좌표(constraints.x/y) 또는 target_description
      - 기판 배경 vs 타겟 신호 구분   → substrate (배경 분리·경험 매칭의 키)

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
