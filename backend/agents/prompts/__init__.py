# -*- coding: utf-8 -*-
"""
시스템 프롬프트 조립 — 두 에이전트가 여기서 프롬프트를 받아 간다.

    from backend.agents.prompts import build_system_prompt
    SYSTEM_PROMPT = build_system_prompt("ReAct")     # 또는 "CoALA"

구성:
    common.py        두 에이전트가 공유하는 도메인 블록 (자율성·안전·절차·복구·좌표계)
    architecture.py  ReAct / CoALA 프로필 (전용 섹션 + 호칭 슬롯)
    이 파일          블록 순서 + 모드 토글 적용

[모드 토글 — 환경변수 2개]
  RAMAN_AUTONOMOUS=0        되묻기 게이트를 되살린 실험실 대화 모드 (기본은 자율)
  RAMAN_EPISODIC_MEMORY=0   CoALA 에피소딕 메모리 ablation (기본은 켜짐)

[왜 '치환'이 아니라 '조립'인가]
예전에는 완성된 프롬프트 문자열에서 섹션을 잘라내고 여러 문장을 replace 했다. 대상 문장이
한 글자만 바뀌어도 치환이 조용히 실패했고, 실제로 그래서 "치환 몇 개가 안 먹었다"를 경고로
알리는 코드까지 양쪽 파일에 있었다. 지금은 필요한 조각만 골라 붙이므로 실패할 지점이 없다
— 슬롯이 비면 KeyError 로 즉시 터진다(조용히 틀리지 않는다).
"""

from __future__ import annotations

import os

from backend.agents.prompts import architecture, common


def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


#: 기본은 자율. 벤치마크는 사람이 답을 줄 수 없는 환경이라, 되묻고 턴을 끝내면
#: 그 문항이 '수행 실패'로 채점된다. 그리드 승인 인터록도 같은 이유로 기본 OFF 다.
AUTONOMOUS = _env_on("RAMAN_AUTONOMOUS")

#: CoALA 에피소딕 메모리(recall_experiences / record_experience) 사용 여부.
EPISODIC_MEMORY = _env_on("RAMAN_EPISODIC_MEMORY")


def build_system_prompt(arch: str, *,
                        autonomous: bool | None = None,
                        episodic: bool | None = None) -> str:
    """arch("ReAct" | "CoALA") 의 시스템 프롬프트를 조립한다.

    autonomous / episodic 을 넘기지 않으면 환경변수 값을 쓴다(인자는 테스트용).
    """
    autonomous = AUTONOMOUS if autonomous is None else autonomous
    episodic = EPISODIC_MEMORY if episodic is None else episodic

    is_coala = arch.strip().lower() == "coala"
    profile = architecture.coala(episodic=episodic) if is_coala else architecture.REACT
    slots = dict(profile["slots"])

    # 모드에 따라 달라지는 문안을 슬롯에 채운다. 자율판은 스스로 판단해 진행하고,
    # 대화판은 사람에게 넘긴다 — 두 에이전트가 같은 문안을 쓴다.
    if autonomous:
        step1 = common.PROCEDURE_STEP1_AUTONOMOUS
        slots["safety_on_block"] = common.SAFETY_ON_BLOCK_AUTONOMOUS
        slots["safety_on_guess"] = common.SAFETY_ON_GUESS_AUTONOMOUS
    else:
        step1 = common.PROCEDURE_STEP1_INTERACTIVE
        slots["safety_on_block"] = common.SAFETY_ON_BLOCK_INTERACTIVE
        slots["safety_on_guess"] = common.SAFETY_ON_GUESS_INTERACTIVE
    slots["step1"] = step1.format(**slots) if "{gather_tools}" in step1 else step1
    slots["safety_on_guess"] = slots["safety_on_guess"].format(**slots) \
        if "{verify_with}" in slots["safety_on_guess"] else slots["safety_on_guess"]

    blocks = [profile["preamble"]]
    if autonomous:
        blocks.append(common.AUTONOMY)          # 대화 모드에서는 통째로 빠진다
    blocks.append(common.VERIFYING_OUTPUT)
    blocks += profile["extra_sections"]         # CoALA: 메모리 구조 + 의사결정 사이클
    blocks += [common.CONVERSATION_TYPES,
               common.ATTACHED_FILES,
               common.MEASUREMENT_PROCEDURE,
               common.HARDWARE_RECOVERY,
               common.SAFETY_RULES]

    return "\n\n".join(b.format(**slots) if "{" in b else b for b in blocks) + "\n"


def describe_mode() -> str:
    """기동 로그 한 줄 — 어떤 프롬프트로 도는지 즉시 보이게."""
    return (f"autonomous={'on' if AUTONOMOUS else 'OFF (ask-the-user gates restored)'}, "
            f"episodic_memory={'on' if EPISODIC_MEMORY else 'OFF (CoALA ablation)'}")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "ReAct"
    print(build_system_prompt(which))
