# -*- coding: utf-8 -*-
"""어떤 에이전트 모듈로 실행할지 고르는 곳. AILA↔CoALA 를 가르는 판단은 여기 '하나'다.

controllers/agent.py 가 /api/experiment/* 에서 쓴다.
"""

from __future__ import annotations

from typing import Optional


def select_agent_module(name: Optional[str]):
    """agent 이름 → (에이전트 모듈, 정규화된 이름).

    새 에이전트를 추가하거나 분기 규칙을 바꿀 때 이 함수만 고치면 라우트는 손대지 않는다.
    두 모듈 모두 동일 공개 API(ALL_TOOLS / stream_experiment / run_experiment / run_stream)를
    노출하므로 호출부는 동일하다. 알 수 없는 값은 기본 AILA 로 폴백한다(회귀 방지).
    """
    coala = (name or "").strip().upper() == "COALA"
    arch = "CoALA" if coala else "AILA"

    if coala:
        from backend.agents.architectures import single_agent_CoALA as mod
        return mod, arch
    from backend.agents.architectures import single_agent_AILA as mod
    return mod, arch
