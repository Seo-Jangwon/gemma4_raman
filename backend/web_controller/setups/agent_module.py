# -*- coding: utf-8 -*-
"""어떤 에이전트 모듈로 실행할지 고르는 곳. AILA↔CoALA 를 가르는 판단은 여기 '하나'다.

controllers/agent.py 가 /api/experiment/* 에서 쓴다. 요청이 아무것도 지정하지 않으면
llm_config.AGENT_ARCH 로 떨어진다 — 기본값을 여기 적지 않는 이유는 아래 참고.
"""

from __future__ import annotations

from typing import Optional

# 기본 아키텍처의 단일 출처. 예전에는 기본값이 세 군데에 각각 적혀 있었고 서로 달랐다
# (ExperimentRequest.agent="CoALA" / 여기 폴백="AILA" / agents_health 기본="AILA").
# 그래서 '아무것도 지정하지 않은' 호출이 어느 입구로 들어오느냐에 따라 다른 아키텍처로
# 돌았는데, 하필 그게 두 아키텍처를 비교하는 실험의 독립변수 자체였다.
from backend.llm_config import AGENT_ARCH

#: 이름 → 모듈 경로. 새 아키텍처는 여기 한 줄만 늘린다.
_MODULES = {
    "COALA": "backend.agents.architectures.single_agent_CoALA",
    "AILA": "backend.agents.architectures.single_agent_AILA",
}
#: 표시용 정식 표기(로그·API 응답에 그대로 실린다).
_DISPLAY = {"COALA": "CoALA", "AILA": "AILA"}


def select_agent_module(name: Optional[str] = None):
    """agent 이름 → (에이전트 모듈, 정규화된 이름).

    Parameters
    ----------
    name : 요청이 명시한 아키텍처. 비었거나 모르는 값이면 llm_config.AGENT_ARCH 를 쓰고,
           그것마저 모르는 값이면 CoALA 로 떨어진다(설정 오타로 서버가 죽지 않게).

    Notes
    -----
    두 모듈 모두 같은 공개 API(ALL_TOOLS / stream_experiment / run_experiment / run_stream)를
    노출하므로 호출부는 어느 쪽을 받든 동일하다. import 를 함수 안에서 하는 이유는 고르지
    않은 쪽 모듈을 불필요하게 로드하지 않기 위해서다.
    """
    import importlib

    key = (name or "").strip().upper()
    if key not in _MODULES:                       # 미지정이거나 오타 → 설정 기본값
        key = AGENT_ARCH.strip().upper()
    if key not in _MODULES:                       # 설정도 오타 → 최후 폴백
        key = "COALA"
    return importlib.import_module(_MODULES[key]), _DISPLAY[key]
