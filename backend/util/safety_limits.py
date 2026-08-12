# -*- coding: utf-8 -*-
"""
[역할] 레이저 조사량(dose) 상한과 그 계산식의 **단일 출처**.

[왜 별도 모듈인가 — 2026-07-30]
같은 상수 1000.0 과 같은 식(power_pct × exposure_s × 0.01 × 점수)이 세 파일에
각각 박혀 있었다:

  raman_tools._GRID_MAX_DOSE_MJ          = 1000.0   (툴 계층, run_grid_scan 1회 기준)
  single_agent_AILA._MAX_DOSE_MJ_PER_TURN = 1000.0   (에이전트 계층, 턴 누계)
  single_agent_CoALA._MAX_DOSE_MJ_PER_TURN= 1000.0   (동일)

한 곳만 고치면 나머지가 조용히 갈라지는데, 갈라지는 대상이 하필 '레이저를 얼마나
쏘게 둘 것인가'다. 두 계층 모두 필요하다(툴 계층은 run_grid_scan 내부 N회 조사를,
에이전트 계층은 턴 누계를 막는다 — 서로를 대체하지 못한다). 그래서 계층은 남기고
**숫자와 식만** 여기로 모은다.

[왜 raman_tools 가 아니라 여기인가]
raman_tools 는 config.py 를 통해 Config.ini 를 읽으므로 그 파일이 없는 환경에서는
import 자체가 실패한다(에이전트의 _get_dispatch() 가 이 경우를 이미 다룬다). 조사량
한계는 하드웨어가 없어도 항상 읽혀야 하는 값이라, 의존성이 없는 자리에 둔다.

[왜 에이전트 두 개가 각자 가드를 갖는가]
AILA/CoALA 는 서로를 import 하지 않는다(비교 실험의 독립변수를 오케스트레이션으로
유지하기 위해서다 — 각 파일 머리말 참고). 이 모듈은 두 에이전트의 공통 상위 의존이지
서로에 대한 결합이 아니므로 그 원칙을 깨지 않는다.
"""
from __future__ import annotations

# 한 턴(대화 1회 요청) 동안 누적 허용하는 조사량. 에이전트 계층이 쓴다.
MAX_DOSE_MJ_PER_TURN = 1000.0

# run_grid_scan 한 번이 낼 수 있는 조사량. 툴 계층이 독립적으로 한 번 더 막는다
# (에이전트 계층 가드는 도구 이름 기준이라, 내부에서 N 번 조사하는 도구를 놓칠 수 있다).
MAX_DOSE_MJ_PER_GRID = 1000.0


def estimate_dose_mj(power_pct: float, exposure_s: float, n_shots: int = 1) -> float:
    """근사 조사량(mJ) = 파워[%] × 노출[s] × 0.01 × 조사 횟수.

    절대적인 물리량이 아니라 '같은 척도로 비교·누적하기 위한 근사치'다. 세 호출부가
    반드시 같은 척도를 써야 하므로(한쪽만 다르면 한계값의 의미가 달라진다) 식을
    함수로 고정한다.
    """
    return float(power_pct) * float(exposure_s) * 0.01 * int(n_shots)
