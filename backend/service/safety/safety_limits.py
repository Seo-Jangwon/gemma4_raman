# -*- coding: utf-8 -*-
"""
[역할] 레이저 조사량(dose) 상한과 그 계산식의 **단일 출처**[cite: 3].

[왜 도구 계층이 아니라 여기인가]
장비 도구 모듈은 config.py를 통해 Config.ini를 읽으므로 파일이 없는 환경에서는 
import 자체가 실패하기 때문에 의존성이 없는 자리에 위치시킴[cite: 3].
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
