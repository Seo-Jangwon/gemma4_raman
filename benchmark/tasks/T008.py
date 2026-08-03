# -*- coding: utf-8 -*-
"""T008 — 계측 제어 (1점)

[문제]
  Instead of the measurement laser, switch to guide-beam mode that checks the sample
  position.

[정답 기준]
  GT=power_armed=False(ND가 차단 위치). 확인=power_percent는 마지막 요청값이 남으므로 그것으로 '무장됨'을 판단하면 오답 —
  power_armed를 읽어야 한다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T008",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Instead of the measurement laser, switch to guide-beam mode that checks the sample "
        "position. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T008 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.state("실행 후 power_armed", after, "power_armed", False),
    ]
