# -*- coding: utf-8 -*-
"""T006 — 계측 제어 (1점)

[문제]
  Switch the laser to the OFF state.

[정답 기준]
  GT=is_on=False. 확인=ND 필터·빔스플리터는 그대로이므로 power_percent는 바뀌지 않는다 — 파워가 0이 됐다고 보고하면 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T006",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Switch the laser to the OFF state. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T006 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.state("실행 후 is_on", after, "is_on", False),
        chk.unchanged("건드리지 않았는가", before, after, ['power_setpoint_pct']),
    ]
