# -*- coding: utf-8 -*-
"""T015 — 계측 제어 (1점)

[문제]
  Set the CCD to internal trigger mode so it starts measurement on the instrument's
  internal signal.

[정답 기준]
  GT=get_ccd_info의 trigger_mode='internal'.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T015",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Set the CCD to internal trigger mode so it starts measurement on the instrument's "
        "internal signal. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T015 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.state("실행 후 trigger_mode", after, "trigger_mode", "internal"),
    ]
