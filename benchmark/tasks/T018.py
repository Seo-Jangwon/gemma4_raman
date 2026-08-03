# -*- coding: utf-8 -*-
"""T018 — 계측 제어 (1점)

[문제]
  Tell me the current CCD temperature.

[정답 기준]
  GT=get_ccd_info의 temperature_C. 확인=보고값이 툴 반환과 일치, temperature_status(안정/냉각중)도 함께 전하면 가점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T018",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Tell me the current CCD temperature. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T018 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "temperature_C", after.get("temperature_C"), tol=0.1, name="보고한 temperature_C"),
    ]
