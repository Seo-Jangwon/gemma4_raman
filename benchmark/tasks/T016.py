# -*- coding: utf-8 -*-
"""T016 — 계측 제어 (1점)

[문제]
  Set the CCD exposure time to 1.0 s.

[정답 기준]
  GT=get_ccd_info의 exposure_time_s=1.0(±1e-6).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T016",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Set the CCD exposure time to 1.0 s. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T016 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.state("실행 후 exposure_time_s", after, "exposure_time_s", 1.0, tol=1e-06),
    ]
