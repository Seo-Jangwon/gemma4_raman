# -*- coding: utf-8 -*-
"""T053 — 데이터 처리 (2점)

[문제]
  Compute the area of 990-1012 cm-1 (inclusive) of the already baseline-corrected T053.csv
  by trapezoidal integration, with no further baseline correction.

[정답 기준]
  GT=np.trapz(y[구간], x[구간]). 추가 보정 금지를 명시해 이중 보정 분기를 없앴다. 확인=상대오차 2% 이내.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T053",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T053.csv'],
    criteria="NUM(2%)",
    prompt=(
        "Compute the area of 990-1012 cm-1 (inclusive) of the already baseline-corrected "
        "T053.csv by trapezoidal integration, with no further baseline correction. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T053 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "area", 7951.014983208794, rel=0.02, name="area"),
    ]
