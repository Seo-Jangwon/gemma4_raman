# -*- coding: utf-8 -*-
"""N03 — 절차 구성 (2점)

[문제]
  Build a CSV table summarising every measurement of this session, one row per measurement.

[정답 기준]
  GT=aggregate_spectra_csv 1회, 반환 CSV의 행 수 = 세션 측정 건수. 확인=run_analysis로 직접 만들면 감점(N02와 같은
  취지).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="N03",
    score=2,
    axis="절차 구성",
    mode="live",
    prompt=(
        "Build a CSV table summarising every measurement of this session, one row per "
        "measurement. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N03 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "aggregate_spectra_csv", times=1),
    ]
