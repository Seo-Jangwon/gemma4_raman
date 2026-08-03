# -*- coding: utf-8 -*-
"""T029 — 절차 구성 (2점)

[문제]
  Acquire one spectrum at each of X = 37.0, 37.2, 37.4, 37.6, 37.8, 38.0 mm with Y=25.25
  and Z=0.

[정답 기준]
  GT=6점의 X 좌표 집합, Y=25.25 / Z=0 고정. 확인=list_results 6건의 meta.x가 GT와 ±0.001 일치.
  run_grid_scan(rows=1, cols=6, spacing=0.2, center_x=37.5)로 풀어도 좌표가 맞으면 정답으로 인정한다(승인 게이트는
  필요).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T029",
    score=2,
    axis="procedure",
    mode="live",
    criteria="SET(coords 6 items, ±0.001)",
    prompt=(
        "Acquire one spectrum at each of X = 37.0, 37.2, 37.4, 37.6, 37.8, 38.0 mm with "
        "Y=25.25 and Z=0. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T029 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    pts = run.positions()
    WANT_X = [37.0, 37.2, 37.4, 37.6, 37.8, 38.0]
    if not pts:
        return [chk.fail("measured X coordinates", "no coordinates in the move-call responses",
                         weight=2.0)]
    xs = [p[0] for p in pts]
    return [
        chk.set_match("measured X coordinates", xs, WANT_X, tol=MM_GRID, weight=2.0),
        chk.ok("Y held fixed", all(abs(p[1] - 25.25) <= MM_GRID for p in pts),
               f"y={[round(p[1], 3) for p in pts[:6]]} (expected 25.25)"),
    ]
