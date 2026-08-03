# -*- coding: utf-8 -*-
"""T060 — 절차 구성 (3점)

[문제]
  Measure a 3x3 grid at X=40.0, 40.1, 40.2 mm and Y=27.0, 27.1, 27.2 mm, and save a table
  of the strongest peak position and intensity at each position together with its
  coordinates.

[정답 기준]
  GT(좌표)=9점, center=(40.1, 27.1), spacing_mm=0.1, rows=cols=3. GT(표)=채점기가 저장된 9개 파일에서 재계산한
  (x, y, 최강피크 위치, 세기) 9행. 확인=meta.x/y ⊆ GT좌표(±0.001mm), 표 9행, 각 행 피크 ±3 cm-1 / 세기 상대오차 5%.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T060",
    score=3,
    axis="절차 구성",
    mode="live",
    prompt=(
        "Measure a 3x3 grid at X=40.0, 40.1, 40.2 mm and Y=27.0, 27.1, 27.2 mm, and save a "
        "table of the strongest peak position and intensity at each position together with "
        "its coordinates. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T060 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [
        chk.arg(run, "run_grid_scan", "center_x", 40.1),
        chk.arg(run, "run_grid_scan", "center_y", 27.1),
        chk.arg(run, "run_grid_scan", "spacing_mm", 0.1),
        chk.arg(run, "run_grid_scan", "rows", 3),
        chk.arg(run, "run_grid_scan", "cols", 3),
    ]
    if len(saved) < 9:
        return out + [chk.fail("격자 피크 표", f"저장 {len(saved)}건 (9건 필요)", weight=2.0)]
    pos = [sp.strongest_peak(x, y) for _, x, y in saved[:9]]
    got = run.answer.get("peak_positions")
    return out + [
        chk.set_match("9점 최강 피크 위치",
                      [float(v) for v in got] if isinstance(got, list) else None,
                      pos, tol=TOL_PEAK_CM1, ordered=True, partial=True, weight=2.0),
    ]
