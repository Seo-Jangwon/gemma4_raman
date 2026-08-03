# -*- coding: utf-8 -*-
"""T031 — 절차 구성 (2점)

[문제]
  Measure a 5x5 grid with 0.1 mm spacing whose points span +0.0 to +0.4 mm in both X and Y
  relative to the current position. Autofocus at every point, then acquire one spectrum.

[정답 기준]
  GT(좌표)=시작 (x0,y0) 기준 25점 {x0+0.0,0.1,0.2,0.3,0.4}×{y0+동일}. 판별점: run_grid_scan은 center 대칭
  격자이므로 center=(x0+0.2, y0+0.2), spacing_mm=0.1, rows=5, cols=5, autofocus='each' 여야 한다.
  center를 x0로 주면 -0.2~+0.2가 되어 전 좌표 오답. GT(승인게이트)=preview_grid_scan이 run보다 앞선 '다른 턴'에 호출될
  것. 같은 턴이면 툴이 거부하므로 run 결과 ok=false 자체가 오답 신호. 확인=list_results 25건 meta.x/y ⊆
  GT좌표(±0.001mm), n_measured=25, n_autofocus_failed 보고 여부.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T031",
    score=2,
    axis="procedure",
    mode="live",
    criteria="PROC(args exact match) + SET(coords 25 items, ±0.001mm)",
    prompt=(
        "Measure a 5x5 grid with 0.1 mm spacing whose points span +0.0 to +0.4 mm in both X "
        "and Y relative to the current position. Autofocus at every point, then acquire one "
        "spectrum. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T031 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    DX, DY = 0.2, 0.2
    out = [
        chk.arg(run, "run_grid_scan", "spacing_mm", 0.1),
        chk.arg(run, "run_grid_scan", "rows", 5),
        chk.arg(run, "run_grid_scan", "cols", 5),
        chk.arg(run, "run_grid_scan", "autofocus", "each"),
        chk.order(run, "preview_grid_scan", "run_grid_scan"),
    ]
    cx = (run.args("run_grid_scan", "center_x") or [None])[0]
    cy = (run.args("run_grid_scan", "center_y") or [None])[0]
    if cx is None or "x" not in before:
        return out + [chk.fail("grid center", "no center argument and no starting coordinate",
                               weight=2.0)]
    # 중심을 시작 좌표 그대로 주면 격자 전체가 한 칸씩 어긋난다.
    return out + [
        chk.near("grid center X", cx, float(before["x"]) + DX, tol=1e-3, weight=2.0),
        chk.near("grid center Y", cy, float(before["y"]) + DY, tol=1e-3, weight=2.0),
    ]
