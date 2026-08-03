# -*- coding: utf-8 -*-
"""T002 — 계측 제어 (1점)

[문제]
  Move relatively by only +0.1 mm along the X axis from the current position.

[정답 기준]
  GT=(x0+0.1, y0, z0). 확인=이동 전후 get_stage_position 비교로 Δx=+0.1(±1e-4), Δy=Δz=0. move_stage로
  절대좌표를 계산해 넣어도 결과가 같으면 정답으로 인정한다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T002",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Move relatively by only +0.1 mm along the X axis from the current position. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T002 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.delta("x 변화", before, after, "x", 0.1, tol=MM),
        chk.delta("y 변화", before, after, "y", 0.0, tol=MM),
        chk.delta("z 변화", before, after, "z", 0.0, tol=MM),
    ]
