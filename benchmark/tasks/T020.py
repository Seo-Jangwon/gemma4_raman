# -*- coding: utf-8 -*-
"""T020 — 계측 제어 (1점)

[문제]
  Return the stage to the minimum-coordinate origin, X=0, Y=0, keeping the current Z.

[정답 기준]
  원본('장비에 설정된 원점')은 T088이 지적하듯 원점이 둘이라 정답이 갈렸다. 최소좌표 원점으로 못박았다. GT=(0, 0, z0).
  확인=get_stage_position이 x=y=0(±1e-4), z는 시작값 유지. z를 0으로 바꾸면 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T020",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="STATE(coords ±1e-4, z unchanged)",
    prompt=(
        "Return the stage to the minimum-coordinate origin, X=0, Y=0, keeping the current Z. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T020 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.state("final x", after, "x", 0.0, tol=MM),
        chk.state("final y", after, "y", 0.0, tol=MM),
        chk.unchanged("left untouched", before, after, ['z']),
    ]
