# -*- coding: utf-8 -*-
"""T001 — 계측 제어 (1점)

[문제]
  Move the stage once to the absolute coordinates X=35.0, Y=25.0, Z=0.5 mm.

[정답 기준]
  GT=(35.0, 25.0, 0.5). 확인=이동 후 get_stage_position이 GT와 ±1e-4 일치. z를 생략하면 현재 Z가 유지되므로
  오답(문항이 Z를 지정했다).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T001",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="STATE(coords ±1e-4)",
    prompt=(
        "Move the stage once to the absolute coordinates X=35.0, Y=25.0, Z=0.5 mm. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T001 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.state("final x", after, "x", 35.0, tol=MM),
        chk.state("final y", after, "y", 25.0, tol=MM),
        chk.state("final z", after, "z", 0.5, tol=MM),
    ]
