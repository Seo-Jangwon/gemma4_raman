# -*- coding: utf-8 -*-
"""T003 — 계측 제어 (1점)

[문제]
  Tell me the current stage X, Y, Z coordinates.

[정답 기준]
  GT=get_stage_position 반환값 3개. 원본 2차가 'None'이었으나, 조회 결과를 그대로 보고했는지는 채점 가능하다. 확인=보고한 3값이 툴
  반환과 ±1e-4 일치(값을 지어내면 오답).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T003",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="NUM(±1e-4 ×3)",
    prompt=(
        "Tell me the current stage X, Y, Z coordinates. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T003 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "get_stage_position", at_least=1, at_most=3),
        chk.reported(run, "x", after.get("x"), tol=0.0001, name="reported x"),
        chk.reported(run, "y", after.get("y"), tol=0.0001, name="reported y"),
        chk.reported(run, "z", after.get("z"), tol=0.0001, name="reported z"),
    ]
