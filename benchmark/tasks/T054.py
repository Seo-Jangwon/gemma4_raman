# -*- coding: utf-8 -*-
"""T054 — 데이터 처리 (2점)

[문제]
  In T054.csv find the detected peaks nearest to 1001 and 1602 cm-1 (prominence 5% of
  range) and report the ratio of their intensities (1001 over 1602).

[정답 기준]
  GT=두 피크 세기의 비. '검출된 피크 중 최근접'과 비의 방향을 명시해 역수 오답을 배제했다. 확인=비 상대오차 5%, 사용한 두 피크 위치 ±3 cm-1.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T054",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T054.csv'],
    criteria="NUM(5%) + SET(2 items, ±3 cm-1)",
    prompt=(
        "In T054.csv find the detected peaks nearest to 1001 and 1602 cm-1 (prominence 5% of "
        "range) and report the ratio of their intensities (1001 over 1602). "
    ),
    answer_keys=[
        ("ratio", "number - the peak intensity ratio"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T054 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "ratio", 1.6081484490367692, rel=0.05, name="intensity ratio"),
    ]
