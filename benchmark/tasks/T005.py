# -*- coding: utf-8 -*-
"""T005 — 계측 제어 (1점)

[문제]
  Switch the laser to the ON state.

[정답 기준]
  GT=is_on=True. 판별점=파워가 무장되지 않았으면 나오는 것은 가이드빔이다. 응답의 beam 필드를 읽고 어느 빔인지 보고하면 가점, '측정빔이
  켜졌다'고 단정하면 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T005",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="STATE(is_on True) + KEYWORD(beam type)",
    prompt=(
        "Switch the laser to the ON state. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T005 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.state("final is_on", after, "is_on", True),
        chk.keywords(run, ['guide', 'measurement beam']),
    ]
