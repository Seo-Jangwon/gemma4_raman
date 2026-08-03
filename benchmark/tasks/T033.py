# -*- coding: utf-8 -*-
"""T033 — 계측 제어 (2점)

[문제]
  Measure a spectrum once each at (X=37, Y=25) and (X=38, Y=26) mm, and save them so the
  two positions can be told apart.

[정답 기준]
  GT(좌표)={(37.0,25.0),(38.0,26.0)}. 확인=list_results 2건의 meta.x/y가 GT와 ±0.001mm 일치, 두 건의
  base(파일명)가 서로 달라 좌표로 구분 가능할 것. 같은 좌표로 두 번 찍으면 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T033",
    score=2,
    axis="instrument control",
    mode="live",
    criteria="SET(coords 2 items, ±0.001mm) + PROC",
    prompt=(
        "Measure a spectrum once each at (X=37, Y=25) and (X=38, Y=26) mm, and save them so "
        "the two positions can be told apart. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T033 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    WANT = [(37.0, 25.0), (38.0, 26.0)]
    out = [chk.called(run, "acquire_spectrum", times=2)]
    pts = run.positions()
    if not pts:
        return out + [chk.fail("measured coordinates", "no coordinates in the move-call responses",
                               weight=2.0)]
    return out + [
        chk.set_match("measured coordinates", [[p[0], p[1]] for p in pts],
                      [list(w) for w in WANT], tol=MM_GRID, ordered=True, weight=2.0),
    ]
