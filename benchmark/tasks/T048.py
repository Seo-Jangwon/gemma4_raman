# -*- coding: utf-8 -*-
"""T048 — 데이터 처리 (2점)

[문제]
  Overlay T048_a.csv and T048_b.csv on the same axes and label which line is which with a
  legend.

[정답 기준]
  GT=두 입력 배열. 확인=그림 1장에 곡선 2개, legend 항목 2개가 각 파일을 식별 가능하게 지칭, 각 곡선 배열이 해당 입력과 일치. 한 파일만
  그리면 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T048",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T048_a.csv', 'T048_b.csv'],
    criteria="ARRAY(rtol 1e-6) ×2 + STATE(legend 2 entries)",
    prompt=(
        "Overlay T048_a.csv and T048_b.csv on the same axes and label which line is which "
        "with a legend. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T048 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    out = [chk.called(run, "run_analysis", at_least=1, at_most=3)]
    a, bb = _input(b, "T048_a.csv"), _input(b, "T048_b.csv")
    if a is None or bb is None:
        return out + [chk.fail("overlay plot", "could not read the inputs T048_a/b.csv")]
    return out + [
        chk.keywords(run, ["legend", "T048_a", "T048_b"], name="the two curves are distinguishable"),
        chk.reported(run, "n_curves", 2.0, tol=0, name="number of curves"),
        chk.reported(run, "a_max", float(a[1].max()), rel=0.02, name="A max"),
        chk.reported(run, "b_max", float(bb[1].max()), rel=0.02, name="B max"),
    ]


def _input(b, name):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / name
    return sp.read_xy(p) if p.exists() else None
