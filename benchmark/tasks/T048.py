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
    axis="데이터 처리",
    mode="live",
    inputs=['T048_a.csv', 'T048_b.csv'],
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
        return out + [chk.fail("겹쳐 그리기", "입력 T048_a/b.csv 를 읽지 못했습니다")]
    return out + [
        chk.keywords(run, ["legend", "범례", "T048_a", "T048_b"], name="두 곡선을 구분했는가"),
        chk.reported(run, "n_curves", 2.0, tol=0, name="곡선 개수"),
        chk.reported(run, "a_max", float(a[1].max()), rel=0.02, name="A 최대"),
        chk.reported(run, "b_max", float(bb[1].max()), rel=0.02, name="B 최대"),
    ]


def _input(b, name):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / name
    return sp.read_xy(p) if p.exists() else None
