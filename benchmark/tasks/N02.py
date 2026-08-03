# -*- coding: utf-8 -*-
"""N02 — 절차 구성 (2점)

[문제]
  Combine every spectrum measured in this session into one grid image, 3 columns wide.

[정답 기준]
  GT=combine_spectra 1회 호출(max_cols=3), 반환에 image_url 존재. 확인=run_analysis로 직접 그리면 감점 — 전용
  툴이 있는데 코드로 우회했는지를 본다. 사전 조건=세션에 측정이 2건 이상 있을 것.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="N02",
    score=2,
    axis="절차 구성",
    mode="live",
    prompt=(
        "Combine every spectrum measured in this session into one grid image, 3 columns "
        "wide. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N02 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "combine_spectra", times=1),
        chk.arg(run, "combine_spectra", "max_cols", 3),
    ]
