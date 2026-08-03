# -*- coding: utf-8 -*-
"""T072 — 데이터 처리 (3점)

[문제]
  Apply IPBSA baseline order 5 and L2 normalization to each spectrum of the map T072.csv,
  mean-center the resulting matrix, run PCA with 3 components and report each component's
  explained variance ratio.

[정답 기준]
  GT=설명분산비 3값. 평균중심화 여부를 명시해 확정했다(생략하면 값이 달라진다). 확인=3값 각각 절대오차 0.01 이내, 내림차순이고 합 <= 1. 주성분의
  부호·방향은 채점 제외.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T072",
    score=3,
    axis="데이터 처리",
    mode="live",
    inputs=['T072.csv'],
    prompt=(
        "Apply IPBSA baseline order 5 and L2 normalization to each spectrum of the map "
        "T072.csv, mean-center the resulting matrix, run PCA with 3 components and report "
        "each component's explained variance ratio. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T072 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.fail("채점 항목 없음", "이 문항의 정답 기준이 아직 옮겨지지 않았습니다"),
    ]
