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
from bench import answer as A
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T072",
    score=3,
    axis="data processing",
    mode="live",
    inputs=['T072.csv'],
    criteria="NUM(±0.01) ×3",
    prompt=(
        "Apply IPBSA baseline order 5 and L2 normalization to each spectrum of the map "
        "T072.csv, mean-center the resulting matrix, run PCA with 3 components and report "
        "each component's explained variance ratio. "
    ),
    answer_keys=[
        ("explained_variance_ratio",
         "list of 3 numbers - the explained variance ratio of each component, "
         "descending"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T072 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    want = [0.984376, 0.000917, 0.000912]
    got = A.seq(run, "explained_variance_ratio", "evr", "explained_variance",
                "variance_ratio", cast=float)
    if not got:
        return [chk.fail("3 explained variance ratios",
                         "the answer carries no explained_variance_ratio", weight=2.0)]
    return [
        chk.set_match("3 explained variance ratios", got, want, tol=0.01, ordered=True,
                      weight=2.0),
        chk.ok("descending and sums to <= 1",
               all(a >= b - 1e-9 for a, b in zip(got, got[1:])) and sum(got) <= 1.0 + 1e-6,
               f"{[round(v, 4) for v in got]} sum={sum(got):.4f}", kind="REL"),
    ]
