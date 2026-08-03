# -*- coding: utf-8 -*-
"""T075 — 데이터 처리 (3점)

[문제]
  Compare the two sessions in T075_a.csv and T075_b.csv and report in one summary: the 1001
  cm-1 peak position difference, the RSD (ddof=1, %) of the peak intensity in each session,
  and the cosine similarity between the two mean spectra (common axis, L2).

[정답 기준]
  GT=4개 수치(위치차 1, RSD 2, 유사도 1). 각 산식을 명시해 확정했다. 확인=위치차 ±1 cm-1, RSD 각 상대오차 5%, 유사도 절대오차
  0.01. 4개 중 누락이 있으면 그만큼 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T075",
    score=3,
    axis="data processing",
    mode="live",
    inputs=['T075_a.csv', 'T075_b.csv'],
    criteria="NUM ×4 (position difference ±1 cm-1 / RSD 5% / similarity ±0.01)",
    prompt=(
        "Compare the two sessions in T075_a.csv and T075_b.csv and report in one summary: "
        "the 1001 cm-1 peak position difference, the RSD (ddof=1, %) of the peak intensity "
        "in each session, and the cosine similarity between the two mean spectra (common "
        "axis, L2). "
    ),
    answer_keys=[
        ("rsd_a_pct", "number - RSD of group A in percent"),
        ("rsd_b_pct", "number - RSD of group B in percent"),
        ("cosine_of_means", "number - cosine similarity between the two mean spectra"),
        ("peak_position_diff", "number - peak position difference in cm-1"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T075 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "rsd_a_pct", 3.2781799854102798, rel=0.05, name="RSD(a)"),
        chk.reported(run, "rsd_b_pct", 3.1057700697693074, rel=0.05, name="RSD(b)"),
        chk.reported(run, "cosine_of_means", 0.9570680913993472, tol=0.01, name="similarity of the means"),
        chk.reported(run, "peak_position_diff", 1.0, tol=1.0, name="peak position difference"),
    ]
