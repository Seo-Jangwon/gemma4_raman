# -*- coding: utf-8 -*-
"""T074 — 절차 구성 (3점)

[문제]
  In the map T074.csv, compute the element-wise median spectrum, judge the positions whose
  cosine similarity to it (after L2 normalization) is below 0.80 as anomalous, and
  re-measure a spectrum once at those positions only.

[정답 기준]
  GT(판정)=조건을 만족하는 좌표 집합 — 입력이 파일이라 결정적. 중앙값·정규화 규약을 명시했다. GT(재측정)=그 좌표에만 측정(초과 금지). 확인=좌표
  집합 완전 일치. 재평가 유사도 값은 채점 제외.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T074",
    score=3,
    axis="procedure",
    mode="live",
    inputs=['T074.csv'],
    criteria="SET(coords EXACT) + PROC",
    prompt=(
        "In the map T074.csv, compute the element-wise median spectrum, judge the positions "
        "whose cosine similarity to it (after L2 normalization) is below 0.80 as anomalous, "
        "and re-measure a spectrum once at those positions only. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T074 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    # 요소별 median 과의 코사인 유사도 < 0.80 인 좌표.
    want = [[37.8, 25.4], [38.0, 25.2]]
    n = run.count("acquire_spectrum")
    pts = [[round(p[0], 4), round(p[1], 4)] for p in run.positions()]
    return [
        chk.set_match("re-measured positions", pts or None, want, tol=1e-3, weight=2.0),
        # 초과 측정도 오답이다 — 필요 없는 자리에 빔을 더 쬐는 것이라서.
        chk.called(run, "acquire_spectrum", times=len(want), weight=2.0),
    ]
