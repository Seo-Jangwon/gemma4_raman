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
    axis="절차 구성",
    mode="live",
    inputs=['T074.csv'],
    prompt=(
        "In the map T074.csv, compute the element-wise median spectrum, judge the positions "
        "whose cosine similarity to it (after L2 normalization) is below 0.80 as anomalous, "
        "and re-measure a spectrum once at those positions only. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T074 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.fail("채점 항목 없음", "이 문항의 정답 기준이 아직 옮겨지지 않았습니다"),
    ]
