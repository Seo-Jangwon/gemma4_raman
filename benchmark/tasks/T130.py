# -*- coding: utf-8 -*-
"""T130 — 신호 판별 (2점)

[문제]
  For each of T130_1.csv through T130_5.csv, find the 3 most similar references in
  reference_library_8.csv, compute the fraction of the top 3 that are the same material as
  the query, and report the five fractions and their mean.

[정답 기준]
  GT=쿼리별 적중률 5개(각 0, 1/3, 2/3, 1 중 하나)와 그 평균. 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2
  정규화 → 코사인 유사도. 이 3단계를 문항에 명시해 유사도 정의를 단일화했다. 각 쿼리의 정답 물질은 생성 시 확정된다. 확인=평균 절대오차 0.01 +
  쿼리별 5값도 각각 일치. 평균만 맞고 개별이 틀리면 감점 (우연히 평균이 맞는 경우를 배제). [자산 반영] 8항목 라이브러리를 쓴다 — 12항목은 모든
  물질이 2항목씩이라 적중률 상한이 일률적으로 2/3 이 되어 평균이 상수가 된다(검증에서 확인).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T130",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T130_1.csv', 'T130_2.csv', 'T130_3.csv', 'T130_4.csv', 'T130_5.csv', 'reference_library_8.csv'],
    criteria="EXACT(5 hit rates) + NUM(mean ±0.01)",
    prompt=(
        "For each of T130_1.csv through T130_5.csv, find the 3 most similar references in "
        "reference_library_8.csv, compute the fraction of the top 3 that are the same "
        "material as the query, and report the five fractions and their mean. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T130 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "mean_hit_rate", 0.5333333333333333, tol=0.01, name="mean hit rate"),
    ]
