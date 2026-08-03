# -*- coding: utf-8 -*-
"""T121 — 신호 판별 (2점)

[문제]
  Compare T121.csv with all 8 references in reference_library_8.csv and sort every
  reference by similarity, descending. If two scores are equal, order them by spectrum_id
  ascending.

[정답 기준]
  GT=8개 전체의 내림차순 정렬 목록(spectrum_id 기준). 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2 정규화 →
  코사인 유사도. 이 3단계를 문항에 명시해 유사도 정의를 단일화했다. 동점 처리 규칙을 명시해 순서를 유일하게 만들었다. 확인=순서 완전 일치, 각 점수
  절대오차 0.01. 8개 미만을 보고하면 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import answer as A
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T121",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T121.csv', 'reference_library_8.csv'],
    criteria="EXACT(8 items in order) + NUM(scores ±0.01)",
    prompt=(
        "Compare T121.csv with all 8 references in reference_library_8.csv and sort every "
        "reference by similarity, descending. If two scores are equal, order them by "
        "spectrum_id ascending. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T121 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    order = ["PS_01", "PS_03", "PS_02", "PET_02", "PMMA_01", "PET_01", "CAL_01", "SI_01"]
    scores = [0.994552, 0.993954, 0.978229, 0.228639, 0.179389, 0.167487, 0.031256, 0.010961]
    got_i = A.seq(run, "ranking", "sorted", "results", "references", field="spectrum_id")
    got_s = A.seq(run, "ranking", "sorted", "results", "references", field="score",
                  cast=float)
    return [
        chk.set_match("all 8 references in order", got_i, order, tol=0, ordered=True,
                      partial=True, weight=2.0),
        chk.set_match("similarity scores", got_s, scores, tol=0.01, ordered=True),
    ]
