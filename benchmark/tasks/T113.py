# -*- coding: utf-8 -*-
"""T113 — 신호 판별 (2점)

[문제]
  Compare the unknown spectrum T113.csv with the reference spectra in reference_library.csv
  (interpolate onto the reference axis, L2 normalize, cosine similarity) and report the 3
  most similar references with their rank and score.

[정답 기준]
  GT=상위 3개의 (spectrum_id, material, 순위, 점수). 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2
  정규화 → 코사인 유사도. 이 3단계를 문항에 명시해 유사도 정의를 단일화했다. 확인=상위 3개의 물질명과 순서가 완전 일치, 각 점수 절대오차 0.01.
  순위가 뒤바뀌면 오답(순위가 문항의 요구).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import answer as A
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T113",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T113.csv', 'reference_library.csv'],
    criteria="EXACT(ordered 3 items) + NUM(scores ±0.01)",
    prompt=(
        "Compare the unknown spectrum T113.csv with the reference spectra in "
        "reference_library.csv (interpolate onto the reference axis, L2 normalize, cosine "
        "similarity) and report the 3 most similar references with their rank and score. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T113 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    top3 = ["polystyrene", "polystyrene", "PET"]
    ids3 = ["PS_01", "PS_02", "PET_02"]
    scores3 = [0.994786, 0.978124, 0.227469]
    got_m = A.seq(run, "top3", "ranking", "matches", "results", field="material")
    got_i = A.seq(run, "top3", "ranking", "matches", "results", field="spectrum_id")
    got_s = A.seq(run, "top3", "ranking", "matches", "results", field="score", cast=float)
    return [
        chk.set_match("top 3 materials in order", got_m or got_i, top3 if got_m else ids3,
                      tol=0, ordered=True, weight=2.0),
        chk.set_match("top 3 scores", got_s, scores3, tol=0.01, ordered=True),
    ]
