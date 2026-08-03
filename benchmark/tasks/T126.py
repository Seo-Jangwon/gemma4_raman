# -*- coding: utf-8 -*-
"""T126 — 신호 판별 (2점)

[문제]
  Identify the material of each of T126_1.csv through T126_5.csv against
  reference_library.csv, and report them in that order.

[정답 기준]
  GT=순서대로의 물질명 5개. 파일명을 번호로 고정해 '제시 순서'의 모호성을 없앴다. 규약=참조축(reference_library의 공통 파수축)으로 선형보간
  → L2 정규화 → 코사인 유사도. 이 3단계를 문항에 명시해 유사도 정의를 단일화했다. 확인=5개 전부 순서까지 일치. 부분 점수=정답 개수/5.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T126",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T126_1.csv', 'T126_2.csv', 'T126_3.csv', 'T126_4.csv', 'T126_5.csv', 'reference_library.csv'],
    criteria="EXACT(5 materials, in order) / partial credit = correct count / 5",
    prompt=(
        "Identify the material of each of T126_1.csv through T126_5.csv against "
        "reference_library.csv, and report them in that order. "
    ),
    answer_keys=[
        ("materials",
         "list of 5 strings - the material of T126_1.csv through T126_5.csv, in "
         "that order"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T126 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.set_match("5 materials", run.answer.get("materials"), ['polystyrene', 'silicon', 'PET', 'calcite', 'PMMA'], tol=0, ordered=True, partial=True, weight=2.0),
    ]
