# -*- coding: utf-8 -*-
"""T125 — 신호 판별 (2점)

[문제]
  Report the material in reference_library.csv most similar to T125.csv, and present at
  least two basis peaks that distinguish it from the second-ranked candidate.

[정답 기준]
  GT=(물질명, 1위에는 있고 2위 후보에는 없는 피크 목록). 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2 정규화 →
  코사인 유사도. 이 3단계를 문항에 명시해 유사도 정의를 단일화했다. 확인=물질명 일치 + 제시한 근거 피크 2개 이상이 GT 판별피크 목록에 포함(각 ±3
  cm-1). 1·2위 공통 피크만 제시하면 근거로 인정하지 않는다(이 문항의 핵심).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T125",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T125.csv', 'reference_library.csv'],
    criteria="EXACT(material) + SET(basis peaks >=2 items, ±3 cm-1)",
    prompt=(
        "Report the material in reference_library.csv most similar to T125.csv, and present "
        "at least two basis peaks that distinguish it from the second-ranked candidate. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T125 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported_label(run, "material", "PET", ['PET'], name="material"),
    ]
