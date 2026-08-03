# -*- coding: utf-8 -*-
"""T120 — 신호 판별 (2점)

[문제]
  Extract the peak positions of T120.csv (prominence 5% of range), treat peaks within 3
  cm-1 as matching, and find the most similar material in reference_library.csv by the
  ratio of matched peaks to reference peaks (1:1 nearest-neighbour pairing, no duplicates).

[정답 기준]
  GT=(물질명, 각 참조의 매칭 점수). 코사인이 아니라 피크 기반 매칭이라는 점이 T113과의 차이다 — 검출 파라미터·매칭 규칙·분모를 모두 명시해
  확정했다. 확인=물질명 완전 일치 + 최상위 매칭 피크 수 일치.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T120",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T120.csv', 'reference_library.csv'],
    criteria="EXACT(material) + EXACT(matched peak count)",
    prompt=(
        "Extract the peak positions of T120.csv (prominence 5% of range), treat peaks within "
        "3 cm-1 as matching, and find the most similar material in reference_library.csv by "
        "the ratio of matched peaks to reference peaks (1:1 nearest-neighbour pairing, no "
        "duplicates). "
    ),
    answer_keys=[
        ("material",
         'string - one of "polystyrene", "PET", "PMMA", "calcite", "aragonite", '
         '"silicon"'),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T120 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported_label(run, "material", "PET", ['PET'], name="material"),
    ]
