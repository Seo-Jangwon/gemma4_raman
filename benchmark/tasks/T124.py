# -*- coding: utf-8 -*-
"""T124 — 신호 판별 (2점)

[문제]
  Distinguish whether T124.csv is calcite or aragonite, using reference_library.csv.

[정답 기준]
  GT=물질명 1개. 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2 정규화 → 코사인 유사도. 이 3단계를 문항에 명시해
  유사도 정의를 단일화했다. 확인=물질명 완전 일치. 두 다형체는 1085 cm-1 부근이 겹치므로, 출제 전에 참조 스펙트럼에 판별 피크(방해석 712 /
  아라고나이트 705·206 부근)가 실제로 들어 있는지 확인할 것 — 없으면 GT는 있으나 풀 수 없는 문항이 된다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T124",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T124.csv', 'reference_library.csv'],
    criteria="EXACT(material)",
    prompt=(
        "Distinguish whether T124.csv is calcite or aragonite, using reference_library.csv. "
    ),
    answer_keys=[
        ("material",
         'string - one of "polystyrene", "PET", "PMMA", "calcite", "aragonite", '
         '"silicon"'),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T124 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported_label(run, "material", 'aragonite', ['PET', 'PMMA', 'aragonite', 'calcite', 'polystyrene', 'silicon'], name="material",
                           weight=2.0),
    ]
