# -*- coding: utf-8 -*-
"""T117 — 신호 판별 (2점)

[문제]
  Identify the material of T117.csv against reference_library.csv, which includes PET and
  PMMA references.

[정답 기준]
  GT=물질명 1개. 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2 정규화 → 코사인 유사도. 이 3단계를 문항에 명시해
  유사도 정의를 단일화했다. 확인=물질명 완전 일치. PET/PMMA는 유사도 차가 작을 수 있으므로, 출제 전에 두 후보의 GT 유사도 차가 0.05 이상
  벌어지는지 확인할 것(차가 없으면 운으로 맞히는 문항이 된다).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T117",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T117.csv', 'reference_library.csv'],
    criteria="EXACT(material)",
    prompt=(
        "Identify the material of T117.csv against reference_library.csv, which includes PET "
        "and PMMA references. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T117 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported_label(run, "material", 'PET', ['PET', 'PMMA', 'aragonite', 'calcite', 'polystyrene', 'silicon'], name="material",
                           weight=2.0),
    ]
