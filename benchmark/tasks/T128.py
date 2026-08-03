# -*- coding: utf-8 -*-
"""T128 — 신호 판별 (2점)

[문제]
  Interpolate T128.csv and reference_library.csv onto the reference axis, apply IPBSA
  baseline order 5 and L2 normalization, then find the material with the highest cosine
  similarity. If two scores are equal, choose the one whose spectrum_id comes first in
  alphabetical order.

[정답 기준]
  GT=물질명 1개(+ 동점 시 선택된 spectrum_id). 전처리 순서와 타이브레이크를 모두 명시했다. 확인=물질명 완전 일치. 타이브레이크 규칙을 실제로
  채점하려면 동점이 발생하도록 참조를 설계해야 한다 — 그렇지 않으면 이 규칙은 채점되지 않는 장식이 된다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T128",
    score=2,
    axis="신호 판별",
    mode="live",
    inputs=['T128.csv', 'reference_library.csv'],
    prompt=(
        "Interpolate T128.csv and reference_library.csv onto the reference axis, apply IPBSA "
        "baseline order 5 and L2 normalization, then find the material with the highest "
        "cosine similarity. If two scores are equal, choose the one whose spectrum_id comes "
        "first in alphabetical order. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T128 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported_label(run, "material", "calcite", ['PET', 'PMMA', 'aragonite', 'calcite', 'polystyrene', 'silicon'], name="물질명"),
    ]
