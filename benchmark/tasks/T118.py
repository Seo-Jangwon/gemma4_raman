# -*- coding: utf-8 -*-
"""T118 — 신호 판별 (2점)

[문제]
  T118.csv is a mixture synthesised from two library materials. Compare it with
  reference_library.csv and report which component dominates the signal, together with the
  similarity of both candidates.

[정답 기준]
  GT=우세 성분 물질명 + 두 후보 유사도. 혼합비를 알고 합성하므로 정답이 확정된다 (우세=유사도 최대로 정의, 혼합비는 70:30 이상으로 벌려 모호성
  제거). 확인=물질명 일치 + 두 유사도 각 ±0.01. 한쪽만 보고하면 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T118",
    score=2,
    axis="신호 판별",
    mode="live",
    inputs=['reference_library.csv'],
    prompt=(
        "T118.csv is a mixture synthesised from two library materials. Compare it with "
        "reference_library.csv and report which component dominates the signal, together "
        "with the similarity of both candidates. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T118 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported_label(run, "dominant", "polystyrene", ['polystyrene', 'PMMA'], name="우세 성분"),
    ]
