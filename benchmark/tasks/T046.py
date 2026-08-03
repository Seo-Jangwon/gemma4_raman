# -*- coding: utf-8 -*-
"""T046 — 데이터 처리 (2점)

[문제]
  Compare the peaks of T046_sample.csv with the polystyrene reference T046_ref.csv and
  compute the match ratio. Detect peaks with prominence at 5% of range, pair them 1:1 by
  nearest neighbour within 3 cm-1, and define the ratio as matched peaks / reference peaks.

[정답 기준]
  GT=(일치율 소수값, 매칭된 피크쌍 목록). 1:1·중복금지·분모를 명시해 확정했다. 확인=일치율 ±0.02, 매칭쌍 집합 완전 일치.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T046",
    score=2,
    axis="데이터 처리",
    mode="live",
    inputs=['T046_sample.csv', 'T046_ref.csv'],
    prompt=(
        "Compare the peaks of T046_sample.csv with the polystyrene reference T046_ref.csv "
        "and compute the match ratio. Detect peaks with prominence at 5% of range, pair them "
        "1:1 by nearest neighbour within 3 cm-1, and define the ratio as matched peaks / "
        "reference peaks. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T046 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "match_ratio", 1.0, tol=0.02, name="일치율"),
    ]
