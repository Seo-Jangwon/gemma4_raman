# -*- coding: utf-8 -*-
"""T052 — 데이터 처리 (2점)

[문제]
  Pair the peaks of T052_a.csv and T052_b.csv (prominence 5% of range, nearest-neighbour
  1:1 within 10 cm-1) and report only the pairs whose position difference is 5 cm-1 or
  more.

[정답 기준]
  GT=조건을 만족하는 쌍 목록(각 쌍의 두 위치와 차이). 매칭 상한 10 cm-1을 명시해 '대응 피크'를 확정했다. 확인=쌍 집합 완전 일치(개수·구성원).
  5 cm-1 미만 쌍을 섞어 보고하면 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T052",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T052_a.csv', 'T052_b.csv'],
    criteria="SET(pairs EXACT, position ±1 cm-1)",
    prompt=(
        "Pair the peaks of T052_a.csv and T052_b.csv (prominence 5% of range, "
        "nearest-neighbour 1:1 within 10 cm-1) and report only the pairs whose position "
        "difference is 5 cm-1 or more. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T052 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    PAIR_MAX, MIN_DIFF = 10.0, 5.0
    a, bb = _input(b, "T052_a.csv"), _input(b, "T052_b.csv")
    if a is None or bb is None:
        return [chk.fail("peak pairs", "could not read the inputs T052_a/b.csv", weight=2.0)]
    pa, pb = sp.peaks(*a), sp.peaks(*bb)
    pairs, diffs = [], []
    for p in pa:
        cand = [q for q in pb if abs(q - p) <= PAIR_MAX]
        if not cand:
            continue
        q = min(cand, key=lambda v: abs(v - p))
        if abs(q - p) >= MIN_DIFF:
            pairs.append([round(float(p), 3), round(float(q), 3)])
            diffs.append(round(float(abs(q - p)), 3))
    if not pairs:
        # '해당 쌍 없음'도 정답일 수 있다. 그때는 없다고 답해야 맞다.
        got = run.answer.get("pairs")
        return [chk.ok("reported that no pair qualifies", not got,
                       f"reported={got}", weight=2.0)]
    got_pairs = run.answer.get("pairs")
    got_diff = run.answer.get("differences")
    return [
        # 차이값만 맞으면 통과하던 자리 — 어느 피크끼리 짝지었는지가 본질이다.
        chk.set_match("peak pairs", got_pairs if isinstance(got_pairs, list) else None,
                      pairs, tol=1.0, weight=2.0),
        chk.set_match("pair differences", [float(v) for v in got_diff] if isinstance(got_diff, list) else None,
                      diffs, tol=1.0),
    ]


def _input(b, name):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / name
    return sp.read_xy(p) if p.exists() else None
