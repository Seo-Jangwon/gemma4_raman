# -*- coding: utf-8 -*-
"""T070 — 신호 판별 (3점)

[문제]
  Measure a 5x5 grid spanning +0.0 to +0.4 mm in X and Y from the current position (0.1 mm
  spacing). Compare each spectrum with the polystyrene reference T070_ref.csv (interpolate
  onto a common axis, L2 normalize, cosine) and draw a 5x5 map marking the positions with
  similarity 0.85 or higher.

[정답 기준]
  GT(좌표)=25점, center=(x0+0.2, y0+0.2), spacing_mm=0.1. GT(유사도)=사후 재계산한 25값(각 절대오차 0.01)과
  0.85 통과 마스크. 규약(공통축·L2·코사인)을 명시해 유사도 정의를 확정했다. 확인=5×5 맵 1장.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T070",
    score=3,
    axis="신호 판별",
    mode="live",
    inputs=['T070_ref.csv'],
    prompt=(
        "Measure a 5x5 grid spanning +0.0 to +0.4 mm in X and Y from the current position "
        "(0.1 mm spacing). Compare each spectrum with the polystyrene reference T070_ref.csv "
        "(interpolate onto a common axis, L2 normalize, cosine) and draw a 5x5 map marking "
        "the positions with similarity 0.85 or higher. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T070 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    THRESHOLD = 0.85
    ref = _input(b, "T070_ref.csv")
    saved = run.spectra()
    if ref is None or len(saved) < 25:
        return [chk.fail("유사도 맵",
                         f"참조 {'있음' if ref else '없음'} / 저장 {len(saved)}건 (25건 필요)",
                         weight=2.0)]
    xr, yr = ref
    sims = []
    for _, x, y in saved[:25]:
        common = sp.on_common_axis(xr, yr, x, y)
        sims.append(0.0 if common is None
                    else sp.cosine(sp.l2(common[2]), sp.l2(common[1])))
    n_pass = sum(1 for s in sims if s >= THRESHOLD)
    got = run.answer.get("similarities")
    got_n = run.answer.get("n_above_threshold")
    # 예전에는 '25점 측정'과 '통과 개수'가 무조건 통과라, 아무 값도 보고하지 않아도
    # 2차 만점이었다. 보고한 값으로만 채점한다.
    return [
        chk.set_match("유사도 25값",
                      [float(v) for v in got] if isinstance(got, list) else None,
                      sims, tol=0.01, ordered=True, partial=True, weight=2.0),
        chk.reported(run, "n_above_threshold", float(n_pass), tol=0,
                     name=f"{THRESHOLD} 이상 개수") if got_n is not None else
        chk.fail(f"{THRESHOLD} 이상 개수", f"보고 없음 (정답 {n_pass}/25)"),
    ]


def _input(b, name):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / name
    return sp.read_xy(p) if p.exists() else None
