# -*- coding: utf-8 -*-
"""T070 — 신호 판별 (3점)

[문제]
  Measure a 5x5 grid spanning +0.0 to +0.4 mm in X and Y from the current position (0.1 mm
  spacing). Compare each spectrum with the polystyrene reference T070_ref.csv (interpolate
  onto a common axis, L2 normalize, cosine) and draw a 5x5 map marking the positions with
  similarity 0.85 or higher.

[정답 기준]
  GT(좌표)=25점, center=(x0+0.2, y0+0.2), spacing_mm=0.1. GT(유사도)=사후 재계산한 25값(각 절대오차 0.01)과
  0.85 통과 마스크. 규약(공통축·L2·코사인)을 명시해 유사도 정의를 확정했다. 확인=25개 좌표 방문, 맵 1장, 25값과 통과 개수.

  [약속만 하고 안 재던 두 판정 — 2026-08-03]
  문서에는 GT 좌표 25점과 '확인=5×5 맵 1장'이 적혀 있는데 evaluate 는 둘 다 보지 않았다.
  남는 판정이 전부 사후 GT(에이전트 자신의 파일로 되계산)라, 엉뚱한 자리에서 25점을 찍고
  그 25점에 대해 일관되게 답하면 통과한다 — 정합성만 재고 정확성은 안 재는 상태였다.
  좌표는 시작 위치와 프롬프트로 완전히 정해지는 절대 GT 이므로 그것으로 짝을 세운다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T070",
    score=3,
    axis="identification",
    mode="live",
    inputs=['T070_ref.csv'],
    criteria="SET(coords 25 items, ±0.001mm) + STATE(figure) + NUM(±0.01) + EXACT(mask) / post-hoc GT",
    prompt=(
        "Measure a 5x5 grid spanning +0.0 to +0.4 mm in X and Y from the current position "
        "(0.1 mm spacing). Compare each spectrum with the polystyrene reference T070_ref.csv "
        "(interpolate onto a common axis, L2 normalize, cosine) and draw a 5x5 map marking "
        "the positions with similarity 0.85 or higher. "
    ),
    answer_keys=[
        ("similarities",
         "list of 25 numbers - the cosine similarity at each grid point, in the "
         "order you measured them"),
        ("n_above_threshold", "number - how many of them are 0.85 or higher"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T070 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    THRESHOLD = 0.85

    # ── 절대 GT 판정 ─────────────────────────────────────────────────────────
    # 좌표는 '문항 시작 위치 + 프롬프트가 정한 격자'로 완전히 정해진다. 사후 GT 와
    # 짝지어야 하는 판정이 바로 이것이다(위 docstring 참조).
    x0, y0 = (before or {}).get("x"), (before or {}).get("y")
    if x0 is None or y0 is None:
        absolute = [chk.blocked("25 grid coordinates",
                                "the starting stage position could not be read")]
    else:
        want_xy = [(float(x0) + 0.1 * i, float(y0) + 0.1 * j)
                   for j in range(5) for i in range(5)]
        absolute = [chk.visited(run, want_xy, tol=MM_GRID, name="25 grid coordinates")]
    absolute.append(chk.figure(run, name="5x5 similarity map saved"))

    ref = _input(b, "T070_ref.csv")
    saved = run.spectra()
    if ref is None or len(saved) < 25:
        return absolute + [chk.fail("similarity map",
                                    f"reference {'present' if ref else 'none'} / saved {len(saved)} files (need 25)",
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
    return absolute + [
        chk.set_match("25 similarity values",
                      [float(v) for v in got] if isinstance(got, list) else None,
                      sims, tol=0.01, ordered=True, partial=True, weight=2.0),
        chk.reported(run, "n_above_threshold", float(n_pass), tol=0,
                     name=f"{THRESHOLD} count at or above") if got_n is not None else
        chk.fail(f"{THRESHOLD} count at or above", f"not reported (expected {n_pass}/25)"),
    ]


def _input(b, name):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / name
    return sp.read_xy(p) if p.exists() else None
