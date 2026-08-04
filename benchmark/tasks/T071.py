# -*- coding: utf-8 -*-
"""T071 — 신호 판별 (3점)

[문제]
  Measure spectra at 20 positions from X=37.0 to 38.9 mm in 0.1 mm steps. Compute each
  spectrum's similarity to T071_ref.csv (common axis, L2, cosine) and report the position
  where the similarity change between adjacent positions is largest.

[정답 기준]
  GT(좌표)=37.0~38.9 의 20점. GT(경계)=사후 재계산한 |Δ유사도| 최대 지점 — 에이전트 자신의 측정 파일 기준이므로 실측이어도 정답이
  유일하다. 확인=방문 좌표 20개 일치(±0.001mm), 보고 위치가 재계산 위치와 동일(인접 1칸까지 허용).

  [약속만 하고 안 재던 좌표 판정 — 2026-08-03]
  문서에는 '확인=meta.x 20개 일치'가 적혀 있는데 evaluate 에는 좌표 판정이 없었다.
  그러면 판정이 사후 GT 하나뿐이라, 아무 20 자리나 훑고 그 20 자리에 대해 일관되게 답한
  실행이 통과한다 — 정합성만 재고 정확성은 안 재는 상태였다. 이 문항의 좌표는 시작 위치와도
  무관한 절대값(X=37.0~38.9)이라 절대 GT 로 가장 깨끗하다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T071",
    score=3,
    axis="identification",
    mode="live",
    inputs=['T071_ref.csv'],
    criteria="SET(coords 20 items, ±0.001mm) + EXACT(boundary index, ±1 steps) / post-hoc GT",
    prompt=(
        "Measure spectra at 20 positions from X=37.0 to 38.9 mm in 0.1 mm steps. Compute "
        "each spectrum's similarity to T071_ref.csv (common axis, L2, cosine) and report the "
        "position where the similarity change between adjacent positions is largest. "
    ),
    answer_keys=[
        ("boundary_index",
         "number - the 0-based index of the boundary position among the 20 "
         "measured points"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T071 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    # 절대 GT 판정 — 사후 GT 와 반드시 짝지어야 한다(위 docstring 참조). Y 는 프롬프트가
    # 정하지 않으므로 시작값 그대로여야 한다.
    y0 = (before or {}).get("y")
    if y0 is None:
        absolute = [chk.blocked("20 scan coordinates",
                                "the starting stage position could not be read")]
    else:
        absolute = [chk.visited(run, [(37.0 + 0.1 * i, float(y0)) for i in range(20)],
                                tol=MM_GRID, name="20 scan coordinates")]

    ref = _input(b, "T071_ref.csv")
    saved = run.spectra()
    if ref is None or len(saved) < 20:
        return absolute + [chk.fail("boundary position",
                                    f"reference {'present' if ref else 'none'} / saved {len(saved)} files (need 20)",
                                    weight=2.0)]
    xr, yr = ref
    sims = []
    for _, x, y in saved[:20]:
        common = sp.on_common_axis(xr, yr, x, y)
        sims.append(0.0 if common is None
                    else sp.cosine(sp.l2(common[2]), sp.l2(common[1])))
    # 유사도가 가장 급하게 꺾이는 곳이 경계다.
    want = int(np.argmax(np.abs(np.diff(sims))))
    got = run.answer.get("boundary_index")
    return absolute + [
        chk.ok("boundary position", got is not None and abs(int(got) - want) <= 1,
               f"reported={got} expected={want}(±1)", weight=2.0),
    ]


def _input(b, name):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / name
    return sp.read_xy(p) if p.exists() else None
