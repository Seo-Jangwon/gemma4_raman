# -*- coding: utf-8 -*-
"""T071 — 신호 판별 (3점)

[문제]
  Measure spectra at 20 positions from X=37.0 to 38.9 mm in 0.1 mm steps. Compute each
  spectrum's similarity to T071_ref.csv (common axis, L2, cosine) and report the position
  where the similarity change between adjacent positions is largest.

[정답 기준]
  GT(좌표)=37.0~38.9 의 20점. GT(경계)=사후 재계산한 |Δ유사도| 최대 지점 — 에이전트 자신의 측정 파일 기준이므로 실측이어도 정답이
  유일하다. 확인=meta.x 20개 일치, 보고 위치가 재계산 위치와 동일(인접 1칸까지 허용).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T071",
    score=3,
    axis="신호 판별",
    mode="live",
    inputs=['T071_ref.csv'],
    prompt=(
        "Measure spectra at 20 positions from X=37.0 to 38.9 mm in 0.1 mm steps. Compute "
        "each spectrum's similarity to T071_ref.csv (common axis, L2, cosine) and report the "
        "position where the similarity change between adjacent positions is largest. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T071 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    ref = _input(b, "T071_ref.csv")
    saved = run.spectra()
    if ref is None or len(saved) < 20:
        return [chk.fail("경계 위치",
                         f"참조 {'있음' if ref else '없음'} / 저장 {len(saved)}건 (20건 필요)",
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
    return [
        chk.ok("경계 위치", got is not None and abs(int(got) - want) <= 1,
               f"보고={got} 정답={want}(±1)", weight=2.0),
    ]


def _input(b, name):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / name
    return sp.read_xy(p) if p.exists() else None
