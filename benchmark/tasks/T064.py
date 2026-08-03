# -*- coding: utf-8 -*-
"""T064 — 절차 구성 (3점)

[문제]
  Scan 10 positions at +0.0 to +0.9 mm in X (0.1 mm steps) from the current position. Apply
  spike removal (5-point median, 5x MAD), IPBSA baseline order 5 and L2 normalization to
  each spectrum, and plot the 1001 cm-1 intensity against position.

[정답 기준]
  GT(좌표)=x0+0.0…+0.9 의 10점, Y/Z 불변. GT(값)=사후 재계산한 10개 세기(각 상대오차 5%). 확인=meta.x가 GT와
  ±0.001mm, 그림 1장(점 10개), 정규화 후 각 스펙트럼 L2=1.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T064",
    score=3,
    axis="절차 구성",
    mode="live",
    windows=[('1001 cm-1 밴드', 995.0, 1007.0, 2)],
    prompt=(
        "Scan 10 positions at +0.0 to +0.9 mm in X (0.1 mm steps) from the current position. "
        "Apply spike removal (5-point median, 5x MAD), IPBSA baseline order 5 and L2 "
        "normalization to each spectrum, and plot the 1001 cm-1 intensity against position. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T064 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    if len(saved) < 10:
        return [chk.fail("10점 스캔", f"저장 {len(saved)}건 (10건 필요)", weight=2.0)]
    # 문항이 요구한 값은 '1001 cm-1 밴드의 세기'다. 스펙트럼 전체 최대값이 아니다 —
    # 예전 규칙은 전역 최대를 썼는데 그건 다른 물리량이라 보고값과 맞을 이유가 없다.
    ints = [sp.band_max(x, y, 995.0, 1007.0) for _, x, y in saved[:10]]
    if any(v is None for v in ints):
        return [chk.blocked("1001 cm-1 세기", "측정 축이 995~1007 cm-1 을 덮지 않습니다")]
    l2ok = all(abs(float(np.linalg.norm(y)) - 1.0) < 0.05 for _, _, y in saved[:10])
    got = run.answer.get("intensities")
    return [
        chk.ok("L2 정규화", l2ok, "각 스펙트럼의 L2 노름이 1"),
        chk.set_match("1001 cm-1 세기 10값",
                      [float(v) for v in got] if isinstance(got, list) else None,
                      ints, tol=max(ints) * 0.05, ordered=True, partial=True, weight=2.0),
    ]
