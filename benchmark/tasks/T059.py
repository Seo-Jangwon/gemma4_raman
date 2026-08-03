# -*- coding: utf-8 -*-
"""T059 — 절차 구성 (3점)

[문제]
  Perform autofocus, measure a spectrum once, apply IPBSA baseline correction with order 5,
  and report the peaks found (prominence 5% of range).

[정답 기준]
  GT(절차)=AF 1회 → 측정 1건 → poly_order=5 → 피크 보고, 순서 준수. GT(값)=사후 GT. 확인=run_autofocus 응답의 z
  이동 기록, bg 버전 1건의 poly_order=5, 피크 ±3 cm-1. AF 실패 시 z_limit_hits를 보고하고 재호출하지 않으면 가점, 반복
  호출은 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T059",
    score=3,
    axis="procedure",
    mode="live",
    windows=[('1602 cm-1 band', 1596.0, 1608.0, 2)],
    criteria="PROC(args EXACT) + SET(peak ±3 cm-1) / post-hoc GT",
    prompt=(
        "Perform autofocus, measure a spectrum once, apply IPBSA baseline correction with "
        "order 5, and report the peaks found (prominence 5% of range). "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T059 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [
        chk.called(run, "run_autofocus", times=1),
        chk.called(run, "acquire_spectrum", times=1),
        chk.arg(run, "apply_background_subtraction", "poly_order", 5),
    ]
    if not saved:
        return out + [chk.fail("peaks after correction", "no saved spectrum", weight=2.0)]
    # 에이전트가 보정 결과를 저장했다면 그것이 곧 답이다. 채점기가 한 번 더 보정하면
    # 이중 처리가 되어, 중간 산출물을 저장했는지 여부에 따라 정답이 달라진다.
    _, x, y = saved[-1]
    want = sp.peaks(x, y)[:5]
    got = run.answer.get("peaks")
    if not want:
        return out + [chk.fail("peaks after correction", "no peak was detected in the saved file",
                               weight=2.0)]
    return out + [
        chk.set_match("peaks after correction", [float(v) for v in got] if isinstance(got, list) else None,
                      want, tol=TOL_PEAK_CM1, partial=True, weight=2.0),
    ]
