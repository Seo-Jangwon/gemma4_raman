# -*- coding: utf-8 -*-
"""T061 — 절차 구성 (3점)

[문제]
  Measure a spectrum once at X=37 and once at X=38 mm (Y=25.25, Z=0). Interpolate both onto
  the wavenumber axis of the first spectrum, normalize each to L2 norm 1, and report their
  cosine similarity.

[정답 기준]
  원문은 유사도가 실측 의존이라 GT 불가였다. 규약(공통축=첫 스펙트럼 축, L2, 코사인)을 명시해 '두 파일에 대한 결정적 함수'로 바꿨다 — 채점기가
  에이전트의 두 측정 파일로 재계산한다. GT(좌표)={(37, 25.25, 0), (38, 25.25, 0)}. GT(값)=사후 재계산 유사도, 절대오차
  0.01.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T061",
    score=3,
    axis="절차 구성",
    mode="live",
    prompt=(
        "Measure a spectrum once at X=37 and once at X=38 mm (Y=25.25, Z=0). Interpolate "
        "both onto the wavenumber axis of the first spectrum, normalize each to L2 norm 1, "
        "and report their cosine similarity. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T061 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [chk.called(run, "acquire_spectrum", times=2)]
    if len(saved) < 2:
        return out + [chk.fail("코사인 유사도", f"저장 {len(saved)}건 (2건 필요)")]
    _, xa, ya = saved[0]
    _, xb, yb = saved[1]
    common = sp.on_common_axis(xa, ya, xb, yb)
    if common is None:
        return out + [chk.fail("코사인 유사도", "두 스펙트럼의 축이 겹치지 않습니다")]
    _, ga, gb = common
    want = sp.cosine(sp.l2(ga), sp.l2(gb))
    return out + [chk.reported(run, "cosine", want, tol=0.01, name="코사인 유사도",
                               weight=2.0)]
