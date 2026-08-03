# -*- coding: utf-8 -*-
"""T068 — 절차 구성 (3점)

[문제]
  Measure a spectrum 5 times at the same position and report the sample relative standard
  deviation (std with ddof=1 divided by the mean, in percent) of the 1001 cm-1 peak
  intensity.

[정답 기준]
  GT=사후 재계산한 RSD(%). ddof=1을 명시해 모/표본 분기를 없앴다. 확인=저장 5건·좌표 불변, 값이 재계산과 상대오차 5% 이내. 부가 GT=0
  <= RSD < 100 (범위를 벗어나면 산식 오류로 즉시 오답).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T068",
    score=3,
    axis="procedure",
    mode="live",
    windows=[('1001 cm-1 band', 995.0, 1007.0, 2)],
    criteria="PROC + NUM(5%) + REL(0<=RSD<100) / post-hoc GT",
    prompt=(
        "Measure a spectrum 5 times at the same position and report the sample relative "
        "standard deviation (std with ddof=1 divided by the mean, in percent) of the 1001 "
        "cm-1 peak intensity. "
    ),
    answer_keys=[
        ("rsd_pct", "number - the relative standard deviation in percent"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T068 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [chk.called(run, "acquire_spectrum", times=5)]
    if len(saved) < 5:
        return out + [chk.fail("RSD", f"saved {len(saved)} files (need 5)")]
    vals = [sp.band_max(x, y, 996.0, 1006.0) for _, x, y in saved[:5]]
    if any(v is None for v in vals):
        return out + [chk.blocked("RSD", "the instrument axis does not cover 996-1006 cm-1")]
    rsd = sp.rsd_percent(vals)
    if rsd is None:
        return out + [chk.fail("RSD", "the mean is 0, so RSD is undefined")]
    return out + [
        chk.reported(run, "rsd_pct", rsd, rel=0.05, name="RSD(%)", weight=2.0),
        chk.ok("value range", 0 <= rsd < 100, f"{rsd:.3g}% ∈ [0,100)"),
    ]
