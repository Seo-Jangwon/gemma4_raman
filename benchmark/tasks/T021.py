# -*- coding: utf-8 -*-
"""T021 — 절차 구성 (2점)

[문제]
  Set the CCD exposure time to 1.0 s and the laser power to 15%, then measure a spectrum
  once at the current position.

[정답 기준]
  GT=측정 1건, 응답의 exposure_time=1.0 / laser_power_pct=15(하드웨어 readback이라 실제 적용값이다). 확인=미리 걸고
  인자 없이 측정하든, acquire_spectrum(exposure=1.0, power=15)로 한 번에 하든 같은 코드를 타므로 둘 다 정답. 다만 둘을 모두
  하면 감점(중복).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T021",
    score=2,
    axis="procedure",
    mode="live",
    criteria="STATE(readback EXACT) + PROC(no duplicate calls)",
    prompt=(
        "Set the CCD exposure time to 1.0 s and the laser power to 15%, then measure a "
        "spectrum once at the current position. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T021 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    got_e, got_p = _last(run, "exposure_time"), _last(run, "laser_power_pct")
    n_pre = run.count("set_ccd_exposure") + run.count("set_laser_power")
    inline = any(c.get("name") == "acquire_spectrum"
                 and ({"exposure", "power"} & set(c.get("args") or {}))
                 for c in run.calls)
    return [
        chk.called(run, "acquire_spectrum", times=1),
        chk.near("exposure read-back", got_e, 1.0, tol=1e-6),
        chk.near("power read-back", got_p, 15.0, rel=0.02),
        # 사전 설정과 인자를 겹쳐 쓰면 어느 쪽이 적용됐는지 알 수 없다.
        chk.ok("no redundant setting", not (n_pre >= 2 and inline),
               f"preset {n_pre} calls + inline arg {'present' if inline else 'none'}"),
    ]


def _last(run, key):
    v = None
    for c in run.calls:
        r = c.get("result")
        if isinstance(r, dict) and key in r:
            v = r[key]
    return v
