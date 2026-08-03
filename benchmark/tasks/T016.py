# -*- coding: utf-8 -*-
"""T016 — 계측 제어 (1점)

[문제]
  Set the CCD exposure time to 1.0 s.

[정답 기준]
  GT=get_ccd_info의 exposure_time_s=1.0(±1e-6).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_EXPOSURE_S, TOL_PEAK_CM1   # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T016",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="STATE(exposure EXACT)",
    prompt=(
        "Set the CCD exposure time to 1.0 s. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T016 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        # tol 은 장비의 노출 클럭 분해능에 맞춘다. 1e-6 을 요구하면 정확히 1.0 s 를
        # 건 실행도 되읽기가 1.00002 s 라 확정 실패한다(TOL_EXPOSURE_S 주석 참고).
        chk.state("final exposure_time_s", after, "exposure_time_s", 1.0,
                  tol=TOL_EXPOSURE_S),
    ]
