# -*- coding: utf-8 -*-
"""T011 — 계측 제어 (1점)

[문제]
  Set the camera exposure time to 10 ms.

[정답 기준]
  GT=set_camera_exposure(ms=10.0) 1회. 원본의 '육안 검증'을 인자 대조로 바꿨다. 확인=인자 값 EXACT. 카메라 노출은 조회 툴이
  없으므로 호출 인자가 유일한 근거다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T011",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="PROC(args EXACT)",
    prompt=(
        "Set the camera exposure time to 10 ms. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T011 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "set_camera_exposure", times=1),
        chk.arg(run, "set_camera_exposure", "ms", 10.0),
    ]
