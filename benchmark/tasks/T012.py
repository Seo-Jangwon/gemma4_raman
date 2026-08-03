# -*- coding: utf-8 -*-
"""T012 — 계측 제어 (1점)

[문제]
  Set the camera exposure time to 50 ms.

[정답 기준]
  GT=set_camera_exposure(ms=50.0) 1회. T011과 같은 기준. 두 문항이 값만 다르므로 하나는 삭제하고 다른 기능으로 돌려도 좋다(중복
  문항).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T012",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Set the camera exposure time to 50 ms. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T012 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "set_camera_exposure", times=1),
        chk.arg(run, "set_camera_exposure", "ms", 50.0),
    ]
