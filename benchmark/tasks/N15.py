# -*- coding: utf-8 -*-
"""N15 — 계측 제어 (1점)

[문제]
  Turn the camera auto exposure on, then turn it off and set a manual exposure of 20 ms.

[정답 기준]
  GT=세 호출이 이 순서로. 자동 노출을 끄지 않고 수동 값을 걸면 설정이 덮어써지므로 순서가 정답의 일부다. 확인=호출 순서와 인자(true → false →
  20.0) 일치.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="N15",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="PROC(order and args EXACT)",
    prompt=(
        "Turn the camera auto exposure on, then turn it off and set a manual exposure of 20 "
        "ms. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N15 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "set_camera_auto_exposure", times=2),
        chk.called(run, "set_camera_exposure", times=1),
        chk.arg(run, "set_camera_exposure", "ms", 20.0),
        chk.arg_set(run, "set_camera_auto_exposure", "enabled", [True, False]),
        chk.order(run, "set_camera_auto_exposure", "set_camera_exposure"),
    ]
