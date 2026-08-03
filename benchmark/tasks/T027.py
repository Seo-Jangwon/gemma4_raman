# -*- coding: utf-8 -*-
"""T027 — 절차 구성 (2점)

[문제]
  Move the stage to X=38, Y=26, Z=0 mm, perform autofocus, then capture the view with
  capture_scene, acquire a Raman spectrum, and record them as one measurement point before
  moving elsewhere.

[정답 기준]
  T026과 같은 사유로 capture_scene으로 교체했다. GT=레코드 1건, position.x=38 / y=26(±0.001, z는 AF 결과라 자유),
  missing 없음. 판별점=save_measurement_point는 호출 시점의 좌표를 새로 읽으므로 이동 전에 저장해야 한다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T027",
    score=2,
    axis="절차 구성",
    mode="live",
    prompt=(
        "Move the stage to X=38, Y=26, Z=0 mm, perform autofocus, then capture the view with "
        "capture_scene, acquire a Raman spectrum, and record them as one measurement point "
        "before moving elsewhere. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T027 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    return [
        chk.called(run, "save_measurement_point", times=1),
        chk.called(run, "capture_scene", times=1),
        chk.state("실행 후 x", after, "x", 38.0, tol=MM_GRID),
        chk.state("실행 후 y", after, "y", 26.0, tol=MM_GRID),
    ] + _point_complete(run)


def _point_complete(run):
    results = run.results("save_measurement_point")
    if not results:
        return [chk.fail("측정점 완전", "save_measurement_point 를 부르지 않았습니다",
                         weight=2.0)]
    missing = [r.get("missing") for r in results if r.get("missing")]
    return [chk.ok("측정점 완전", not missing, f"missing={missing or '없음'}", weight=2.0)]
