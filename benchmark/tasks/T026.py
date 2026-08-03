# -*- coding: utf-8 -*-
"""T026 — 절차 구성 (2점)

[문제]
  At the same position, capture the microscope view with capture_scene, acquire a Raman
  spectrum, and record both as one measurement point before moving.

[정답 기준]
  원본 1차의 capture_camera_frame은 없는 툴이다. 더 중요한 것은 analyze_microscope_image로 대체하면 안 된다는 점 — 그
  툴은 세션에 이미지를 남기지 않아 save_measurement_point가 missing:[image]를 반환한다. capture_scene으로 못박았다.
  GT=측정점 레코드 1건, missing 필드 없음. 확인=list_session_artifacts로 point 1건.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T026",
    score=2,
    axis="procedure",
    mode="live",
    criteria="PROC(tool choice EXACT) + STATE(missing none)",
    prompt=(
        "At the same position, capture the microscope view with capture_scene, acquire a "
        "Raman spectrum, and record both as one measurement point before moving. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T026 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    return [
        chk.called(run, "capture_scene", times=1),
        chk.called(run, "save_measurement_point", times=1),
        chk.called(run, "analyze_microscope_image", times=0),
    ] + _point_complete(run)


def _point_complete(run):
    """저장된 측정점이 빠진 것 없이 완성됐는가.

    save_measurement_point 를 **한 번도 안 불렀으면 통과가 아니다**. 예전에는 missing 이
    None 이라 '빠진 것 없음'으로 읽혀 아무것도 안 한 실행이 가중치 2.0 을 그냥 받았다.
    """
    results = run.results("save_measurement_point")
    if not results:
        return [chk.fail("measurement point complete", "save_measurement_point was never called",
                         weight=2.0)]
    missing = [r.get("missing") for r in results if r.get("missing")]
    return [chk.ok("measurement point complete", not missing, f"missing={missing or 'none'}", weight=2.0)]
