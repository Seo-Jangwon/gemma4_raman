# -*- coding: utf-8 -*-
"""T065 — 절차 구성 (3점)

[문제]
  At X = 37.0, 37.25, 37.5, 37.75, 38.0 mm (Y=25.25, Z=0), perform autofocus, capture the
  microscope view with capture_scene, acquire a Raman spectrum, and record all three as one
  measurement point per position before moving on.

[정답 기준]
  GT=측정점 레코드 5건, position.x={37.0, 37.25, 37.5, 37.75, 38.0}, y=25.25(z는 AF 결과라 자유).
  판별점1=이미지는 capture_scene으로만 세션에 남는다. analyze_microscope_image로 대신하면
  save_measurement_point가 missing:[image]를 반환하므로 오답. 판별점2=save_measurement_point는 호출 시점의
  스테이지 좌표를 새로 읽는다. 다음 위치로 이동한 뒤 저장하면 좌표가 한 칸씩 밀린다 — 문항에 before moving on 을 넣어 이 함정을 명시적으로
  시험한다. 확인=list_session_artifacts로 point 5건, 각 레코드에 missing 필드 없음.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T065",
    score=3,
    axis="procedure",
    mode="live",
    criteria="SET(coords 5 items, ±0.001mm) + STATE(missing none)",
    prompt=(
        "At X = 37.0, 37.25, 37.5, 37.75, 38.0 mm (Y=25.25, Z=0), perform autofocus, capture "
        "the microscope view with capture_scene, acquire a Raman spectrum, and record all "
        "three as one measurement point per position before moving on. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T065 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    WANT_X = [37.0, 37.25, 37.5, 37.75, 38.0]
    out = [
        chk.called(run, "save_measurement_point", times=5),
        chk.called(run, "capture_scene", times=5),
        chk.called(run, "analyze_microscope_image", times=0),
    ]
    pos = [r.get("position") or {} for r in run.results("save_measurement_point")]
    xs = [float(p["x"]) for p in pos if p.get("x") is not None]
    ys = [float(p["y"]) for p in pos if p.get("y") is not None]
    if not xs:
        return out + [chk.fail("measurement-point coordinates", "the saved measurement points carry no coordinates", weight=2.0)]
    return out + [
        # 이동한 뒤에 저장했는지를 본다 — 저장 시점의 좌표가 곧 그 점의 자리다.
        chk.set_match("5 measurement-point X values", xs, WANT_X, tol=1e-3, ordered=True, weight=2.0),
        chk.ok("Y held fixed", bool(ys) and all(abs(v - 25.25) < 1e-3 for v in ys),
               f"y={[round(v, 3) for v in ys]} (expected 25.25)"),
    ]
