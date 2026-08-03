# -*- coding: utf-8 -*-
"""T030 — 계측 제어 (2점)

[문제]
  Compute the stage coordinates corresponding to pixel (320, 240) in the camera image and
  move there.

[정답 기준]
  원본 2차가 '육안 검증'이었다. 변환식이 결정적이므로 좌표로 채점할 수 있다. GT=optics_map.pixel_to_stage(320, 240, x0,
  y0) — 이미지 중심이 현재 위치에 대응하고 픽셀 +X는 스테이지 -X다(부호 반전이 판별점). 확인=이동 후 get_stage_position이 GT와
  ±0.001 일치. 부호를 반대로 계산하면 대칭 위치로 가서 걸린다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T030",
    score=2,
    axis="instrument control",
    mode="live",
    criteria="STATE(coords ±0.001)",
    prompt=(
        "Compute the stage coordinates corresponding to pixel (320, 240) in the camera image "
        "and move there. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T030 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    PIXEL = (320, 240)
    out = [
        chk.called(run, "move_to_pixel", times=1),
        chk.arg(run, "move_to_pixel", "pixel_x", PIXEL[0]),
        chk.arg(run, "move_to_pixel", "pixel_y", PIXEL[1]),
    ]
    if "x" not in before or "x" not in after:
        return out + [chk.fail("pixel to stage", "could not read the instrument state")]
    try:
        from backend.hw_tools import optics_map
        wx, wy = optics_map.pixel_to_stage(PIXEL[0], PIXEL[1],
                                           float(before["x"]), float(before["y"]))
    except Exception as e:
        return out + [chk.fail("pixel to stage", f"conversion failed: {type(e).__name__}: {e}")]
    # 판별점은 부호다. 화면 좌표계와 스테이지 좌표계의 방향이 반대라, 부호를 뒤집으면
    # 정확히 반대편으로 간다.
    return out + [
        chk.near("X after the move", after.get("x"), wx, tol=1e-3, weight=2.0),
        chk.near("Y after the move", after.get("y"), wy, tol=1e-3, weight=2.0),
    ]
