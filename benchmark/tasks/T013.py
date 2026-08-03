# -*- coding: utf-8 -*-
"""T013 — 계측 제어 (1점)

[문제]
  Tell me the CCD sensor's width and height in pixels.

[정답 기준]
  GT=get_ccd_info의 detector_Nx / detector_Ny. 원본 2차가 'None'이었다. 확인=보고한 두 값이 툴 반환과 완전 일치.
  readout_pixels_Nx(현재 읽기 영역)와 detector_Nx(센서 전체)를 혼동하면 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T013",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Tell me the CCD sensor's width and height in pixels. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T013 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "detector_Nx", after.get("detector_Nx"), tol=0, name="보고한 detector_Nx"),
        chk.reported(run, "detector_Ny", after.get("detector_Ny"), tol=0, name="보고한 detector_Ny"),
    ]
