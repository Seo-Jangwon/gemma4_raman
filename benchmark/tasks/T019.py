# -*- coding: utf-8 -*-
"""T019 — 데이터 처리 (1점)

[문제]
  Capture the current camera view, look at it, and describe what you see.

[정답 기준]
  원본 1차의 capture_camera_frame은 지금 존재하지 않는 툴이다(제거됨). 목적이 '보고 묘사하기'이므로
  analyze_microscope_image로 교체했다 — 이 툴만 이미지를 모델에게 준다. GT=analyze_microscope_image 1회 + 밝기
  통계(min/max/mean_intensity)를 근거로 인용한 묘사. 확인=capture_scene을 쓰면 이미지가 반환되지 않아 묘사가 불가능하므로 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T019",
    score=1,
    axis="data processing",
    mode="live",
    criteria="PROC(tool choice EXACT) + KEYWORD(cited statistics)",
    prompt=(
        "Look at the current microscope view and describe what you see. Back the "
        "description with the image intensity statistics (mean and maximum) rather "
        "than impressions alone. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T019 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "analyze_microscope_image", at_least=1, at_most=2),
        chk.called(run, "capture_scene", times=0),
        chk.keywords(run, ['intensity', 'bright', 'mean']),
    ]
