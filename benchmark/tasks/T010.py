# -*- coding: utf-8 -*-
"""T010 — 계측 제어 (1점)

[문제]
  Start real-time camera streaming and report whether it was already running.

[정답 기준]
  GT=start_camera_stream 1회 + already_streaming 값을 그대로 보고. 확인=응답 필드를 인용했는지(이 값이 이후 stop 여부를
  좌우하므로 읽는 습관을 본다).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T010",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="PROC + EXACT(already_streaming reported)",
    prompt=(
        "Start real-time camera streaming and report whether it was already running. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T010 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "start_camera_stream", times=1),
        chk.has_answer_key(run, "already_streaming"),
    ]
