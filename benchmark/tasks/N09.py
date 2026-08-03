# -*- coding: utf-8 -*-
"""N09 — 계측 제어 (2점)

[문제]
  Set the stage speed to X=9.0 mm/s and Z=0.5 mm/s, then report the speeds that will
  actually be used.

[정답 기준]
  GT=X는 상한 5.0, Z는 상한 0.1로 클리핑되고 응답의 clipped 목록에 두 축이 모두 실린다. 이 툴은 거부가 아니라 클리핑이라는 점이
  T078/T081(거부)과의 대비다. 확인=get_stage_speed가 5.0 / 0.1을 보고, 에이전트가 '요청대로 9.0이 걸렸다'고 답하면 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="N09",
    score=2,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Set the stage speed to X=9.0 mm/s and Z=0.5 mm/s, then report the speeds that will "
        "actually be used. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N09 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.arg(run, "set_stage_speed", "x_speed_mm_s", 9.0),
        chk.arg(run, "set_stage_speed", "z_speed_mm_s", 0.5),
        chk.state("실행 후 x_speed_mm_s", after, "x_speed_mm_s", 5.0),
        chk.state("실행 후 z_speed_mm_s", after, "z_speed_mm_s", 0.1),
        chk.keywords(run, ['클리핑', 'clip', '상한', 'limit', '제한']),
    ]
