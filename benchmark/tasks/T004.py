# -*- coding: utf-8 -*-
"""T004 — 계측 제어 (1점)

[문제]
  Set the stage movement speed to 2.0 mm/s.

[정답 기준]
  GT=x_speed=y_speed=2.0(상한 5.0 이내라 클리핑 없음). z는 상한 0.1이므로 2.0을 z까지 걸면 클리핑되고 clipped에 실린다 —
  축을 지정하지 않은 요청이라 x/y만 거는 것이 정답. 확인=get_stage_speed의 x/y가 2.0(±1e-6).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T004",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Set the stage movement speed to 2.0 mm/s. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T004 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.state("실행 후 x_speed_mm_s", after, "x_speed_mm_s", 2.0, tol=1e-06),
        chk.state("실행 후 y_speed_mm_s", after, "y_speed_mm_s", 2.0, tol=1e-06),
    ]
