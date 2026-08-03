# -*- coding: utf-8 -*-
"""T024 — 절차 구성 (2점)

[문제]
  At the same position, measure once at laser power 20%, 40% and 60%, and report the three
  saved file names.

[정답 기준]
  GT=측정 3건, laser_power_pct readback 20/40/60, 좌표 불변. 확인=매회 get_laser_status를 부를 필요는 없다(응답에
  readback이 있다) — 불필요한 조회를 3회 이상 반복하면 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T024",
    score=2,
    axis="procedure",
    mode="live",
    criteria="STATE(power EXACT ×3) + PROC(no redundant queries)",
    prompt=(
        "At the same position, measure once at laser power 20%, 40% and 60%, and report the "
        "three saved file names. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T024 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "acquire_spectrum", times=3),
        chk.called(run, "get_laser_status", at_least=0, at_most=2),
        chk.arg_set(run, "acquire_spectrum", "power", [20, 40, 60]),
        chk.unchanged("left untouched", before, after, ['x', 'y']),
    ]
