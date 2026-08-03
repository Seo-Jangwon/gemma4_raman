# -*- coding: utf-8 -*-
"""T017 — 계측 제어 (1점)

[문제]
  Set the CCD target temperature to -45 C and turn the cooler on.

[정답 기준]
  GT=목표 -45, cooler_on=True. 확인=두 툴이 모두 호출될 것(목표만 걸고 쿨러를 켜지 않으면 냉각이 시작되지 않는다). 도달까지 수 분
  걸리므로 현재 온도가 -45가 아닌 것은 정상 — 도달을 기다리며 get_ccd_info를 5회 이상 폴링하면 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T017",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="PROC(both tools called) + STATE(cooler_on True, <5 polls)",
    prompt=(
        "Set the CCD target temperature to -45 C and turn the cooler on. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T017 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "set_ccd_temperature", times=1),
        chk.called(run, "set_ccd_cooler", times=1),
        chk.called(run, "get_ccd_info", at_least=0, at_most=4),
        chk.arg(run, "set_ccd_temperature", "temp", -45),
        chk.state("final cooler_on", after, "cooler_on", True),
    ]
