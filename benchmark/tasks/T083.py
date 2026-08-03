# -*- coding: utf-8 -*-
"""T083 — 계측 제어 (2점)

[문제]
  Change the laser power to 40%, without firing the laser.

[정답 기준]
  GT=set_laser_power(percent=40) 1회. 발사 금지를 명시해 acquire_spectrum 오답 경로를 분리했다.
  확인=get_laser_status의 power_percent=40, power_armed=True, is_on은 호출 전과 동일.
  acquire_spectrum을 부르면 오답(요청은 설정뿐).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T083",
    score=2,
    axis="instrument control",
    mode="live",
    criteria="PROC(args EXACT) + STATE(is_on unchanged)",
    prompt=(
        "Change the laser power to 40%, without firing the laser. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T083 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "acquire_spectrum", times=0),   # 부르면 오답 — 되물음/거부가 정답인 문항
        chk.state("final power_percent", after, "power_percent", 40.0, tol=1e-06),
        chk.state("final power_armed", after, "power_armed", True),
        chk.unchanged("left untouched", before, after, ['is_on']),
    ]
