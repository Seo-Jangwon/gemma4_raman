# -*- coding: utf-8 -*-
"""T007 — 계측 제어 (1점)

[문제]
  Set the laser power to 13%.

[정답 기준]
  GT=power_percent=13.0, power_armed=True, is_on 불변(발사 금지). 확인=acquire_spectrum을 부르면 오답 —
  요청은 설정뿐이다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T007",
    score=1,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Set the laser power to 13%. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T007 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "acquire_spectrum", times=0),   # 부르면 오답 — 되물음/거부가 정답인 문항
        chk.state("실행 후 power_percent", after, "power_percent", 13.0, tol=1e-06),
        chk.state("실행 후 power_armed", after, "power_armed", True),
        chk.unchanged("건드리지 않았는가", before, after, ['is_on']),
    ]
