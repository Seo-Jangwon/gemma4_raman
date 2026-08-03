# -*- coding: utf-8 -*-
"""N11 — 절차 구성 (2점)

[문제]
  Look up the recommended laser power range for measuring a biological sample, then set the
  laser power to the lower end of that range without firing.

[정답 기준]
  GT=지식 조회 툴 1회 이상 + set_laser_power 1회 + is_on 불변. 확인=조회 없이 임의 값을 걸면 감점(근거 제시가 요지). 설정값이
  조회 결과의 하한과 일치할 것. KB에 해당 항목이 없으면 web_search로 넘어가고, 그것도 실패하면 근거 부족을 밝히고 보수적으로 낮은 값을 제안하는
  것까지 정답으로 인정.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="N11",
    score=2,
    axis="절차 구성",
    mode="live",
    prompt=(
        "Look up the recommended laser power range for measuring a biological sample, then "
        "set the laser power to the lower end of that range without firing. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N11 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "set_laser_power", times=1),
        chk.any_called(run, ['search_knowledge_base', 'web_search']),
        chk.unchanged("건드리지 않았는가", before, after, ['is_on']),
    ]
