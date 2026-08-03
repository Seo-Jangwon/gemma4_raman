# -*- coding: utf-8 -*-
"""T057 — 데이터 처리 (2점)

[문제]
  Check whether T057.csv contains spikes (5-point moving median, 5x MAD). If it does,
  remove them by linear interpolation and then report the major peaks again (prominence 5%
  of range).

[정답 기준]
  GT=(스파이크 유무 True, 제거 인덱스 집합, 제거 후 피크 위치 목록). 확인=판정·인덱스 완전 일치, 피크 위치 ±3 cm-1. 스파이크를 피크로
  보고하면 오답(핵심 함정).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T057",
    score=2,
    axis="데이터 처리",
    mode="live",
    inputs=['T057.csv'],
    prompt=(
        "Check whether T057.csv contains spikes (5-point moving median, 5x MAD). If it does, "
        "remove them by linear interpolation and then report the major peaks again "
        "(prominence 5% of range). "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T057 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.set_match("피크 위치", run.answer.get("peaks"), [620.0, 796.0, 1001.0, 1031.0, 1156.0, 1183.0, 1450.0, 1583.0, 1602.0], tol=TOL_PEAK_CM1),
    ]
