# -*- coding: utf-8 -*-
"""T014 — 계측 제어 (1점)

[문제]
  Set the CCD readout mode to FVB.

[정답 기준]
  GT=get_ccd_info의 read_mode가 fvb. 확인=드라이버 표기와 인자 표기가 다를 수 있으므로 채점기는 정규화 후 비교한다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T014",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="STATE(read_mode EXACT)",
    prompt=(
        "Set the CCD readout mode to FVB. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T014 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.state("final read_mode", after, "read_mode", "fvb"),
    ]
