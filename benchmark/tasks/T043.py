# -*- coding: utf-8 -*-
"""T043 — 데이터 처리 (2점)

[문제]
  Find the 7 major Raman peaks of T043.csv (polystyrene). Use scipy.signal.find_peaks with
  prominence set to 5% of the intensity range, and report the 7 largest by prominence in
  ascending order of position, with their intensities.

[정답 기준]
  GT=규약대로 검출한 7개 (위치, 세기). prominence 규약을 넣어 '주요 7개'의 모호성을 없앴다. 확인=7개 위치가 GT와 1:1 매칭(±3
  cm-1), 세기 상대오차 5%.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T043",
    score=2,
    axis="데이터 처리",
    mode="live",
    inputs=['T043.csv'],
    prompt=(
        "Find the 7 major Raman peaks of T043.csv (polystyrene). Use scipy.signal.find_peaks "
        "with prominence set to 5% of the intensity range, and report the 7 largest by "
        "prominence in ascending order of position, with their intensities. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T043 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.set_match("피크 위치", run.answer.get("peaks"), [620.0, 793.0, 1001.0, 1031.0, 1154.0, 1450.0, 1602.0], tol=TOL_PEAK_CM1),
    ]
