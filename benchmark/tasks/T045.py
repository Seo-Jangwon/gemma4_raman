# -*- coding: utf-8 -*-
"""T045 — 데이터 처리 (2점)

[문제]
  Compute the FWHM of the strongest peak inside 980-1020 cm-1 of T045.csv. Take the half
  height as (peak intensity - interval minimum)/2 and find the crossings by linear
  interpolation.

[정답 기준]
  GT=규약대로 계산한 FWHM(cm-1). 반높이 기준을 명시해 구현차를 제거했다. 확인=상대오차 5% 이내.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T045",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T045.csv'],
    criteria="NUM(5%)",
    prompt=(
        "Compute the FWHM of the strongest peak inside 980-1020 cm-1 of T045.csv. Take the "
        "half height as (peak intensity - interval minimum)/2 and find the crossings by "
        "linear interpolation. "
    ),
    answer_keys=[
        ("fwhm_cm1", "number - FWHM in cm-1"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T045 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "fwhm_cm1", 5.834533146562421, rel=0.05, name="FWHM"),
    ]
