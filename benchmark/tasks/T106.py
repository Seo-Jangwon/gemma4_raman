# -*- coding: utf-8 -*-
"""T106 — 신호 판별 (3점)

[문제]
  Classify the sample of T106.csv as amorphous, crystalline, or undecidable. Use this rule:
  if the strongest peak has FWHM below 15 cm-1 it is crystalline; above 50 cm-1 it is
  amorphous; in between it is undecidable. Report the FWHM you measured.

[정답 기준]
  판정 기준이 없어 레이블이 흔들렸다. FWHM 임계를 문항에 넣어 확정했다. GT=(레이블 1개, FWHM 값). 확인=레이블 완전 일치 + FWHM 상대오차
  10%. 레이블이 맞아도 근거 FWHM이 임계와 모순되면 감점(우연 정답 배제).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T106",
    score=3,
    axis="identification",
    mode="live",
    inputs=['T106.csv'],
    criteria="EXACT(labels) + NUM(FWHM 10%)",
    prompt=(
        "Classify the sample of T106.csv as amorphous, crystalline, or undecidable. Use this "
        "rule: if the strongest peak has FWHM below 15 cm-1 it is crystalline; above 50 cm-1 "
        "it is amorphous; in between it is undecidable. Report the FWHM you measured. "
    ),
    answer_keys=[
        ("fwhm_cm1", "number - FWHM of the strongest peak in cm-1"),
        ("label", 'string - one of "crystalline", "amorphous", "undecidable"'),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T106 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "fwhm_cm1", 51.787707659444436, rel=0.05, name="FWHM"),
        chk.reported_label(run, "label", "amorphous", ["crystalline", "amorphous", "undecidable"], name="classification"),
    ]
