# -*- coding: utf-8 -*-
"""T025 — 절차 구성 (2점)

[문제]
  Perform autofocus once, then measure a spectrum once at the focus position it found.
  Report the Z before and after the autofocus.

[정답 기준]
  원본 2차가 '육안 검증'이었다. AF 전후 Z를 보고하게 해 자동 채점이 가능해졌다. GT=AF 1회 + 측정 1건, 보고한 z_before/z_after가
  get_stage_position 및 run_autofocus 반환과 일치. 확인=AF 실패 시 z_limit_hits를 보고하고 재호출하지 않을 것.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T025",
    score=2,
    axis="procedure",
    mode="live",
    criteria="NUM(z ±1e-4 ×2) + PROC(1 autofocus call)",
    prompt=(
        "Perform autofocus once, then measure a spectrum once at the focus position it "
        "found. Report the Z before and after the autofocus. "
    ),
    answer_keys=[
        ("z_after", "number - stage Z in mm after the move"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T025 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "run_autofocus", times=1),
        chk.called(run, "acquire_spectrum", times=1),
        chk.reported(run, "z_after", after.get("z"), tol=0.0001, name="reported z_after"),
    ]
