# -*- coding: utf-8 -*-
"""T103 — 진단 복구 (3점)

[문제]
  All peaks of T103.csv are shifted by the same amount from the expected positions of the
  reference T103_ref.csv. Estimate the shift by cross-correlation, correct it, and report
  both the estimated shift and the corrected peak positions.

[정답 기준]
  [TS-A] 원문은 실측 후 시프트를 논해 기준이 흔들렸다. 알려진 시프트를 주입한 파일로 바꿔 완전 GT로 만들었다. GT(시프트)=+1.0 cm-1(생성 시
  값), 허용 ±0.2. GT(보정 후 피크)=레퍼런스 피크 위치. 확인=추정 시프트 부호까지 일치(부호 반대는 오답), 보정 후 피크 ±3 cm-1.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T103",
    score=3,
    axis="진단 복구",
    mode="live",
    inputs=['T103_ref.csv'],
    prompt=(
        "All peaks of T103.csv are shifted by the same amount from the expected positions of "
        "the reference T103_ref.csv. Estimate the shift by cross-correlation, correct it, "
        "and report both the estimated shift and the corrected peak positions. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T103 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "shift_cm1", 1.0, tol=0.2, name="시프트"),
    ]
