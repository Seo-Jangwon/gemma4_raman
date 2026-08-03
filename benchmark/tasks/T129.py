# -*- coding: utf-8 -*-
"""T129 — 신호 판별 (2점)

[문제]
  All peaks of T129.csv are shifted by the same unknown amount. Estimate the shift by
  cross-correlation against reference_library.csv, correct it, identify the material, and
  report the estimated shift.

[정답 기준]
  GT=(추정 시프트 Δ, 물질명). 생성 시 Δ를 기록하므로 완전 GT다. 확인=Δ가 GT와 ±0.2 cm-1이고 부호까지 일치(부호 반대는 오답), 물질명
  완전 일치. Δ는 축 간격의 정수배가 아닌 값으로 잡아 보간 능력까지 시험할 것.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T129",
    score=2,
    axis="신호 판별",
    mode="live",
    inputs=['reference_library.csv'],
    prompt=(
        "All peaks of T129.csv are shifted by the same unknown amount. Estimate the shift by "
        "cross-correlation against reference_library.csv, correct it, identify the material, "
        "and report the estimated shift. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T129 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "shift_cm1", 2.7, tol=0.2, name="시프트"),
        chk.reported_label(run, "material", "polystyrene", ['polystyrene'], name="물질명"),
    ]
