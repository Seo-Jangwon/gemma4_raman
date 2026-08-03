# -*- coding: utf-8 -*-
"""T119 — 신호 판별 (2점)

[문제]
  Apply Savitzky-Golay smoothing (window_length=11, polyorder=3) to the low-signal spectrum
  T119.csv, interpolate onto the reference axis, L2 normalize, and identify the material
  against reference_library.csv.

[정답 기준]
  GT=물질명 1개. 규약=SG(11,3) → 참조축 보간 → L2 → 코사인. 평활을 건너뛰면 노이즈 때문에 순위가 흔들리도록 SNR을 낮게 설계할 것.
  확인=물질명 완전 일치.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T119",
    score=2,
    axis="신호 판별",
    mode="live",
    inputs=['reference_library.csv'],
    prompt=(
        "Apply Savitzky-Golay smoothing (window_length=11, polyorder=3) to the low-signal "
        "spectrum T119.csv, interpolate onto the reference axis, L2 normalize, and identify "
        "the material against reference_library.csv. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T119 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.fail("채점 항목 없음", "이 문항의 정답 기준이 아직 옮겨지지 않았습니다"),
    ]
