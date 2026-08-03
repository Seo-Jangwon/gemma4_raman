# -*- coding: utf-8 -*-
"""T050 — 데이터 처리 (2점)

[문제]
  From T050.csv compute the signal-to-noise ratio as max(intensity in 990-1012 cm-1)
  divided by the sample standard deviation (ddof=1) of 1800-1900 cm-1. Both intervals are
  inclusive.

[정답 기준]
  GT=규약대로 계산한 SNR 1값. ddof와 경계 포함을 명시해 확정했다. 확인=상대오차 5% 이내.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T050",
    score=2,
    axis="데이터 처리",
    mode="live",
    windows=[('SNR 신호창', 990.0, 1012.0, 3), ('SNR 잡음창', 1800.0, 1900.0, 2)],
    inputs=['T050.csv'],
    prompt=(
        "From T050.csv compute the signal-to-noise ratio as max(intensity in 990-1012 cm-1) "
        "divided by the sample standard deviation (ddof=1) of 1800-1900 cm-1. Both intervals "
        "are inclusive. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T050 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "snr", 158.99276556166572, rel=0.05, name="SNR"),
    ]
