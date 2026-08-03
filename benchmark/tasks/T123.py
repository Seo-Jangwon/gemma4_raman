# -*- coding: utf-8 -*-
"""T123 — 신호 판별 (2점)

[문제]
  T123.csv has one strong peak near 520 cm-1. Identify the material against
  reference_library.csv and report the peak position you used as the basis.

[정답 기준]
  GT=(물질명=silicon, 근거 피크 위치≈520.7 cm-1). 확인=물질명 완전 일치 + 보고한 피크 위치가 GT와 ±3 cm-1. 근거 피크를 제시하지
  않으면 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T123",
    score=2,
    axis="신호 판별",
    mode="live",
    inputs=['reference_library.csv'],
    prompt=(
        "T123.csv has one strong peak near 520 cm-1. Identify the material against "
        "reference_library.csv and report the peak position you used as the basis. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T123 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.fail("채점 항목 없음", "이 문항의 정답 기준이 아직 옮겨지지 않았습니다"),
    ]
