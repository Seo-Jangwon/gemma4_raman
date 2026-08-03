# -*- coding: utf-8 -*-
"""T108 — 진단 복구 (3점)

[문제]
  T108.csv holds 10 consecutive measurements (column frame_index 0-9) whose baseline drifts
  over time. Quantify the drift as the linear regression slope of the median intensity per
  frame, and state whether it is increasing or decreasing.

[정답 기준]
  [TS-A] 드리프트를 실제로 만들려면 장시간이 필요해 재현이 어렵다. 주입 파일로 대체했다. GT=기울기 값(생성 시 값, 상대오차 10%)과 증감 방향.
  확인=방향이 일치해야 하고, 기울기 정의(프레임별 중앙값의 회귀)를 문항에 명시해 산식을 확정했다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T108",
    score=3,
    axis="진단 복구",
    mode="live",
    prompt=(
        "T108.csv holds 10 consecutive measurements (column frame_index 0-9) whose baseline "
        "drifts over time. Quantify the drift as the linear regression slope of the median "
        "intensity per frame, and state whether it is increasing or decreasing. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T108 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "drift_slope", 17.98994735783482, rel=0.1, name="드리프트 기울기"),
    ]
