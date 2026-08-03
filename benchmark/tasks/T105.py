# -*- coding: utf-8 -*-
"""T105 — 진단 복구 (3점)

[문제]
  T105.csv holds 10 consecutive measurements of the same spot (column frame_index 0-9).
  Determine whether the Raman signal decreases and the background increases over time,
  quantify both trends by linear regression slope, and name the cause.

[정답 기준]
  [TS-A] 실제 광표백은 시료를 손상시키므로 재현하지 않는다. 추세를 주입한 시계열 파일로 대체했다. GT(원인)=광표백(photobleaching) —
  키워드 채점. GT(수치)=신호 기울기<0, 배경 기울기>0 (각 기울기 값은 상대오차 10%). 확인=두 기울기의 부호가 GT와 일치할 것(부호가 핵심,
  크기는 부차).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T105",
    score=3,
    axis="diagnostics",
    mode="live",
    inputs=['T105.csv'],
    criteria="KEYWORD(photobleaching) + NUM(slope 10%) + REL(sign)",
    prompt=(
        "T105.csv holds 10 consecutive measurements of the same spot (column frame_index "
        "0-9). Determine whether the Raman signal decreases and the background increases "
        "over time, quantify both trends by linear regression slope, and name the cause. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T105 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "signal_slope", -49.28848874219068, rel=0.1, name="signal slope"),
        chk.reported(run, "background_slope", 24.855960678628154, rel=0.1, name="background slope"),
    ]
