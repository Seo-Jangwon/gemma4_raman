# -*- coding: utf-8 -*-
"""T035 — 절차 구성 (2점)

[문제]
  Acquire a kinetic time series of 3 frames at the current position with a 5 s frame
  interval.

[정답 기준]
  원문('t=0,5,10s에 3회')은 3회 개별 호출과 kinetic 1회 모두로 읽혀 GT가 갈렸다. kinetic으로 못박았다. GT=저장 CSV가
  frame_index 0/1/2를 가진 kinetic 롱포맷, num_frames=3, kinetic_cycle_time=5. 확인=응답의
  mode='kinetic', num_frames=3. load_spectrum으로 읽으려 하면 거부되는 것이 정상(1D 전용) — 거부 후
  run_analysis로 전환하면 가점, 거부를 고장으로 오판하면 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T035",
    score=2,
    axis="절차 구성",
    mode="live",
    prompt=(
        "Acquire a kinetic time series of 3 frames at the current position with a 5 s frame "
        "interval. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T035 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    out = [
        chk.arg(run, "acquire_spectrum", "kinetic_count", 3),
        chk.arg(run, "acquire_spectrum", "kinetic_cycle_time", 5),
        chk.called(run, "reconnect_hardware", times=0),
    ]
    frames, mode = None, None
    for c in run.calls:
        r = c.get("result")
        if isinstance(r, dict):
            if isinstance(r.get("frames"), list) and r["frames"]:
                frames = r["frames"]
            if "mode" in r:
                mode = r["mode"]
    return out + [
        chk.ok("kinetic 모드", mode == "kinetic", f"mode={mode}"),
        chk.ok("프레임 3개", frames is not None and len(frames) == 3,
               f"{len(frames) if frames else 0}개", weight=2.0),
    ]
