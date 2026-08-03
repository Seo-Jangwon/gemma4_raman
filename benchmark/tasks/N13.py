# -*- coding: utf-8 -*-
"""N13 — 계측 제어 (2점)

[문제]
  Read the available vertical and horizontal shift speeds, then set the vertical shift
  speed to the slowest available index and report both the index and its speed value.

[정답 기준]
  GT=get_ccd_info를 먼저 호출해 vs_speeds_us 리스트를 읽고, 가장 느린(=값이 큰) 항목의 인덱스를 vs_index로 설정. 확인=조회
  없이 인덱스를 지어내면 오답. 참고=hs_speeds_conventional_mhz는 채널 0의 속도 '리스트'다(스칼라가 아니다) — 이 필드를 단일 값으로
  오해하는지도 함께 관찰한다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="N13",
    score=2,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Read the available vertical and horizontal shift speeds, then set the vertical "
        "shift speed to the slowest available index and report both the index and its speed "
        "value. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N13 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    speeds = _last(run, "vs_speeds_us") or before.get("vs_speeds_us")
    out = [chk.order(run, "get_ccd_info", "set_ccd_shift_speeds")]
    if not isinstance(speeds, list) or not speeds:
        return out + [chk.fail("vs 인덱스", "vs_speeds_us 를 관측하지 못했습니다")]
    # 세로 전송이 느릴수록(값이 클수록) 읽기 잡음이 작다. 가장 큰 값의 인덱스가 정답이다.
    want = int(np.argmax([float(v) for v in speeds]))
    got = (run.args("set_ccd_shift_speeds", "vs_index") or [None])[0]
    return out + [
        chk.ok("가장 느린 vs 선택", got == want,
               f"선택={got} 정답={want} (speeds={speeds})", weight=2.0),
    ]


def _last(run, key):
    """도구 응답들에서 마지막으로 관측된 필드."""
    v = None
    for c in run.calls:
        r = c.get("result")
        if isinstance(r, dict) and key in r:
            v = r[key]
    return v
