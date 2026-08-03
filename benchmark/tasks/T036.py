# -*- coding: utf-8 -*-
"""T036 — 계측 제어 (2점)

[문제]
  Keep the current X and Y, and acquire once at each of Z = -0.002, -0.001, 0, 0.001, 0.002
  mm.

[정답 기준]
  GT(좌표)=Z 5값 {-0.002,-0.001,0,0.001,0.002}(가동범위 -1~1 안이라 전부 유효), X/Y는 시작값 불변. 확인=각 이동 후
  get_stage_position의 z가 GT와 ±1e-4 일치, 저장 5건, x/y 변동 없음(±0.001).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T036",
    score=2,
    axis="instrument control",
    mode="live",
    criteria="SET(5 Z values, ±1e-4) + STATE(x/y unchanged)",
    prompt=(
        "Keep the current X and Y, and acquire once at each of Z = -0.002, -0.001, 0, 0.001, "
        "0.002 mm. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T036 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    WANT_Z = [-0.002, -0.001, 0.0, 0.001, 0.002]
    out = [chk.unchanged("X/Y left untouched", before, after, ["x", "y"])]
    # 절대 이동이면 인자의 z, 상대 이동이면 시작 Z 에서 누적한다.
    zs = [float(v) for v in run.args("move_stage", "z")]
    if not zs:
        base, dz = before.get("z"), run.args("move_stage_relative", "dz")
        if base is not None and dz:
            cur = float(base)
            for d in dz:
                cur += float(d)
                zs.append(cur)
    if not zs:
        return out + [chk.fail("5 Z coordinates", "no Z move was recorded", weight=2.0)]
    return out + [chk.set_match("5 Z coordinates", zs, WANT_Z, tol=MM, ordered=True, weight=2.0)]
