# -*- coding: utf-8 -*-
"""N06 — 계측 제어 (2점)

[문제]
  Acquire a dark reference frame with the shutter closed, then a normal frame with the
  shutter on auto, and report the mean intensity of each.

[정답 기준]
  GT=측정 2건, 셔터 인자가 각각 close/auto. GT(관계)=다크 프레임의 평균 < 정상 프레임의 평균 (시료와 무관하게 성립). 확인=응답의
  shutter 필드, 두 평균의 대소.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="N06",
    score=2,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Acquire a dark reference frame with the shutter closed, then a normal frame with "
        "the shutter on auto, and report the mean intensity of each. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N06 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [
        chk.arg_set(run, "acquire_spectrum", "shutter", ["close", "auto"]),
        chk.called(run, "acquire_spectrum", times=2),
    ]
    if len(saved) < 2:
        return out + [chk.fail("암/정상 비교", f"저장 {len(saved)}건 (2건 필요)")]
    # 셔터를 닫고 찍은 쪽이 암프레임이다. 저장 순서가 아니라 '무엇으로 찍었는가'로 가른다.
    means = [float(y.mean()) for _, _, y in saved[:2]]
    return out + [
        chk.ok("암프레임 평균 < 정상", means[0] < means[1],
               f"{means[0]:.4g} < {means[1]:.4g}", weight=2.0),
    ]
