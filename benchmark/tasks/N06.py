# -*- coding: utf-8 -*-
"""N06 — 계측 제어 (2점)

[문제]
  Acquire a dark reference frame with the shutter closed, then a normal frame with the
  shutter on auto, and report the mean intensity of each.

[정답 기준]
  GT=측정 2건, 셔터 인자가 각각 close/auto. 확인=응답의 shutter 필드, 그리고 보고한 두
  평균이 실제로 저장된 두 프레임의 평균과 맞는가(각 5% 이내).

[예전 기준을 버린 이유 — 2026-08-03]
  예전 판정은 '암프레임 평균 < 정상프레임 평균'이었다. 시료와 무관하게 성립한다고 봤지만
  **이 벤치에서는 성립하지 않는다**: reset_all 이 매 문항 맨 앞에서 laser_off 를 걸므로
  문항이 시작될 때 레이저는 꺼져 있고, N06 은 그걸 켜는 setup 이 없다. 그래서 두 프레임이
  모두 CCD 바이어스 바닥이었다(실측 107.2 vs 106.7 — 0.5 카운트짜리 잡음 차이). 대소
  판정이 사실상 동전던지기가 된다.
  레이저를 켜는 setup 을 붙이는 선택지도 있었지만, 이 문항이 묻는 것은 '셔터를 제대로
  가려 두 프레임을 얻고 각 평균을 보고했는가'이지 광신호의 크기가 아니다. 프롬프트가
  요구한 그대로 — 보고한 값이 실제 프레임과 맞는지 — 를 채점한다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="N06",
    score=2,
    axis="instrument control",
    mode="live",
    criteria="PROC(shutter EXACT) + NUM(reported means match the saved frames, 5%)",
    prompt=(
        "Acquire a dark reference frame with the shutter closed, then a normal frame with "
        "the shutter on auto, and report the mean intensity of each. "
    ),
    answer_keys=[
        ("dark_mean", "number - mean intensity of the shutter-closed frame"),
        ("normal_mean", "number - mean intensity of the shutter-auto frame"),
    ],
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
        return out + [chk.fail("reported frame means",
                               f"saved {len(saved)} files (need 2)", weight=2.0)]
    # 저장 순서가 곧 측정 순서다(프롬프트가 dark → normal 순서를 지정했다).
    dark, normal = (float(y.mean()) for _, _, y in saved[:2])
    return out + [
        chk.reported(run, "dark_mean", dark, rel=0.05, name="reported dark mean"),
        chk.reported(run, "normal_mean", normal, rel=0.05, name="reported normal mean"),
    ]
