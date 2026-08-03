# -*- coding: utf-8 -*-
"""T034 — 절차 구성 (2점)

[문제]
  Set the detector to accumulate 10 exposures, measure once, then divide the accumulated
  spectrum by 10 with run_analysis and save that per-exposure average.

[정답 기준]
  원문('10회 평균을 저장')은 장비 동작과 어긋난다 — accumulate는 누적합이라 평균이 되려면 /10 단계가 필요하다. 그 단계를 문항에 명시해 GT를
  확정했다. GT(설정)=get_ccd_info의 acquisition_mode='accumulate', num_accumulations=10. GT(값)=측정
  파일 intensity / 10 (채점기가 재계산). 확인=저장 2건(원본+평균), 평균 파일이 원본/10과 일치.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T034",
    score=2,
    axis="절차 구성",
    mode="live",
    prompt=(
        "Set the detector to accumulate 10 exposures, measure once, then divide the "
        "accumulated spectrum by 10 with run_analysis and save that per-exposure average. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T034 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [
        chk.arg(run, "set_ccd_acquisition_mode", "mode", "accumulate"),
        chk.arg(run, "set_ccd_acquisition_mode", "num_accumulations", 10),
        chk.state("실행 후 acquisition_mode", after, "acquisition_mode", "accumulate"),
        chk.state("실행 후 num_accumulations", after, "num_accumulations", 10),
    ]
    if len(saved) < 2:
        return out + [chk.fail("누적/10", f"저장 {len(saved)}건 (2건 필요)")]
    # 방향이 중요하다. 예전에는 모든 (i,j) 순서쌍을 다 시도해서 '10배 곱하기'도 통과했다.
    # 누적 결과를 10 으로 **나눈** 것이 뒤에 저장된다고 보고, 그 방향만 인정한다.
    acc, per = saved[0][2], saved[-1][2]
    n = min(len(acc), len(per))
    return out + [
        chk.array("누적을 10으로 나눔", per[:n], acc[:n] / 10.0, mode="exact", weight=2.0),
    ]
