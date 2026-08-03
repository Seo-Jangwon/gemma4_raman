# -*- coding: utf-8 -*-
"""T051 — 데이터 처리 (2점)

[문제]
  T051.csv holds a 3x3 Raman map (columns: x, y, raman_shift_cm-1, intensity). At each
  position take the intensity of the sample nearest to 1000 cm-1 and save a spatial
  heatmap.

[정답 기준]
  GT=좌표별 9개 값과 그 (x,y) 배치. 입력 포맷을 문항에 명시해 파싱 모호성을 없앴다. 확인=히트맵 데이터 배열이 GT 9값과 rtol 1e-6 일치,
  축이 좌표에 대응(전치되면 오답).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T051",
    score=2,
    axis="데이터 처리",
    mode="live",
    inputs=['T051.csv'],
    prompt=(
        "T051.csv holds a 3x3 Raman map (columns: x, y, raman_shift_cm-1, intensity). At "
        "each position take the intensity of the sample nearest to 1000 cm-1 and save a "
        "spatial heatmap. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T051 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.fail("채점 항목 없음", "이 문항의 정답 기준이 아직 옮겨지지 않았습니다"),
    ]
