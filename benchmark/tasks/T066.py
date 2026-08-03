# -*- coding: utf-8 -*-
"""T066 — 절차 구성 (3점)

[문제]
  T066.csv is a Raman map (columns: x, y, raman_shift_cm-1, intensity). Find the positions
  whose SNR (T050 definition) is below 10 and re-measure a spectrum once at each of those
  positions only.

[정답 기준]
  GT(판정)=입력 맵에서 SNR<10 인 좌표 집합 — 파일 입력이라 결정적. GT(재측정)=그 좌표에만 이동·측정. 초과 측정도 오답으로 본다(불필요한 조사량
  증가). 확인=재측정 건수와 meta 좌표가 GT 집합과 완전 일치. 재측정 스펙트럼 값은 채점 대상 아님.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T066",
    score=3,
    axis="절차 구성",
    mode="live",
    inputs=['T066.csv'],
    prompt=(
        "T066.csv is a Raman map (columns: x, y, raman_shift_cm-1, intensity). Find the "
        "positions whose SNR (T050 definition) is below 10 and re-measure a spectrum once at "
        "each of those positions only. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T066 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.fail("채점 항목 없음", "이 문항의 정답 기준이 아직 옮겨지지 않았습니다"),
    ]
