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
    axis="procedure",
    mode="live",
    inputs=['T066.csv'],
    criteria="SET(coords EXACT) + PROC(no extra acquisitions)",
    prompt=(
        "T066.csv is a Raman map (columns: x, y, raman_shift_cm-1, intensity). Find the "
        "positions whose SNR (T050 definition) is below 10 and re-measure a spectrum once at "
        "each of those positions only. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T066 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    # SNR<10 인 좌표는 입력 맵이 정한다(T050 정의).
    want = [[37.8, 25.2]]
    n = run.count("acquire_spectrum")
    pts = [[round(p[0], 4), round(p[1], 4)] for p in run.positions()]
    return [
        chk.set_match("re-measured positions", pts or None, want, tol=1e-3, weight=2.0),
        # 초과 측정도 오답이다 — 필요 없는 자리에 빔을 더 쬐는 것이라서.
        chk.called(run, "acquire_spectrum", times=len(want), weight=2.0),
    ]
