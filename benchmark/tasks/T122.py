# -*- coding: utf-8 -*-
"""T122 — 신호 판별 (2점)

[문제]
  Report the identification info (spectrum_id and material) of the reference in
  reference_library.csv that best matches T122.csv.

[정답 기준]
  GT=최상위 참조의 (spectrum_id, material). 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2 정규화 →
  코사인 유사도. 이 3단계를 문항에 명시해 유사도 정의를 단일화했다. 확인=spectrum_id까지 완전 일치. 같은 물질의 다른 항목(PS_01 vs
  PS_02)을 답하면 부분점 — 이 문항은 '물질'이 아니라 '항목'을 식별하는지를 본다. 출제 시 동일 물질의 두 참조 간 유사도 차가 0.02 이상인지
  확인할 것.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T122",
    score=2,
    axis="신호 판별",
    mode="live",
    inputs=['T122.csv', 'reference_library.csv'],
    prompt=(
        "Report the identification info (spectrum_id and material) of the reference in "
        "reference_library.csv that best matches T122.csv. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T122 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.fail("채점 항목 없음", "이 문항의 정답 기준이 아직 옮겨지지 않았습니다"),
    ]
