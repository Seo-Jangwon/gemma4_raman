# -*- coding: utf-8 -*-
"""T044 — 데이터 처리 (2점)

[문제]
  In T044.csv, report the positions of the 3 highest-intensity peaks in descending order of
  intensity.

[정답 기준]
  GT=세기 내림차순 상위 3개 피크 위치(입력 파일로 확정). 확인=순서까지 일치, 각 위치 ±3 cm-1. 순서가 틀리면 오답(순위가 문항의 요구).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T044",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T044.csv'],
    criteria="SET(ordered 3 items, ±3 cm-1)",
    prompt=(
        "In T044.csv, report the positions of the 3 highest-intensity peaks in descending "
        "order of intensity. "
    ),
    answer_keys=[
        ("peaks",
         "list of 3 numbers - peak positions in cm-1, ordered by descending "
         "intensity. Numbers only, not objects"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T044 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        # ordered=True 가 빠져 있었다. 프롬프트·criteria·answer_keys 가 모두 '세기
        # 내림차순'을 요구하는데 집합 일치로만 봐서, 세 값의 6 가지 순열이 전부
        # 통과했다 — 순위가 이 문항의 요구인데 순위를 안 본 셈이다.
        chk.set_match("peak positions", run.answer.get("peaks"),
                      [1001.0, 1602.0, 1031.0], tol=TOL_PEAK_CM1, ordered=True),
    ]
