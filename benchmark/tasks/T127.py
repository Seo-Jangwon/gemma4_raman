# -*- coding: utf-8 -*-
"""T127 — 신호 판별 (2점)

[문제]
  T127.csv is presented as being the material named in its metadata. Judge whether it
  really matches, using reference_library.csv: accept if the cosine similarity to that
  material's reference is 0.85 or higher, otherwise reject.

[정답 기준]
  GT=(일치 여부, 근거 유사도 값). 임계 0.85를 명시해 판정을 확정했다. 출제 시 참/거짓 사례를 모두 만들어야 판정 능력이 측정된다(거짓 사례만 있으면
  항상 reject가 정답이 된다). 확인=판정 일치 + 유사도 절대오차 0.01.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T127",
    score=2,
    axis="신호 판별",
    mode="live",
    inputs=['reference_library.csv'],
    prompt=(
        "T127.csv is presented as being the material named in its metadata. Judge whether it "
        "really matches, using reference_library.csv: accept if the cosine similarity to "
        "that material's reference is 0.85 or higher, otherwise reject. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T127 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.fail("채점 항목 없음", "이 문항의 정답 기준이 아직 옮겨지지 않았습니다"),
    ]
