# -*- coding: utf-8 -*-
"""T114 — 신호 판별 (2점)

[문제]
  Compute the highest cosine similarity between T114.csv and the references in
  reference_library.csv (interpolate onto the reference axis, L2 normalize) and state
  whether it is 0.85 or higher.

[정답 기준]
  GT=(최고 유사도 값, 0.85 이상 여부). 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2 정규화 → 코사인 유사도. 이
  3단계를 문항에 명시해 유사도 정의를 단일화했다. 확인=값 절대오차 0.01, 임계 판정 일치. 값과 판정이 서로 모순되면 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import answer as A
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T114",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T114.csv', 'reference_library.csv'],
    criteria="NUM(±0.01) + EXACT(threshold verdict)",
    prompt=(
        "Compute the highest cosine similarity between T114.csv and the references in "
        "reference_library.csv (interpolate onto the reference axis, L2 normalize) and state "
        "whether it is 0.85 or higher. "
    ),
    answer_keys=[
        ("best_score", "number - the highest cosine similarity"),
        ("above_threshold", "true or false - is it 0.85 or higher"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T114 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    best = 0.9958523269835462
    said = A.flag(run, "above_threshold", "is_match", "above", "match")
    if said is None:
        said = run.last_mention(["below", "above"]) == "above"
    return [
        chk.reported(run, "best_score", best, tol=0.01, name="highest similarity",
                     weight=2.0),
        chk.ok("threshold verdict", said is True,
               f"reported={said} (expected True, "
               f"best={best:.4f} vs 0.85)", weight=2.0, kind="EXACT"),
    ]
