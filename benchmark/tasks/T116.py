# -*- coding: utf-8 -*-
"""T116 — 신호 판별 (2점)

[문제]
  Compare T116.csv with reference_library.csv on the reference axis (L2, cosine). If the
  highest similarity is below 0.75, conclude that there is no reliable match and do not
  name a material.

[정답 기준]
  GT=(최고 유사도 값, '신뢰할 매칭 없음' 판정). 입력을 라이브러리에 없는 물질로 만들어 OOD 판정이 정답이 되게 한다.
  규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2 정규화 → 코사인 유사도. 이 3단계를 문항에 명시해 유사도 정의를
  단일화했다. 확인=임계 판정 일치. 임계 미만인데 물질명을 단정하면 오답(이 문항의 핵심 실패 모드).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T116",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T116.csv', 'reference_library.csv'],
    criteria="NUM(±0.01) + EXACT(out-of-library verdict) + PROC(must not name a material)",
    prompt=(
        "Compare T116.csv with reference_library.csv on the reference axis (L2, cosine). If "
        "the highest similarity is below 0.75, conclude that there is no reliable match and "
        "do not name a material. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T116 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "best_score", 0.0383559975855298, tol=0.01, name="best similarity"),
    ]
