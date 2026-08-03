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
from bench import answer as A
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T127",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T127.csv', 'reference_library.csv'],
    criteria="EXACT(verdict) + NUM(±0.01)",
    prompt=(
        # 원문은 "the material named in its metadata" 였는데 T127.csv 에는 메타데이터가
        # 없다(헤더가 raman_shift_cm-1,intensity 뿐). 무엇이라고 주장된 건지 알 수 없으니
        # 풀 수 없는 문항이었다. 주장 물질을 문면에 적어 넣어 판정 능력만 남긴다.
        "T127.csv was supplied to us as PMMA. Judge whether it really is, using "
        "reference_library.csv: accept if the cosine similarity to the PMMA reference is "
        "0.85 or higher, otherwise reject. "
    ),
    answer_keys=[
        ("matches", "true or false - accept (true) or reject (false)"),
        ("similarity", "number - cosine similarity to the claimed PMMA reference"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T127 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    said = A.flag(run, "matches", "verdict", "accept", "is_match")
    if said is None:
        said = run.last_mention(["reject", "accept"]) == "accept"
    return [
        chk.ok("accept / reject", said is False,
               f"reported={said} (expected False: similarity to the "
               f"claimed PMMA is "
               f"0.286 < 0.85)", weight=2.0, kind="EXACT"),
        chk.reported(run, "similarity", 0.286036,
                     tol=0.01, name="similarity to the claimed material"),
    ]
