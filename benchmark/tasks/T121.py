# -*- coding: utf-8 -*-
"""T121 — 신호 판별 (2점)

[문제]
  Compare T121.csv with all 8 references in reference_library_8.csv and sort every
  reference by similarity, descending. Use cosine similarity computed on the reference
  wavenumber axis (linear interpolation) after L2-normalizing both spectra. If two scores
  are equal, order them by spectrum_id ascending.

[정답 기준]
  GT=8개 전체의 내림차순 정렬 목록(spectrum_id 기준). 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2 정규화 →
  코사인 유사도. 이 3단계를 문항에 명시해 유사도 정의를 단일화했다. 동점 처리 규칙을 명시해 순서를 유일하게 만들었다. 확인=순서 완전 일치, 각 점수
  절대오차 0.01. 8개 미만을 보고하면 감점.

  [규약이 GT json 에만 있었다 — 2026-08-06]
  이 docstring 은 "3단계를 문항에 명시해 유사도 정의를 단일화했다"고 적었지만 실제
  TASK.prompt 에는 한 글자도 없었다 — 규약은 gt/T121.json 의 rule 필드에만 있어서
  사람만 읽는 주석이었다. 프롬프트가 말한 것은 "sort by similarity" 뿐이라, 똑같이
  표준적인 피어슨 상관으로 푼 실행이 이렇게 나온다(입력으로 실측):
      PET_01 코사인 0.1675 vs 피어슨 0.0670   CAL_01 0.0313 vs **-0.0271**
      SI_01  0.0110 vs **-0.0258**
  점수는 ±0.01 에서 다섯 개가 탈락하고 **CAL_01/SI_01 은 순서까지 뒤집혀** 순서 판정도
  죽는다. 판정이 기대하는 계산을 문항이 말하게 한다.
  (덧붙여 코사인은 배율 불변이라 L2 단계는 점수를 바꾸지 않고, 두 축이 200~2000 step 1
  로 같아 보간도 항등이다. 실제로 답을 가르는 것은 "어떤 유사도 함수인가" 하나였다.)

  [동점 규칙은 이 데이터에서 한 번도 발동하지 않는다]
  8개 점수에 동점이 없으므로 tie_break 문구는 장식이다. T128 은 같은 문제를 알고
  쿼리를 두 참조의 등거리에 놓아 규칙이 실제로 발동하게 만들었다. 이 문항도 그렇게
  할 수 있으면 좋겠지만, 그건 입력 재생성이라 별건으로 둔다 — 문구는 무해하므로 남긴다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import answer as A
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T121",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T121.csv', 'reference_library_8.csv'],
    criteria="EXACT(8 items in order) + NUM(scores ±0.01)",
    prompt=(
        "Compare T121.csv with all 8 references in reference_library_8.csv and sort every "
        "reference by similarity, descending. Use cosine similarity computed on the "
        "reference wavenumber axis (linear interpolation) after L2-normalizing both "
        "spectra. If two scores are equal, order them by spectrum_id ascending. "
    ),
    answer_keys=[
        ("ranking",
         'list of 8 objects, most similar first, each {"spectrum_id": string, '
         '"score": number}'),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T121 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    order = ["PS_01", "PS_03", "PS_02", "PET_02", "PMMA_01", "PET_01", "CAL_01", "SI_01"]
    scores = [0.994552, 0.993954, 0.978229, 0.228639, 0.179389, 0.167487, 0.031256, 0.010961]
    got_i = A.seq(run, "ranking", "sorted", "results", "references", field="spectrum_id")
    got_s = A.seq(run, "ranking", "sorted", "results", "references", field="score",
                  cast=float)
    return [
        chk.set_match("all 8 references in order", got_i, order, tol=0, ordered=True,
                      partial=True, weight=2.0),
        chk.set_match("similarity scores", got_s, scores, tol=0.01, ordered=True),
    ]
