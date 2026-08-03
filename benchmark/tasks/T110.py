# -*- coding: utf-8 -*-
"""T110 — 절차 구성 (3점)

[문제]
  Measure at the current position and then at +0.1 mm in X, -0.1 mm in X, +0.1 mm in Y and
  -0.1 mm in Y, in that order. Report at which of the five positions the spectrum differs
  most from the first one, using cosine distance after L2 normalization.

[정답 기준]
  원문('520 cm-1만 보이는 상태'에서 '시료 신호가 처음 나타나는 위치')은 전제와 답이 모두 실측 의존이었다. '첫 스펙트럼과 가장 다른 위치'로 바꿔
  에이전트 자신의 측정 파일로 정답이 정해지게 했다. GT(좌표·순서)=현재 → X+0.1 → X-0.1 → Y+0.1 → Y-0.1 의 5점, 순서 준수.
  GT(답)=사후 재계산한 최대 코사인 거리 위치. 확인=좌표 5개와 호출 순서, 보고 위치 일치.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T110",
    score=3,
    axis="procedure",
    mode="live",
    criteria="SET(coords 5 items, order EXACT) + EXACT(max-distance index) / post-hoc GT",
    prompt=(
        "Measure at the current position and then at +0.1 mm in X, -0.1 mm in X, +0.1 mm in "
        "Y and -0.1 mm in Y, in that order. Report at which of the five positions the "
        "spectrum differs most from the first one, using cosine distance after L2 "
        "normalization. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T110 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    LABELS = ["current", "x+0.1", "x-0.1", "y+0.1", "y-0.1"]
    saved = run.spectra()
    out = [chk.called(run, "acquire_spectrum", times=5)]
    if len(saved) < 5:
        return out + [chk.fail("position of maximum distance", f"saved {len(saved)} files (need 5)")]
    _, x0, y0 = saved[0]
    base = sp.l2(y0)
    dist = []
    for _, x, y in saved[:5]:
        common = sp.on_common_axis(x0, y0, x, y)
        dist.append(0.0 if common is None
                    else sp.cosine_distance(sp.l2(common[1]), sp.l2(common[2])))
    want = int(np.argmax(dist))
    got = run.answer.get("index")
    if got is None:
        # 본문 폴백은 '마지막으로 언급된 라벨' 이어야 한다. 예전에는 '인덱스가 가장 큰
        # 라벨' 을 골라, 다섯 개를 모두 나열하면 항상 4 가 나왔다.
        lab = run.last_mention(LABELS)
        got = LABELS.index(lab) if lab else None
    return out + [
        chk.ok("most different position", got is not None and int(got) == want,
               f"chosen={got} expected={want} (distance={[round(v, 4) for v in dist]})", weight=2.0),
    ]
