# -*- coding: utf-8 -*-
"""T032 — 절차 구성 (2점)

[문제]
  Independently acquire a spectrum 5 times at the current position, compute the
  arithmetic-mean spectrum, and save it with save_result inside run_analysis.

[정답 기준]
  GT=저장된 원본 5건 intensity의 요소별 산술평균(채점기가 원본 5개 파일로 재계산). 실측이라 절대값은 미정이지만 '5건의 평균'이라는 관계는
  결정적이므로 값 채점이 성립한다. 확인=산출 파일 6건(원본5+평균1), 평균 파일을 load_spectrum 해 재계산 배열과 비교.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T032",
    score=2,
    axis="procedure",
    mode="live",
    criteria="ARRAY(rtol 1e-6) - the mean is deterministic",
    prompt=(
        "Independently acquire a spectrum 5 times at the current position, compute the "
        "arithmetic-mean spectrum, and save it with save_result inside run_analysis. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T032 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [chk.called(run, "acquire_spectrum", times=5)]
    if len(saved) < 6:
        return out + [chk.fail("mean spectrum",
                               f"saved {len(saved)} files (5 originals + 1 mean = need 6)")]
    ys = [y for _, _, y in saved]
    n = min(len(y) for y in ys)
    ys = [y[:n] for y in ys]
    # 어느 파일이 평균인지 이름으로는 알 수 없다. '나머지 5개의 평균'과 가장 잘 맞는 하나를
    # 찾아 그것으로 판정한다. 평균은 구현이 유일하므로 exact 로 본다.
    best = None
    for i in range(len(ys)):
        others = [y for j, y in enumerate(ys) if j != i][:5]
        if len(others) < 5:
            continue
        c = chk.array("mean spectrum", ys[i], np.mean(others, axis=0), mode="exact",
                      weight=2.0)
        if best is None or c.score > best.score:
            best = c
    return out + [best or chk.fail("mean spectrum", "no combination available to compare", weight=2.0)]
