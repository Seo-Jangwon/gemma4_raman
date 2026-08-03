# -*- coding: utf-8 -*-
"""N05 — 데이터 처리 (3점)

[문제]
  Apply IPBSA baseline correction to N05.csv at polynomial orders 3, 5 and 7, label the
  versions v_o3, v_o5 and v_o7, compare them, and say which order you keep and why.

[정답 기준]
  GT=bg 버전 3건(라벨·poly_order 정확히 3/5/7)과 비교 근거 제시. 핵심 판별점=list_bg_versions는 요약만 주고,
  get_bg_version을 불러도 배열은 관측 축약기가 버려서 숫자가 도착하지 않는다. 배열을 읽으려 반복 호출하면 감점, 요약 통계로 판단하거나
  save_result=True로 저장해 run_analysis로 넘기면 가점. 확인=poly_order 인자 3값 EXACT, get_bg_version 반복
  호출 3회 이상이면 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="N05",
    score=3,
    axis="data processing",
    mode="live",
    inputs=['N05.csv'],
    criteria="PROC(args EXACT, repeats<3) + KEYWORD(basis for the choice)",
    prompt=(
        "Apply IPBSA baseline correction to N05.csv at polynomial orders 3, 5 and 7, label "
        "the versions v_o3, v_o5 and v_o7, compare them, and say which order you keep and "
        "why. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N05 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "apply_background_subtraction", times=3),
        chk.called(run, "get_bg_version", at_least=0, at_most=2),
        chk.arg_set(run, "apply_background_subtraction", "poly_order", [3, 5, 7]),
        chk.keywords(run, ['keep', 'choose']),
    ]
