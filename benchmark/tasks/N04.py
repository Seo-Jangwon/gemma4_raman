# -*- coding: utf-8 -*-
"""N04 — 절차 구성 (1점)

[문제]
  Bundle this session's results into a single zip for download.

[정답 기준]
  GT=bundle_results 1회, 반환에 zip_url 존재. 확인=scope 기본값(session)을 유지했는지 — scope='all'을 주면 다른
  세션 산출물까지 섞이므로 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="N04",
    score=1,
    axis="procedure",
    mode="live",
    criteria="PROC(tool and args EXACT)",
    prompt=(
        "Bundle this session's results into a single zip for download. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N04 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "bundle_results", times=1),
        chk.arg_not(run, "bundle_results", "scope", "all"),
    ]
