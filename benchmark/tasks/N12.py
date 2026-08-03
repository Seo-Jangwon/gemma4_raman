# -*- coding: utf-8 -*-
"""N12 — 절차 구성 (2점)

[문제]
  List everything you have produced in this session and give the data/-relative path of
  each.

[정답 기준]
  GT=list_session_artifacts 1회, 반환 artifacts 목록의 각 항목 경로를 그대로 인용. 확인=list_results(원측정)와
  혼동하면 감점 — 세 개의 목록 툴(list_results / list_session_artifacts / list_uploaded_files)을 목적에 맞게
  고르는지가 요지. 경로를 지어내면 오답.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="N12",
    score=2,
    axis="절차 구성",
    mode="live",
    prompt=(
        "List everything you have produced in this session and give the data/-relative path "
        "of each. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N12 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "list_session_artifacts", times=1),
        chk.called(run, "list_results", times=0),
    ]
