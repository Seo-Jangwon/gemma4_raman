# -*- coding: utf-8 -*-
"""T085 — 계측 제어 (2점)

[문제]
  Report the current stage, laser, CCD and camera states comprehensively.

[정답 기준]
  GT=4개 구성요소(stage/laser/ccd/camera)가 모두 보고에 포함. 개별 값은 환경 의존이라 '빠짐없이 보고했는가'를 채점 대상으로 삼는다.
  확인=응답에 4개 키가 모두 존재. get_hardware_status 대신 개별 툴 4개를 불러도 정답으로 인정한다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T085",
    score=2,
    axis="계측 제어",
    mode="live",
    prompt=(
        "Report the current stage, laser, CCD and camera states comprehensively. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T085 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.keywords(run, ['stage', '스테이지']),
        chk.keywords(run, ['laser', '레이저']),
        chk.keywords(run, ['ccd', 'detector', '검출기']),
        chk.keywords(run, ['camera', '카메라']),
    ]
