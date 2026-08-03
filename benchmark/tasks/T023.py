# -*- coding: utf-8 -*-
"""T023 — 절차 구성 (2점)

[문제]
  At the same position, measure once at CCD exposure 0.5 s, 1.0 s and 2.0 s. The results
  are auto-saved; report the three saved file names.

[정답 기준]
  GT=측정 3건, 각 응답의 exposure_time이 0.5/1.0/2.0. 좌표 불변. '별도 파일로 저장'은 자동 저장이라 별도 조치가 필요 없다 — 저장
  툴을 찾아 헤매지 않도록 문항에 명시했다. 확인=list_results 3건의 파일명 보고.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T023",
    score=2,
    axis="procedure",
    mode="live",
    criteria="STATE(exposure EXACT ×3) + EXACT(filenames 3 items)",
    prompt=(
        "At the same position, measure once at CCD exposure 0.5 s, 1.0 s and 2.0 s. The "
        "results are auto-saved; report the three saved file names. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T023 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "acquire_spectrum", times=3),
        chk.arg_set(run, "acquire_spectrum", "exposure", [0.5, 1.0, 2.0]),
        chk.unchanged("left untouched", before, after, ['x', 'y']),
    ]
