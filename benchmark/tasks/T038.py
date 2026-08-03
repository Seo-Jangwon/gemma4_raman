# -*- coding: utf-8 -*-
"""T038 — 데이터 처리 (2점)

[문제]
  Load T038.csv (a polystyrene Raman spectrum) and display it as a line graph of Raman
  shift versus intensity.

[정답 기준]
  GT=입력의 (raman_shift, intensity) 전 구간. 확인=그림 1장 + 플롯에 쓰인 배열의 길이·min·max·평균이 입력과 일치(그림 픽셀
  비교 아님). stdout에 요약 통계를 찍게 하면 자동 대조가 쉬워진다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T038",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T038.csv'],
    criteria="ARRAY(rtol 1e-6) on the plotted array",
    prompt=(
        "Load T038.csv (a polystyrene Raman spectrum) and display it as a line graph of "
        "Raman shift versus intensity. "
    ),
    answer_keys=[
        ("n_points", "number - how many data points the spectrum has"),
        ("x_min", "number - lowest Raman shift in cm-1"),
        ("x_max", "number - highest Raman shift in cm-1"),
        ("y_max", "number - maximum intensity"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T038 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    out = [chk.called(run, "run_analysis", at_least=1, at_most=3)]
    ref = _input(b, "T038.csv")
    if ref is None:
        return out + [chk.fail("plotted arrays", "could not read the input T038.csv")]
    # 문항이 요구한 것은 그림 한 장이다. 저장 CSV 를 강제하지 않는다 —
    # 대신 '입력의 실제 통계를 인용했는가'로 본다(그림만 그리고 값을 지어내면 걸린다).
    x, y = ref
    return out + [
        chk.reported(run, "n_points", float(len(x)), tol=0, name="number of points"),
        chk.reported(run, "x_min", float(x.min()), tol=1.0, name="x min"),
        chk.reported(run, "x_max", float(x.max()), tol=1.0, name="x max"),
        chk.reported(run, "y_max", float(y.max()), rel=0.02, name="max intensity"),
    ]


def _input(b, name):
    """벤치 입력 파일을 읽는다 — 정답이 그 파일로 정해지는 문항용."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / name
    return sp.read_xy(p) if p.exists() else None
