# -*- coding: utf-8 -*-
"""T028 — 절차 구성 (2점)

[문제]
  Measure a 3x3 grid at X=37.8, 37.9, 38.0 mm and Y=25.2, 25.3, 25.4 mm, saving each
  spectrum with its coordinates.

[정답 기준]
  GT=9점, center=(37.9, 25.3), spacing_mm=0.1, rows=cols=3. 중심 좌표를 명시적으로 계산해야 한다(격자가 중심
  대칭이므로 모서리 좌표를 center로 주면 전부 어긋난다). GT(게이트)=preview가 run보다 앞선 다른 턴에 호출될 것. 확인=list_results
  9건의 meta.x/y ⊆ GT좌표(±0.001).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T028",
    score=2,
    axis="procedure",
    mode="live",
    criteria="PROC(args EXACT) + SET(coords 9 items, ±0.001)",
    prompt=(
        "Measure a 3x3 grid at X=37.8, 37.9, 38.0 mm and Y=25.2, 25.3, 25.4 mm, saving each "
        "spectrum with its coordinates. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T028 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.arg(run, "run_grid_scan", "center_x", 37.9),
        chk.arg(run, "run_grid_scan", "center_y", 25.3),
        chk.arg(run, "run_grid_scan", "spacing_mm", 0.1),
        chk.arg(run, "run_grid_scan", "rows", 3),
        chk.arg(run, "run_grid_scan", "cols", 3),
        # preview→턴 종료→사람 승인 절차는 여기서 요구하지 않는다. 도구 스키마가
        # 요구하는 그 절차에는 "다음 턴"이 있어야 하는데 벤치에는 사람도 다음 턴도
        # 없다 — 문서를 지키면 스캔을 영영 못 돌리고, 돌리면 이 판정에 걸리는
        # 모순이었다. 인터록 자체는 N01 이 enforce_grid_gate=True 로 잰다.
    ]
