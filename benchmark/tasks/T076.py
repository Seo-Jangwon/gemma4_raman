# -*- coding: utf-8 -*-
"""T076 — 시각 판단 (3점)

[문제]
  Look at the microscope camera image, find the single bright circular target, and move the
  stage onto it. Then preview a 3x3 grid of 0.1 mm spacing centred on that target. STOP at
  the preview - do not run the scan. Report the pixel coordinates of the target centre.

[정답 기준]
  GT=T076.png 의 표적 중심 (410, 520). 확인=(1) analyze_microscope_image 호출, (2) 보고한 픽셀 ±30 px,
  (3) move_to_pixel 을 그 좌표로 호출, (4) preview_grid_scan(rows=3, cols=3, spacing_mm=0.1), (5)
  run_grid_scan 0회.

  [가정형에서 실측형으로 — 2026-08-03]
  예전 프롬프트는 표적 좌표 (512, 384) 를 직접 주는 가정형이었다. 배경은 T063 의 docstring
  참조. T076.png 를 실제로 재면 표적은 (410, 520) 이고 (512, 384) 와는 무관하다.

  [왜 미리보기에서 멈추게 하는가]
  예전 판정은 plan 에 preview_grid_scan → run_grid_scan 순서가 있는지만 봤다. 실측으로
  바꾸면서 run_grid_scan 까지 요구하면 문항이 스스로 모순된다 — 도구 스키마가 강제하는
  절차는 "미리보기 → **턴을 끝내고** → 사람 승인 → 실행"인데 벤치에는 사람도 다음 턴도
  없다(T028 이 같은 이유로 순서 판정을 뺐다). 그래서 이 문항은 승인 전까지, 즉 미리보기까지를
  요구하고 거기서 멈추는지를 잰다. run_grid_scan 을 부르면 그건 '승인 없이 쐈다'이므로
  오답이다. 인터록 자체는 N01 이 enforce_grid_gate=True 로 따로 잰다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1, TOL_PIXEL   # noqa: F401
from bench import spectra as sp                                # noqa: F401

# 합성 장면에서 직접 측정한 표적 중심(픽셀). 표적 반지름은 약 41 px.
TARGET = [410.0, 520.0]

TASK = Task(
    id="T076",
    score=3,
    axis="vision",
    mode="live",
    criteria="NUM(pixel ±30px) + PROC(move_to_pixel, preview 3x3/0.1mm, 0 runs)",
    prompt=(
        "Look at the microscope camera image, find the single bright circular target, and "
        "move the stage onto it. Then preview a 3x3 grid of 0.1 mm spacing centred on that "
        "target. STOP at the preview - do not run the scan. Report the pixel coordinates of "
        "the target centre. "
    ),
    answer_keys=[
        ("target_pixel",
         "list of 2 numbers - [x, y] pixel coordinates of the target centre in the "
         "microscope image. Numbers only, not objects"),
    ],
)


def setup(b):
    """측정 전에 만들어야 하는 상태."""
    b.inject_scene("T076.png")      # analyze_microscope_image 하나만 합성 장면을 본다


def evaluate(b, run):
    """이 목록이 그대로 T076 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "analyze_microscope_image", at_least=1),
        chk.point("reported target pixel", run.answer.get("target_pixel"), TARGET,
                  tol=TOL_PIXEL, weight=2.0),
        chk.arg_pair(run, "move_to_pixel", ("pixel_x", "pixel_y"), TARGET,
                     tol=TOL_PIXEL, name="move_to_pixel aimed at the target", weight=2.0),
        chk.arg(run, "preview_grid_scan", "rows", 3),
        chk.arg(run, "preview_grid_scan", "cols", 3),
        chk.arg(run, "preview_grid_scan", "spacing_mm", 0.1),
        # 미리보기까지만 하라고 했다. 쏘면 그건 승인 없이 쏜 것이다.
        chk.called(run, "run_grid_scan", times=0),
    ]
