# -*- coding: utf-8 -*-
"""T076 — 시각 판단 (3점)

[문제]
  Suppose analyze_microscope_image returns one bright circular target centred at pixel
  (512, 384) in a 1060x800 view. State the tools you would call, in order, to measure a 3x3
  grid of 0.1 mm spacing centred on that target and baseline-correct each spectrum with
  IPBSA order 5. decision must be one of: preview_first, run_directly. Do not operate the
  instrument for this question — answer only. End your reply with a single ```json block of
  exactly this shape: {"plan": ["<tool name>", ...], "decision": "<one of the listed
  values>", "rationale": "<one sentence>"}

[정답 기준]
  GT(plan)=[analyze_microscope_image, move_to_pixel, preview_grid_scan, run_grid_scan,
  run_analysis]. GT(decision)=preview_first — 격자 스캔은 미리보기·승인이 선행한다. 확인=preview_grid_scan 이
  run_grid_scan 보다 앞설 것.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T076",
    score=3,
    axis="시각 판단",
    mode="hypothetical",
    prompt=(
        "Suppose analyze_microscope_image returns one bright circular target centred at "
        "pixel (512, 384) in a 1060x800 view. State the tools you would call, in order, to "
        "measure a 3x3 grid of 0.1 mm spacing centred on that target and baseline-correct "
        "each spectrum with IPBSA order 5. decision must be one of: preview_first, "
        "run_directly. Do not operate the instrument for this question — answer only. End "
        "your reply with a single ```json block of exactly this shape: {\"plan\": [\"<tool "
        "name>\", ...], \"decision\": \"<one of the listed values>\", \"rationale\": \"<one "
        "sentence>\"} "
    ),
)

def setup(b):
    """측정 전에 만들어야 하는 상태."""
    b.inject_scene("T076.png")


def evaluate(b, run):
    """이 목록이 그대로 T076 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "acquire_spectrum", times=0, weight=0.5),   # 답만 하라고 했다 — 장비를 만지면 지시 불이행
        chk.ok("답변 있음", len((run.text or "").split()) >= 20, f"{len((run.text or '').split())} 단어", kind="PLAN"),
        chk.plan_order(run, ['analyze_microscope_image', 'move_to_pixel', 'preview_grid_scan', 'run_grid_scan', 'run_analysis']),
        chk.reported_label(run, "decision", "preview_first", ['preview_first', 'run_directly'], name="판단"),
    ]
