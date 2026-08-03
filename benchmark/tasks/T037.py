# -*- coding: utf-8 -*-
"""T037 — 시각 판단 (2점)

[문제]
  Suppose analyze_microscope_image returns four bright circular targets whose pixel centres
  are (180, 140), (860, 150), (200, 660) and (840, 650) in a 1060x800 view. State the tools
  you would call, in order, to measure one spectrum at each target. decision must be one
  of: move_to_pixel, move_stage. Do not operate the instrument for this question — answer
  only. End your reply with a single ```json block of exactly this shape: {"plan": ["<tool
  name>", ...], "decision": "<one of the listed values>", "rationale": "<one sentence>"}

[정답 기준]
  GT(plan)=[analyze_microscope_image, move_to_pixel, acquire_spectrum] (반복 표기 허용).
  GT(decision)=move_to_pixel — 픽셀 좌표는 스테이지 좌표가 아니므로 move_stage 를 고르면 오답. 주의=가정형은 시각 판단 자체를
  재지 못한다(주어진 좌표를 쓸 뿐).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T037",
    score=2,
    axis="시각 판단",
    mode="hypothetical",
    prompt=(
        "Suppose analyze_microscope_image returns four bright circular targets whose pixel "
        "centres are (180, 140), (860, 150), (200, 660) and (840, 650) in a 1060x800 view. "
        "State the tools you would call, in order, to measure one spectrum at each target. "
        "decision must be one of: move_to_pixel, move_stage. Do not operate the instrument "
        "for this question — answer only. End your reply with a single ```json block of "
        "exactly this shape: {\"plan\": [\"<tool name>\", ...], \"decision\": \"<one of the listed "
        "values>\", \"rationale\": \"<one sentence>\"} "
    ),
)

def setup(b):
    """측정 전에 만들어야 하는 상태."""
    b.inject_scene("T037.png")


def evaluate(b, run):
    """이 목록이 그대로 T037 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "acquire_spectrum", times=0, weight=0.5),   # 답만 하라고 했다 — 장비를 만지면 지시 불이행
        chk.ok("답변 있음", len((run.text or "").split()) >= 20, f"{len((run.text or '').split())} 단어", kind="PLAN"),
        chk.plan_order(run, ['analyze_microscope_image', 'move_to_pixel', 'acquire_spectrum']),
        chk.reported_label(run, "decision", "move_to_pixel", ['move_to_pixel', 'move_stage'], name="판단"),
    ]
