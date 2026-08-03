# -*- coding: utf-8 -*-
"""T063 — 시각 판단 (3점)

[문제]
  Suppose analyze_microscope_image returns one bright circular target centred at pixel
  (512, 384) in a 1060x800 view. State the tools you would call, in order, to acquire one
  spectrum at that target. decision must be one of: move_to_pixel, move_stage. Do not
  operate the instrument for this question — answer only. End your reply with a single
  ```json block of exactly this shape: {"plan": ["<tool name>", ...], "decision": "<one of
  the listed values>", "rationale": "<one sentence>"}

[정답 기준]
  GT(plan)=[analyze_microscope_image, move_to_pixel, acquire_spectrum].
  GT(decision)=move_to_pixel. 주의=시각 판단은 재지 못한다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T063",
    score=3,
    axis="vision",
    mode="hypothetical",
    criteria="NUM(pixel ±25px) + PROC",
    prompt=(
        "Suppose analyze_microscope_image returns one bright circular target centred at pixel "
        "(512, 384) in a 1060x800 view. State the tools you would call, in order, to acquire one "
        "spectrum at that target. decision must be one of: move_to_pixel, move_stage. Do not "
        "operate the instrument for this question — answer only. "
    ),
    answer_keys=[
        ("plan", "list of tool-name strings, in the order you would call them"),
        ("decision", 'string - either "move_to_pixel" or "move_stage"'),
    ],
)

# 장면 주입(b.inject_scene("T063.png"))은 여기 없다 — 이 문항은 가정형이라
# 러너가 setup 을 부르지 않는다(장비를 건드리지 말라는 문항이므로). 프롬프트가
# 픽셀 좌표를 직접 주므로 그림 없이도 답할 수 있고, 그래서 이 문항은 "무엇을
# 보고 어떻게 판단하는가"가 아니라 "픽셀→스테이지 변환을 아는가"를 잰다.
# 진짜 시각 판단을 재려면 mode="live" 로 바꾸고 장면을 주입해야 한다.


def evaluate(b, run):
    """이 목록이 그대로 T063 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "acquire_spectrum", times=0, weight=0.5),   # 답만 하라고 했다 — 장비를 만지면 지시 불이행
        chk.ok("answer present", len((run.text or "").split()) >= 20, f"{len((run.text or '').split())} words", kind="PLAN"),
        # 프롬프트가 analyze_microscope_image 의 결과를 이미 줬다(T037 과 같은 이유).
        chk.plan_order(run, ['move_to_pixel', 'acquire_spectrum']),
        chk.reported_label(run, "decision", "move_to_pixel", ['move_to_pixel', 'move_stage'], name="decision"),
    ]
