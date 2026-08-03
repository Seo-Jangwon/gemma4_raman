# -*- coding: utf-8 -*-
"""T107 — 진단 복구 (3점)

[문제]
  Suppose a spectrum shows a strong broad component unrelated to the sample and you suspect
  room light entering the spectrometer. State the tools you would call, in order, to
  quantify the external-light contribution, and state the subtraction you would perform.
  decision must be one of: normal_minus_dark, dark_minus_normal. Do not operate the
  instrument for this question — answer only. End your reply with a single ```json block of
  exactly this shape: {"plan": ["<tool name>", ...], "decision": "<one of the listed
  values>", "rationale": "<one sentence>"}

[정답 기준]
  GT(plan)=[acquire_spectrum, acquire_spectrum, run_analysis] — 셔터를 닫고 한 번,
  열고 한 번 찍은 뒤 차감. 여분 단계는 용서한다(chk.plan_order).
  GT(decision)=normal_minus_dark. 차감 방향이 뒤집히면 오답 — 그게 이 문항의 핵심이다.

[실제 측정을 채점하지 않는 이유 — 2026-08-03]
  예전 채점기는 mode="hypothetical" 인데도 acquire_spectrum(shutter=…) 실호출 2 건과
  저장 파일 2 개를 요구하고 그 배열로 차감 결과를 검증했다. 프롬프트는 "장비를 만지지
  말고 부를 도구를 순서대로 말하라"고 한다. 계획을 정확히 낸 실행이 '측정을 안 했다'는
  이유로 0 점이었다. 명세가 요구한 것(계획과 차감 방향)만 본다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T107",
    score=3,
    axis="diagnostics",
    mode="hypothetical",
    criteria="PROC(shutter args EXACT) + ARRAY(post-hoc GT, rtol 1e-6) + NUM(ratio 5%)",
    needs=(
        "Optional: turning the room light on lets real stray light in. Grading works "
        "without it (the fraction is simply near 0)."
    ),
    prompt=(
        "Suppose a spectrum shows a strong broad component unrelated to the sample and you "
        "suspect room light entering the spectrometer. State the tools you would call, in order, "
        "to quantify the external-light contribution, and state the subtraction you would "
        "perform. decision must be one of: normal_minus_dark, dark_minus_normal. "
        "Do not operate the instrument for this question — answer only. "
    ),
    answer_keys=[
        ("plan", "list of tool-name strings, in the order you would call them"),
        ("decision", 'string - either "normal_minus_dark" or "dark_minus_normal"'),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T107 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        # 답만 하라고 했다 — 장비를 만지면 지시 불이행.
        chk.called(run, "acquire_spectrum", times=0),
        chk.ok("answer present", len((run.text or "").split()) >= 20,
               f"{len((run.text or '').split())} words", kind="PLAN"),
        # 셔터 닫고 한 번, 열고 한 번, 그리고 차감. 확인 단계를 끼워 넣는 것은 용서한다.
        chk.plan_order(run, ['acquire_spectrum', 'acquire_spectrum', 'run_analysis']),
        # 방향이 뒤집히면 배경이 아니라 신호를 빼는 것이다 — 이 문항의 핵심.
        chk.reported_label(run, "decision", "normal_minus_dark",
                           ['normal_minus_dark', 'dark_minus_normal'], name="decision"),
    ]
