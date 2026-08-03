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
  GT(plan)=[acquire_spectrum, acquire_spectrum, run_analysis] with shutter close→auto
  (set_ccd_shutter 를 먼저 부르는 변형도 인정). GT(decision)=normal_minus_dark. 확인=차감 방향이 뒤집히면 오답.
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
        "perform. Do not operate the instrument for this question — answer only. "
    ),
    answer_keys=[
        ("external_fraction", "number - the stray-light fraction, 0 to 1"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T107 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [chk.arg_set(run, "acquire_spectrum", "shutter", ["close", "auto"])]
    if len(saved) < 2:
        return out + [chk.fail("dark-frame subtraction", f"saved {len(saved)} files (need 2)")]
    n = min(len(saved[0][2]), len(saved[1][2]))
    first, second = saved[0][2][:n], saved[1][2][:n]
    # 차감 방향은 판정 대상이다. 예전에는 채점기가 평균으로 알아서 정렬한 뒤
    # '방향 맞음'을 무조건 통과시켜, 뒤집어 뺀 답도 만점이었다.
    dark_first = float(first.mean()) < float(second.mean())
    dark, normal = (first, second) if dark_first else (second, first)
    want = normal - dark
    tot = float(np.abs(normal).sum())
    ratio = float(np.abs(dark).sum() / tot) if tot > 0 else 0.0

    cands = [y for _, _, y in saved if len(y) == n]
    best = max((chk.array("subtracted array", y, want, mode="exact", weight=2.0) for y in cands),
               key=lambda c: c.score, default=None)
    wrong = max((chk.array("reversed direction", y, dark - normal, mode="exact") for y in cands),
                key=lambda c: c.score, default=None)
    return out + [
        best or chk.fail("subtracted array", "no saved array has a matching length", weight=2.0),
        chk.ok("subtraction direction", not (wrong and wrong.passed and not (best and best.passed)),
               "normal minus dark", weight=2.0),
        chk.reported(run, "external_fraction", ratio, rel=0.05, name="stray-light fraction"),
    ]
