# -*- coding: utf-8 -*-
"""T111 — 진단 복구 (3점)

[문제]
  You measure a silicon reference once and the strongest peak lands at 524.1 cm-1. The
  expected position is 520.7 cm-1 and the tolerance is 2 cm-1. Report the offset and
  whether a wavenumber recalibration is required. decision must be one of:
  recalibration_required, within_tolerance. Do not operate the instrument for this question
  — answer only. End your reply with a single ```json block of exactly this shape: {"plan":
  ["<tool name>", ...], "decision": "<one of the listed values>", "rationale": "<one
  sentence>"}

[정답 기준]
  GT(offset)=524.1-520.7=+3.4 cm-1 (허용 ±0.5). GT(recalibration_required)=True (|3.4|>2).
  이 가정형은 하드웨어도 시료도 필요 없이 완전히 결정적이다 — 판정 규칙을 적용할 줄
  아는지만 본다(측정 능력은 못 잰다). 장비를 만지면 지시 불이행.

[숫자를 다시 측정하지 않는 이유 — 2026-08-03]
  예전 채점기는 mode="hypothetical" 인데도 acquire_spectrum 을 정확히 1 회 부를 것을
  요구하고, 저장된 스펙트럼에서 최강 피크를 다시 찾아 offset 을 계산했다. 프롬프트는
  "장비를 만지지 말고 답만 하라"면서 피크 위치(524.1)까지 이미 알려 준다. 프롬프트대로
  산수만 해서 정확히 답한 실행이 '측정을 안 했다'는 이유로 0 점이었다. 명세가 준 숫자로
  채점한다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import answer as A
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T111",
    score=3,
    axis="diagnostics",
    mode="hypothetical",
    windows=[('silicon reference line', 515.0, 526.0, 2)],
    criteria="NUM(offset ±0.5 cm-1) + EXACT(verdict) / post-hoc GT",
    needs=(
        "Optional: mounting a silicon reference sample also tests absolute accuracy. "
        "Without it, verdict consistency is still graded."
    ),
    prompt=(
        "You measure a silicon reference once and the strongest peak lands at 524.1 cm-1. The "
        "expected position is 520.7 cm-1 and the tolerance is 2 cm-1. Report the offset and "
        "whether a wavenumber recalibration is required. Do not operate the instrument for this "
        "question — answer only. "
    ),
    answer_keys=[
        ("offset", "number - measured peak minus 520.7, in cm-1"),
        ("recalibration_required", "true or false"),
    ],
)


MEASURED_CM1, REFERENCE_CM1, TOLERANCE_CM1 = 524.1, 520.7, 2.0


def evaluate(b, run):
    """이 목록이 그대로 T111 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    off = MEASURED_CM1 - REFERENCE_CM1          # +3.4
    need = abs(off) > TOLERANCE_CM1             # True
    # 판정은 answer 로 받는다. 본문에서 '필요'/'불필요' 를 찾으면 서로 부분문자열이라
    # 두 답이 다 걸려 판정이 무효가 된다(= 무조건 통과).
    said = A.flag(run, "recalibration_required")
    return [
        # 답만 하라고 했다 — 장비를 만지면 지시 불이행.
        chk.called(run, "acquire_spectrum", times=0),
        chk.reported(run, "offset", off, tol=0.5, name="offset (cm-1)", weight=2.0),
        chk.ok("verdict consistent with the offset", said is not None and bool(said) == need,
               f"verdict={said} (offset {off:+.2f} cm-1, threshold +-{TOLERANCE_CM1}, "
               f"expected {need})", weight=2.0, kind="EXACT"),
    ]
