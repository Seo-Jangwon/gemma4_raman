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
  GT(offset)=+3.4 cm-1 (허용 ±0.05). GT(decision)=recalibration_required. 이 가정형은 하드웨어도 시료도 필요
  없이 완전히 결정적이다 — 판정 규칙을 적용할 줄 아는지만 본다(측정 능력은 못 잰다).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
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
        "You measure a silicon reference once and the strongest peak lands at 524.1 cm-1. "
        "The expected position is 520.7 cm-1 and the tolerance is 2 cm-1. Report the offset "
        "and whether a wavenumber recalibration is required. decision must be one of: "
        "recalibration_required, within_tolerance. Do not operate the instrument for this "
        "question — answer only. End your reply with a single ```json block of exactly this "
        "shape: {\"plan\": [\"<tool name>\", ...], \"decision\": \"<one of the listed values>\", "
        "\"rationale\": \"<one sentence>\"} "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T111 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    REFERENCE_CM1, TOLERANCE_CM1 = 520.7, 2.0
    saved = run.spectra()
    out = [chk.called(run, "acquire_spectrum", times=1)]
    if not saved:
        return out + [chk.fail("silicon offset", "no saved spectrum")]
    _, x, y = saved[-1]
    if not sp.covers(x, 515.0, 526.0, 2):
        return out + [chk.blocked("silicon offset",
                                  "the instrument axis does not cover ~520.7 cm-1")]
    peak = sp.strongest_peak(x, y)        # 전역 최대가 아니라 규약대로 검출한 최강 피크
    off = peak - REFERENCE_CM1
    need = abs(off) > TOLERANCE_CM1
    # 판정은 answer 로 받는다. 본문에서 '필요'/'불필요' 를 찾으면 서로 부분문자열이라
    # 두 답이 다 걸려 판정이 무효가 된다(= 무조건 통과).
    said = run.answer.get("recalibration_required")
    return out + [
        chk.reported(run, "offset", off, tol=0.5, name="offset (cm-1)", weight=2.0),
        chk.ok("verdict consistent with the offset", said is not None and bool(said) == need,
               f"verdict={said} (offset {off:+.2f} cm-1, threshold +-{TOLERANCE_CM1})", weight=2.0),
    ]
