# -*- coding: utf-8 -*-
"""T100 — 진단 복구 (3점)

[문제]
  The signal is weaker and the peaks broader than usual. Diagnose whether this is a focus
  problem: record the SNR (T050 definition) at the current Z, run autofocus, measure again,
  and report both SNR values and whether focus was the cause.

[정답 기준]
  [TS-B] 원인 후보가 여럿이라 자유서술로는 GT가 없었다. '초점 가설을 검증하라'로 좁혀 SNR 전후 비교라는 결정적 기준을 만들었다. 사람 개입=시작 전
  Z를 초점에서 0.05mm 어긋나게 둔다. GT=측정 2건(AF 전/후), SNR 2값(사후 재계산), 결론=SNR_after > SNR_before 이면 초점
  원인. 확인=결론이 두 SNR 값과 일관될 것(값과 결론이 어긋나면 오답).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T100",
    score=3,
    axis="diagnostics",
    mode="live",
    windows=[('SNR signal window', 990.0, 1012.0, 3), ('SNR noise window', 1800.0, 1900.0, 2)],
    criteria="REL(SNR before/after) + NUM(5%) + PROC / post-hoc GT",
    prompt=(
        "The signal is weaker and the peaks broader than usual. Diagnose whether this is a "
        "focus problem: record the SNR (T050 definition) at the current Z, run autofocus, "
        "measure again, and report both SNR values and whether focus was the cause. "
    ),
    answer_keys=[
        ("snr_before", "number - SNR before autofocus"),
        ("snr_after", "number - SNR after autofocus"),
        ("focus_was_cause", "true or false - was focus the cause"),
    ],
)

def setup(b):
    """측정 전에 만들어야 하는 상태."""
    b.hw("run_autofocus")
    b.hw("move_stage_relative", dz=0.05)          # 일부러 흐트러뜨린다


def evaluate(b, run):
    """이 목록이 그대로 T100 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [
        chk.called(run, "run_autofocus", times=1),
        chk.called(run, "acquire_spectrum", times=2),
    ]
    snrs = [v for v in (sp.snr(x, y) for _, x, y in saved) if v is not None]
    if len(snrs) < 2:
        return out + [chk.fail("SNR before/after", f"SNR computable from {len(snrs)} files (need 2)")]
    s0, s1 = snrs[0], snrs[-1]
    # 결론은 answer 로 받는다. 본문에서 '초점'/'focus' 를 찾는 방식은 두 답이 다 걸려
    # 판정이 거의 항상 무효가 됐다(= 무조건 통과).
    said = run.answer.get("focus_was_cause")
    return out + [
        chk.reported(run, "snr_before", s0, rel=0.05, name="SNR before autofocus"),
        chk.reported(run, "snr_after", s1, rel=0.05, name="SNR after autofocus"),
        chk.ok("conclusion consistent with the data", said is not None and bool(said) == (s1 > s0),
               f"conclusion={said} (SNR {s0:.1f} → {s1:.1f})", weight=2.0),
    ]
