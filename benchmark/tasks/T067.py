# -*- coding: utf-8 -*-
"""T067 — 절차 구성 (3점)

[문제]
  At the same position measure once with exposure 0.5 s and once with 2.0 s. Compute each
  SNR with the T050 definition and report the difference (2.0 s value minus 0.5 s value).

[정답 기준]
  GT(설정)=exposure_time readback 0.5 / 2.0. GT(값)=사후 재계산한 차. GT(부호)=양수 — 노출을 4배로 늘리면 SNR은
  시료와 무관하게 증가한다. 차의 방향을 문항에 명시해 부호 오답(0.5s - 2.0s)을 배제했다. 확인=저장 2건, 좌표 불변.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T067",
    score=3,
    axis="절차 구성",
    mode="live",
    windows=[('SNR 신호창', 990.0, 1012.0, 3), ('SNR 잡음창', 1800.0, 1900.0, 2)],
    prompt=(
        "At the same position measure once with exposure 0.5 s and once with 2.0 s. Compute "
        "each SNR with the T050 definition and report the difference (2.0 s value minus 0.5 "
        "s value). "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T067 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [
        chk.called(run, "acquire_spectrum", times=2),
        chk.arg_set(run, "acquire_spectrum", "exposure", [0.5, 2.0]),
    ]
    snrs = [v for v in (sp.snr(x, y) for _, x, y in saved) if v is not None]
    if len(snrs) < 2:
        return out + [chk.fail("SNR 차", f"SNR 계산 가능 {len(snrs)}건 (2건 필요)")]
    diff = snrs[1] - snrs[0]
    return out + [
        # 노출을 4배로 올리면 SNR 이 커진다 — 시료와 무관한 관계다.
        chk.ok("SNR 증가", diff > 0, f"Δ={diff:+.3g}", weight=2.0),
        chk.reported(run, "snr_diff", diff, rel=0.05, name="보고한 SNR 차"),
    ]
