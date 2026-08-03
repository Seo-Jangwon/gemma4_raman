# -*- coding: utf-8 -*-
"""T062 — 절차 구성 (3점)

[문제]
  At the same position measure once at laser power 20%, 40% and 60%. Compute the SNR of
  each with the T050 definition (signal = max in 990-1012, noise = std ddof=1 in 1800-1900)
  and plot SNR versus laser power.

[정답 기준]
  GT(설정)=laser_power_pct readback 20/40/60. GT(SNR)=사후 재계산 3값(각 상대오차 5%). GT(경향)=SNR이 파워에
  대해 단조 증가 — 시료와 무관하게 성립하는 물리 관계라 실측에도 쓸 수 있는 기준. 확인=저장 3건, 그림 1장(점 3개).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T062",
    score=3,
    axis="절차 구성",
    mode="live",
    windows=[('SNR 신호창', 990.0, 1012.0, 3), ('SNR 잡음창', 1800.0, 1900.0, 2)],
    prompt=(
        "At the same position measure once at laser power 20%, 40% and 60%. Compute the SNR "
        "of each with the T050 definition (signal = max in 990-1012, noise = std ddof=1 in "
        "1800-1900) and plot SNR versus laser power. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T062 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [
        chk.called(run, "acquire_spectrum", times=3),
        chk.arg_set(run, "acquire_spectrum", "power", [20, 40, 60]),
    ]
    snrs = [v for v in (sp.snr(x, y) for _, x, y in saved) if v is not None]
    if len(snrs) < 3:
        return out + [chk.fail("SNR 3값",
                               f"SNR 을 계산할 수 있는 저장 파일 {len(snrs)}건 (3건 필요)")]
    s = snrs[:3]
    got = run.answer.get("snr")
    return out + [
        # 파워를 올리면 SNR 이 커진다 — 어떤 시료에서도 성립하는 관계다.
        chk.increasing("SNR 단조 증가", s, weight=2.0),
        chk.set_match("보고한 SNR 3값", [float(v) for v in got] if isinstance(got, list) else None,
                      s, tol=max(s) * 0.05, ordered=True, weight=2.0),
    ]
