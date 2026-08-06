# -*- coding: utf-8 -*-
"""T062 — 절차 구성 (3점)

[문제]
  At the same position measure once at laser power 20%, 40% and 60%. Compute the SNR of
  each with the T050 definition (signal = max in 990-1012, noise = std ddof=1 in 1800-1900)
  and plot SNR versus laser power.

[정답 기준]
  GT(설정)=laser_power_pct readback 20/40/60. GT(SNR)=사후 재계산 3값(각 상대오차 5%). GT(경향)=SNR이 파워에
  대해 단조 증가 — 시료와 무관하게 성립하는 물리 관계라 실측에도 쓸 수 있는 기준. 확인=저장 3건, 그림 1장(점 3개).

  [정답 기준이 readback 이라고 적어 놓고 인자를 보고 있었다 — 2026-08-06]
  판정은 chk.arg_set(run, "acquire_spectrum", "power", ...) 였다. set_laser_power(60)
  으로 걸고 acquire_spectrum() 을 인자 없이 부른 실행은 run.args 가 비어 **확정 실패**
  였다 — 프롬프트가 금지한 적 없는 도구 경로를 벌한 것이다. T099 는 같은 문제를 반대
  방향에서 이미 고쳤다("예전에는 인자 경로만 봐서 설정 툴로 조정하면 검사가 통째로
  통과했다"). acquire_spectrum 결과에는 그 측정에 **실제로 걸린** laser_power_pct 가
  실려 오므로, 정답 기준이 원래 적어 둔 대로 되읽기로 본다.

  [저장 순서를 조건 순서로 가정했다 — 2026-08-06]
  snrs = [모든 저장 CSV 의 SNR]; s = snrs[:3] 이었다. run.spectra() 는 **모든 CSV
  산출물**을 주므로 중간 산출물을 하나만 저장해도 20/40/60 매핑이 어긋난다.
  run.acquisitions() 로 (파워 ↔ 스펙트럼)을 결과에 실린 저장 경로로 짝지어, 순서 가정을
  없애고 파워 오름차순으로 정렬해 단조성을 본다.

  [프롬프트가 요구한 그림을 안 재고 있었다 — 2026-08-06]
  프롬프트가 "plot SNR versus laser power" 를 요구하고 정답 기준도 "그림 1장" 을 적어
  뒀는데 판정이 없었다. 그림 **내용**(점이 3개인지)까지는 chk.figure 가 약속하지
  않지만, 요구한 산출물을 실제로 남겼는지는 그것으로 볼 수 있다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T062",
    score=3,
    axis="procedure",
    mode="live",
    windows=[('SNR signal window', 990.0, 1012.0, 3), ('SNR noise window', 1800.0, 1900.0, 2)],
    criteria="PROC(power EXACT) + NUM(5%) + REL(monotonic increase) / post-hoc GT",
    prompt=(
        "At the same position measure once at laser power 20%, 40% and 60%. Compute the SNR "
        "of each with the T050 definition (signal = max in 990-1012, noise = std ddof=1 in "
        "1800-1900) and plot SNR versus laser power. "
    ),
    answer_keys=[
        ("snr", "list of 3 numbers - the SNR at 20, 40 and 60 % laser power, in that order"),
    ],
)

WANT_POWERS = [20.0, 40.0, 60.0]


def evaluate(b, run):
    """이 목록이 그대로 T062 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    acqs = run.acquisitions()
    out = [
        chk.called(run, "acquire_spectrum", times=3),
        # 되읽기로 본다 — 인자로 넘겼든 set_laser_power 로 걸었든 같은 값이 나온다.
        chk.set_match("laser power set (readback)",
                      [a["power"] for a in acqs if a["power"] is not None] or None,
                      WANT_POWERS, tol=1e-6, weight=2.0),
        chk.figure(run),
    ]

    # 파워 ↔ 스펙트럼은 측정 결과에 실린 저장 경로로 짝지어진다(저장 순서 가정이 없다).
    pairs = []
    for a in acqs:
        if a["power"] is None or a["x"] is None:
            continue
        v = sp.snr(a["x"], a["y"])
        if v is not None:
            pairs.append((float(a["power"]), v))
    if len(pairs) < 3:
        return out + [chk.fail("3 SNR values",
                               f"acquisitions with computable SNR: {len(pairs)} (need 3)")]

    pairs.sort(key=lambda t: t[0])          # 파워 오름차순 — 보고 순서도 이 순서다
    s = [v for _, v in pairs[:3]]
    got = run.answer.get("snr")
    return out + [
        # 파워를 올리면 SNR 이 커진다 — 어떤 시료에서도 성립하는 관계다.
        chk.increasing("SNR increases monotonically", s, weight=2.0),
        chk.set_match("reported 3 SNR values",
                      [float(v) for v in got] if isinstance(got, list) else None,
                      s, tol=max(s) * 0.05, ordered=True, weight=2.0),
    ]
