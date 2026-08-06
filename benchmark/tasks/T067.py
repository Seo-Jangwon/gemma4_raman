# -*- coding: utf-8 -*-
"""T067 — 절차 구성 (3점)

[문제]
  At the same position measure once with exposure 0.5 s and once with 2.0 s. Compute each
  SNR with the T050 definition and report the difference (2.0 s value minus 0.5 s value).

[정답 기준]
  GT(설정)=exposure_time readback 0.5 / 2.0. GT(값)=사후 재계산한 차. GT(부호)=양수 — 노출을 4배로 늘리면 SNR은
  시료와 무관하게 증가한다. 차의 방향을 문항에 명시해 부호 오답(0.5s - 2.0s)을 배제했다. 확인=저장 2건, 좌표 불변.

  [정답 기준이 readback 이라고 적어 놓고 인자를 보고 있었다 — 2026-08-06]
  판정은 chk.arg_set(run, "acquire_spectrum", "exposure", ...) 였다. set_ccd_exposure
  로 걸고 acquire_spectrum() 을 인자 없이 부른 실행은 확정 실패였다 — 프롬프트가 금지한
  적 없는 도구 경로를 벌한 것이다(T062 와 같은 결함, T099 는 이미 고쳤다).
  acquire_spectrum 결과에 그 측정의 실효 exposure_time 이 실려 오므로 되읽기로 본다.

  [측정 순서를 바꾸면 정답이 오답이 됐다 — 2026-08-06]
  예전에는 diff = snrs[1] - snrs[0] 이었다. snrs 는 **모든 저장 CSV** 의 SNR 을 저장
  순으로 모은 것이라, 2.0 s 를 먼저 잰 실행은 차가 음수가 되어 "SNR increased" 와
  보고값 판정이 동시에 죽었다. 측정 순서는 프롬프트가 요구한 바가 아니고, 프롬프트가
  못박은 것은 **차의 방향(2.0 s 값 - 0.5 s 값)** 이다. 그러니 노출값으로 어느 것이
  2.0 s 인지 식별해서 빼야 그 명시가 뜻을 갖는다. 중간 산출물을 저장해도 안 흔들린다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T067",
    score=3,
    axis="procedure",
    mode="live",
    windows=[('SNR signal window', 990.0, 1012.0, 3), ('SNR noise window', 1800.0, 1900.0, 2)],
    criteria="PROC(exposure EXACT) + NUM(5%) + REL(positive sign) / post-hoc GT",
    prompt=(
        "At the same position measure once with exposure 0.5 s and once with 2.0 s. Compute "
        "each SNR with the T050 definition and report the difference (2.0 s value minus 0.5 "
        "s value). "
    ),
    answer_keys=[
        ("snr_diff", "number - SNR at 2.0 s minus SNR at 0.5 s"),
    ],
)

SHORT, LONG = 0.5, 2.0


def evaluate(b, run):
    """이 목록이 그대로 T067 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    acqs = run.acquisitions()
    out = [
        chk.called(run, "acquire_spectrum", times=2),
        # 되읽기로 본다 — 인자든 set_ccd_exposure 든 실효 설정은 같은 값으로 돌아온다.
        chk.set_match("exposure set (readback)",
                      [a["exposure"] for a in acqs if a["exposure"] is not None] or None,
                      [SHORT, LONG], tol=1e-3, weight=2.0),
    ]

    # 노출값으로 식별한다 — 저장/측정 순서와 무관하게 '2.0 s 값 - 0.5 s 값'이 된다.
    by_exp = {}
    for a in acqs:
        if a["exposure"] is None or a["x"] is None:
            continue
        v = sp.snr(a["x"], a["y"])
        if v is None:
            continue
        for want in (SHORT, LONG):
            if abs(a["exposure"] - want) <= 1e-3:
                by_exp[want] = v
    if SHORT not in by_exp or LONG not in by_exp:
        return out + [chk.fail(
            "SNR difference",
            f"need one acquisition at {SHORT} s and one at {LONG} s with computable SNR; "
            f"got {sorted(by_exp)}")]

    diff = by_exp[LONG] - by_exp[SHORT]
    return out + [
        # 노출을 4배로 올리면 SNR 이 커진다 — 시료와 무관한 관계다.
        chk.ok("SNR increased", diff > 0,
               f"Δ={diff:+.3g}  ({SHORT}s={by_exp[SHORT]:.3g}, {LONG}s={by_exp[LONG]:.3g})",
               weight=2.0),
        chk.reported(run, "snr_diff", diff, rel=0.05, name="reported SNR difference"),
    ]
