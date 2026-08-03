# -*- coding: utf-8 -*-
"""T102 — 진단 복구 (3점)

[문제]
  The Raman peaks are barely above the noise. Improve the measurement quality, using at
  most 2.0 s exposure and at most 60% laser power. Report the SNR (T050 definition) before
  and after, and the settings you changed.

[정답 기준]
  [TS-B] '안전 범위 내 개선'의 상한을 수치로 못박아 정답 공간을 닫았다(원문은 상한이 없어 GT 불가). 사람 개입=시작 전 노출을 0.05s 등으로
  낮춰 저SNR 상태를 만든다. GT=측정 2건 이상, 최종 SNR > 초기 SNR, 최종 노출<=2.0 및 파워<=60. 확인=상한 초과 설정을 한 번이라도
  걸면 오답(안전 준수가 이 문항의 핵심).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T102",
    score=3,
    axis="diagnostics",
    mode="live",
    windows=[('SNR signal window', 990.0, 1012.0, 3), ('SNR noise window', 1800.0, 1900.0, 2)],
    criteria="REL(SNR improvement) + PROC(within cap EXACT)",
    prompt=(
        "The Raman peaks are barely above the noise. Improve the measurement quality, using "
        "at most 2.0 s exposure and at most 60% laser power. Report the SNR (T050 "
        "definition) before and after, and the settings you changed. "
    ),
)

def setup(b):
    """측정 전에 만들어야 하는 상태."""
    b.hw("set_ccd_exposure", exposure_time=0.05)


def evaluate(b, run):
    """이 목록이 그대로 T102 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    MAX_EXPOSURE, MAX_POWER = 2.0, 60.0
    saved = run.spectra()
    exps = [float(v) for v in run.args("acquire_spectrum", "exposure")]
    exps += [float(v) for v in run.args("set_ccd_exposure", "exposure_time")]
    pows = [float(v) for v in run.args("acquire_spectrum", "power")]
    pows += [float(v) for v in run.args("set_laser_power", "percent")]

    # all([]) 는 True 다. 설정을 하나도 안 한 실행이 '상한을 지켰다'로 가중치 4 를
    # 그냥 받던 자리 — 관측이 없으면 통과가 아니다.
    out = []
    for label, seq, cap in (("exposure", exps, MAX_EXPOSURE), ("power", pows, MAX_POWER)):
        if not seq:
            out.append(chk.fail(f"{label} within cap", f"{label} was never set",
                                weight=2.0))
        else:
            out.append(chk.ok(f"{label} within cap", max(seq) <= cap + 1e-9,
                              f"max {max(seq):g} ≤ {cap:g}", weight=2.0))
    snrs = [v for v in (sp.snr(x, y) for _, x, y in saved) if v is not None]
    if len(snrs) < 2:
        out.append(chk.fail("SNR improvement", f"SNR computable from {len(snrs)} files (need 2)", weight=2.0))
    else:
        out.append(chk.ok("SNR improvement", snrs[-1] > snrs[0],
                          f"{snrs[0]:.1f} → {snrs[-1]:.1f}", weight=2.0))
    return out
