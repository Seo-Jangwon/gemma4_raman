# -*- coding: utf-8 -*-
"""T077 — 절차 구성 (3점)

[문제]
  Test at most 8 of the combinations of exposure 0.25, 0.5, 1.0, 2.0 s and laser power 20,
  40, 60 %. Among the conditions satisfying SNR >= 20 (T050 definition), no saturated
  pixel, and at least 90% of the T077_ref.csv reference peaks detected, select the one with
  the lowest dose, where dose = power x exposure x 0.01 mJ.

[정답 기준]
  GT(탐색)=측정 8회 이하, 각 조합이 정의된 격자 안. dose 산식을 명시해 선택 기준을 확정했다. GT(선택)=사후에 3조건을 재평가해 얻은 최소
  dose 조합. 확인=선택 근거로 3조건을 모두 제시했는지. 9회 이상 측정하면 오답(조사량 통제가 이 문항의 요구).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T077",
    score=3,
    axis="procedure",
    mode="live",
    inputs=['T077_ref.csv'],
    criteria="PROC(<=8 calls) + EXACT(chosen combination) / post-hoc GT",
    prompt=(
        "Test at most 8 of the combinations of exposure 0.25, 0.5, 1.0, 2.0 s and laser "
        "power 20, 40, 60 %. Among the conditions satisfying SNR >= 20 (T050 definition), no "
        "saturated pixel, and at least 90% of the T077_ref.csv reference peaks detected, "
        "select the one with the lowest dose, where dose = power x exposure x 0.01 mJ. "
    ),
    answer_keys=[
        ("exposure", "number - the exposure time in seconds you chose"),
        ("power", "number - the laser power in percent you chose"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T077 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    GRID_E, GRID_P = (0.25, 0.5, 1.0, 2.0), (20, 40, 60)
    n = run.count("acquire_spectrum")
    exps = [c["args"].get("exposure") for c in run.calls
            if c.get("name") == "acquire_spectrum"]
    pows = [c["args"].get("power") for c in run.calls
            if c.get("name") == "acquire_spectrum"]
    off_grid = [(e, p) for e, p in zip(exps, pows)
                if e not in GRID_E or p not in GRID_P]

    saved = run.spectra()
    out = [
        chk.called(run, "acquire_spectrum", at_least=1, at_most=8),
        chk.ok("stayed inside the defined grid", not off_grid, f"off-grid {len(off_grid)} files"),
    ]
    # 문항의 핵심은 '조건을 만족하는 것 중 dose 가 최소인 조합'이다. 예전에는 그 선택을
    # 아예 채점하지 않아 탐색만 하면 통과였다.
    ok_pairs = []
    for (e, p), (_, x, y) in zip(zip(exps, pows), saved):
        if e is None or p is None:
            continue
        s = sp.snr(x, y)
        if s is not None and s >= 10.0 and sp.saturated_count(y) == 0:
            ok_pairs.append((float(e), float(p)))
    got_e = run.answer.get("exposure")
    got_p = run.answer.get("power")
    if not ok_pairs:
        return out + [chk.fail("minimum-dose combination",
                               "no acquisition met the condition (SNR >= 10, 0 saturated)", weight=2.0)]
    want = min(ok_pairs, key=lambda t: t[0] * t[1])
    return out + [
        chk.ok("chose the minimum-dose combination",
               got_e is not None and got_p is not None
               and abs(float(got_e) - want[0]) < 1e-6 and abs(float(got_p) - want[1]) < 1e-6,
               f"reported=({got_e}, {got_p}) expected={want} "
               f"(qualifying combinations={ok_pairs})", weight=2.0),
    ]
