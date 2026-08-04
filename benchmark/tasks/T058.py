# -*- coding: utf-8 -*-
"""T058 — 절차 구성 (3점)

[문제]
  Measure a spectrum once at the current position and save it. Then apply to the SAVED
  file, in order: spike removal (5-point moving median, 5x MAD), IPBSA baseline order 5,
  Savitzky-Golay (11, 3), and 0-1 normalization, and report the 3 highest-intensity peaks
  of the result.

[정답 기준]
  실측이라 피크 절대값은 사전 GT가 될 수 없다. 대신 에이전트가 저장한 그 파일을 채점기가 읽어 규약대로 재계산한 값을 GT로 삼는다(사후 GT) — 정답이
  유일해진다. GT=재계산한 상위 3피크(위치·순서). 확인=측정 정확히 1회, 최종 배열 min=0/max=1, 피크 ±3 cm-1 및 순서 일치.

  [사후 GT 는 혼자 쓰면 정합성만 잰다 — 2026-08-03]
  사후 GT 는 '에이전트 자신의 파일'로 기대값을 만들므로, 그 하나만으로는 정확성이 아니라
  자기 정합성을 잰다. 그래서 절대 GT 판정과 반드시 짝지어야 한다. 이 문항에서 절대적으로
  정해지는 것은 (1) 프롬프트가 "measure ... once" 라고 못박은 측정 횟수와 (2) 마지막
  단계로 지정한 0-1 정규화 두 가지이므로 그 둘을 절차·상태 판정으로 세운다.
  측정 횟수 판정이 없으면 여러 번 쏘고 마음에 드는 것만 남긴 실행이 그냥 통과한다 —
  비가역인 레이저 조사를 더 한 것이라 봐주면 안 되는 종류다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T058",
    score=3,
    axis="procedure",
    mode="live",
    windows=[('1602 cm-1 band', 1596.0, 1608.0, 2)],
    criteria="PROC(1 measurement) + STATE(min0/max1) + SET(3 items, ±3 cm-1, order) / post-hoc GT",
    prompt=(
        "Measure a spectrum once at the current position and save it. Then apply to the "
        "SAVED file, in order: spike removal (5-point moving median, 5x MAD), IPBSA baseline "
        "order 5, Savitzky-Golay (11, 3), and 0-1 normalization, and report the 3 "
        "highest-intensity peaks of the result. "
    ),
    answer_keys=[
        ("peaks", "list of 3 numbers - peak positions in cm-1, ordered by descending intensity"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T058 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    # 절대 GT 판정 — 사후 GT 와 반드시 짝지어야 한다(위 docstring 참조).
    once = chk.called(run, "acquire_spectrum", times=1)

    saved = run.spectra()
    if not saved:
        return [once, chk.fail("top 3 peaks", "no saved spectrum", weight=2.0)]
    _, x, y = saved[-1]
    pk = sp.peaks(x, y)
    if not pk:
        return [once,
                chk.ok("final min/max normalization",
                       abs(y.min()) < 1e-6 and abs(y.max() - 1) < 1e-6,
                       f"min={y.min():.3g} max={y.max():.3g}"),
                chk.fail("top 3 peaks", "no peak was detected", weight=2.0)]
    # 세기 순 상위 3개
    order = sorted(pk, key=lambda p: -float(y[int(np.argmin(abs(x - p)))]))[:3]
    got = run.answer.get("peaks")
    return [
        once,
        chk.ok("final min/max normalization", abs(y.min()) < 1e-6 and abs(y.max() - 1) < 1e-6,
               f"min={y.min():.3g} max={y.max():.3g}"),
        chk.set_match("top 3 peaks", [float(v) for v in got] if isinstance(got, list) else None,
                      order, tol=TOL_PEAK_CM1, ordered=True, partial=True, weight=2.0),
    ]
