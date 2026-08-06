# -*- coding: utf-8 -*-
"""T058 — 절차 구성 (3점)

[문제]
  Measure a spectrum once at the current position and save it. Then apply to the SAVED
  file, in order: spike removal (5-point moving median, 100x MAD), IPBSA baseline order 5,
  Savitzky-Golay (11, 3), and 0-1 normalization. Save the processed spectrum, and report
  the 3 highest-intensity peaks of the result.

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

  [저장하라고 하지 않고 저장물로 채점했다 — 2026-08-06]
  채점기는 saved[-1] 에 min=0/max=1 을 요구하고 그 배열에서 피크를 재계산해 GT 로
  삼는데, 프롬프트는 "처리를 적용하고 상위 3피크를 **보고**하라" 였을 뿐 저장하라는
  말이 없었다. 도구 목록에 정규화 도구가 없어 마지막 단계는 run_analysis 코드로 해야
  하고 그 결과를 파일로 남기는 것은 별도 행위라, 코드로 계산해 정확히 보고한 실행은
  saved[-1] 이 원시 측정본이라 **판정 3 개 중 2 개가 죽었다**. 사후 GT 는 에이전트가
  남긴 파일이 있어야 성립하므로 그 파일을 만들라는 지시를 프롬프트에 넣는다.

  [정규화 후 2·3위가 붙으면 순서가 실력과 무관해진다 — 2026-08-06]
  상위 3피크를 ordered=True 로 봤다. 0-1 정규화 뒤 2위와 3위의 세기가 근소하면 순서
  뒤집힘이 잡음으로 일어난다. 두 값이 충분히 벌어졌을 때만 순서를 따진다.

  [스파이크 규약을 T040 과 맞췄다 — 2026-08-06]
  '5x MAD' 는 폭이 좁은 진짜 라만 밴드까지 스파이크로 집는 규약이다(T040 docstring 과
  synth.detect_spikes 참고 — 실측으로 6 개 대신 53 개가 검출된다). 이 문항은 스파이크
  위치를 채점하지 않지만, 시킨 대로 하면 실제 밴드를 지워 뒤 단계와 보고할 피크가
  망가진다. 벤치 안에서 같은 규약이 두 값을 갖지 않도록 100x 로 통일한다.
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
        "SAVED file, in order: spike removal (5-point moving median, 100x MAD), IPBSA "
        "baseline order 5, Savitzky-Golay (11, 3), and 0-1 normalization. Save the "
        "processed spectrum, and report the 3 highest-intensity peaks of the result. "
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
    height = lambda p: float(y[int(np.argmin(abs(x - p)))])
    order = sorted(pk, key=lambda p: -height(p))[:3]
    # 2·3위가 붙어 있으면 순서는 잡음이 정한다 — 그때는 순서를 묻지 않는다.
    # 배열은 0-1 정규화되어 있으므로 0.02 는 전체 범위의 2% 다.
    ranked = len(order) >= 3 and (height(order[1]) - height(order[2])) > 0.02
    got = run.answer.get("peaks")
    return [
        once,
        chk.ok("final min/max normalization", abs(y.min()) < 1e-6 and abs(y.max() - 1) < 1e-6,
               f"min={y.min():.3g} max={y.max():.3g}"),
        chk.set_match("top 3 peaks", [float(v) for v in got] if isinstance(got, list) else None,
                      order, tol=TOL_PEAK_CM1, ordered=ranked, partial=True, weight=2.0),
    ]
