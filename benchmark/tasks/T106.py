# -*- coding: utf-8 -*-
"""T106 — 신호 판별 (3점)

[문제]
  Classify the sample of T106.csv as amorphous, crystalline, or undecidable. Use this rule:
  if the strongest peak has FWHM below 15 cm-1 it is crystalline; above 40 cm-1 it is
  amorphous; in between it is undecidable. Measure the FWHM inside 960-1060 cm-1, taking
  the half height as (peak intensity - interval minimum)/2 with crossings by linear
  interpolation. Report the FWHM you measured.

[정답 기준]
  판정 기준이 없어 레이블이 흔들렸다. FWHM 임계를 문항에 넣어 확정했다. GT=(레이블 1개, FWHM 값). 확인=레이블 완전 일치 + FWHM 상대오차
  5%. 레이블이 맞아도 근거 FWHM이 임계와 모순되면 감점(우연 정답 배제).

  [FWHM 규약이 없어 값이 30 cm-1 넘게 흔들렸다 — 2026-08-06]
  GT 는 synth.fwhm_spec(y, x, 960, 1060) — **구간 최소를 기준선으로** 잡은 51.79 인데,
  프롬프트는 창도 기준선도 말하지 않고 "the strongest peak" 라고만 했다. 입력으로 직접
  재 보면 합리적 해석마다 값이 이렇게 갈린다:
      창 960-1060 / 구간최소 기준   51.79 (GT)      창 900-1100 / 구간최소  67.08
      창 970-1030 / 구간최소        44.73           전 구간 / 반높이=최대/2  73.71
  허용 밴드(±5%)에 드는 것은 GT 가 쓴 그 조합 하나뿐이라, 가장 소박한 해석(73.71)은
  40% 벗어나 확정 오답이었다. T045 가 같은 FWHM 을 물으면서 창과 반높이 기준을 프롬프트에
  적어 두고 있으므로, **그 문장을 그대로 가져와** 규약을 통일한다.

  [GT 가 분류 경계에 붙어 있어 문항이 자기모순이었다 — 2026-08-06]
  임계가 50 인데 GT 가 51.79 — 여유가 3.5% 뿐이라 허용 밴드 [49.2, 54.4] 의 아래쪽이
  **임계 밑**이었다. FWHM 49.5 를 보고한 답은 수치 판정은 통과하지만 그 값의 논리적
  귀결은 undecidable 이므로, amorphous 라 하면 자기 근거와 모순되고 undecidable 이라
  하면 레이블 판정에서 죽는다 — 어느 쪽을 골라도 지는 문항이었다.
  임계를 40 으로 내리면 밴드 전체가 임계 위에 놓여(51.79 는 40 의 1.29 배) 근거와 결론이
  항상 같은 방향을 가리킨다. 데이터·GT 값·레이블은 그대로다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T106",
    score=3,
    axis="identification",
    mode="live",
    inputs=['T106.csv'],
    criteria="EXACT(labels) + NUM(FWHM 5%)",
    prompt=(
        "Classify the sample of T106.csv as amorphous, crystalline, or undecidable. Use this "
        "rule: if the strongest peak has FWHM below 15 cm-1 it is crystalline; above 40 cm-1 "
        "it is amorphous; in between it is undecidable. Measure the FWHM inside 960-1060 "
        "cm-1, taking the half height as (peak intensity - interval minimum)/2 and finding "
        "the crossings by linear interpolation. Report the FWHM you measured. "
    ),
    answer_keys=[
        ("fwhm_cm1", "number - FWHM of the strongest peak in cm-1"),
        ("label", 'string - one of "crystalline", "amorphous", "undecidable"'),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T106 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "fwhm_cm1", 51.787707659444436, rel=0.05, name="FWHM"),
        chk.reported_label(run, "label", "amorphous", ["crystalline", "amorphous", "undecidable"], name="classification"),
    ]
