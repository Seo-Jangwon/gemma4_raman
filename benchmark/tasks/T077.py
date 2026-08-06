# -*- coding: utf-8 -*-
"""T077 — 절차 구성 (3점)

[문제]
  Test at most 8 of the combinations of exposure 0.25, 0.5, 1.0, 2.0 s and laser power 20,
  40, 60 %. Among the conditions satisfying SNR >= 20 (T050 definition), no saturated
  pixel, and at least 90% of the T077_ref.csv reference peaks detected, select the one with
  the lowest dose, where dose = power x exposure x 0.01 mJ. If two combinations give the
  same dose, choose the one with the lower laser power.

[정답 기준]
  GT(탐색)=측정 8회 이하, 각 조합이 정의된 격자 안. dose 산식을 명시해 선택 기준을 확정했다. GT(선택)=사후에 3조건을 재평가해 얻은 최소
  dose 조합. 확인=선택 근거로 3조건을 모두 제시했는지. 9회 이상 측정하면 오답(조사량 통제가 이 문항의 요구).

  [채점기가 프롬프트보다 느슨한 조건을 쓰고 있었다 — 2026-08-06]
  프롬프트의 자격 조건은 셋(SNR>=20, 포화 0, 참조 피크 90% 검출)인데 evaluate 는
      if s is not None and s >= 10.0 and sp.saturated_count(y) == 0:
  이었다. **임계가 10 이고 피크 재현율은 아예 안 봤다.** 조건이 느슨하면 자격 집합이
  커지고 min(dose) 가 고르는 조합은 체계적으로 더 낮은 dose 쪽으로 내려가므로,
  프롬프트대로 SNR>=20 을 적용한 실행이 채점기 기대값과 달라 **정확히 이해할수록
  틀리는** 문항이었다. 임계값 세 개는 gt/T077.json 에 이미 min_snr / peak_recall /
  reference_peaks 로 적혀 있었는데 evaluate 가 안 읽고 상수를 따로 박아 둔 것이 원인이라,
  이제 GT 파일 하나에서 읽는다(두 벌 관리를 없앤다).

  [dose 동점에 규칙이 없었다 — 2026-08-06]
  12 조합의 power x exposure 곱에는 동점이 세 쌍 있다:
      10 → (0.25s, 40%) / (0.5s, 20%)    20 → (0.5s, 40%) / (1.0s, 20%)
      40 → (1.0s, 40%) / (2.0s, 20%)
  min() 은 동점이면 리스트 첫 원소, 즉 **에이전트가 먼저 잰 쪽**을 돌려주므로 측정
  순서가 정답을 바꿨다. 프롬프트에 타이브레이크(파워가 낮은 쪽)를 넣고 정렬 키를 맞춘다.

  [자격 조합이 하나도 없으면 오답이 아니라 채점 불가다]
  장비 상태에 따라 12 조합 중 어느 것도 SNR>=20 을 못 넘을 수 있다. 그때 기대 답 자체가
  정해지지 않으므로 chk.blocked 로 분모에서 뺀다 — 장비 사정을 에이전트 실력으로
  기록하는 것이 이 프레임워크에서 가장 나쁜 고장이다(check.blocked 주석 참고).
"""
import json
from pathlib import Path

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
        "select the one with the lowest dose, where dose = power x exposure x 0.01 mJ. If "
        "two combinations give the same dose, choose the one with the lower laser power. "
    ),
    answer_keys=[
        ("exposure", "number - the exposure time in seconds you chose"),
        ("power", "number - the laser power in percent you chose"),
    ],
)

# 자격 조건 세 개는 GT 파일이 원본이다. evaluate 에 숫자를 다시 박으면 프롬프트·GT·채점기가
# 서로 다른 말을 하게 된다(그것이 이 문항이 오래 틀려 있던 이유다).
_GT = json.loads((Path(__file__).resolve().parent.parent / "gt" / "T077.json")
                 .read_text(encoding="utf-8"))
MIN_SNR = float(_GT["min_snr"])                 # 20.0
PEAK_RECALL = float(_GT["peak_recall"])         # 0.9
REF_PEAKS = [float(v) for v in _GT["reference_peaks"]]

GRID_E, GRID_P = (0.25, 0.5, 1.0, 2.0), (20, 40, 60)


def _recall(x, y):
    """참조 피크 중 몇 할이 이 스펙트럼에서 검출되는가(위치 ±3 cm-1)."""
    found = sp.peaks(x, y)
    hit = sum(1 for p in REF_PEAKS
              if any(abs(p - f) <= TOL_PEAK_CM1 for f in found))
    return hit / len(REF_PEAKS) if REF_PEAKS else 0.0


def evaluate(b, run):
    """이 목록이 그대로 T077 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    # 설정은 되읽기로 본다 — set_ccd_exposure/set_laser_power 로 걸고 인자 없이 부른
    # 실행을 벌하지 않기 위해서다. 측정과 스펙트럼의 짝도 여기서 확정된다.
    acqs = run.acquisitions()
    off_grid = [(a["exposure"], a["power"]) for a in acqs
                if a["exposure"] not in GRID_E or a["power"] not in GRID_P]

    out = [
        chk.called(run, "acquire_spectrum", at_least=1, at_most=8),
        chk.ok("stayed inside the defined grid", not off_grid,
               f"off-grid {len(off_grid)} acquisitions: {off_grid[:4]}"),
    ]

    # 프롬프트가 적은 세 조건을 그대로 적용한다.
    ok_pairs, seen = [], []
    for a in acqs:
        x, y, e, p = a["x"], a["y"], a["exposure"], a["power"]
        if x is None or e is None or p is None:
            continue
        s = sp.snr(x, y)
        rec = _recall(x, y)
        sat = sp.saturated_count(y)
        seen.append(f"({e}s,{p}%) SNR={s if s is None else round(s, 1)} "
                    f"recall={rec:.0%} sat={sat}")
        if s is not None and s >= MIN_SNR and sat == 0 and rec >= PEAK_RECALL:
            ok_pairs.append((float(e), float(p)))

    if not acqs:
        return out + [chk.fail("minimum-dose combination", "no acquisition was recorded",
                               weight=2.0)]
    if not ok_pairs:
        # 기대 답이 정해지지 않는다 — 오답이 아니라 채점 불가다.
        return out + [chk.blocked(
            "minimum-dose combination",
            f"no acquisition met SNR>={MIN_SNR:g} / 0 saturated / recall>={PEAK_RECALL:.0%}; "
            f"tested: {'; '.join(seen[:8])}", weight=2.0)]

    # dose = power x exposure x 0.01 (곱만 비교하면 되고, 동점은 파워가 낮은 쪽)
    want = min(ok_pairs, key=lambda t: (t[0] * t[1], t[1]))
    got_e, got_p = _num(run.answer.get("exposure")), _num(run.answer.get("power"))
    return out + [
        chk.ok("chose the minimum-dose combination",
               got_e is not None and got_p is not None
               and abs(got_e - want[0]) < 1e-6 and abs(got_p - want[1]) < 1e-6,
               f"reported=({got_e}, {got_p}) expected={want} "
               f"(qualifying combinations={ok_pairs})", weight=2.0),
    ]


def _num(v):
    """답변 값을 숫자로. 못 읽으면 None — 여기서 예외가 나면 문항이 통째로 채점 불가가 된다."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
