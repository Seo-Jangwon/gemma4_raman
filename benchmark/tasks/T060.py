# -*- coding: utf-8 -*-
"""T060 — 절차 구성 (3점)

[문제]
  Measure a 3x3 grid at X=40.0, 40.1, 40.2 mm and Y=27.0, 27.1, 27.2 mm, and save a table
  of the strongest peak position and intensity at each position together with its
  coordinates.

[정답 기준]
  GT(좌표)=9점, center=(40.1, 27.1), spacing_mm=0.1, rows=cols=3. GT(표)=채점기가 저장된 9개 파일에서 재계산한
  (x, y, 최강피크 위치, 세기) 9행. 확인=meta.x/y ⊆ GT좌표(±0.001mm), 표 9행, 각 행 피크 ±3 cm-1 / 세기 상대오차 5%.

  [도구를 지정하지 않고 도구로 채점하고 있었다 — 2026-08-06]
  판정 7개 중 5개가 chk.arg(run, "run_grid_scan", ...) 였다. 그런데 프롬프트는 "3x3
  격자를 재라"일 뿐 어느 도구로 풀라고 하지 않는다 — move_stage 9회 + acquire 9회로
  그 아홉 자리를 정확히 훑은 실행은 **명세를 100% 지키고도 5개 판정에서 죽었다**.
  check.chk.visited 가 정확히 이 문제를 위해 만들어져 있고("격자 도구로 푼 실행이 좌표를
  안 남겼다는 이유로 오답이 되면 도구 선택을 벌하는 셈") 격자 인자에서 방문 좌표를
  복원해 두 경로를 다 받는다. 이 문항만 그걸 안 쓰고 있었다.

  [saved[:9] 는 앞에 뭐 하나만 끼어도 통째로 어긋났다 — 2026-08-06]
  예전에는 저장물 앞 9개가 격자 9점이라고 가정했다. 격자 전에 시험 삼아 한 장만 저장해도
  9개 대응이 전부 밀려 확정 오답이 된다. run.acquisitions() 는 acquire_spectrum 결과에
  실려 오는 저장 경로로 (측정 ↔ 스펙트럼)을 짝지으므로 추측이 필요 없고, 격자 도구가 만든
  파일은 호출 기록에 안 남으므로 그때는 저장물 **뒤에서** 9개를 결론으로 본다.

  [프롬프트의 주 산출물을 안 재고 있었다 — 2026-08-06]
  프롬프트가 요구한 것은 "표를 저장하라"인데 표 산출물도, 정답 기준이 약속한 세기
  상대오차 5% 도 evaluate 에 없었다. 둘 다 세운다. 순서는 요구한 적이 없으므로 집합으로
  본다(좌표는 chk.visited 가 절대 GT 로 잡으므로 변별력은 그대로다).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

# 프롬프트가 좌표를 직접 적었으므로 정답은 이 아홉 자리로 확정된다.
GRID = [(x, y) for y in (27.0, 27.1, 27.2) for x in (40.0, 40.1, 40.2)]

TASK = Task(
    id="T060",
    score=3,
    axis="procedure",
    mode="live",
    criteria="SET(coords 9 items) + NUM(peak ±3 cm-1, intensity 5%) / post-hoc GT",
    prompt=(
        "Measure a 3x3 grid at X=40.0, 40.1, 40.2 mm and Y=27.0, 27.1, 27.2 mm, and save a "
        "table of the strongest peak position and intensity at each position together with "
        "its coordinates. "
    ),
    answer_keys=[
        ("peak_positions",
         "list of 9 numbers - the strongest peak position in cm-1 at each grid "
         "point, one per point"),
        ("peak_intensities",
         "list of 9 numbers - the intensity of that strongest peak at each grid "
         "point, in the same order as peak_positions"),
    ],
)


def _grid_spectra(run):
    """격자 9점의 스펙트럼 — 어느 도구로 풀었든 같은 것을 돌려준다.

    acquire_spectrum 으로 푼 실행은 호출 결과에 저장 경로가 실려 있어 정확히 짝지어지고,
    run_grid_scan 은 내부 루프라 호출 기록에 개별 파일이 안 남으므로 저장물 뒤에서 9개를
    그 실행의 결론으로 본다.
    """
    acq = [(a["x"], a["y"]) for a in run.acquisitions() if a["x"] is not None]
    if len(acq) >= 9:
        return acq[-9:]
    return [(x, y) for _, x, y in run.spectra()[-9:]]


def evaluate(b, run):
    """이 목록이 그대로 T060 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    out = [
        # 어느 자리를 훑었는가 — 격자 도구/개별 이동 두 경로를 모두 인정한다.
        chk.visited(run, GRID),
        # 표를 남겼는가. 내용까지는 여기서 못 보지만 '요구한 산출물을 냈는가'는 볼 수 있다.
        chk.any_called(run, ["aggregate_spectra_csv", "run_analysis", "combine_spectra"]),
    ]

    pts = _grid_spectra(run)
    if len(pts) < 9:
        return out + [chk.fail("grid peak table",
                               f"{len(pts)} spectra available (need 9)", weight=2.0)]

    pos = [sp.strongest_peak(x, y) for x, y in pts]
    inten = [float(y[int(np.argmin(np.abs(x - p)))]) for (x, y), p in zip(pts, pos)]

    got_p = run.answer.get("peak_positions")
    got_i = run.answer.get("peak_intensities")
    return out + [
        # 순서는 프롬프트가 요구한 적이 없다 — 아홉 값의 집합으로 본다.
        chk.set_match("strongest peak position at 9 points",
                      [float(v) for v in got_p] if isinstance(got_p, list) else None,
                      pos, tol=TOL_PEAK_CM1, partial=True, weight=2.0),
        # 정답 기준이 약속하고 재지 않던 판정. 세기는 실측 절대값이라 사후 GT 다.
        chk.set_match("peak intensity at 9 points",
                      [float(v) for v in got_i] if isinstance(got_i, list) else None,
                      inten, tol=max(abs(v) for v in inten) * 0.05, partial=True),
    ]
