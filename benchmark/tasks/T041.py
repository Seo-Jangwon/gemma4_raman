# -*- coding: utf-8 -*-
"""T041 — 데이터 처리 (2점)

[문제]
  Apply Savitzky-Golay smoothing (window_length=11, polyorder=3, default 'interp' edge
  handling) to T041.csv and save it.

[정답 기준]
  GT=scipy.signal.savgol_filter(y, 11, 3). 구현이 유일해 완전 결정적. 확인=저장 배열이 GT와 rtol 1e-6 일치, 길이
  불변.
"""
from pathlib import Path

from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T041",
    score=2,
    axis="데이터 처리",
    mode="live",
    inputs=['T041.csv'],
    prompt=(
        "Apply Savitzky-Golay smoothing (window_length=11, polyorder=3, default 'interp' "
        "edge handling) to T041.csv and save it. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T041 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.fail("채점 항목 없음", "이 문항의 정답 기준이 아직 옮겨지지 않았습니다"),
    ] + _array_check(run)


# ── 배열 GT ──────────────────────────────────────────────────────────────────
GT_ARRAY = "T041_smoothed.csv"
GT_MODE = "exact"      # exact = 결정적 변환(구현이 유일) / similar = 알고리즘 자유도 있음


def _gt_array():
    """정답 배열. 입력 파일로부터 결정되므로 미리 계산해 둘 수 있다."""
    p = Path(__file__).resolve().parent.parent / "gt" / "arrays" / GT_ARRAY
    if not p.exists():
        return None
    import csv
    rows = list(csv.DictReader(p.read_text(encoding="utf-8-sig").splitlines()))
    if not rows:
        return None
    col = "intensity" if "intensity" in rows[0] else list(rows[0])[-1]
    try:
        return np.array([float(r[col]) for r in rows])
    except (TypeError, ValueError):
        return None


def _array_check(run):
    """에이전트가 **마지막에 저장한** 배열을 정답과 비교한다.

    예전 채점기는 저장물 중 '가장 잘 맞는 하나'를 골랐다. 그러면 파라미터를 훑어 가며
    전부 저장한 실행이 항상 만점을 받는다 — 고르는 능력을 안 재게 된다.
    마지막 저장물이 그 실행의 결론이므로 그것으로 판정한다.
    """
    want = _gt_array()
    if want is None:
        return [chk.fail("결과 배열", f"정답 배열을 읽지 못했습니다: {GT_ARRAY}", weight=2.0)]
    saved = run.spectra()
    if not saved:
        return [chk.fail("결과 배열", "저장한 스펙트럼이 없습니다", weight=2.0)]
    got = saved[-1][2]
    out = [chk.array("결과 배열", got[:len(want)] if len(got) >= len(want) else got,
                     want, mode=GT_MODE, weight=2.0)]
    if len(saved) > 3 and not out[0].passed:
        # 여러 개를 저장했는데 마지막이 답이 아니라면, 어느 것이 맞았는지 알려 준다.
        hit = [i for i, (_, _, y) in enumerate(saved)
               if len(y) >= len(want)
               and chk.array("", y[:len(want)], want, mode=GT_MODE).passed]
        if hit:
            out.append(chk.fail("결론 지목",
                                f"{len(saved)}건 중 {hit}번째가 정답인데 마지막({len(saved)-1}번째)을 "
                                f"결론으로 저장했습니다"))
    return out
