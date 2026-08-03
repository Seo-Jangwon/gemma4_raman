# -*- coding: utf-8 -*-
"""T095 — 데이터 처리 (2점)

[문제]
  Normalize T095.csv to L2 norm 1 and report its peaks (prominence 5% of range).

[정답 기준]
  원문은 정규화 방식이 없어 0-1과 L2로 갈렸다. L2로 못박았다. GT=(정규화 배열 y/||y||, 피크 위치 목록). 확인=L2 노름 1(±1e-9),
  배열 rtol 1e-6, 피크 ±3 cm-1.
"""
from pathlib import Path

from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T095",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T095.csv'],
    criteria="ARRAY(rtol 1e-6) + SET(peak ±3 cm-1)",
    prompt=(
        "Normalize T095.csv to L2 norm 1 and report its peaks (prominence 5% of range). "
    ),
    answer_keys=[
        ("peaks", "list of numbers - peak positions in cm-1, ascending"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T095 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.set_match("peak positions", run.answer.get("peaks"), [620.0, 795.0, 1001.0, 1031.0, 1155.0, 1184.0, 1450.0, 1583.0, 1602.0], tol=TOL_PEAK_CM1),
    ] + _array_check(run)


# ── 배열 GT ──────────────────────────────────────────────────────────────────
GT_ARRAY = "T095_l2.csv"
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
        return [chk.fail("result array", f"could not read the expected array: {GT_ARRAY}", weight=2.0)]
    saved = run.spectra()
    if not saved:
        return [chk.fail("result array", "no spectrum was saved", weight=2.0)]
    got = saved[-1][2]
    out = [chk.array("result array", got[:len(want)] if len(got) >= len(want) else got,
                     want, mode=GT_MODE, weight=2.0)]
    if len(saved) > 3 and not out[0].passed:
        # 여러 개를 저장했는데 마지막이 답이 아니라면, 어느 것이 맞았는지 알려 준다.
        hit = [i for i, (_, _, y) in enumerate(saved)
               if len(y) >= len(want)
               and chk.array("", y[:len(want)], want, mode=GT_MODE).passed]
        if hit:
            out.append(chk.fail("which save is the conclusion",
                                f"{len(saved)} saved, #{hit} matches the expected array but the last one (#{len(saved)-1}) "
                                f"was saved as the conclusion"))
    return out
