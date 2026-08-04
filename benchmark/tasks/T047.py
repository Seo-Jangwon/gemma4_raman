# -*- coding: utf-8 -*-
"""T047 — 데이터 처리 (2점)

[문제]
  Apply to T047.csv, in this order: spike removal (5-point moving median, 5x MAD, linear
  interpolation), IPBSA baseline correction with order 5, Savitzky-Golay (11, 3), and 0-1
  normalization. Save the result.

[정답 기준]
  GT=4단계를 이 순서로 적용한 배열(각 단계 규약은 T040/T039/T041/T042와 동일). 순서도 GT의 일부다. 확인=저장 배열 대조 +
  min=0/max=1. 순서를 바꾸면 결과가 달라져 배열 비교에서 걸린다.

  [criteria 가 약속만 하고 안 재던 판정 — 2026-08-03]
  criteria 는 'ARRAY + STATE(min0/max1)' 인데 evaluate 에는 배열 판정 하나뿐이었다.
  0-1 정규화는 이 문항이 명시적으로 요구한 마지막 단계이고 저장 배열에서 바로 확인되므로,
  약속한 대로 판정한다.

  [차수 5 는 여기서도 못 가린다 — 실측]
  파이프라인 전체를 order 를 바꿔 가며 돌려 GT 와 비교하면 4~10 차가 전부 통과한다
  (order 4: cos 0.9984 / order 10: cos 0.9965). 2·3 차만 탈락한다. 즉 배열 판정이 가리는
  것은 '차수 5'가 아니라 '알고리즘 계열과 차수가 너무 낮지 않은가'다 — T039 와 같다.
  T039 와 달리 여기서는 프롬프트에 order 5 를 남겨 둔다: 이 문항의 요구는 '지정된 4단계
  파이프라인을 지정된 순서로 재현하라'이고, 파라미터를 고정해 둬야 GT 배열이 하나로
  정해지기 때문이다. 다만 채점이 차수를 개별로 가린다고 주장하지는 않는다.
"""
from pathlib import Path

from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T047",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T047.csv'],
    criteria="ARRAY(cos>=0.99 AND NRMSE<=0.02) + STATE(min0/max1)",
    prompt=(
        "Apply to T047.csv, in this order: spike removal (5-point moving median, 5x MAD, "
        "linear interpolation), IPBSA baseline correction with order 5, Savitzky-Golay (11, "
        "3), and 0-1 normalization. Save the result. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T047 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    out = _array_check(run)
    saved = run.spectra()
    if not saved:
        return out + [chk.fail("final 0-1 normalization", "no spectrum was saved")]
    y = saved[-1][2]
    return out + [
        chk.ok("final 0-1 normalization",
               abs(float(y.min())) < 1e-6 and abs(float(y.max()) - 1.0) < 1e-6,
               f"min={float(y.min()):.6g} max={float(y.max()):.6g} (expected 0 and 1)",
               kind="STATE"),
    ]


# ── 배열 GT ──────────────────────────────────────────────────────────────────
GT_ARRAY = "T047_result.csv"
GT_MODE = "similar"      # exact = 결정적 변환(구현이 유일) / similar = 알고리즘 자유도 있음


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
