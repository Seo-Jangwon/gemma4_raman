# -*- coding: utf-8 -*-
"""T040 — 데이터 처리 (2점)

[문제]
  Remove the narrow spikes from T040.csv and save the corrected spectrum. Treat a point as
  a spike when it deviates from the 5-point moving median by more than 5x the MAD, and
  replace it by linear interpolation from its neighbours.

[정답 기준]
  GT=규약대로 계산한 (제거 위치 집합, 보정 배열). 규약을 문항에 넣어 구현 자유도를 없앴다. 확인=저장 파일을 load_spectrum 해 대조 +
  보고한 스파이크 위치가 GT 6개와 완전 일치(개수가 적어 부분 일치를 허용하면 변별력이 없다).

  [criteria 가 약속만 하고 안 재던 판정 — 2026-08-03]
  criteria 는 'SET(index EXACT) + ARRAY' 였는데 evaluate 에는 배열 판정 하나뿐이었다.
  스파이크는 6 점(1801 점 중)이라 배열 판정만으로는 변별력이 거의 없다 — 아무것도 안 고치고
  원본을 그대로 저장해도 cos 는 1 에 가깝다. 실제로 재 보면 GT 배열은 입력과 정확히 6 곳에서만
  다르고(상대차 1e-4 이상 기준), 그 자리는 synth.remove_spikes 의 검출 결과와 정확히 같다:
      index    122   386   697   776   1241   1693
      cm-1     322   586   897   976   1441   1893
  이 위치들을 보고하게 하고 완전 일치를 요구하면, 규약을 실제로 적용했는지가 드러난다.
"""
from pathlib import Path

from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T040",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T040.csv'],
    criteria="SET(index EXACT) + ARRAY(cos>=0.99 AND NRMSE<=0.02)",
    prompt=(
        "Remove the narrow spikes from T040.csv and save the corrected spectrum. Treat a "
        "point as a spike when it deviates from the 5-point moving median by more than 5x "
        "the MAD, and replace it by linear interpolation from its neighbours. Report the "
        "positions of the spikes you removed. "
    ),
    answer_keys=[
        ("spikes",
         "list of numbers - the Raman shift in cm-1 of every point you treated as a "
         "spike, ascending. Numbers only, not objects"),
    ],
)

# 입력 파일과 규약으로 확정된 정답(위 docstring 참조).
GT_SPIKES_CM1 = [322.0, 586.0, 897.0, 976.0, 1441.0, 1893.0]


def evaluate(b, run):
    """이 목록이 그대로 T040 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return _array_check(run) + [
        # 축 간격이 1 cm-1 이라 ±0.5 면 '이웃 점을 지웠다'는 여전히 오답이다. 0 으로 두지
        # 않는 이유는 부동소수 표기(321.99998)로 갈리게 하지 않기 위해서다.
        chk.set_match("spike positions", run.answer.get("spikes"), GT_SPIKES_CM1,
                      tol=0.5, weight=2.0),
    ]


# ── 배열 GT ──────────────────────────────────────────────────────────────────
GT_ARRAY = "T040_cleaned.csv"
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
