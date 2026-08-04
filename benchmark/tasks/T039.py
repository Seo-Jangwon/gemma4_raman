# -*- coding: utf-8 -*-
"""T039 — 데이터 처리 (2점)

[문제]
  Remove the fluorescence background from T039.csv with IPBSA (iterative polynomial
  background subtraction, max_iterations=100, threshold=0.001). Choose the polynomial order
  yourself from the shape of this spectrum's background, save the corrected spectrum, and
  compare before and after in one graph.

[정답 기준]
  GT=IPBSA 보정 배열. 확인=마지막 저장 배열이 cos>=0.99 AND NRMSE<=0.02, 그림 1장.

  [왜 프롬프트에서 'order 5' 를 뺐는가 — 2026-08-03]
  예전 프롬프트는 'polynomial order 5' 를 지정했고 문서에는 '차수는 1차(툴 인자)에서
  채점한다'고 적혀 있었다. 그런데 (1) evaluate 에 chk.arg 가 없었고, (2) 있었어도 이
  문항은 업로드 CSV 를 run_analysis 안의 자작 코드로 처리하므로 poly_order 가 툴 인자로
  나타나지 않는다. 즉 **구조적으로 검증 불가능한 요구**를 프롬프트에 적어 두고 있었다.
  배열 비교로 차수를 가릴 수 있는지 직접 재 봤다(GT=order 5 기준):
      order  2: cos 0.7863 NRMSE 0.0713  탈락
      order  3: cos 0.9743 NRMSE 0.0176  탈락
      order  4: cos 0.9972 NRMSE 0.0055  통과
      order  5: cos 1.0000 NRMSE 0.0000  통과
      order  6: cos 0.9999 NRMSE 0.0003  통과   … order 10 까지 전부 통과
      단순 polyfit(3/5/7차): cos 0.917~0.933  전부 탈락
      보정 안 함:            cos 0.4064      탈락
  배열 판정이 실제로 가리는 것은 '차수 5'가 아니라 **알고리즘 계열(IPBSA 인가)과
  차수가 너무 낮지 않은가(4 이상)** 다. 그래서 못 재는 것을 요구하는 대신, 도구 스키마가
  "차수 선택이 이 작업의 핵심"이라고 말하는 그대로 에이전트에게 고르게 하고, 채점은
  실제로 가릴 수 있는 것만 약속한다. 낮은 차수(2·3)를 고르면 배경이 남아 탈락한다 —
  이건 문체가 아니라 진짜 오답이다.
"""
from pathlib import Path

from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T039",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T039.csv'],
    criteria="ARRAY(cos>=0.99 AND NRMSE<=0.02) + STATE(figure)",
    prompt=(
        "Remove the fluorescence background from T039.csv with IPBSA (iterative polynomial "
        "background subtraction, max_iterations=100, threshold=0.001). Choose the polynomial "
        "order yourself from the shape of this spectrum's background, save the corrected "
        "spectrum, and compare before and after in one graph. "
    ),
    answer_keys=[
        ("poly_order", "number - the polynomial order you chose"),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T039 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    # poly_order 판정은 '차수가 5 인가'가 아니다 — 위 docstring 대로 그건 이 문항에서
    # 잴 수 없다. 여기서 재는 것은 '차수를 실제로 정해서 밝혔는가'이고, 그 선택이
    # 쓸 만했는지는 배열 판정이 가린다(2·3차는 탈락한다).
    o = run.answer.get("poly_order")
    try:
        ok = o is not None and 2 <= float(o) <= 10
    except (TypeError, ValueError):
        ok = False
    return _array_check(run) + [
        chk.ok("polynomial order stated", ok,
               f"reported poly_order={o!r} (must be a number in the tool's 2-10 range)",
               kind="NUM"),
        # 프롬프트가 그림을 요구한다 — 요구했으면 확인한다.
        chk.figure(run, name="before/after graph saved"),
    ]


# ── 배열 GT ──────────────────────────────────────────────────────────────────
GT_ARRAY = "T039_corrected.csv"
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
