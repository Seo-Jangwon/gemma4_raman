# -*- coding: utf-8 -*-
"""T115 — 신호 판별 (2점)

[문제]
  Apply IPBSA baseline correction with order 5 to the raw spectrum T115.csv and save the
  corrected spectrum. Then apply L2 normalization, compare it with reference_library.csv on
  the reference axis using cosine similarity, and report the most similar material.

[정답 기준]
  GT=물질명 1개. 규약=IPBSA(order=5) → 참조축 보간 → L2 → 코사인. 배경을 얹은 원시 스펙트럼이라 보정을 건너뛰면 유사도 순위가 뒤집히도록
  입력을 설계할 것 (그래야 전처리 단계가 실제로 채점된다). 확인=물질명 완전 일치 + 저장한 보정 배열이 GT와 모양 일치(배율 무관).

  ['가점'이라는 것이 이 프레임워크에 없다 — 2026-08-06]
  예전 정답 기준은 배열 판정을 "가점"이라고 적었지만 check.py 가 밝히듯 이 벤치는
  **이진 채점이고 weight 는 장식**이다 — 판정 하나만 떨어져도 그 문항은 fail 이다.
  즉 배열 판정은 가점이 아니라 필수 관문이었는데, **프롬프트에는 저장하라는 말이
  없었다.** 메모리에서 처리하고 calcite 를 정확히 맞힌 실행이 "no spectrum was saved"
  로 죽었다. 채점할 것은 프롬프트가 요구해야 하므로 저장 지시를 문항에 넣는다.

  [시킨 대로 저장하면 배열 판정이 죽었다 — 2026-08-06]
  프롬프트는 "IPBSA 및 **L2 정규화**"를 요구하는데 GT 배열(gt/arrays/T115_corrected.csv)은
  정규화 전 값(-14.2 ~ 994.8 counts)이다. 최종 산출물(최대 ~0.02)을 저장하면 코사인은
  1.00000 인데 NRMSE 는 GT 의 range 로 나뉘어 0.0622(상한 0.02) → 확정 실패였다.
  정규화 여부는 이 문항이 자유도로 남긴 것이므로 배율을 맞추고 모양을 보는
  mode="shape" 로 판정한다. 실측으로 확인한 판정 표(shape):
      보정본 저장 PASS(cos 1.00000) / 정규화본 저장 PASS(cos 1.00000)
      보정 안 한 원시 FAIL(cos 0.25255) / 원시 정규화본 FAIL(cos 0.25255)
  즉 저장 형태의 자유도만 흡수하고 **보정을 건너뛴 것**은 그대로 걸린다.

  [유사도 정의가 프롬프트에 없었다 — 2026-08-06]
  규약이 GT json 에만 있었다. 물질이 calcite 로 크게 갈려 순위가 뒤집힐 위험은 낮지만,
  판정이 기대하는 계산을 문항이 말하지 않는 상태를 남겨 둘 이유가 없어 코사인을 명시한다.
"""
from pathlib import Path

from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T115",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T115.csv', 'reference_library.csv'],
    criteria="EXACT(material) + ARRAY(corrected array, shape)",
    prompt=(
        "Apply IPBSA baseline correction with order 5 to the raw spectrum T115.csv and save "
        "the corrected spectrum. Then apply L2 normalization, compare it with "
        "reference_library.csv on the reference axis using cosine similarity, and report the "
        "most similar material. "
    ),
    answer_keys=[
        ("material",
         'string - one of "polystyrene", "PET", "PMMA", "calcite", "aragonite", '
         '"silicon"'),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T115 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported_label(run, "material", "calcite", ['calcite'], name="material"),
    ] + _array_check(run)


# ── 배열 GT ──────────────────────────────────────────────────────────────────
GT_ARRAY = "T115_corrected.csv"
# shape = 세로 배율에 자유도가 있는 변환. 이 문항은 L2 정규화까지 요구하므로 저장물이
# 보정본일 수도 정규화본일 수도 있다 — 배율을 맞추고 모양을 본다(check.array 주석 참고).
GT_MODE = "shape"


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
