# -*- coding: utf-8 -*-
"""T043 — 데이터 처리 (2점)

[문제]
  Find the major Raman peaks of T043.csv (polystyrene). Use scipy.signal.find_peaks with
  prominence set to 5% of the intensity range, and report the 6 largest by prominence in
  ascending order of position, with their intensities.

[정답 기준]
  GT=규약대로 검출한 6개 (위치, 세기). 확인=6개 위치가 GT와 1:1 매칭(±3 cm-1), 세기 상대오차 5%.

  [왜 7개가 아니라 6개인가 — 2026-08-03]
  이 파일에서 prominence 5% 를 넘는 피크는 9개고, prominence 순위의 여유는 이렇다:
      1위 1001 (1015.03) → 2위 1602 (643.44) → 3위 1031 (375.83) → 4위 1450 (233.77)
      → 5위 620 (196.10) → 6위 1154 (161.77) → 7위 793 (117.21) → 8위 1183 (111.54)
  6위와 7위 사이는 27.5% 벌어져 있는데 **7위와 8위 사이는 4.8% 밖에 안 된다**. 상위 7개로
  자르면 입력이나 prominence 규약이 조금만 흔들려도 정답 집합이 793 ↔ 1183 으로 뒤집힌다 —
  에이전트가 규약대로 계산하고도 GT 가 바뀌어 틀리는 문항이 된다. 6개로 자르면 경계 여유가
  27.5% 라 그런 뒤집힘이 없다.
  덤으로 상위 6개(620/1001/1031/1154/1450/1602)는 폴리스티렌의 표준 특성밴드 그대로고,
  7·8위(793/1183)는 그렇지 않다 — 문제의 물리적 의도와도 6개가 맞는다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T043",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T043.csv'],
    criteria="SET(6 items, ±3 cm-1) + NUM(intensity 5%)",
    prompt=(
        "Find the major Raman peaks of T043.csv (polystyrene). Use scipy.signal.find_peaks "
        "with prominence set to 5% of the intensity range, and report the 6 largest by "
        "prominence in ascending order of position, with their intensities. "
    ),
    answer_keys=[
        ("peaks",
         "list of 6 numbers - peak positions in cm-1, ascending. Numbers only, not "
         "objects"),
        ("intensities",
         "list of 6 numbers - the intensity at each of those peaks, in the same "
         "order. Numbers only, not objects"),
    ],
)

# 입력 파일로 확정된 정답. 위 docstring 의 규약(find_peaks, prominence=5% of range,
# prominence 상위 6개, 위치 오름차순)을 그대로 계산한 값이다.
GT_POS = [620.0, 1001.0, 1031.0, 1154.0, 1450.0, 1602.0]
GT_INT = [180.56, 996.92, 429.49, 149.29, 225.02, 627.53]


def evaluate(b, run):
    """이 목록이 그대로 T043 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    out = [
        chk.set_match("peak positions", run.answer.get("peaks"), GT_POS, tol=TOL_PEAK_CM1),
    ]
    # criteria 가 'NUM(intensity 5%)' 라고 적어 놓고 실제로는 세기를 안 보고 있었다.
    # 프롬프트가 세기까지 요구하므로 판정도 한다.
    got = run.answer.get("intensities")
    if not isinstance(got, list) or len(got) != len(GT_INT):
        out.append(chk.fail("peak intensities",
                            f"expected {len(GT_INT)} numbers, got {got!r}"))
    else:
        try:
            vals = [float(v) for v in got]
        except (TypeError, ValueError):
            out.append(chk.fail("peak intensities", f"not numbers: {got!r}"))
        else:
            bad = [(i, g, w) for i, (g, w) in enumerate(zip(vals, GT_INT))
                   if abs(g - w) > abs(w) * 0.05]
            out.append(chk.ok("peak intensities", not bad,
                              "all within 5%" if not bad else
                              f"off by more than 5% at index {[i for i, _, _ in bad]}: "
                              f"{[(round(g, 2), w) for _, g, w in bad]}",
                              kind="NUM"))
    return out
