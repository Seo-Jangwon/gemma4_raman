# -*- coding: utf-8 -*-
"""T122 — 신호 판별 (2점)

[문제]
  Report the identification info (spectrum_id and material) of the reference in
  reference_library.csv that best matches T122.csv.

[정답 기준]
  GT=최상위 참조의 (spectrum_id, material). 규약=참조축(reference_library의 공통 파수축)으로 선형보간 → L2 정규화 →
  코사인 유사도. 이 3단계를 문항에 명시해 유사도 정의를 단일화했다. 확인=spectrum_id까지 완전 일치. 같은 물질의 다른 항목(PS_01 vs
  PS_02)을 답하면 부분점 — 이 문항은 '물질'이 아니라 '항목'을 식별하는지를 본다. 출제 시 동일 물질의 두 참조 간 유사도 차가 0.02 이상인지
  확인할 것.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import answer as A
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T122",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T122.csv', 'reference_library.csv'],
    criteria="EXACT(spectrum_id) / partial credit if only the material is right",
    prompt=(
        "Report the identification info (spectrum_id and material) of the reference in "
        "reference_library.csv that best matches T122.csv. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T122 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    want_id, want_mat = 'PMMA_02', 'PMMA'
    got_id = A.value(run, "spectrum_id", "id", "best_id")
    got_mat = A.value(run, "material", "best_material") or run.last_mention(['PET', 'PMMA', 'aragonite', 'calcite', 'polystyrene', 'silicon'])
    # 같은 물질의 다른 항목(PMMA_01)을 답하면 절반. 이 문항은 '물질'이 아니라
    # '어느 참조 항목인지'를 식별하는지를 본다.
    exact = str(got_id or "").strip().upper() == want_id.upper()
    partial = not exact and str(got_mat or "").strip().lower() == want_mat.lower()
    return [
        chk.ok("best-matching reference", exact,
               f"spectrum_id={got_id!r} (expected {want_id!r})"
               + (" - right material, wrong entry" if partial else ""),
               weight=2.0, kind="EXACT", score=1.0 if exact else (0.5 if partial else 0.0)),
    ]
