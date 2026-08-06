# -*- coding: utf-8 -*-
"""T129 — 신호 판별 (2점)

[문제]
  All peaks of T129.csv are shifted by the same unknown amount. Estimate the shift by
  cross-correlation against reference_library.csv, correct it, identify the material, and
  report the estimated shift. Report the shift as positive when the sample peaks sit at
  higher wavenumbers than the reference, i.e. the value you subtract to correct them.

[정답 기준]
  GT=(추정 시프트 Δ, 물질명). 생성 시 Δ를 기록하므로 완전 GT다. 확인=Δ가 GT와 ±0.2 cm-1이고 부호까지 일치(부호 반대는 오답), 물질명
  완전 일치. Δ는 축 간격의 정수배가 아닌 값으로 잡아 보간 능력까지 시험할 것.

  [부호를 채점하면서 부호 규약을 안 줬다 — 2026-08-06]
  GT 는 +2.7(시료가 참조보다 고파수 쪽), 허용오차 ±0.2 라 부호가 뒤집히면 확정 오답인데
  프롬프트는 "report the estimated shift" 뿐이었다. "내가 관측한 시프트"(+2.7)와
  "보정하려고 적용한 시프트"(-2.7) 는 둘 다 자연스러운 보고이고, 프롬프트가 "correct
  it" 까지 요구하므로 오히려 후자가 더 자연스러울 수 있었다. 부호를 가를 근거가 문항에
  없으면 그건 실력이 아니라 관례 추측을 재는 것이라, 규약을 한 구절로 못박는다.
  (같은 구조가 T103 에도 있다 — 거기도 "NUM(시프트 ±0.2 cm-1, 부호 포함)" 이면서
  프롬프트에 부호 규약이 없다. 이번 검토 범위 밖이라 손대지 않았지만 같이 봐야 한다.)
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T129",
    score=2,
    axis="identification",
    mode="live",
    inputs=['T129.csv', 'reference_library.csv'],
    criteria="NUM(delta ±0.2 cm-1, sign included) + EXACT(material)",
    prompt=(
        "All peaks of T129.csv are shifted by the same unknown amount. Estimate the shift by "
        "cross-correlation against reference_library.csv, correct it, identify the material, "
        "and report the estimated shift. Report the shift as positive when the sample peaks "
        "sit at higher wavenumbers than the reference, i.e. the value you subtract to "
        "correct them. "
    ),
    answer_keys=[
        ("shift_cm1", "number - the estimated shift in cm-1"),
        ("material",
         'string - one of "polystyrene", "PET", "PMMA", "calcite", "aragonite", '
         '"silicon"'),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T129 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "shift_cm1", 2.7, tol=0.2, name="shift"),
        chk.reported_label(run, "material", "polystyrene", ['polystyrene'], name="material"),
    ]
