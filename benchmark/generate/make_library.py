# -*- coding: utf-8 -*-
"""참조 라이브러리 생성 + 판별 가능성 검증.

생성만 하고 끝내면 'GT 는 있는데 풀 수 없는 문항'이 남는다. 설계 단계에서 경고한 세 가지를
여기서 수치로 확인하고, 미달이면 실패시켜 다음 단계로 못 넘어가게 한다.
"""
from __future__ import annotations

import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from materials import LIBRARY_12, LIBRARY_8                       # noqa: E402
import synth                                                       # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "inputs"

# 검증 기준 — 설계문서 §9 의 리스크와 1:1 대응한다.
GAP_CONFUSABLE = 0.05   # PET vs PMMA, calcite vs aragonite: 정답과 오답의 유사도 차
GAP_SAME_MAT = 0.02   # 같은 물질의 두 항목: T122 가 '항목'을 식별하려면 벌어져야 한다


def build(entries):
    x = synth.axis()
    return x, {sid: (mat, synth.scale_counts(synth.pure(mat, x, broaden=bd)))
               for sid, mat, bd in entries}


def write_library(path, x, lib):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["spectrum_id", "material", "raman_shift_cm-1", "intensity"])
        for sid, (mat, y) in lib.items():
            for xv, yv in zip(x, y):
                w.writerow([sid, mat, f"{xv:.4f}", f"{yv:.6f}"])


def verify(x, lib):
    """세 가지 판별 가능성을 확인하고 (통과여부, 보고문) 을 돌려준다."""
    ids = list(lib)
    norm = {sid: synth.l2(y) for sid, (m, y) in lib.items()}
    S = {a: {b: synth.cosine(norm[a], norm[b]) for b in ids} for a in ids}
    mat = {sid: m for sid, (m, y) in lib.items()}
    lines, ok = [], True

    lines.append("[1] 혼동쌍이 실제로 구별되는가 (정답 자기유사도 vs 최고 오답)")
    for a, b in [("PET_01", "PMMA_01"), ("CAL_01", "ARA_01")]:
        if a not in S or b not in S:
            continue
        cross = S[a][b]
        same = max(S[a][o] for o in ids if o != a and mat[o] == mat[a])
        gap = same - cross
        good = gap >= GAP_CONFUSABLE
        ok &= good
        lines.append(f"    {a} vs {b}: 같은물질 {same:.4f} / 다른물질 {cross:.4f} "
                     f"→ 차 {gap:.4f} {'OK' if good else 'FAIL'}")

    lines.append("[2] 같은 물질의 두 항목이 구별되는가 (T122)")
    for m in sorted(set(mat.values())):
        same_ids = [s for s in ids if mat[s] == m]
        if len(same_ids) < 2:
            continue
        a, b = same_ids[0], same_ids[1]
        d = 1.0 - S[a][b]
        good = d >= GAP_SAME_MAT
        ok &= good
        lines.append(f"    {a} vs {b}: 유사도 {S[a][b]:.4f} → 차 {d:.4f} "
                     f"{'OK' if good else 'FAIL'}")

    lines.append("[3] 모든 물질이 자기 물질을 1위로 찾는가")
    for a in ids:
        best = max((o for o in ids if o != a), key=lambda o: S[a][o])
        good = mat[best] == mat[a]
        ok &= good
        lines.append(f"    {a:8s} 최근접={best:8s}({mat[best]}) {'OK' if good else 'FAIL'}")
    return ok, "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, lib12 = build(LIBRARY_12)
    _, lib8 = build(LIBRARY_8)
    write_library(OUT / "reference_library.csv", x, lib12)
    write_library(OUT / "reference_library_8.csv", x, lib8)
    print(f"reference_library.csv     : {len(lib12)}개 항목 × {len(x)}점")
    print(f"reference_library_8.csv   : {len(lib8)}개 항목 × {len(x)}점\n")
    ok, report = verify(x, lib12)
    print(report)
    print("\n검증:", "통과" if ok else "실패 — 밴드 정의를 고쳐야 한다")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
