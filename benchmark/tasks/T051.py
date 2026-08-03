# -*- coding: utf-8 -*-
"""T051 — 데이터 처리 (2점)

[문제]
  T051.csv holds a 3x3 Raman map (columns: x, y, raman_shift_cm-1, intensity). At each
  position take the intensity of the sample nearest to 1000 cm-1 and save a spatial
  heatmap.

[정답 기준]
  GT=좌표별 9개 값과 그 (x,y) 배치. 입력 포맷을 문항에 명시해 파싱 모호성을 없앴다. 확인=히트맵 데이터 배열이 GT 9값과 rtol 1e-6 일치,
  축이 좌표에 대응(전치되면 오답).
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import answer as A
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T051",
    score=2,
    axis="data processing",
    mode="live",
    inputs=['T051.csv'],
    criteria="ARRAY(rtol 1e-6, 9 values) + STATE(axis layout)",
    prompt=(
        "T051.csv holds a 3x3 Raman map (columns: x, y, raman_shift_cm-1, intensity). At "
        "each position take the intensity of the sample nearest to 1000 cm-1 and save a "
        "spatial heatmap. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T051 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    # 좌표는 x 오름차순 → 그 안에서 y 오름차순. 값 9개는 입력 파일이 정한다.
    xs = [37.8, 37.9, 38.0]
    ys = [25.2, 25.3, 25.4]
    want = [[701.4696, 778.875261, 823.44237], [881.307976, 938.69264, 993.712893], [1049.683145, 1097.546382, 1153.229065]]
    got = A.grid(run, "heatmap", "values", "map", "intensities", "grid")
    if not got:
        return [chk.fail("9 heatmap values", "the answer carries no heatmap values",
                         weight=2.0)]
    flat = [c for row in got for c in row] if isinstance(got[0], list) else list(got)
    want_flat = [c for row in want for c in row]
    out = [chk.set_match("9 heatmap values", flat, want_flat, tol=0.5, weight=2.0)]
    # 축이 좌표에 대응하는가 — 전치되면 같은 9값이라도 지도가 뒤집힌다.
    if isinstance(got[0], list) and len(got) == 3 and all(len(r) == 3 for r in got):
        rows_ok = all(abs(got[i][j] - want[i][j]) <= 0.5 for i in range(3) for j in range(3))
        tr_ok = all(abs(got[j][i] - want[i][j]) <= 0.5 for i in range(3) for j in range(3))
        out.append(chk.ok("axes match the coordinates", rows_ok,
                          f"rows=x{xs}, cols=y{ys}"
                          + ("  - transposed" if tr_ok and not rows_ok else ""),
                          kind="STATE"))
    return out
