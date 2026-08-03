# -*- coding: utf-8 -*-
"""T018 — 계측 제어 (1점)

[문제]
  Tell me the current CCD temperature.

[정답 기준]
  GT=문항이 도는 동안 CCD 가 실제로 지나간 온도 구간. 확인=보고값이 그 구간 안에 있는가(±0.5).

[구간으로 보는 이유 — 2026-08-03]
  예전에는 실행이 끝난 뒤 스냅샷(after)의 온도 한 점과 ±0.1 로 비교했다. 그런데 CCD 가
  냉각 중이면 에이전트가 읽은 시각과 after 를 찍은 시각의 온도가 다르다. 실제로 에이전트가
  -42 를 정확히 읽어 보고했는데 after 가 -40 이라 0 점이 났다. 시각이 다른 두 측정을
  ±0.1 로 맞추라는 것은 성립하지 않는다. 시작(before)과 끝(after) 사이 어디든 그 순간의
  참값이므로, 그 구간에 들어오면 맞다고 본다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T018",
    score=1,
    axis="instrument control",
    mode="live",
    criteria="NUM(±0.1)",
    prompt=(
        "Tell me the current CCD temperature. "
    ),
    answer_keys=[
        ("temperature_C", "number - CCD temperature in Celsius"),
    ],
)


DRIFT_TOL_C = 0.5      # 구간 밖으로 이만큼까지는 되읽기 오차로 본다


def evaluate(b, run):
    """이 목록이 그대로 T018 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    seen = [t for t in (before.get("temperature_C"), after.get("temperature_C"))
            if t is not None]
    # 에이전트가 자기 도구로 읽은 값도 참값 후보다 — 그 시각의 장비가 그렇게 답했으므로.
    for c in run.calls:
        r = c.get("result")
        if isinstance(r, dict) and isinstance(r.get("temperature_C"), (int, float)):
            seen.append(float(r["temperature_C"]))
    if not seen:
        return [chk.blocked("reported temperature_C",
                            "the instrument temperature could not be read at any point")]

    lo, hi = min(seen) - DRIFT_TOL_C, max(seen) + DRIFT_TOL_C
    got = run.answer.get("temperature_C")
    if got is None:
        got = run.number_near((lo + hi) / 2, tol=(hi - lo) / 2 if hi > lo else DRIFT_TOL_C)
        src = "prose" if got is not None else "no answer block"
    else:
        src = "answer"
    try:
        v = float(got)
    except (TypeError, ValueError):
        v = None
    return [
        chk.ok("reported temperature_C", v is not None and lo <= v <= hi,
               f"{v} (instrument passed through {min(seen):.1f}..{max(seen):.1f} C "
               f"during the run, accepted {lo:.1f}..{hi:.1f})  [{src}]",
               weight=2.0, kind="NUM"),
    ]
