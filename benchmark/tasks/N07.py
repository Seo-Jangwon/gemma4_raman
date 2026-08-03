# -*- coding: utf-8 -*-
"""N07 — 데이터 처리 (3점)

[문제]
  Acquire a kinetic series of 5 frames (1 s interval) at the current position, then plot
  how the total intensity of each frame changes over the series.

[정답 기준]
  GT(설정)=num_frames=5. GT(분석)=run_analysis 안에서 spectra[i]['frames'] (n_frames, n_pixels) 2D
  배열로 프레임별 합을 계산 — kinetic을 사후 분석하는 유일한 경로다. 확인=그림 1장(점 5개), 프레임 합 5값이 사후 재계산과 rtol 1e-6.
  load_spectrum으로 읽으려다 거부당한 뒤 run_analysis로 전환하면 감점 없음, 거부를 고장으로 오판하고 reconnect_hardware를
  부르면 감점.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="N07",
    score=3,
    axis="data processing",
    mode="live",
    criteria="PROC(args EXACT) + ARRAY(rtol 1e-6) / post-hoc GT",
    prompt=(
        "Acquire a kinetic series of 5 frames (1 s interval) at the current position, then "
        "plot how the total intensity of each frame changes over the series. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 N07 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    out = [
        chk.arg(run, "acquire_spectrum", "kinetic_count", 5),
        chk.called(run, "acquire_spectrum", times=1),
        chk.called(run, "reconnect_hardware", times=0),
    ]
    frames = _frames(run)
    if frames is None:
        return out + [chk.fail("frame sums", "no frames in the tool response")]
    sums = [float(np.sum(f)) for f in frames]
    got = run.answer.get("frame_sums")
    return out + [
        chk.ok("5 frames", len(frames) == 5, f"{len(frames)} items"),
        chk.set_match("per-frame sums", [float(v) for v in got] if isinstance(got, list) else None,
                      sums, tol=max(abs(max(sums)) * 1e-6, 1e-9), ordered=True, weight=2.0),
    ]


def _frames(run):
    """kinetic 응답의 프레임 배열들. 없으면 None."""
    for c in run.calls:
        r = c.get("result")
        if isinstance(r, dict) and isinstance(r.get("frames"), list) and r["frames"]:
            return [np.asarray(f.get("intensity") if isinstance(f, dict) else f, float)
                    for f in r["frames"]]
    return None
