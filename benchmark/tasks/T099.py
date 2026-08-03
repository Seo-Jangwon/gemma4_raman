# -*- coding: utf-8 -*-
"""T099 — 진단 복구 (3점)

[문제]
  The spectrum you are about to measure may be clipped at the detector maximum. Acquire
  once, check for saturation, and if the maximum is clipped, halve the exposure (or the
  power) and re-acquire until no pixel is saturated. Report the final settings.

[정답 기준]
  [TS-B] 포화는 노출·파워를 크게 걸면 실제 하드웨어에서 안전하게 재현된다(사람 개입=시작 전 노출을 포화가 나도록 세팅). 원문의 '이미 포화된 스펙트럼'
  전제를 '측정→판정→조정' 루프로 바꿨다. GT(판정)=최대값이 검출기 상한에 붙은 점의 개수>0. GT(종료조건)=최종 측정에 포화 픽셀 0개.
  GT(방향)=최종 노출<=초기 노출(또는 최종 파워<=초기 파워). 확인=최종 측정 파일에 포화 없음, 조정 방향이 감소, 최종 설정 보고.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T099",
    score=3,
    axis="diagnostics",
    mode="live",
    criteria="REL(0 saturated points in the final acquisition, settings reduced) + PROC",
    prompt=(
        "The spectrum you are about to measure may be clipped at the detector maximum. "
        "Acquire once, check for saturation, and if the maximum is clipped, halve the "
        "exposure (or the power) and re-acquire until no pixel is saturated. Report the "
        "final settings. "
    ),
)

def setup(b):
    """측정 전에 만들어야 하는 상태."""
    b.hw("set_ccd_exposure", exposure_time=5.0)   # 노출만 — 파워는 시료를 태운다


def evaluate(b, run):
    """이 목록이 그대로 T099 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    if not saved:
        return [chk.fail("saturation resolved", "no saved spectrum", weight=2.0)]
    y = saved[-1][2]
    n_clip = int(np.sum(y >= y.max() - 1e-9))

    # 조정 방향은 두 경로 모두 본다 — 인자로 준 노출과 set_ccd_exposure 로 건 노출.
    # 예전에는 인자 경로만 봐서, 설정 툴로 조정하면 검사가 통째로 통과했다.
    exps = [float(v) for v in run.args("acquire_spectrum", "exposure")]
    exps += [float(v) for v in run.args("set_ccd_exposure", "exposure_time")]
    pows = [float(v) for v in run.args("acquire_spectrum", "power")]
    pows += [float(v) for v in run.args("set_laser_power", "percent")]

    out = [chk.ok("no saturation in the final acquisition", n_clip <= 2, f"points pinned at the ceiling: {n_clip} items",
                  weight=2.0)]
    if len(exps) >= 2 or len(pows) >= 2:
        seq = exps if len(exps) >= 2 else pows
        what = "exposure" if len(exps) >= 2 else "power"
        out.append(chk.ok(f"adjusted downward ({what})", seq[-1] <= seq[0],
                          f"{seq[0]} → {seq[-1]}", weight=2.0))
    else:
        # 조정을 한 번도 안 했으면 '방향이 옳다'고 볼 근거가 없다. 통과시키면 안 된다.
        out.append(chk.fail("adjusted downward",
                            f"not enough exposure/power adjustments recorded (exposure {len(exps)} calls, "
                            f"power {len(pows)} calls)", weight=2.0))
    return out
