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

  [포화 판정이 포화를 안 보고 있었다 — 2026-08-06]
  예전 판정은 n_clip = sum(y >= y.max() - 1e-9) 였다. 검출기 상한이 아니라 **그 배열
  자신의 최대값**과 비교하므로 어떤 정상 스펙트럼도 n_clip == 1 이고, 상한이 2 라
  사실상 자동 통과였다. 정답 기준이 "검출기 상한에 붙은 점" 이라고 적어 놓았고
  spectra.saturated_count(y, ceiling=65535) 가 이미 그 뜻으로 있으며 T077 은 그걸
  쓰는데 이 문항만 달랐다. 규약은 한 곳에만 둔다.

  [조건부 지시를 무조건 요구로 채점했다 — 2026-08-06]
  프롬프트는 "**포화면**(if the maximum is clipped) 절반으로 줄여라" 인데, 예전 판정은
  노출·파워 조정 기록이 2 건 미만이면 무조건 실패였다. setup 이 5.0 s 를 걸어 두지만
  에이전트가 첫 acquire_spectrum(exposure=...) 에 자기 값을 명시하면 그 설정은 덮이고
  포화가 안 날 수 있다. 그러면 "조정하지 않음" 이 정답인데 확정 오답이었다 — 조건
  분기를 옳게 처리한 실행을 벌하는 판정이다. 포화가 실제로 났을 때만 조정을 요구한다.

  [두 조정 경로를 이어붙여 순서가 뒤섞였다 — 2026-08-06]
  exps = [acquire 인자들] + [set_ccd_exposure 인자들] 로 **도구별로 뭉쳐서** 이어붙인
  뒤 seq[0] 과 seq[-1] 을 비교했다. 두 경로를 섞어 쓰면 시간 순서가 아니게 되어
  방향 판정이 엉뚱한 두 값을 본다. run.calls 순서를 유지한 채 병합한다.
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


def _timeline(run, pairs):
    """(도구, 인자) 여러 경로의 값을 **호출 순서 그대로** 하나의 시계열로."""
    out = []
    for c in run.calls:
        a = c.get("args") or {}
        for tool, key in pairs:
            if c.get("name") == tool and a.get(key) is not None:
                try:
                    out.append((tool, float(a[key])))
                except (TypeError, ValueError):
                    pass
    return out


def evaluate(b, run):
    """이 목록이 그대로 T099 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    if not saved:
        return [chk.fail("saturation resolved", "no saved spectrum", weight=2.0)]

    # 검출기 상한 기준으로 센다 — 규약은 spectra.saturated_count 한 곳에만 있다.
    n_clip = sp.saturated_count(saved[-1][2])
    ever_saturated = any(sp.saturated_count(y) > 0 for _, _, y in saved)

    out = [chk.ok("no saturation in the final acquisition", n_clip == 0,
                  f"points pinned at the detector ceiling: {n_clip} items", weight=2.0)]

    exps = _timeline(run, [("acquire_spectrum", "exposure"),
                           ("set_ccd_exposure", "exposure_time")])
    pows = _timeline(run, [("acquire_spectrum", "power"),
                           ("set_laser_power", "percent")])

    if not ever_saturated:
        # 프롬프트는 포화일 때만 줄이라고 했다. 안 났으면 안 줄인 것이 정답이다.
        out.append(chk.ok("adjustment only when clipped", True,
                          "no acquisition was saturated, so no reduction was required"))
    elif len(exps) >= 2 or len(pows) >= 2:
        seq = exps if len(exps) >= 2 else pows
        what = "exposure" if len(exps) >= 2 else "power"
        vals = [v for _, v in seq]
        out.append(chk.ok(f"adjusted downward ({what})", vals[-1] <= vals[0],
                          f"{vals[0]} → {vals[-1]}  (path {[t for t, _ in seq]})", weight=2.0))
    else:
        # 포화가 났는데 조정 기록이 없다. 통과시키면 안 된다.
        out.append(chk.fail("adjusted downward",
                            f"an acquisition was saturated but no exposure/power reduction was "
                            f"recorded (exposure {len(exps)} calls, power {len(pows)} calls)",
                            weight=2.0))
    return out
