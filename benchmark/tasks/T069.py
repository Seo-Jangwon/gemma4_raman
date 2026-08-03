# -*- coding: utf-8 -*-
"""T069 — 계측 제어 (3점)

[문제]
  Measure once at each pre-amplifier gain index 0, 1 and 2, and select the index with the
  highest SNR (T050 definition) among the results that are not saturated. If tied, select
  the lower index.

[정답 기준]
  원문(이득 100/200/300)은 이 장비에 없는 기능이라 실행 자체가 불가능했다 — iDus여서 EM gain·MCP gain 툴은 제거됐고
  set_ccd_preamp_gain(인덱스 0~2)만 남는다. 실행 가능한 인덱스 비교로 전면 교체했다. GT(설정)=preamp_gain_index가
  0,1,2로 각각 걸린 3회 측정. GT(선택)=사후 재계산한 최고 SNR 인덱스(동점이면 낮은 인덱스).
  확인=get_ccd_info.preamp_gains_available로 사용 가능 인덱스를 먼저 확인하면 가점, 포화 판정 근거 제시.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T069",
    score=3,
    axis="계측 제어",
    mode="live",
    windows=[('SNR 신호창', 990.0, 1012.0, 3), ('SNR 잡음창', 1800.0, 1900.0, 2)],
    prompt=(
        "Measure once at each pre-amplifier gain index 0, 1 and 2, and select the index with "
        "the highest SNR (T050 definition) among the results that are not saturated. If "
        "tied, select the lower index. "
    ),
)


def evaluate(b, run):
    """이 목록이 그대로 T069 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    saved = run.spectra()
    out = [
        chk.arg_set(run, "set_ccd_preamp_gain", "index", [0, 1, 2]),
        chk.called(run, "acquire_spectrum", times=3),
    ]
    if len(saved) < 3:
        return out + [chk.fail("이득 선택", f"저장 {len(saved)}건 (3건 필요)")]
    # 저장 순서 i 를 그대로 gain 인덱스로 쓰면 안 된다 — 에이전트가 어떤 순서로 돌렸는지
    # 모른다. 실제로 건 gain 인자를 순서대로 읽어 짝짓는다.
    gains = run.args("set_ccd_preamp_gain", "index")[:3]
    if len(gains) < 3:
        return out + [chk.fail("이득 선택", f"gain 설정 {len(gains)}회 (3회 필요)")]
    cand = []
    for gi, (_, x, y) in zip(gains, saved[:3]):
        if sp.saturated_count(y) > 2:
            continue                       # 포화된 것은 후보에서 뺀다
        s = sp.snr(x, y)
        if s is not None:
            cand.append((int(gi), s))
    if not cand:
        return out + [chk.fail("이득 선택", "포화되지 않은 측정이 없습니다")]
    want = min(cand, key=lambda t: (-t[1], t[0]))[0]     # 동점이면 낮은 인덱스
    got = run.answer.get("index")
    return out + [
        chk.ok("선택한 gain 인덱스", got is not None and int(got) == want,
               f"선택={got} 정답={want} (SNR={[(g, round(s, 1)) for g, s in cand]})",
               weight=2.0),
    ]
