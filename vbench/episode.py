# -*- coding: utf-8 -*-
"""에피소드 하나를 실행한다 — 월드 생성 → 정답 확정 → 에이전트 구동 → 채점.

[에이전트를 왜 인프로세스로 부르나]
기존 벤치마크는 HTTP 서버를 경유한다. 장비 DLL 을 한 프로세스만 쥘 수 있어서다.
가상 환경에는 DLL 이 없으므로 그 제약이 사라진다 — 그냥 import 해서 부르면 된다.
서버·SSE·폴링이 통째로 없어지고, 하드웨어 없는 PC 에서 그대로 돈다.

[기준선(baseline)이 왜 여기 있나]
LLM 없이 도는 참조 전략들이다. 두 가지 일을 한다:
  1. LLM 을 띄우지 않고 파이프라인 전체를 검증한다.
  2. 점수를 해석할 눈금을 준다 — `careful` 이 잘 못 받으면 **벤치마크가 잘못된 것**이지
     에이전트가 못하는 것이 아니다.
기준선도 에이전트와 똑같이 TOOL_DISPATCH 를 통해서만 장비를 만진다.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vbench.world import VirtualWorld, TARGET_CM1, VIEW_W, VIEW_H
from vbench import drivers as D
from vbench import oracle as O
from vbench import score as S


# ══════════════════════════════════════════════════════════════════════════════
# 과제 프롬프트
# ══════════════════════════════════════════════════════════════════════════════

PROMPT = """You are operating a Raman microscope on an unfamiliar sample.

GOAL — find all three of these and report them:
  1. WHERE on the sample the target material gives the strongest signal,
  2. what LASER POWER to use there,
  3. what CCD EXPOSURE TIME to use there.

The target material is polystyrene; its marker band is at {target:.1f} cm-1.
"Strongest signal" means the highest signal-to-noise ratio of that band - not simply the
largest raw counts.

WHAT YOU ARE WORKING WITH
  - The sample is a mixture of three components: bare substrate, buffer, and the polystyrene
    target. Their proportions vary across the sample, and the camera image shows this as
    colour and texture. Part of your job is to work out which visual appearance corresponds
    to a high target signal.
  - Searchable area (stage coordinates, mm):  X {x0:.3f} to {x1:.3f},  Y {y0:.3f} to {y1:.3f}
    The camera sees {vw} x {vh} pixels covering about {fov_w:.3f} x {fov_h:.3f} mm.
    You are starting at the centre of this area.

TWO THINGS CAN RUIN A MEASUREMENT, AND BOTH ARE IRREVERSIBLE FOR THAT SPOT
  - PHOTODAMAGE. Too much laser dose burns the sample: the target band collapses and a broad
    fluorescence/carbon background rises in its place. Before it collapses there is usually a
    milder warning - the background creeps up while the band is still intact. How much dose a
    sample tolerates DIFFERS FROM SAMPLE TO SAMPLE, so a recipe that was safe on another
    sample may destroy this one. Burnt spots never recover.
  - SATURATION. The detector clips at 65535 counts. A saturated spectrum has a flat top and
    its peak height cannot be measured.

BUDGET for this episode
  - at most {n_meas} spectra
  - at most {dose:.0f} reference-seconds of total laser dose, where
        dose = (power / 20) ** 3.5  *  exposure
    Note the exponent: doubling the power costs about 11x more dose than doubling the
    exposure does. What this instrument does NOT tell you is how much dose THIS sample
    tolerates - that is what you have to find out.
  - at most {t_budget:.0f} s of instrument time
Tools will tell you when a budget is spent. Spend it on finding the answer, not on repeating
measurements you have already made.

REPORT
  stage_x, stage_y   the stage coordinates you recommend measuring at (mm)
  laser_power        the laser power you recommend (%)
  ccd_exposure       the CCD exposure time you recommend (s)
  visual_signature   in words, how a person would recognise a good spot by looking at the
                     camera image

The power and exposure you report must be SAFE at the position you report: applying them to a
fresh spot must neither damage the sample nor saturate the detector. A recipe that burns or
saturates scores zero no matter how good the position is.

End your reply with a JSON block:

```json
{{"stage_x": 0.0, "stage_y": 0.0, "laser_power": 0.0, "ccd_exposure": 0.0,
 "visual_signature": "..."}}
```
"""


def build_prompt(world: VirtualWorld) -> str:
    from backend.hw_tools import optics_map as om
    x0, x1, y0, y1 = world.bounds_mm()
    fw, fh = om.fov_mm(VIEW_W, VIEW_H)
    return PROMPT.format(
        target=TARGET_CM1, x0=x0, x1=x1, y0=y0, y1=y1,
        vw=VIEW_W, vh=VIEW_H, fov_w=fw, fov_h=fh,
        n_meas=D.VirtualRig.max_measurements, dose=D.VirtualRig.max_dose,
        t_budget=D.VirtualRig.max_virtual_time_s)


# ══════════════════════════════════════════════════════════════════════════════
# 에이전트 구동
# ══════════════════════════════════════════════════════════════════════════════

def _run_llm_agent(agent: str, prompt: str, session_id: str) -> str:
    """AILA / CoALA 를 한 턴 돌리고 최종 보고 문자열을 돌려준다."""
    if agent.upper() == "AILA":
        from backend.agents import single_agent_AILA_bench as mod
    elif agent.upper() == "COALA":
        from backend.agents import single_agent_CoALA_bench as mod
    else:
        raise ValueError(f"unknown agent: {agent}")
    return (mod.run_experiment(prompt, session_id=session_id) or {}).get("final_report", "")


# ══════════════════════════════════════════════════════════════════════════════
# 기준선 — LLM 없이 도는 참조 전략
# ══════════════════════════════════════════════════════════════════════════════

def _tools():
    from backend.hw_tools.raman_tools import TOOL_DISPATCH
    return TOOL_DISPATCH


def _answer_block(x, y, p, t, note) -> str:
    return ("done\n```json\n" + json.dumps(
        {"stage_x": round(float(x), 5), "stage_y": round(float(y), 5),
         "laser_power": round(float(p), 3), "ccd_exposure": round(float(t), 3),
         "visual_signature": note}) + "\n```")


def _amberness(bgr) -> float:
    """타겟 색(호박색)에 얼마나 가까운가. 이미지만 보고 판단하는 대용치."""
    b, g, r = bgr[..., 0].astype(float), bgr[..., 1].astype(float), bgr[..., 2].astype(float)
    return float(np.mean(r - b))


def _aim(T):
    """지금 화면에서 **가장 호박색인 군집**으로 스테이지를 옮긴다.

    [왜 화면 중심으로는 안 되나 — 실측]
    처음에는 격자로 훑어 가장 호박색인 '화면'을 고르고 그 중심을 답했다. 그런데 군집이
    31 µm 밖에 안 돼서 화면 중심은 대개 군집 사이의 어두운 자리에 떨어진다. 그래서
    화면을 훑은 기준선이 아무 데도 안 움직인 기준선과 위치 점수가 같게 나왔다.
    실제 에이전트라면 화면을 보고 군집을 찍어 move_to_pixel 로 간다 — 그것과 같게 맞춘다.
    """
    import cv2
    r = _rig()
    frame = r.world.render(r.x, r.y, VIEW_W, VIEW_H)
    amber = cv2.GaussianBlur(
        (frame[..., 2].astype(np.float32) - frame[..., 0].astype(np.float32)), (31, 31), 0)
    py, px = np.unravel_index(int(np.argmax(amber)), amber.shape)
    T["move_to_pixel"]({"pixel_x": int(px), "pixel_y": int(py)})
    return float(np.max(amber))


def _survey(T, world, n=9):
    """맵을 격자로 훑어 호박색이 진한 화면들을 찾는다. (좋은 순서로 정렬해 돌려준다)"""
    x0, x1, y0, y1 = world.bounds_mm()
    k = int(np.sqrt(n))
    xs = np.linspace(x0 + 0.15 * (x1 - x0), x1 - 0.15 * (x1 - x0), k)
    ys = np.linspace(y0 + 0.15 * (y1 - y0), y1 - 0.15 * (y1 - y0), k)
    seen = []
    for yy in ys:
        for xx in xs:
            T["move_stage"]({"x": float(xx), "y": float(yy)})
            seen.append((_amberness(_rig().world.render(_rig().x, _rig().y, VIEW_W, VIEW_H)),
                         float(xx), float(yy)))
    seen.sort(key=lambda s: -s[0])
    return seen


_RIG = {"cur": None}


def _rig():
    return _RIG["cur"]


def baseline_oracle(T, world, orc) -> str:
    """상한 — 정답을 그대로 답한다. 점수 1.00 이 나와야 채점기가 정상이다."""
    bx, by = orc.best_spots_mm[0]
    T["move_stage"]({"x": bx, "y": by})
    T["acquire_spectrum"]({"exposure": orc.exposure_s, "power": orc.power_pct})
    return _answer_block(bx, by, orc.power_pct, orc.exposure_s,
                         "amber / golden clusters against the dark substrate")


def baseline_naive(T, world, orc) -> str:
    """하한 — 출발 자리에서 무난한 값으로 한 번 재고 그대로 답한다."""
    r = _rig()
    T["acquire_spectrum"]({"exposure": 1.0, "power": 20.0})
    return _answer_block(r.x, r.y, 20.0, 1.0, "did not look")


def baseline_greedy(T, world, orc) -> str:
    """흔한 실패형 — 좋아 보이는 자리를 찾은 뒤 파워를 계속 올려 최고 신호를 쫓는다.

    시료를 태우고, 태우는 레시피를 권고하게 된다. 이 전략이 낮은 점수를 받아야
    '신호 최대화'와 '시료 보존'의 긴장이 실제로 작동하는 것이다.
    """
    seen = _survey(T, world)
    T["move_stage"]({"x": seen[0][1], "y": seen[0][2]})
    _aim(T)
    r0 = _rig()
    bx, by = r0.x, r0.y
    best = (0.0, 20.0, 2.0)
    for p in (20.0, 35.0, 50.0, 70.0):
        r = T["acquire_spectrum"]({"exposure": 2.0, "power": p})
        if not r.get("ok"):
            break
        snr = _rig().log[-1]["snr"]
        if snr > best[0]:
            best = (snr, p, 2.0)
    return _answer_block(bx, by, best[1], best[2], "brightest orange area")


def baseline_careful(T, world, orc) -> str:
    """의도한 정답 절차. **이 전략이 높은 점수를 못 받으면 벤치마크가 잘못된 것이다.**

    1) 카메라로 후보 자리를 훑어 가장 호박색인 곳을 고른다
    2) 그 옆의 '희생 자리'에서 한계를 잰다 — 먼저 노출을 늘려 포화 한계를,
       다음으로 파워를 올려 손상 경고를 찾는다
    3) 알아낸 한계에서 물러선 조건을 **새 자리**에 권고한다
    희생 자리를 따로 두는 것이 핵심이다. 한계를 재는 행위 자체가 그 자리를 망가뜨리므로,
    좋은 자리에서 한계를 재면 정작 쓸 자리가 남지 않는다.
    """
    seen = _survey(T, world)

    # 희생 자리 — **두 번째로 좋은 화면**에서 잰다. 한계를 재는 행위 자체가 그 자리를
    # 망가뜨리므로, 정작 쓸 자리에서 재면 남는 게 없다.
    T["move_stage"]({"x": seen[1][1], "y": seen[1][2]})
    _aim(T)

    FW, HEAD = 65535.0, 0.86            # 포화 상한과, 거기서 남길 여유

    # ① 두 파워에서 짧게 재서 '카운트가 파워에 어떻게 늘어나는지'를 잡는다.
    #    카운트 ≈ t·(a·P + b) 이므로 두 점이면 a, b 가 나오고, 어떤 파워에서든
    #    포화 직전 노출을 계산할 수 있다. 노출만 늘려 한계를 재면 그 값이 그 파워에서만
    #    유효해서, 파워를 올린 순간 곧바로 포화된다(실측으로 그렇게 실패했다).
    t0 = 1.0
    m = []
    for p in (6.0, 18.0):
        r = T["acquire_spectrum"]({"exposure": t0, "power": p})
        if not r.get("ok"):
            return _answer_block(_rig().x, _rig().y, 10.0, 2.0, "probe failed")
        m.append(float(r.get("max_intensity", 0.0)))
    a = (m[1] - m[0]) / (t0 * 12.0)
    b = max(m[0] / t0 - a * 6.0, 1.0)

    def t_for(p):
        return max(0.05, HEAD * FW / max(a * p + b, 1.0))

    # ② 파워를 올리며 **측정된 SNR 이 가장 좋은 조건**을 고른다.
    #    각 파워마다 그 파워에서의 포화 직전 노출을 쓰고, 매번 옆으로 조금 옮겨
    #    아직 안 쏜 자리에서 잰다(같은 자리에 계속 쏘면 뒤로 갈수록 이미 상한 시료를 잰다).
    #
    #    [왜 '경고 뜨면 정지' 가 아닌가 — 실측으로 걸린 문제]
    #    경고는 u=0.5 부터 뜨는데 정답은 u≈0.65 에 있다. 즉 **정답 자체가 경고 구간 안**이라
    #    (설계문서 §5.3 의 의도) 첫 경고에서 멈추면 반드시 못 미친다. 실제로 파워 8.7 에서
    #    멈춰 정답 14.06 을 한참 밑돌았다. 경고는 '한계가 가깝다'는 정보이지 정지 신호가
    #    아니다 — 정지 조건은 **실제 손상**이고, 선택 기준은 **측정된 SNR** 이다.
    p_ok, best_snr = 6.0, -1.0
    for p in (6.0, 8.0, 10.0, 12.0, 14.0, 16.5, 19.0, 22.0):
        r = T["acquire_spectrum"]({"exposure": t_for(p), "power": p})
        if not r.get("ok"):
            break
        rec = _rig().log[-1]
        if rec["survival"] < 0.98:
            break                       # 실제로 태웠다 — 더 올리면 안 된다
        if not rec["saturated"] and rec["snr"] > best_snr:
            best_snr, p_ok = rec["snr"], p
        T["move_stage_relative"]({"dx": 0.008, "dy": 0.0})     # 다음 점은 새 시료로

    # ③ 최종 자리로 옮겨 **거기서 다시 확인한다.**
    #    자리마다 밝기도 손상 임계도 다르다(타겟이 진할수록 더 잘 탄다). 희생 자리에서
    #    잰 값을 그대로 쓰면 더 밝은 자리에서 포화한다 — 실측으로 그렇게 0점을 받았다.
    T["move_stage"]({"x": seen[0][1], "y": seen[0][2]})
    _aim(T)
    r0 = _rig()
    r = T["acquire_spectrum"]({"exposure": 1.0, "power": p_ok})
    t_ans = t_for(p_ok)
    if r.get("ok"):
        rate = max(float(r.get("max_intensity", 0.0)), 1.0)     # counts/s at this spot
        t_ans = HEAD * FW / rate
    # 자리마다 손상 임계가 다르므로 **여유를 남긴다.** 임계는 타겟 비율에 따라
    # 최대 1.3 배까지 차이나고(E = E0/(1+3.2·f_tgt)), 조사량은 파워의 3.5 제곱이라
    # 파워를 12% 낮추면 조사량이 34% 줄어 그 편차를 덮는다.
    p_ans = p_ok / 1.12
    t_ans *= 0.92
    return _answer_block(r0.x, r0.y, p_ans, t_ans,
                         "amber / golden clusters, clearly warmer than the pale buffer "
                         "and the dark substrate")


BASELINES = {"oracle": baseline_oracle, "naive": baseline_naive,
             "greedy": baseline_greedy, "careful": baseline_careful}


# ══════════════════════════════════════════════════════════════════════════════
# 에피소드
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Episode:
    agent: str
    seed: int
    level: str
    result: dict


def run_episode(agent: str, seed: int, level: str = "medium",
                out_dir: Path | None = None, session_id: str = "") -> dict:
    """에피소드 하나. 정답은 에이전트가 시작하기 **전에** 확정된다."""
    t0 = time.time()
    world = VirtualWorld(seed=seed, level=level)
    orc = O.build(world)                       # ← 여기서 정답이 정해진다

    rig = D.attach(world)
    _RIG["cur"] = rig
    prompt = build_prompt(world)
    report, err = "", None
    try:
        if agent in BASELINES:
            report = BASELINES[agent](_tools(), world, orc)
        else:
            report = _run_llm_agent(agent, prompt,
                                    session_id or f"vbench_{agent}_{level}_{seed}")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        _RIG["cur"] = None
        D.detach()

    ans = S.parse_answer(report)
    graded = S.grade(ans, orc, world, rig.log)
    out = {
        "agent": agent, "seed": seed, "level": level,
        "oracle": orc.to_json(),
        "report": report,
        "error": err,
        "wall_time_s": round(time.time() - t0, 1),
        "virtual_time_s": round(rig.virtual_time_s, 1),
        "stage_moves": rig.moves,
        **graded,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{agent}_{level}_seed{seed}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
