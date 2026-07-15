"""
HWManagerNode — 하드웨어를 만지는 모든 동작을 담당하는 단일 실행 에이전트.

[에이전트 통합 노트 — roi_detector → locate_target task 흡수]
과거에는 "측정 위치 결정"이 별도 노드(roi_detector)였다. 하지만 위치 탐색과
스펙트럼 측정은 둘 다 "스테이지/카메라/레이저를 움직이는 손"이고, 위치 추적·
dose 부기·시뮬레이션 강등 같은 인프라를 통째로 공유한다. 노드를 나눠서 얻는
것은 없고 ablation 단위와 그래프 왕복만 늘었다. 이제 task 하나로 흡수한다 —
ablation 시 "시각 탐색 유무"는 locate_target의 mode 플래그로 통제하면 된다.

[구조 개요]
step.params["task"]에 따라 4가지 실행 경로로 분기한다:

  0. "locate_target"      — 측정 위치(ROI) 결정 (구 roi_detector)
     * 좌표가 있으면 manual: 그대로 채택.
     * 없으면 visual_search: 현미경 프레임 → vision LLM으로 타겟 픽셀 식별 →
       move_to_pixel → 오토포커스. 시야(~0.3mm)씩 나선형으로 최대 5개 영역 탐색.
       같은 프레임에서 "빈 기판" 픽셀도 함께 받아 배경 측정 위치로 저장한다 —
       타겟을 눈으로 확인한 그 화면에서 고른 기판이 맹목적 오프셋(+0.5mm)보다
       "진짜 기판"일 확률이 높다 (기판 배경 분리 문제의 입구).
     * 결과는 state.next_roi에 기록 → acquire_target이 그 위치로 이동해 측정.

  1. "acquire_target"     — 적응형 스펙트럼 획득 (결정적 코드)
     타겟 물질별 최적 레이저 파워/노출을 모른다는 문제에 대응:
     저출력 프로브 측정 → 최대 ADU 평가 → 파워/노출 스케일링 → 재측정.
     목표 신호 창(5,000~50,000 ADU)에 들어올 때까지 최대 4회 반복.
     확정된 파라미터는 state.acquisition_params에 기록되고
     experience store에도 축적되어 다음 실험의 시작점이 된다.

  2. "acquire_background" — 기판 배경 참조 측정 (결정적 코드)
     기판 background와 타겟 신호 구분이 어렵다는 문제에 대응:
     타겟 옆 기판 위치로 이동 → "타겟과 동일 조건"으로 측정 → 원위치 복귀.
     결과는 state.background_reference에 저장되어 analyst가
     타겟 고유 피크를 분리하는 기준이 된다.

  3. task 없음            — 기존 LLM tool-calling 루프 (유연한 자유 작업)

[왜 1·2를 LLM이 아닌 결정적 코드로 구현했나]
- 레이저 조사가 걸린 반복 루프를 LLM 판단에 맡기면 매 실행마다 다른 순서/횟수로
  조사가 일어나 재현성이 없고, 시편 손상 위험을 통제할 수 없다.
- 포화/신호부족 판정은 숫자 비교라 LLM이 필요 없다. LLM은 "무엇을 할지"(Planner)와
  "결과가 무엇을 의미하는지"(Specialist)에만 쓰고, "어떻게 안전하게 측정하는지"는
  코드로 고정한다.

[Failure 처리 원칙]
- 모든 레이저 조사 직전에 Critic C2(위치별 dose) pre-check — 프로브 측정도 예외 없음.
- 어떤 경로로 끝나든 finally에서 laser_off (acquire_spectrum 내부 OFF와 이중 안전).
- 실패 시 step을 "failed"로 표시하고 failure_log에 원인을 기록하되,
  current_step_idx를 전진시키지 않는다 → 재시도/재계획 결정은 Planner의 몫.
- 성공 시에도 idx를 전진시키지 않는다 → Planner가 C3(스펙트럼 품질) 게이트를
  거친 뒤 전진시킨다. (이 노드가 전진까지 해버리면 품질 게이트를 끼워 넣을 수 없다)

LLM: Claude claude-opus-4-8 (교체: 파일 상단 _llm 변수).
"""

from __future__ import annotations

import json
import math
import random
import re
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from backend.agents import experience
from backend.agents.critic import check_c2_hardware_safety
from backend.agents.state import ExperimentState

# ── LLM 설정 (교체 포인트 — Ollama 사용 시 ChatOllama로 교체) ─────────────────
# from langchain_community.chat_models import ChatOllama
# _llm = ChatOllama(model="gemma4:31b", base_url="http://192.168.1.16:11434")
_llm = ChatAnthropic(model="claude-opus-4-8")

# ── 스테이지/카메라/렌즈 상수 (배경 위치 클램프 + 픽셀→mm 변환용) ─────────────
# Config.ini가 없는 개발 PC에서도 import가 죽지 않도록 방어한다.
try:
    from backend.config import (
        CALIB_FACTOR_X, CALIB_FACTOR_Y,
        CAMERA_HEIGHT, CAMERA_WIDTH,
        LENS_HEIGHT_UM, LENS_WIDTH_UM,
        STAGE_MAX_X, STAGE_MAX_Y,
    )
except Exception:
    STAGE_MAX_X, STAGE_MAX_Y = 75.3, 50.2  # 장비 스펙 기본값
    CAMERA_WIDTH, CAMERA_HEIGHT = 1060, 800
    LENS_WIDTH_UM, LENS_HEIGHT_UM = 305.0, 230.0
    CALIB_FACTOR_X, CALIB_FACTOR_Y = 1.4, 1.285

# ── 적응형 튜닝 상수 ──────────────────────────────────────────────────────────
# CCD 16-bit(65535). 신호 목표 창을 [5000, 50000]으로 잡은 이유:
#  - 하한 5000: 피크 귀속에 충분한 SNR을 확보하는 경험적 최소선
#  - 상한 50000: 포화(≈60000)까지 여유를 둬 강한 피크가 clip되지 않게 함
_ADU_LOW, _ADU_HIGH = 5000.0, 50000.0
_ADU_MID = 30000.0            # 튜닝 목표점 (창의 중앙 부근)
_SATURATION_ADU = 60000.0

# 프로브(첫 탐색) 측정 기본값 — "일단 낮게 쏘고 본다".
# 물질을 모르는 상태에서 고출력으로 시작하면 광민감 시료(생체 등)가 손상되므로
# 항상 저출력에서 출발해 위로 올라가는 방향만 허용한다.
_PROBE_POWER_PCT = 5.0
_PROBE_EXPOSURE_S = 0.2

_MAX_TUNE_ITERS = 4           # 프로브 포함 최대 조사 횟수 (dose 통제)
_MIN_POWER_PCT, _MAX_POWER_PCT = 0.004, 100.0   # ND 필터 물리 한계
_MIN_EXPOSURE_S, _MAX_EXPOSURE_S = 0.05, 10.0   # 실용적 노출 범위
_MAX_SCALE_PER_ITER = 8.0     # 1회 반복당 최대 증폭 배율 — 급격한 조사량 점프 방지

_BIO_KEYWORDS = {"exosome", "cell", "lipid", "tissue", "bacteria", "protein", "membrane"}
_MAX_POWER_BIO = 40.0         # critic.py와 동일 값 — bio 시료 튜닝 상한

# ── locate_target(시각 탐색) 상수 (구 roi_detector) ───────────────────────────
# vision LLM — 현미경 이미지에서 타겟 픽셀 좌표를 읽는 용도
_llm_vision = ChatAnthropic(model="claude-opus-4-8")

# 픽셀→mm 변환 — hw_tools.raman_tools.move_to_pixel과 반드시 동일 수식.
# (다르면 vision이 고른 배경 위치가 엉뚱한 스테이지 좌표가 된다)
_UM_PER_PX_X = LENS_WIDTH_UM / CAMERA_WIDTH
_UM_PER_PX_Y = LENS_HEIGHT_UM / CAMERA_HEIGHT
_SIGN_X = -1   # pixel +X(right) → stage -X  (raman_tools.move_to_pixel과 동일)
_SIGN_Y = +1   # pixel +Y(down)  → stage +Y

# 나선형 탐색 오프셋 (mm) — 시야(~0.3mm)와 겹치지 않게 0.25mm 간격.
# (0,0)=현재 화면부터 시작해 좌우/대각으로 확장. 5개로 제한하는 이유:
# 탐색은 가이드빔/조명만 쓰므로 시편 손상은 없지만 무한 탐색은 실험 시간을
# 폭주시킨다. 5회 안에 못 찾으면 "탐색 실패"를 명확히 선언하고 Planner의
# 재계획에 넘기는 것이 어정쩡한 위치에서 레이저를 쏘는 것보다 안전하다.
_SEARCH_OFFSETS_MM = [(0.0, 0.0), (0.25, 0.0), (-0.5, 0.0), (0.25, 0.25), (0.0, -0.5)]

_MIN_CONFIDENCE = 0.5   # vision LLM 확신도가 이보다 낮으면 "못 찾음" 취급 —
                        # 낮은 확신으로 이동하면 엉뚱한 곳에 레이저를 쏘게 된다.

_VISION_SYSTEM = """\
당신은 광학 현미경 이미지에서 측정 타겟을 찾는 시각 분석 전문가입니다.
이미지를 보고 요청된 타겟이 있는지 판단하고, 있으면 중심 픽셀 좌표를 알려주세요.
또한 타겟이 아닌 '빈 기판(background)' 영역의 픽셀 좌표도 하나 골라주세요.
빈 기판 좌표는 타겟에서 충분히 떨어져 있고, 다른 타겟/이물질이 없는 균일한 영역이어야 합니다.

JSON으로만 답변하세요:
{"found": true/false, "pixel_x": 정수, "pixel_y": 정수, "confidence": 0.0~1.0,
 "background_pixel_x": 정수, "background_pixel_y": 정수, "reason": "판단 근거 한 문장"}
타겟이 없으면 found=false로 하고 나머지 좌표는 null로 두세요."""

_SYSTEM = """\
당신은 라만 분광기 하드웨어를 제어하는 전문 에이전트입니다.
사용 가능한 도구를 이용해 요청된 하드웨어 작업을 안전하게 수행하세요.
- 레이저를 켜기 전에 반드시 출력을 설정하세요
- 스펙트럼 획득은 acquire_spectrum 단일 도구를 사용하세요 (laser_on/off 체이닝 금지 —
  추론 시간 동안 레이저가 시편에 계속 조사되어 손상됩니다)
- 모든 작업 후 결과를 명확히 보고하세요
- 오류 발생 시 즉시 레이저를 끄세요"""

# 허용할 하드웨어 도구 이름 집합 (LLM 루프에 바인딩할 도구)
_HW_TOOL_NAMES = {
    "acquire_spectrum", "move_stage", "move_stage_relative", "get_stage_position",
    "set_stage_speed", "set_laser_power", "laser_on", "laser_off",
    "set_ccd_exposure", "get_ccd_info", "capture_camera_frame",
    "analyze_focus_quality", "start_camera_stream", "stop_camera_stream",
    "run_autofocus", "apply_background_subtraction",
}

# ── 하드웨어 "질의" 경로에서만 허용하는 도구 (run_hardware_query 참고) ──────────
# 실험 파이프라인 밖(채팅 한 줄)에서 도는 경로라 Critic C1(계획 안전 검증)을
# 거치지 않는다. 따라서 레이저 조사·스테이지 이동처럼 시편을 손상시키거나
# 물리 상태를 바꾸는 도구는 절대 넣지 않는다 — 관측 전용(read-only)만 허용.
# "화면 좀 보여줘" 수준의 요청에 필요한 건 이것으로 충분하고, 그 이상(측정·이동)은
# 실험 요청으로 분류돼 Planner+Critic 게이트를 타야 한다.
_QUERY_TOOL_NAMES = {
    "get_stage_position", "get_ccd_info", "capture_camera_frame",
    "analyze_focus_quality", "start_camera_stream", "stop_camera_stream",
}

# 허용 이름 집합(frozenset)별 도구 빌드 결과 캐시.
# _HW_TOOL_NAMES(실험용)와 _QUERY_TOOL_NAMES(질의용)가 서로 다른 목록을 쓰므로
# 단일 캐시로는 둘이 섞인다 → 집합을 키로 분리해 캐싱한다.
_lc_tools_cache: dict[frozenset, tuple[list, dict]] = {}


def _get_dispatch() -> dict | None:
    """raman_tools.TOOL_DISPATCH 로드. 하드웨어 모듈이 없으면 None(시뮬레이션 모드).
    ImportError만이 아니라 Exception 전체를 잡는 이유: raman_tools가 import하는
    config.py는 장비 PC의 Config.ini를 읽는데, 파일이 없는 개발 PC에서는
    NoSectionError(ImportError 아님)가 발생한다 — 이 경우도 시뮬레이션으로 강등."""
    try:
        from backend.hw_tools.raman_tools import TOOL_DISPATCH
        return TOOL_DISPATCH
    except Exception:
        return None


def _build_lc_tools(allowed: set | None = None):
    """hw_tools.TOOL_DISPATCH의 함수를 LangChain @tool로 래핑해 반환 (LLM 루프용).

    allowed: 바인딩할 도구 이름 집합. None이면 실험용 전체(_HW_TOOL_NAMES).
             질의 경로는 관측 전용 부분집합(_QUERY_TOOL_NAMES)을 넘긴다.
    """
    names = frozenset(allowed if allowed is not None else _HW_TOOL_NAMES)
    cached = _lc_tools_cache.get(names)
    if cached is not None:
        return cached

    try:
        from backend.hw_tools.raman_tools import TOOL_DISPATCH
        from backend.hw_tools.raman_tool_schemas import RAMAN_TOOLS
    except Exception:   # _get_dispatch와 동일한 이유 — Config.ini 부재 등도 포함
        return [], {}

    lc_tools = []
    tool_map: dict[str, callable] = {}

    for schema in RAMAN_TOOLS:
        fn_info = schema.get("function", {})
        name = fn_info.get("name", "")
        if name not in names:
            continue
        raw_fn = TOOL_DISPATCH.get(name)
        if raw_fn is None:
            continue

        desc = fn_info.get("description", name)

        def _make_tool(fn, n, d):
            @tool(n, description=d)
            def _t(**kwargs):
                return fn(kwargs)
            return _t

        lc_tools.append(_make_tool(raw_fn, name, desc))
        tool_map[name] = raw_fn

    _lc_tools_cache[names] = (lc_tools, tool_map)
    return lc_tools, tool_map


# ══════════════════════════════════════════════════════════════════════════════
# 실행 컨텍스트 (ctx)
# ──────────────────────────────────────────────────────────────────────────────
# LangGraph 노드는 부분 상태 dict를 "반환"해야 상태가 갱신된다.
# 실행 도중의 dose/위치/관측 기록을 매번 반환할 수 없으므로, 노드 실행 동안의
# 변경분을 ctx dict에 누적했다가 마지막에 한 번에 상태 반환값으로 조립한다.
# ══════════════════════════════════════════════════════════════════════════════

def _make_ctx(state: ExperimentState, step: dict) -> dict:
    return {
        "base_state": state,                      # C2 체크용 읽기 전용 참조
        "dispatch": _get_dispatch(),              # None이면 시뮬레이션 모드
        "step_id": step.get("step_id", "?"),
        "observations": [],
        "dose": state.get("cumulative_dose_mj", 0.0),
        "dose_map": dict(state.get("cumulative_dose_map", {})),
        "position": dict(state.get("stage_position") or {}),
        "c2_abort": None,                         # C2 HARD VETO 발생 시 entry 저장
        "acquisition_params": None,               # 튜닝 확정 시 채워짐
        "background_reference": None,             # 배경 측정 성공 시 채워짐
        "next_roi": None,                         # locate_target 성공 시 채워짐
        "_sim_last_data": None,                   # 시뮬레이션 IPBSA용 캐시
    }


def _pos_key(pos: dict) -> str:
    """critic.check_c2와 동일한 위치 키 포맷 — 불일치 시 dose 추적이 깨진다."""
    return f"{pos.get('x', 0):.1f}_{pos.get('y', 0):.1f}"


# ── 시뮬레이션 (하드웨어 미연결 개발/테스트 환경) ─────────────────────────────
# 파이프라인 전체(탐색→측정→분석→보고)를 하드웨어 없이 검증할 수 있어야
# 에이전트 로직 자체의 버그를 실험실 밖에서 잡을 수 있다.

def _synthetic_spectrum(power: float, exposure: float, background_only: bool = False) -> dict:
    """파워×노출에 선형 비례하는 합성 스펙트럼. 포화(clip)도 재현해
    적응형 튜닝 루프가 시뮬레이션에서도 실제처럼 동작하게 한다."""
    n = 1024
    scale = power * exposure * 3000.0          # 프로브(5%, 0.2s) → 피크 ~3000 ADU
    baseline = 800.0
    data = []
    for i in range(n):
        v = baseline + random.uniform(-30, 30)
        # 넓은 형광 hump (기판 배경 성분)
        v += 0.25 * scale * math.exp(-((i - 300) ** 2) / (2 * 250.0 ** 2))
        if not background_only:
            # 타겟 고유 sharp 피크 3개
            for c, h in ((280, 1.0), (520, 0.7), (760, 0.5)):
                v += h * scale * math.exp(-((i - c) ** 2) / (2 * 6.0 ** 2))
        data.append(min(v, 65535.0))           # 16-bit 포화 재현
    shift = [200.0 + i * (2800.0 / n) for i in range(n)]
    return {
        "ok": True, "simulated": True, "mode": "single",
        "length": n, "data": data,
        "max_intensity": float(max(data)), "sum_intensity": float(sum(data)),
        "raman_shift_cm-1": shift, "calibrated": True,
        "exposure_time": exposure, "laser_power_pct": power,
    }


def _simulate_tool(name: str, args: dict, ctx: dict) -> dict:
    """시뮬레이션 모드 도구 응답. 실제 도구의 반환 스키마를 그대로 흉내 내
    다운스트림(specialist/critic) 코드가 분기 없이 동작하게 한다."""
    if name == "acquire_spectrum":
        # 배경 측정 여부는 호출부가 args에 심어준 힌트로 구분 (실장비엔 없는 인자)
        res = _synthetic_spectrum(
            float(args.get("power", 40.0)), float(args.get("exposure", 0.2)),
            background_only=bool(args.get("_sim_background", False)),
        )
        ctx["_sim_last_data"] = res["data"]
        return res
    if name in ("move_stage", "move_to_pixel"):
        pos = {"x": float(args.get("x", 0)), "y": float(args.get("y", 0)),
               "z": args.get("z", ctx["position"].get("z"))}
        return {"ok": True, "simulated": True, "position": pos}
    if name == "move_stage_relative":
        cur = ctx["position"]
        pos = {"x": cur.get("x", 0) + float(args.get("dx", 0)),
               "y": cur.get("y", 0) + float(args.get("dy", 0)),
               "z": cur.get("z")}
        return {"ok": True, "simulated": True, "position": pos}
    if name == "get_stage_position":
        cur = ctx["position"]
        return {"ok": True, "simulated": True,
                "x": cur.get("x", 0), "y": cur.get("y", 0), "z": cur.get("z", 0)}
    if name == "apply_background_subtraction":
        data = ctx.get("_sim_last_data")
        if not data:
            return {"ok": False, "error": "시뮬레이션: 저장된 스펙트럼 없음"}
        m = sorted(data)[len(data) // 2]
        corrected = [v - m for v in data]
        return {"ok": True, "simulated": True,
                "version_label": args.get("version_label", "default"),
                "corrected_data": corrected,
                "max_corrected_intensity": float(max(corrected)),
                "raman_shift_cm-1": [200.0 + i * (2800.0 / len(data)) for i in range(len(data))]}
    # 나머지 도구(laser_off, autofocus 등)는 성공으로 간주
    return {"ok": True, "simulated": True, "tool": name}


# ── 도구 호출 단일 관문 ───────────────────────────────────────────────────────

def _call(ctx: dict, name: str, args: dict, record: bool = True) -> dict:
    """
    모든 도구 호출은 이 함수를 거친다 (단일 관문).
    이유: 관측 기록·스테이지 위치 추적·dose 누적·예외 방어를 한 곳에 모아야
    어떤 실행 경로(적응형/배경/LLM 루프)에서도 부기(bookkeeping)가 누락되지 않는다.
    """
    if ctx["dispatch"] is None:
        result = _simulate_tool(name, args, ctx)
    else:
        fn = ctx["dispatch"].get(name)
        if fn is None:
            result = {"ok": False, "error": f"도구 없음: {name}"}
        else:
            try:
                result = fn(dict(args))
            except Exception as e:
                # 도구 예외는 노드를 죽이지 않고 실패 결과로 변환 —
                # 상위(Planner)가 retry/replan을 결정할 수 있게 한다.
                result = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    if record:
        ctx["observations"].append({
            "tool": name, "args": dict(args), "result": result,
            "step_id": ctx["step_id"],
        })

    if isinstance(result, dict) and result.get("ok"):
        # 위치 추적: 이동 계열 도구의 반환 position을 신뢰 소스로 삼는다
        if name in ("move_stage", "move_stage_relative", "move_to_pixel") and result.get("position"):
            ctx["position"] = dict(result["position"])
        elif name == "get_stage_position":
            ctx["position"] = {"x": result.get("x"), "y": result.get("y"), "z": result.get("z")}
        # dose 누적: 레이저가 실제 조사된 호출(acquire_spectrum)만
        if name == "acquire_spectrum":
            p = float(args.get("power", 40.0))
            e = float(args.get("exposure", 0.2))
            n_acc = int(args.get("num_accumulations", 1)) if args.get("acq_mode") == "accumulate" else 1
            dose_inc = p * e * 0.01 * n_acc
            ctx["dose"] += dose_inc
            key = _pos_key(ctx["position"])
            ctx["dose_map"][key] = ctx["dose_map"].get(key, 0.0) + dose_inc

    return result


def _acquire_once(ctx: dict, power: float, exposure: float,
                  acq_mode: str = "single", num_acc: int = 1,
                  sim_background: bool = False) -> dict | None:
    """
    C2 안전 pre-check → 1회 스펙트럼 획득.
    반환 None = C2 HARD VETO (ctx["c2_abort"]에 entry 저장됨).
    프로브 측정을 포함한 '모든' 조사가 이 함수를 통과한다 — 우회 경로 없음.
    """
    c2_state = {
        **ctx["base_state"],
        "stage_position": ctx["position"],
        "cumulative_dose_map": ctx["dose_map"],   # 이번 노드에서 누적된 dose 반영
    }
    c2 = check_c2_hardware_safety(c2_state, power_pct=power, exposure_s=exposure)
    if c2["verdict"] == "ABORT":
        ctx["c2_abort"] = c2
        return None

    args: dict = {"power": power, "exposure": exposure}
    if acq_mode != "single":
        args["acq_mode"] = acq_mode
        args["num_accumulations"] = num_acc
    if ctx["dispatch"] is None and sim_background:
        args["_sim_background"] = True
    return _call(ctx, "acquire_spectrum", args)


# ── 적응형 튜닝 루프 (결정적) ─────────────────────────────────────────────────

def _resolve_limits(state: ExperimentState) -> tuple[float, float]:
    """
    이번 실험의 파워/노출 상한 결정.
    우선순위: bio 안전 한도(하드코딩) < 사용자 constraints — 항상 더 작은 쪽 채택.
    (사용자가 60%를 허용해도 bio 시료면 40%가 상한 — 안전이 사용자 지시보다 우선)
    """
    intent = state.get("intent") or {}
    constraints = intent.get("constraints", {}) or {}
    max_power = float(constraints.get("max_laser_power_pct") or _MAX_POWER_PCT)
    max_exposure = float(constraints.get("max_exposure_s") or _MAX_EXPOSURE_S)

    sample = (intent.get("sample_type") or "").lower()
    if any(kw in sample for kw in _BIO_KEYWORDS):
        max_power = min(max_power, _MAX_POWER_BIO)

    return min(max_power, _MAX_POWER_PCT), min(max_exposure, _MAX_EXPOSURE_S)


def _tune_acquire(ctx: dict, start_power: float, start_exposure: float,
                  max_power: float, max_exposure: float) -> tuple[dict | None, float, float, bool, list]:
    """
    적응형 획득의 핵심 루프. 반환: (마지막 결과, 확정 파워, 확정 노출, 수렴 여부, 이력).

    알고리즘 (모두 결정적 — 같은 시료에는 같은 동작):
      1. 현재 (power, exposure)로 측정
      2. max ADU가 목표 창 [_ADU_LOW, _ADU_HIGH] 안이면 수렴 → 종료
      3. 초과(포화 포함): 목표 중앙(_ADU_MID)까지의 비율로 축소.
         노출을 먼저 줄이는 이유 — 노출 변경은 즉각적이고 부작용이 없지만
         파워 변경은 ND 필터 기계 구동이 필요해 느리고 마모가 있다.
         포화 시에는 실제 광량을 알 수 없으므로(clip) 보수적으로 최소 1/4로 줄인다.
      4. 미달: 비율만큼 증폭하되 한 번에 _MAX_SCALE_PER_ITER 배 이하로 제한
         (선형 외삽이 틀렸을 때 조사량이 폭주하는 것을 방지).
         노출을 먼저 올리고(광손상 관점에서 피크 파워 증가보다 안전),
         노출이 상한이면 남은 배율만큼 파워를 올린다.
      5. 더 조정할 수 없으면(모두 한계) 미수렴 상태로 마지막 결과 반환 —
         "약한 신호라도 있는 게 없는 것보다 낫다". 미수렴은 이력에 남아
         Critic C3와 보고서가 인지한다.
    """
    power = min(max(start_power, _MIN_POWER_PCT), max_power)
    exposure = min(max(start_exposure, _MIN_EXPOSURE_S), max_exposure)
    history: list[dict] = []
    result: dict | None = None

    for i in range(_MAX_TUNE_ITERS):
        result = _acquire_once(ctx, power, exposure)
        if result is None:                       # C2 HARD VETO
            return None, power, exposure, False, history
        if not result.get("ok"):                 # 하드웨어 오류 — 상위에서 실패 처리
            return result, power, exposure, False, history

        m = float(result.get("max_intensity", 0.0))
        history.append({"iter": i + 1, "power_pct": round(power, 3),
                        "exposure_s": round(exposure, 3), "max_adu": round(m, 1)})

        if _ADU_LOW <= m <= _ADU_HIGH:
            return result, power, exposure, True, history

        prev_power, prev_exposure = power, exposure

        if m > _ADU_HIGH:                        # 과다/포화 → 축소
            scale = _ADU_MID / max(m, 1.0)
            if m >= _SATURATION_ADU:
                scale = min(scale, 0.25)
            new_exposure = exposure * scale
            if new_exposure >= _MIN_EXPOSURE_S:
                exposure = new_exposure
            else:                                # 노출이 바닥 → 파워도 줄임
                exposure = _MIN_EXPOSURE_S
                power = max(_MIN_POWER_PCT, power * scale)
        else:                                    # 신호 부족 → 증폭
            scale = min(_ADU_MID / max(m, 1.0), _MAX_SCALE_PER_ITER)
            exp_gain = min(scale, max_exposure / exposure)
            exposure = exposure * exp_gain
            remaining = scale / max(exp_gain, 1e-9)
            if remaining > 1.01:
                power = min(max_power, power * remaining)

        # 더 이상 조정 여지가 없으면 반복해도 같은 결과 — 즉시 중단해 dose 절약
        if abs(power - prev_power) < 1e-6 and abs(exposure - prev_exposure) < 1e-6:
            break

    return result, power, exposure, False, history


# ── Task 0: 측정 위치 결정 (구 roi_detector) ──────────────────────────────────

def _ask_vision(img_b64: str, target_description: str, width: int, height: int) -> dict:
    """
    현미경 프레임을 vision LLM에 보여주고 타겟/기판 픽셀 좌표를 받는다.
    실패(파싱/API 오류)는 found=False로 강등 — 탐색 루프는 다음 오프셋으로 계속.
    """
    prompt = (
        f"찾을 타겟: {target_description}\n"
        f"이미지 해상도: 가로 {width}px × 세로 {height}px.\n"
        "타겟의 중심 픽셀 좌표와, 비교 측정용 '빈 기판' 픽셀 좌표를 알려주세요."
    )
    try:
        resp = _llm_vision.invoke([
            SystemMessage(content=_VISION_SYSTEM),
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                # anthropic 네이티브 이미지 블록 — langchain-anthropic이 그대로 전달
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": img_b64}},
            ]),
        ])
        cleaned = re.sub(r"```(?:json)?", "", resp.content).strip().rstrip("`").strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return {"found": False}
        return parsed
    except Exception:
        return {"found": False}


def _pixel_delta_to_mm(dpx: float, dpy: float) -> tuple[float, float]:
    """픽셀 변위 → 스테이지 mm 변위. move_to_pixel과 반드시 동일한 수식."""
    dx_mm = dpx * _UM_PER_PX_X * CALIB_FACTOR_X / 1000.0 * _SIGN_X
    dy_mm = dpy * _UM_PER_PX_Y * CALIB_FACTOR_Y / 1000.0 * _SIGN_Y
    return dx_mm, dy_mm


def _visual_search(ctx: dict, target_description: str) -> dict | None:
    """
    나선형 탐색으로 타겟을 찾아 스테이지를 타겟 위로 이동시키고 ROI를 반환.
    실패 시 None. (내가 켠 카메라 스트림은 반드시 정리)

    도구 호출이 ctx의 _call 관문을 지나므로 스테이지 위치 추적이 유지된다.
    카메라/이미지 호출은 record=False — base64 프레임을 observations에 쌓으면
    상태가 수 MB로 불어나고 보고서 프롬프트에도 도움이 안 되기 때문.
    """
    stream = _call(ctx, "start_camera_stream", {}, record=False)
    if not stream.get("ok"):
        return None  # 카메라 자체가 안 됨 — 호출부가 fallback 판단
    # 내가 켠 스트림만 내가 끈다 — _look_at_microscope의 [스트림 소유권] 주석 참고.
    owns_stream = not stream.get("already_streaming", False)

    try:
        for i, (ox, oy) in enumerate(_SEARCH_OFFSETS_MM):
            # 첫 시도(0,0)는 현재 화면 그대로, 이후는 시야만큼 이동해 인접 탐색
            if i > 0:
                mv = _call(ctx, "move_stage_relative", {"dx": ox, "dy": oy})
                if not mv.get("ok"):
                    continue  # 이동 실패한 오프셋은 건너뛰고 다음 후보 시도

            img = _call(ctx, "analyze_microscope_image",
                        {"question": f"타겟 탐색: {target_description}"}, record=False)
            if not img.get("ok") or not img.get("image_base64"):
                continue

            found = _ask_vision(
                img["image_base64"], target_description,
                img.get("width", CAMERA_WIDTH), img.get("height", CAMERA_HEIGHT))

            if not found.get("found") or float(found.get("confidence") or 0) < _MIN_CONFIDENCE:
                continue

            px, py = int(found["pixel_x"]), int(found["pixel_y"])

            # 타겟 픽셀로 스테이지 이동 (이미지 중심 = 현재 스테이지 위치)
            mv = _call(ctx, "move_to_pixel", {"pixel_x": px, "pixel_y": py})
            if not mv.get("ok"):
                continue

            # 오토포커스 — 실패해도 진행 (초점이 조금 나가도 스펙트럼은 나오며,
            # 품질 문제는 C3의 신호부족/배경우세 검사가 다시 걸러준다)
            _call(ctx, "run_autofocus", {}, record=False)

            pos = _call(ctx, "get_stage_position", {}, record=False)
            if not pos.get("ok"):
                continue
            target_pos = {"x": pos.get("x"), "y": pos.get("y"), "z": pos.get("z")}

            # ── 배경(기판) 위치 계산 ──────────────────────────────────────────
            # vision LLM이 같은 프레임에서 고른 "빈 기판" 픽셀을,
            # 타겟 픽셀과의 변위 → mm 변위로 변환해 스테이지 좌표를 만든다.
            # (이동하지 않고 좌표만 계산 — 실제 이동은 acquire_background가 한다)
            background_position = None
            bgx, bgy = found.get("background_pixel_x"), found.get("background_pixel_y")
            if bgx is not None and bgy is not None:
                dmm_x, dmm_y = _pixel_delta_to_mm(float(bgx) - px, float(bgy) - py)
                background_position = {
                    "x": min(max(target_pos["x"] + dmm_x, 0.0), STAGE_MAX_X),
                    "y": min(max(target_pos["y"] + dmm_y, 0.0), STAGE_MAX_Y),
                }

            return {
                "x": target_pos["x"], "y": target_pos["y"], "z": target_pos["z"],
                "mode": "visual_search",
                "confidence": float(found.get("confidence") or 0),
                "reason": found.get("reason", ""),
                "background_position": background_position,
                "search_attempts": i + 1,
            }

        return None  # 모든 오프셋 소진 — 탐색 실패
    finally:
        # 내가 켠 스트림만 정리 — 프론트 MJPEG 뷰가 켜 둔 스트림을 끄면 그 화면이 죽는다.
        if owns_stream:
            _call(ctx, "stop_camera_stream", {}, record=False)


def _task_locate_target(state: ExperimentState, step: dict, ctx: dict) -> tuple[bool, str]:
    """
    측정 위치(ROI) 결정. 성공 시 ctx["next_roi"]에 저장 → 노드 반환값으로 상태 반영.

    [3가지 동작 모드]
      1. manual           — step params 또는 intent.constraints에 x/y가 있으면 그대로.
                            (위치를 아는데 시각 탐색을 강제하면 시간 낭비 + 오탐 위험)
      2. visual_search    — 좌표가 없고 카메라가 있으면 vision 탐색 (위 _visual_search).
      3. current_position — 둘 다 불가하면 현재 위치 유지 (마지막 fallback).
                            단, Planner가 visual_search를 명시 요구했는데 실패한
                            경우는 fallback 대신 정직하게 실패 보고 → 재계획 유도.
    """
    intent = state.get("intent") or {}
    constraints = intent.get("constraints", {}) or {}
    params = step.get("params", {}) or {}

    # ── 모드 1: manual ────────────────────────────────────────────────────────
    x = params.get("x", constraints.get("x"))
    y = params.get("y", constraints.get("y"))
    z = params.get("z", constraints.get("z"))
    if x is not None or y is not None:
        ctx["next_roi"] = {
            "x": float(x) if x is not None else 0.0,
            "y": float(y) if y is not None else 0.0,
            "z": float(z) if z is not None else None,
            "mode": "manual",
            "confidence": 1.0,
            "background_position": None,  # manual 모드는 acquire_background가
                                          # 오프셋 규칙(+0.5mm)으로 기판 위치를 정한다
        }
        return True, f"타겟 위치 확정 (manual): ({ctx['next_roi']['x']}, {ctx['next_roi']['y']})"

    # ── 모드 2: visual_search ─────────────────────────────────────────────────
    requested_visual = params.get("mode") == "visual_search"
    target_description = (
        params.get("target_description")
        or intent.get("target_description")
        or intent.get("sample_type", "측정 대상 시료")
    )

    if ctx["dispatch"] is None:
        # 하드웨어 미연결(시뮬레이션): 현재 위치를 타겟으로 가정하고 진행 —
        # 개발 환경에서 파이프라인 나머지를 검증할 수 있게 한다.
        current = ctx["position"] or {}
        ctx["next_roi"] = {
            "x": current.get("x", 0.0), "y": current.get("y", 0.0),
            "z": current.get("z"),
            "mode": "simulated_search", "confidence": 1.0,
            "background_position": {"x": current.get("x", 0.0) + 0.5,
                                    "y": current.get("y", 0.0)},
        }
        return True, "타겟 위치 (시뮬레이션): 현재 위치를 타겟으로 가정"

    roi = _visual_search(ctx, target_description)
    if roi is not None:
        ctx["next_roi"] = roi
        return True, (f"타겟 시각 탐색 성공: ({roi['x']}, {roi['y']}) "
                      f"확신도 {roi['confidence']:.2f}, 시도 {roi['search_attempts']}회")

    # ── 탐색 실패 처리 ────────────────────────────────────────────────────────
    if requested_visual:
        # 어정쩡한 위치에서 측정을 강행하는 것보다, Planner가 재계획
        # (예: 다른 target_description, 사용자 좌표 확인, 중심 좌표 측정)
        # 하도록 실패를 정직하게 보고하는 것이 데이터 신뢰성에 낫다.
        return False, (f"시각 탐색 실패: '{target_description}'을(를) "
                       f"{len(_SEARCH_OFFSETS_MM)}개 시야에서 찾지 못함")

    # ── 모드 3: current_position (시각 탐색이 '요구'된 게 아닌 일반 요청) ─────
    current = ctx["position"] or {}
    ctx["next_roi"] = {
        "x": current.get("x", 0.0), "y": current.get("y", 0.0),
        "z": current.get("z"),
        "mode": "current_position", "confidence": 0.0,
        "background_position": None,
    }
    return True, "타겟 위치: 현재 스테이지 위치 유지 (fallback)"


# ── Task 1: 타겟 적응형 획득 ──────────────────────────────────────────────────

def _task_acquire_target(state: ExperimentState, step: dict, ctx: dict) -> tuple[bool, str]:
    params = step.get("params", {}) or {}

    # 1. ROI로 이동 — locate_target step이 정한 타겟 위치가 있고 현재 위치와 다르면 이동.
    #    (visual_search가 이미 타겟 위로 이동시켰으면 0-이동으로 스킵된다)
    roi = state.get("next_roi") or {}
    if roi.get("x") is not None:
        cur = ctx["position"]
        if (abs(float(roi["x"]) - float(cur.get("x") or 0)) > 1e-3
                or abs(float(roi.get("y", 0)) - float(cur.get("y") or 0)) > 1e-3):
            mv_args = {"x": float(roi["x"]), "y": float(roi.get("y", 0))}
            if roi.get("z") is not None:
                mv_args["z"] = float(roi["z"])
            mv = _call(ctx, "move_stage", mv_args)
            if not mv.get("ok"):
                return False, f"ROI 이동 실패: {mv.get('error')}"

    # 2. 시작 파라미터 결정 — 우선순위와 그 이유:
    #    (a) step params: Planner가 C3 재시도 시 보정 계수를 곱해 넣은 값이므로 최우선
    #    (b) experience store: 컨텍스트(시료+기판+타겟 외형+기판 위치 영역)가
    #        유사한 과거 성공 에피소드의 확정 조건 → 튜닝 반복(=레이저 조사) 절약.
    #        정확 일치가 아닌 유사도 매칭인 이유: 기판/영역이 조금 달라도
    #        "가장 비슷한 과거"가 저출력 프로브 기본값보다 좋은 시작점이다.
    #        유사도가 임계값 미만이면 None → 안전한 프로브 기본값으로 시작.
    #    (c) 프로브 기본값: 아무 정보도 없으면 무조건 저출력에서 시작
    exp = experience.recall_params(experience.build_context(state)) or {}
    start_power = float(params.get("power_pct") or exp.get("power_pct") or _PROBE_POWER_PCT)
    start_exposure = float(params.get("exposure_s") or exp.get("exposure_s") or _PROBE_EXPOSURE_S)

    max_power, max_exposure = _resolve_limits(state)
    result, power, exposure, tuned, history = _tune_acquire(
        ctx, start_power, start_exposure, max_power, max_exposure)

    if ctx["c2_abort"] is not None:
        return False, f"C2 HARD VETO: {ctx['c2_abort']['reason']}"
    if result is None or not result.get("ok"):
        return False, f"스펙트럼 획득 실패: {(result or {}).get('error', '알 수 없음')}"

    # 3. IPBSA 형광 배경 제거 — 기판/시료의 넓은 형광 hump 위에 앉은 피크를 세운다.
    #    실패해도 실험은 계속한다: 원본 스펙트럼으로도 분석은 가능하며,
    #    배경 제거는 품질 향상 수단이지 필수 경로가 아니다 (best-effort).
    _call(ctx, "apply_background_subtraction",
          {"source": "last", "version_label": "target", "poly_order": 5})

    # 4. 확정 파라미터 기록 — 이후 acquire_background가 동일 조건으로 측정하고,
    #    실험 종료 시 experience store에 노하우로 축적된다.
    ctx["acquisition_params"] = {
        "power_pct": round(power, 3),
        "exposure_s": round(exposure, 3),
        "tuned": tuned,
        "history": history,
    }
    note = (f"적응형 획득 {'수렴' if tuned else '미수렴(한계 도달)'} — "
            f"power {power:.2f}%, exposure {exposure:.2f}s, "
            f"max {result.get('max_intensity', 0):.0f} ADU, 반복 {len(history)}회")
    return True, note


# ── Task 2: 기판 배경 참조 측정 ───────────────────────────────────────────────

def _task_acquire_background(state: ExperimentState, step: dict, ctx: dict) -> tuple[bool, str]:
    params = step.get("params", {}) or {}
    roi = state.get("next_roi") or {}

    # 복귀 지점 기억 — 이후 step들은 "스테이지가 타겟 위에 있다"고 가정하므로
    # 배경 측정 후 반드시 원위치로 돌아가야 한다.
    origin = dict(ctx["position"] or {})

    # 배경 위치 결정 우선순위:
    #  (a) params.position     — Planner/사용자가 명시한 기판 위치
    #  (b) roi.background_position — visual_search가 이미지에서 "빈 기판"으로 판단한 곳
    #  (c) 타겟 +0.5mm(x)      — 카메라 시야(~0.3mm)를 벗어난 인접 지점.
    #      타겟이 시야 크기보다 크면 실패할 수 있으나, (a)(b)가 없을 때의
    #      마지막 수단이며 C3의 배경우세 검사가 이상 여부를 다시 걸러준다.
    bg_pos = params.get("position") or roi.get("background_position")
    if not bg_pos:
        bg_pos = {"x": (origin.get("x") or 0) + 0.5, "y": origin.get("y") or 0}
    # 스테이지 물리 한계 클램프 — 한계 초과 좌표는 move_stage가 거부하므로 미리 보정
    bg_x = min(max(float(bg_pos.get("x", 0)), 0.0), STAGE_MAX_X)
    bg_y = min(max(float(bg_pos.get("y", 0)), 0.0), STAGE_MAX_Y)

    ok = False
    note = ""
    try:
        mv = _call(ctx, "move_stage", {"x": bg_x, "y": bg_y})
        if not mv.get("ok"):
            return False, f"배경 위치 이동 실패: {mv.get('error')}"

        # 측정 조건 = 타겟과 "동일" (재튜닝하지 않음!)
        # 이유: 배경 스펙트럼의 용도는 타겟 스펙트럼과의 강도 비교인데,
        # 파워/노출이 다르면 강도 축이 달라져 비교 자체가 무의미해진다.
        acq = ctx.get("acquisition_params") or state.get("acquisition_params") or {}
        power = float(acq.get("power_pct", _PROBE_POWER_PCT))
        exposure = float(acq.get("exposure_s", _PROBE_EXPOSURE_S))

        result = _acquire_once(ctx, power, exposure, sim_background=True)
        if ctx["c2_abort"] is not None:
            return False, f"C2 HARD VETO: {ctx['c2_abort']['reason']}"
        if result is None or not result.get("ok"):
            return False, f"배경 스펙트럼 획득 실패: {(result or {}).get('error', '알 수 없음')}"

        # 배경도 IPBSA 처리 (타겟과 같은 전처리 — 비교 일관성)
        _call(ctx, "apply_background_subtraction",
              {"source": "last", "version_label": "background", "poly_order": 5})

        # analyst가 프롬프트에 바로 넣을 수 있는 압축 요약을 만들어 둔다.
        # (1024포인트 원본을 상태에 두 벌 들고 다니지 않기 위한 다운샘플)
        data = result.get("data") or []
        shift = result.get("raman_shift_cm-1") or list(range(len(data)))
        summary = ""
        if data:
            stride = max(1, len(data) // 60)
            summary = ", ".join(
                f"{shift[i]:.0f}:{data[i]:.0f}" for i in range(0, len(data), stride))

        ctx["background_reference"] = {
            "position": {"x": bg_x, "y": bg_y},
            "max_intensity": float(result.get("max_intensity", 0.0)),
            "power_pct": power,
            "exposure_s": exposure,
            "summary": summary,
        }
        ok = True
        note = (f"기판 배경 측정 완료 — 위치 ({bg_x:.2f}, {bg_y:.2f}), "
                f"max {result.get('max_intensity', 0):.0f} ADU (타겟과 동일 조건)")
        return True, note
    finally:
        # 성공/실패 무관 원위치 복귀 — 복귀 실패는 이후 step 전부를 오염시키므로
        # 복귀가 실패하면 이 step 자체를 실패로 강등한다.
        if origin.get("x") is not None:
            back = _call(ctx, "move_stage",
                         {"x": origin["x"], "y": origin.get("y", 0)})
            if not back.get("ok") and ok:
                # note/ok는 이미 확정됐지만 복귀 실패는 치명적 — 실패로 덮어씀
                # (finally에서 return하면 원래 return을 삼키므로 플래그만 조작 불가 →
                #  아래처럼 예외로 승격시켜 상위 except에서 실패 처리하게 한다)
                raise RuntimeError(f"배경 측정 후 원위치 복귀 실패: {back.get('error')}")


# ── Task 3: 일반 LLM tool-calling 루프 (기존 동작 유지) ───────────────────────

def _run_llm_tool_loop(state: ExperimentState, step: dict, ctx: dict) -> tuple[bool, str]:
    """
    task가 지정되지 않은 자유 형식 하드웨어 작업.
    (예: "오토포커스 실행", "스테이지를 중심으로 이동" 같은 단발 작업)
    도구 실행은 전부 _call 관문을 거쳐 dose/위치 추적이 유지된다.
    """
    params = step.get("params", {}) or {}

    lc_tools, _ = _build_lc_tools()
    if not lc_tools:
        # 하드웨어 미연결 — 시뮬레이션 성공 처리 (파이프라인 검증용)
        ctx["observations"].append({
            "tool": "hw_manager_sim", "args": {},
            "result": {"ok": True, "simulated": True, "message": "하드웨어 미연결 — 시뮬레이션"},
            "step_id": ctx["step_id"],
        })
        return True, "시뮬레이션 완료"

    llm_with_tools = _llm.bind_tools(lc_tools)
    prompt = (
        f"작업: {step['action']}\n"
        f"파라미터: {json.dumps(params, ensure_ascii=False)}\n"
        f"현재 스테이지 위치: {ctx['position']}\n"
        "위 작업을 수행하세요."
    )
    messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]

    last_tool_ok = True
    max_turns = 8                                 # LLM 무한 루프 방지
    for _ in range(max_turns):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            name, args = tc["name"], tc["args"] or {}

            # 레이저 조사 도구는 LLM 루프에서도 C2 pre-check를 우회할 수 없다.
            if name == "acquire_spectrum":
                result = _acquire_once(
                    ctx,
                    float(args.get("power", 40.0)),
                    float(args.get("exposure", 0.2)),
                    acq_mode=args.get("acq_mode", "single"),
                    num_acc=int(args.get("num_accumulations", 1)),
                )
                if result is None:               # C2 VETO — LLM에게 알리고 루프 종료
                    veto_msg = {"ok": False, "error": f"안전 거부: {ctx['c2_abort']['reason']}"}
                    messages.append(ToolMessage(
                        content=json.dumps(veto_msg, ensure_ascii=False), tool_call_id=tc["id"]))
                    return False, f"C2 HARD VETO: {ctx['c2_abort']['reason']}"
            else:
                result = _call(ctx, name, args)

            last_tool_ok = bool(result.get("ok"))
            # 대용량 배열(data 등)은 LLM 컨텍스트에 넣지 않는다 —
            # 토큰 낭비 + 모델 혼란. 요약 필드만 전달한다.
            slim = {k: v for k, v in result.items()
                    if not (isinstance(v, list) and len(v) > 32)}
            messages.append(ToolMessage(
                content=json.dumps(slim, ensure_ascii=False), tool_call_id=tc["id"]))

    # 성공 판정 휴리스틱: "마지막 도구 호출"이 성공했는지를 본다.
    # 중간 실패는 LLM이 스스로 복구했을 수 있으므로(예: 범위 초과 → 좌표 수정 재시도)
    # 마지막 결과만이 step의 최종 상태를 대표한다.
    if not last_tool_ok:
        err = ctx["observations"][-1]["result"].get("error", "?") if ctx["observations"] else "?"
        return False, f"마지막 도구 호출 실패: {err}"
    return True, "작업 완료"


# ══════════════════════════════════════════════════════════════════════════════
# 하드웨어 질의 (실험 파이프라인 밖 — 채팅에서 바로 답하는 경로)
# ──────────────────────────────────────────────────────────────────────────────
# 왜 여기(hw_manager)에 두는가:
#   "지금 현미경 화면 보여?" 같은 질문에 답하려면 "어떤 도구가 있고 무엇을 할 수
#   있는지"를 알아야 한다. 그걸 아는 곳은 도구를 실제로 들고 있는 이 모듈뿐이다.
#   translator는 도구를 바인딩하지 않으므로(순수 JSON 변환기), 그쪽에서 답하면
#   "그런 기능 없습니다" 같은 사실과 다른 답을 지어낸다 — 실제로 그랬다.
#   → orchestrator는 하드웨어 질의를 여기로 넘기고, 답변은 도구 실행 결과로 만든다.
#
# 안전 경계:
#   이 경로는 Planner/Critic C1 게이트를 타지 않는다. 그래서 _QUERY_TOOL_NAMES
#   (관측 전용)만 바인딩한다 — 레이저·이동은 실험 요청으로 분류돼야 한다.
# ══════════════════════════════════════════════════════════════════════════════

_QUERY_SYSTEM = """\
당신은 라만 분광기 하드웨어의 상태를 사용자에게 알려주는 에이전트입니다.
사용 가능한 도구로 실제 장비 상태를 확인한 뒤, 그 결과에 근거해 한국어로 답하세요.

규칙:
- 반드시 도구를 먼저 호출해 실제 값을 확인하고 답하세요. 추측하지 마세요.
- 당신이 가진 도구로 할 수 없는 일이면 "할 수 없다"가 아니라,
  무엇이 가능한지(어떤 도구가 있는지) 알려주세요.
- 레이저 조사나 스테이지 이동이 필요한 요청이면, 그건 이 대화창에서 바로 하지 않고
  "측정 요청으로 말씀해 주시면 계획을 세워 안전 검증 후 수행한다"고 안내하세요.
- 답변은 2~4문장으로 간결하게. JSON이 아닌 자연스러운 한국어 문장으로 답하세요."""

_LOOK_SYSTEM = """\
당신은 현미경 카메라 이미지를 보고 사용자에게 설명하는 전문가입니다.
이미지에 실제로 보이는 것만 근거로, 한국어 2~4문장으로 설명하세요.
시야에 보이는 물체의 모양·밝기·분포, 초점 상태를 구체적으로 언급하세요."""


def _look_at_microscope(ctx: dict, question: str) -> str:
    """현미경 프레임 1장을 캡처해 vision LLM에게 보여주고 설명을 받는다.

    analyze_microscope_image를 tool-calling 루프에 직접 바인딩하지 않는 이유:
    이 도구의 반환값에는 PNG base64 문자열이 들어 있어(수 MB) ToolMessage로
    LLM 컨텍스트에 그대로 실리면 토큰이 폭발한다. 그래서 이미지 처리는 코드가
    맡고, LLM에게는 "설명 텍스트"만 돌려준다.

    [스트림 소유권 — 끄기 전에 반드시 확인]
    프론트의 MJPEG 뷰(/api/camera/stream)가 열려 있으면 카메라는 "이미" 스트리밍
    중이다. 그 상태에서 우리가 끝나고 stop_camera_stream을 부르면 TUCam의
    AbortWait/Cap_Stop/Buf_Release가 실행되어 MJPEG 쪽 화면이 죽는다(그리고 그
    루프는 프레임이 None이 되면 yield를 못 해 접속 종료도 감지 못 하고 영원히 돈다).
    → 내가 켠 스트림만 내가 끈다.
    """
    stream = _call(ctx, "start_camera_stream", {}, record=False)
    if not stream.get("ok"):
        return f"카메라 스트림을 켤 수 없습니다: {stream.get('error', '알 수 없는 오류')}"
    # 이 호출이 스트림을 켰다면 우리 소유 → 끝나고 정리한다.
    # 이미 켜져 있었다면 남의 것 → 건드리지 않는다.
    owns_stream = not stream.get("already_streaming", False)

    try:
        img = _call(ctx, "analyze_microscope_image", {"question": question}, record=False)
        if not img.get("ok") or not img.get("image_base64"):
            return f"현미경 프레임을 가져오지 못했습니다: {img.get('error', '알 수 없는 오류')}"

        resp = _llm_vision.invoke([
            SystemMessage(content=_LOOK_SYSTEM),
            HumanMessage(content=[
                {"type": "text", "text": question},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": img["image_base64"]}},
            ]),
        ])
        return (resp.content or "").strip() or "이미지를 해석하지 못했습니다."
    except Exception as e:
        return f"이미지 해석 중 오류: {type(e).__name__}: {e}"
    finally:
        if owns_stream:
            try:
                _call(ctx, "stop_camera_stream", {}, record=False)
            except Exception:
                pass


def run_hardware_query(user_msg: str) -> str:
    """실험이 아닌 "장비 상태 질문"에 도구를 실제로 호출해 답한다.

    orchestrator가 intent.request_type == "hardware"일 때 호출한다.
    반환: 사용자에게 그대로 보여줄 한국어 답변 문장.
    """
    ctx = _make_ctx({}, {"step_id": "query"})
    lc_tools, _ = _build_lc_tools(_QUERY_TOOL_NAMES)

    if ctx["dispatch"] is None or not lc_tools:
        return ("지금은 장비에 연결돼 있지 않아(시뮬레이션 모드) 실제 화면이나 상태를 "
                "읽을 수 없습니다. 장비 PC에서 실행하면 현미경 화면 확인, 스테이지 위치, "
                "CCD 상태, 초점 품질을 조회할 수 있습니다.")

    # 화면을 "보는" 능력은 코드가 감싼 전용 도구로 제공한다(위 _look_at_microscope 참고).
    @tool("look_at_microscope",
          description="현미경 카메라의 현재 화면을 실제로 보고 무엇이 보이는지 설명한다. "
                      "사용자가 화면/시야/샘플 모습을 물어보면 이 도구를 사용하라. "
                      "question 인자에 사용자가 궁금해하는 내용을 넣어라.")
    def _look(question: str = "현재 현미경 화면에 무엇이 보이나요?") -> str:
        return _look_at_microscope(ctx, question)

    llm_with_tools = _llm.bind_tools(lc_tools + [_look])
    messages = [SystemMessage(content=_QUERY_SYSTEM), HumanMessage(content=user_msg)]

    try:
        for _ in range(6):                     # LLM 무한 루프 방지
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                return (response.content or "").strip() or "장비 상태를 확인하지 못했습니다."

            for tc in response.tool_calls:
                name, args = tc["name"], tc["args"] or {}
                if name == "look_at_microscope":
                    result = _look_at_microscope(ctx, args.get("question") or user_msg)
                else:
                    result = _call(ctx, name, args)
                # 대용량 배열은 컨텍스트에서 제외 (_run_llm_tool_loop와 동일 정책)
                if isinstance(result, dict):
                    result = {k: v for k, v in result.items()
                              if not (isinstance(v, list) and len(v) > 32)}
                messages.append(ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=tc["id"]))
        return "장비 상태 확인이 길어져 중단했습니다. 다시 질문해 주세요."
    except Exception as e:
        return f"장비 상태를 확인하는 중 오류가 발생했습니다: {type(e).__name__}: {e}"
    # 카메라 스트림 정리는 _look_at_microscope가 자기가 켠 경우에만 수행한다.
    # 여기서 무조건 stop_camera_stream을 부르면 프론트 MJPEG 뷰가 켜 둔 스트림까지
    # 꺼서 화면이 죽는다 (_look_at_microscope의 [스트림 소유권] 주석 참고).


# ══════════════════════════════════════════════════════════════════════════════
# 노드 진입점
# ══════════════════════════════════════════════════════════════════════════════

def hw_manager_node(state: ExperimentState) -> dict:
    plan = state.get("plan", [])
    idx = state.get("current_step_idx", 0)
    step = plan[idx] if idx < len(plan) else None

    if step is None:
        # 방어적 처리: Planner 라우팅 버그로 여기 도달해도 그래프가 죽지 않게 한다
        return {"observations": [{"tool": "hw_manager", "args": {},
                                  "result": {"ok": False, "error": "plan step 없음"},
                                  "step_id": "?"}]}

    task = (step.get("params", {}) or {}).get("task", "")
    ctx = _make_ctx(state, step)

    success, note = False, ""
    try:
        if task == "locate_target":
            success, note = _task_locate_target(state, step, ctx)
        elif task == "acquire_target":
            success, note = _task_acquire_target(state, step, ctx)
        elif task == "acquire_background":
            success, note = _task_acquire_background(state, step, ctx)
        else:
            success, note = _run_llm_tool_loop(state, step, ctx)
    except Exception as e:
        # 노드 내부 예외는 절대 그래프를 죽이지 않는다 — 실패 step으로 변환해
        # Planner의 failure 정책(retry/replan/skip/abort)에 넘긴다.
        success, note = False, f"hw_manager 내부 예외: {type(e).__name__}: {e}"
    finally:
        # 레이저 이중 안전망: acquire_spectrum 내부 finally에서도 끄지만,
        # LLM 루프가 laser_on을 개별 호출한 채 예외로 중단되는 경로까지 막는다.
        # record=False — 안전망 동작은 관측 데이터가 아니므로 기록하지 않는다.
        try:
            _call(ctx, "laser_off", {}, record=False)
        except Exception:
            pass

    # ── 상태 반환값 조립 ──────────────────────────────────────────────────────
    updated_plan = list(plan)

    # step.result에는 요약만 저장 — 1024포인트 배열을 plan에 넣으면
    # 매 planner 왕복마다 상태 복사 비용이 커지고 로그 가독성이 죽는다.
    # 원본 데이터는 observations에 남아 있으므로 정보 손실은 없다.
    last_result = ctx["observations"][-1]["result"] if ctx["observations"] else {}
    slim_result = {k: v for k, v in (last_result or {}).items()
                   if not (isinstance(v, list) and len(v) > 32)}

    out: dict = {
        "observations": ctx["observations"],
        "cumulative_dose_mj": ctx["dose"],
        "cumulative_dose_map": ctx["dose_map"],
    }
    if ctx["position"]:
        out["stage_position"] = ctx["position"]
    if ctx["acquisition_params"]:
        out["acquisition_params"] = ctx["acquisition_params"]
    if ctx["background_reference"]:
        out["background_reference"] = ctx["background_reference"]
    if ctx["next_roi"]:
        out["next_roi"] = ctx["next_roi"]

    # C2 HARD VETO는 실패 처리보다 우선 — 즉시 실험 중단 (Tier-A)
    if ctx["c2_abort"] is not None:
        updated_plan[idx] = {**step, "status": "failed",
                             "result": {"c2_abort": ctx["c2_abort"]["reason"]}}
        out.update({
            "plan": updated_plan,
            "critic_log": [ctx["c2_abort"]],
            "abort_reason": ctx["c2_abort"]["reason"],
            "next_node": "__end__",
        })
        return out

    if success:
        # 주의: current_step_idx를 전진시키지 않는다.
        # Planner가 C3 품질 게이트를 통과시킨 뒤 전진시킨다 (모듈 docstring 참고).
        updated_plan[idx] = {**step, "status": "done",
                             "result": {**slim_result, "note": note}}
        out["plan"] = updated_plan
    else:
        updated_plan[idx] = {**step, "status": "failed", "result": {"error": note}}
        out["plan"] = updated_plan
        out["failure_log"] = [{
            "step_id": step.get("step_id", "?"),
            "agent": "hw_manager",
            "action": step.get("action", ""),
            "task": task,
            "error": note,
            "timestamp": time.time(),
        }]

    return out
