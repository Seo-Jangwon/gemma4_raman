"""
ROIDetectorNode — 측정 위치(ROI) 결정.

[3가지 동작 모드]
  1. manual          — intent.constraints에 x/y 좌표가 있으면 그대로 사용 (기존 동작).
  2. visual_search   — 좌표가 없고 카메라가 있으면: 현미경 이미지 → vision LLM으로
                       타겟 픽셀 좌표 식별 → move_to_pixel → 오토포커스 → 확정.
                       "타겟의 위치를 육안으로도 못 찾는다"는 문제에 대한 직접 대응.
  3. current_position— 좌표도 카메라도 없으면 현재 위치 유지 (마지막 fallback).

[visual_search 설계 결정]
- 나선형 탐색 오프셋: 카메라 시야는 렌즈 기준 ~0.3mm에 불과하다. 첫 화면에
  타겟이 없으면 시야 크기만큼 스테이지를 옮겨가며 인접 영역을 훑는다.
  오프셋 수를 5개로 제한한 이유: 탐색은 가이드빔/조명만 쓰므로 시편 손상은 없지만,
  무한 탐색은 실험 시간을 폭주시킨다. 5회 안에 못 찾으면 "탐색 실패"를 명확히
  선언하고 Planner의 재계획(예: 사용자 좌표 요청, 중심 좌표 측정)에 넘기는 것이
  어정쩡한 위치에서 레이저를 쏘는 것보다 안전하다.
- 배경 위치 동시 결정: vision LLM에게 타겟과 함께 "빈 기판" 픽셀도 요청한다.
  타겟을 눈으로 확인한 바로 그 프레임에서 기판 위치를 고르는 것이,
  나중에 맹목적 오프셋(+0.5mm)으로 찍는 것보다 "진짜 기판"일 확률이 높다.
  → 기판 background 신호 분리 문제의 입구를 여기서 해결한다.
- 픽셀→mm 변환 수식은 hw_tools.raman_tools.move_to_pixel과 동일해야 한다.
  (다르면 배경 위치가 엉뚱한 곳이 된다) — 같은 config 상수를 import해 계산한다.

[Failure 처리]
- 탐색 실패/카메라 오류 시 step을 "failed"로 표시하고 failure_log에 남긴다.
  idx는 전진시키지 않는다 — Planner의 on_fail 정책(replan)이 다음 행동을 결정.
- 시뮬레이션 모드(하드웨어 미접속)에서는 성공으로 간주해 파이프라인 검증을 돕는다.
"""

from __future__ import annotations

import json
import re
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.state import ExperimentState

# ── vision LLM (이미지에서 타겟 픽셀 좌표를 읽는 용도) ────────────────────────
_llm_vision = ChatAnthropic(model="claude-opus-4-8")

# ── 픽셀→mm 변환 상수 (move_to_pixel과 동일 수식 — 주석의 [설계 결정] 참고) ──
# Config.ini가 없는 개발 환경에서도 import가 죽지 않도록 방어.
try:
    from backend.config import (
        CALIB_FACTOR_X, CALIB_FACTOR_Y,
        CAMERA_HEIGHT, CAMERA_WIDTH,
        LENS_HEIGHT_UM, LENS_WIDTH_UM,
        STAGE_MAX_X, STAGE_MAX_Y,
    )
except Exception:
    CAMERA_WIDTH, CAMERA_HEIGHT = 1060, 800
    LENS_WIDTH_UM, LENS_HEIGHT_UM = 305.0, 230.0
    CALIB_FACTOR_X, CALIB_FACTOR_Y = 1.4, 1.285
    STAGE_MAX_X, STAGE_MAX_Y = 75.3, 50.2

_UM_PER_PX_X = LENS_WIDTH_UM / CAMERA_WIDTH
_UM_PER_PX_Y = LENS_HEIGHT_UM / CAMERA_HEIGHT
_SIGN_X = -1   # pixel +X(right) → stage -X  (raman_tools.move_to_pixel과 동일)
_SIGN_Y = +1   # pixel +Y(down)  → stage +Y

# 나선형 탐색 오프셋 (mm) — 시야(~0.3mm)와 겹치지 않게 0.25mm 간격.
# (0,0)=현재 화면부터 시작해 좌우/대각으로 확장.
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


def _fail_step(state: ExperimentState, error: str) -> dict:
    """step을 failed로 표시 + failure_log 기록. idx는 전진 안 함 (Planner가 결정)."""
    idx = state.get("current_step_idx", 0)
    plan = list(state.get("plan", []))
    step = plan[idx] if idx < len(plan) else {}
    if idx < len(plan):
        plan[idx] = {**plan[idx], "status": "failed", "result": {"error": error}}
    return {
        "plan": plan,
        "failure_log": [{
            "step_id": step.get("step_id", "?"),
            "agent": "roi_detector",
            "action": step.get("action", ""),
            "error": error,
            "timestamp": time.time(),
        }],
    }


def _done_step(state: ExperimentState, roi: dict) -> dict:
    """step을 done으로 표시하고 idx 전진. (ROI 결정에는 C3 게이트가 필요 없으므로
    hw_manager와 달리 스스로 전진한다 — Planner 왕복 1회 절약)"""
    idx = state.get("current_step_idx", 0)
    plan = list(state.get("plan", []))
    if idx < len(plan):
        plan[idx] = {**plan[idx], "status": "done", "result": {"roi": roi}}
    return {"next_roi": roi, "plan": plan, "current_step_idx": idx + 1}


# ── 도구 호출 헬퍼 ────────────────────────────────────────────────────────────

def _get_dispatch() -> dict | None:
    """하드웨어 도구 로드. Exception 전체를 잡는 이유: Config.ini가 없는 개발 PC에서
    config.py가 NoSectionError(ImportError 아님)를 던지므로 — 시뮬레이션으로 강등."""
    try:
        from backend.hw_tools.raman_tools import TOOL_DISPATCH
        return TOOL_DISPATCH
    except Exception:
        return None


def _call(dispatch: dict, name: str, args: dict) -> dict:
    """도구 예외를 실패 dict로 변환 — 탐색 루프가 예외로 끊기지 않게 한다."""
    fn = dispatch.get(name)
    if fn is None:
        return {"ok": False, "error": f"도구 없음: {name}"}
    try:
        return fn(dict(args))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


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


# ── visual search 본체 ────────────────────────────────────────────────────────

def _visual_search(state: ExperimentState, dispatch: dict, target_description: str) -> dict | None:
    """
    나선형 탐색으로 타겟을 찾아 스테이지를 타겟 위로 이동시키고 ROI를 반환.
    실패 시 None. (스트림 정리는 항상 수행)
    """
    stream = _call(dispatch, "start_camera_stream", {})
    if not stream.get("ok"):
        return None  # 카메라 자체가 안 됨 — 호출부가 fallback 판단

    try:
        for i, (ox, oy) in enumerate(_SEARCH_OFFSETS_MM):
            # 첫 시도(0,0)는 현재 화면 그대로, 이후는 시야만큼 이동해 인접 탐색
            if i > 0:
                mv = _call(dispatch, "move_stage_relative", {"dx": ox, "dy": oy})
                if not mv.get("ok"):
                    continue  # 이동 실패한 오프셋은 건너뛰고 다음 후보 시도

            img = _call(dispatch, "analyze_microscope_image",
                        {"question": f"타겟 탐색: {target_description}"})
            if not img.get("ok") or not img.get("image_base64"):
                continue

            found = _ask_vision(
                img["image_base64"], target_description,
                img.get("width", CAMERA_WIDTH), img.get("height", CAMERA_HEIGHT))

            if not found.get("found") or float(found.get("confidence") or 0) < _MIN_CONFIDENCE:
                continue

            px, py = int(found["pixel_x"]), int(found["pixel_y"])

            # 타겟 픽셀로 스테이지 이동 (이미지 중심 = 현재 스테이지 위치)
            mv = _call(dispatch, "move_to_pixel", {"pixel_x": px, "pixel_y": py})
            if not mv.get("ok"):
                continue

            # 오토포커스 — 실패해도 진행 (초점이 조금 나가도 스펙트럼은 나오며,
            # 품질 문제는 C3의 신호부족/배경우세 검사가 다시 걸러준다)
            _call(dispatch, "run_autofocus", {})

            pos = _call(dispatch, "get_stage_position", {})
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
        # 스트림은 반드시 정리 — 켜둔 채 방치하면 다음 카메라 사용이 충돌한다
        _call(dispatch, "stop_camera_stream", {})


# ══════════════════════════════════════════════════════════════════════════════
# 노드 진입점
# ══════════════════════════════════════════════════════════════════════════════

def roi_detector_node(state: ExperimentState) -> dict:
    intent = state.get("intent") or {}
    constraints = intent.get("constraints", {}) or {}

    plan = state.get("plan", [])
    idx = state.get("current_step_idx", 0)
    step_params = (plan[idx].get("params", {}) or {}) if idx < len(plan) else {}

    # ── 모드 1: manual — 좌표가 명시돼 있으면 탐색 없이 그대로 사용 ───────────
    # (사용자가 위치를 알고 있는데 시각 탐색을 강제하면 시간 낭비 + 오탐 위험)
    x = step_params.get("x", constraints.get("x"))
    y = step_params.get("y", constraints.get("y"))
    z = step_params.get("z", constraints.get("z"))
    if x is not None or y is not None:
        roi = {
            "x": float(x) if x is not None else 0.0,
            "y": float(y) if y is not None else 0.0,
            "z": float(z) if z is not None else None,
            "mode": "manual",
            "confidence": 1.0,
            "background_position": None,  # manual 모드는 acquire_background가
                                          # 오프셋 규칙(+0.5mm)으로 기판 위치를 정한다
        }
        return _done_step(state, roi)

    # ── 모드 2: visual_search — 좌표가 없으면 카메라로 타겟을 찾는다 ──────────
    requested_visual = step_params.get("mode") == "visual_search"
    target_description = (
        step_params.get("target_description")
        or intent.get("target_description")
        or intent.get("sample_type", "측정 대상 시료")
    )

    dispatch = _get_dispatch()
    if dispatch is None:
        # 하드웨어 미연결(시뮬레이션): 현재 위치를 타겟으로 가정하고 진행 —
        # 개발 환경에서 파이프라인 나머지를 검증할 수 있게 한다.
        current = state.get("stage_position") or {}
        roi = {
            "x": current.get("x", 0.0), "y": current.get("y", 0.0),
            "z": current.get("z"),
            "mode": "simulated_search", "confidence": 1.0,
            "background_position": {"x": current.get("x", 0.0) + 0.5,
                                    "y": current.get("y", 0.0)},
        }
        return _done_step(state, roi)

    roi = _visual_search(state, dispatch, target_description)
    if roi is not None:
        return _done_step(state, roi)

    # ── 탐색 실패 처리 ────────────────────────────────────────────────────────
    if requested_visual:
        # Planner가 명시적으로 시각 탐색을 요구했는데 실패 → step failed.
        # 어정쩡한 위치에서 측정을 강행하는 것보다, Planner가 재계획
        # (예: 다른 target_description, 사용자 좌표 확인, 중심 좌표 측정)
        # 하도록 실패를 정직하게 보고하는 것이 데이터 신뢰성에 낫다.
        return _fail_step(
            state,
            f"시각 탐색 실패: '{target_description}'을(를) "
            f"{len(_SEARCH_OFFSETS_MM)}개 시야에서 찾지 못함",
        )

    # 시각 탐색이 '요구'된 게 아니라면 (좌표 없는 일반 요청) — 현재 위치 유지.
    # 기존 동작과의 호환 유지: 실패보다 관대한 fallback이 적절한 경우다.
    current = state.get("stage_position") or {}
    roi = {
        "x": current.get("x", 0.0), "y": current.get("y", 0.0),
        "z": current.get("z"),
        "mode": "current_position", "confidence": 0.0,
        "background_position": None,
    }
    return _done_step(state, roi)
