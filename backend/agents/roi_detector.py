"""
ROIDetectorNode — 다음 측정 위치(ROI) 결정.

MVP: intent.constraints에서 x, y, z 좌표를 직접 읽어 next_roi 설정.
V1: 카메라 이미지 분석 + LLM으로 자동 ROI 탐색 (exploratory/event-chase 모드).
"""

from __future__ import annotations

from backend.agents.state import ExperimentState


def _advance_plan(state: ExperimentState) -> dict:
    """현재 step을 done으로 표시하고 idx를 올린다."""
    idx = state.get("current_step_idx", 0)
    plan = list(state.get("plan", []))
    if idx < len(plan):
        plan[idx] = {**plan[idx], "status": "done"}
    return {"plan": plan, "current_step_idx": idx + 1}


def roi_detector_node(state: ExperimentState) -> dict:
    intent = state.get("intent") or {}
    constraints = intent.get("constraints", {})

    x = constraints.get("x")
    y = constraints.get("y")
    z = constraints.get("z")

    if x is not None or y is not None:
        roi = {
            "x": float(x) if x is not None else 0.0,
            "y": float(y) if y is not None else 0.0,
            "z": float(z) if z is not None else None,
            "mode": "manual",
        }
    else:
        # 좌표 미지정 — 현재 스테이지 위치 유지
        current = state.get("stage_position") or {}
        roi = {
            "x": current.get("x", 0.0),
            "y": current.get("y", 0.0),
            "z": current.get("z"),
            "mode": "current_position",
        }

    return {"next_roi": roi, **_advance_plan(state)}
