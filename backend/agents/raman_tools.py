"""
Raman 하드웨어 tool 래퍼
- LLM agent가 호출할 수 있는 단순 함수들
- 각 함수는 dict를 반환 (LLM에게 결과를 텍스트로 전달하기 위해)
"""

from __future__ import annotations
import sys
import os

STAGE_MAX_X =  75.3169
STAGE_MAX_Y =  50.1879
STAGE_MIN_Z =  -1.0
STAGE_MAX_Z =   1.0

_stage = None
_laser = None
_ccd   = None


def init_hardware(stage=None, laser=None, ccd=None):
    """하드웨어 객체를 주입. run_scan.py 등에서 초기화 후 호출."""
    global _stage, _laser, _ccd
    _stage = stage
    _laser = laser
    _ccd = ccd


# ──────────────────────────────────────────
# 스테이지
# ──────────────────────────────────────────

def move_stage(x: float, y: float, z: float = None) -> dict:
    """스테이지를 절대 좌표(mm)로 이동."""
    if _stage is None:
        return {"ok": False, "error": "스테이지가 초기화되지 않았습니다."}

    # 범위 검증
    if not (0 <= x <= STAGE_MAX_X):
        return {"ok": False, "error": f"X 범위 초과: {x} (허용: 0~{STAGE_MAX_X})"}
    if not (0 <= y <= STAGE_MAX_Y):
        return {"ok": False, "error": f"Y 범위 초과: {y} (허용: 0~{STAGE_MAX_Y})"}
    if z is not None and not (STAGE_MIN_Z <= z <= STAGE_MAX_Z):
        return {"ok": False, "error": f"Z 범위 초과: {z} (허용: {STAGE_MIN_Z}~{STAGE_MAX_Z})"}

    try:
        kw = {"x": x, "y": y, "wait": True}
        if z is not None:
            kw["z"] = z
        _stage.move_absolute(**kw)
        pos = _stage.get_position()
        return {"ok": True, "position": {"x": pos[0], "y": pos[1], "z": pos[2]}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_stage_position() -> dict:
    """현재 스테이지 위치를 반환."""
    if _stage is None:
        return {"ok": False, "error": "스테이지가 초기화되지 않았습니다."}
    try:
        pos = _stage.get_position()
        return {"ok": True, "x": pos[0], "y": pos[1], "z": pos[2]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def move_stage_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
    """스테이지를 현재 위치 기준 상대 이동(mm)."""
    if _stage is None:
        return {"ok": False, "error": "스테이지가 초기화되지 않았습니다."}
    try:
        _stage.move_relative(dx, dy, dz, 0)
        pos = _stage.get_position()
        return {"ok": True, "position": {"x": pos[0], "y": pos[1], "z": pos[2]}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 레이저
# ──────────────────────────────────────────

def laser_on() -> dict:
    """레이저를 켠다."""
    if _laser is None:
        return {"ok": False, "error": "레이저가 초기화되지 않았습니다."}
    try:
        _laser.laser_on()
        return {"ok": True, "status": "레이저 ON"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def laser_off() -> dict:
    """레이저를 끈다."""
    if _laser is None:
        return {"ok": False, "error": "레이저가 초기화되지 않았습니다."}
    try:
        _laser.laser_off()
        return {"ok": True, "status": "레이저 OFF"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_laser_power(percent: int) -> dict:
    """레이저 출력 설정. percent: 20, 40, 60, 80, 100 중 하나."""
    if _laser is None:
        return {"ok": False, "error": "레이저가 초기화되지 않았습니다."}
    valid = {20, 40, 60, 80, 100}
    if percent not in valid:
        return {"ok": False, "error": f"유효한 출력값: {valid}"}
    try:
        _laser.set_power(percent)
        return {"ok": True, "power_percent": percent}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 스펙트럼 수집 (Andor CCD)
# ──────────────────────────────────────────

def acquire_spectrum(exposure: float = 0.1) -> dict:
    """스펙트럼 1회 수집. exposure: 노출시간(초)."""
    if _ccd is None:
        return {"ok": False, "error": "분광기가 초기화되지 않았습니다."}
    try:
        _ccd.setup_acquisition(exposure_time=exposure)
        data = _ccd.start_acquisition_cycle()
        return {
            "ok": True,
            "length": len(data),
            "max_intensity": float(max(data)),
            "sum_intensity": float(sum(data)),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# tool dispatch 테이블 (agent loop에서 사용)
# ──────────────────────────────────────────

TOOL_DISPATCH = {
    "move_stage":          lambda a: move_stage(**a),
    "get_stage_position":  lambda a: get_stage_position(),
    "move_stage_relative": lambda a: move_stage_relative(**a),
    "laser_on":            lambda a: laser_on(),
    "laser_off":           lambda a: laser_off(),
    "set_laser_power":     lambda a: set_laser_power(**a),
    "acquire_spectrum":    lambda a: acquire_spectrum(**a),
}
