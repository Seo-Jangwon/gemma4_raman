"""
Raman 하드웨어 tool 래퍼
- LLM agent가 호출할 수 있는 단순 함수들
- 각 함수는 dict를 반환 (LLM에게 결과를 텍스트로 전달하기 위해)
"""

from __future__ import annotations
import sys
import os
import time

# Andor SDK 상수 (USE_andor_test.py 와 동일한 값)
_READ_MODE_FVB       = 0   # Full Vertical Binning — 스펙트럼 1D
_TRIGGER_MODE_INTERNAL = 0 # 내부 트리거

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
        else:
            kw["z"] = _stage.get_position()[2]  # 현재 Z 유지
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

def acquire_spectrum(
    exposure: float = 0.1,
    power: int = 100,
    stabilize_sec: float = 0.5,
) -> dict:
    """
    라만 스펙트럼 1회 수집.

    원자적 실행 흐름:
      1. 레이저 출력 설정 (ND filter motor 이동, 블로킹)
      2. 레이저 ON
      3. 출력 안정화 대기 (stabilize_sec)
      4. CCD 파라미터 설정 (FVB, 내부 트리거)
      5. CCD 촬영 (StartAcquisition → 폴링 → GetAcquiredData)
      6. 레이저 OFF (성공/실패 무관하게 반드시 실행)

    Parameters
    ----------
    exposure : float
        CCD 노출 시간 [초]. 기본 0.1.
    power : int
        레이저 출력 [%]. 20/40/60/80/100. 기본 100.
    stabilize_sec : float
        레이저 ON 후 안정화 대기 [초]. 기본 0.5.

    Returns
    -------
    dict
        ok: bool
        length: int           — pixel 수
        max_intensity: float  — 최대 강도
        sum_intensity: float  — 합산 강도
        data: list[int]       — raw intensity 배열
        calibrated: bool      — 캘리브레이션 적용 여부
        raman_shift_cm-1: list[float]  — (calibrated=True 시)
        wavelength_nm: list[float]     — (calibrated=True 시)
        laser_nm: float                — (calibrated=True 시)
    """
    if _ccd is None:
        return {"ok": False, "error": "분광기가 초기화되지 않았습니다."}
    if _laser is None:
        return {"ok": False, "error": "레이저가 초기화되지 않았습니다."}

    valid_powers = {20, 40, 60, 80, 100}
    if power not in valid_powers:
        return {"ok": False, "error": f"유효한 출력값: {valid_powers}"}

    data = None
    try:
        # 1. 레이저 출력 설정 (ND filter motor 이동 — 블로킹)
        _laser.set_power(power)

        # 2. 레이저 ON
        _laser.laser_on()

        # 3. 출력 안정화 대기 (레이저 모드 전환 후 출력 안정화에 필요)
        time.sleep(stabilize_sec)

        # 4. CCD 파라미터 설정
        #    VSSpeed/HSSpeed/PreAmpGain 은 initialize() 에서 이미 설정됨.
        #    여기서는 노출/모드/트리거만 변경.
        _ccd.setup_acquisition(
            read_mode=_READ_MODE_FVB,
            exposure_time=exposure,
            trigger_mode=_TRIGGER_MODE_INTERNAL,
        )

        # 5. 촬영 → dict 반환 (pixel, intensity, + calibrated axes)
        data = _ccd.start_acquisition_cycle()

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        # 6. 레이저 OFF (성공/실패 무관 — 안전 보장)
        try:
            _laser.laser_off()
        except Exception:
            pass

    if data is None:
        return {"ok": False, "error": "CCD 데이터 수집 실패"}

    # ── 결과 조립 ──
    intensity = data["intensity"]
    result = {
        "ok": True,
        "length": len(intensity),
        "max_intensity": float(max(intensity)),
        "sum_intensity": float(sum(intensity)),
        "data": intensity,
        "calibrated": data.get("calibrated", False),
    }

    # 캘리브레이션 된 경우 축 정보 추가
    if data.get("calibrated"):
        result["raman_shift_cm-1"] = data["raman_shift_cm-1"]
        result["wavelength_nm"]    = data["wavelength_nm"]
        result["laser_nm"]         = data["laser_nm"]

    return result



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
