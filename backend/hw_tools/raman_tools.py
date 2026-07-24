"""
Raman 하드웨어 tool 래퍼
- LLM agent가 호출할 수 있는 단순 함수들
- 각 함수는 dict를 반환 (LLM에게 결과를 텍스트로 전달하기 위해)
"""

from __future__ import annotations
import sys
import os
import time
import json
import csv
from pathlib import Path

from backend.config import (
    STAGE_MAX_X, STAGE_MAX_Y,
    CAMERA_WIDTH, CAMERA_HEIGHT,
    LENS_WIDTH_UM, LENS_HEIGHT_UM,
    CALIB_FACTOR_X, CALIB_FACTOR_Y,
)
# 이 모듈에도 save_spectrum(CSV 저장 툴)이 있어 이름이 겹친다 → 별칭으로 import.
from backend.spectrum_store import (
    save_spectrum as _store_save_spectrum,
    list_results as _store_list_results,
    combine_spectra as _store_combine_spectra,
    aggregate_spectra_csv as _store_aggregate_csv,
    bundle_results as _store_bundle_results,
    save_scene as _store_save_scene,
    save_preview_png as _store_save_preview,
)
from backend.analysis_sandbox import run_analysis as _run_analysis
from backend.web_search import web_search as _web_search

STAGE_MIN_Z =  -1.0
STAGE_MAX_Z =   1.0

_stage = None
_laser = None
_ccd   = None
_camera = None

# ── 배경 제거 세션 상태 ──────────────────────────────────────────────────────
_last_spectrum: dict | None = None     # 가장 최근 acquire_spectrum() 결과 캐시
_bg_versions:   dict        = {}       # version_label → 처리 결과


def _cache_and_return(result: dict) -> dict:
    """acquire_spectrum() 결과를 변경하지 않고 _last_spectrum에 캐시 후 그대로 반환."""
    global _last_spectrum
    if result.get("ok") and "data" in result:
        _last_spectrum = result
    return result


def init_hardware(stage=None, laser=None, ccd=None, camera=None):
    """하드웨어 객체를 주입. run_scan.py 등에서 초기화 후 호출."""
    global _stage, _laser, _ccd, _camera
    _stage = stage
    _laser = laser
    _ccd = ccd
    _camera = camera
    print(f"[DEBUG] raman_tools.init_hardware() 호출됨: stage={_stage}, laser={_laser}, ccd={_ccd}, camera={_camera}")


# ──────────────────────────────────────────
# 스테이지
# ──────────────────────────────────────────

def get_stage_speed() -> dict:
    """현재 스테이지 이동 속도를 반환한다. 단위는 mm/s."""
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    try:
        speeds = _stage.get_velocity()
        return {"ok": True, "x_speed_mm_s": speeds[0], "y_speed_mm_s": speeds[1], "z_speed_mm_s": speeds[2]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def set_stage_speed(x_speed_mm_s: float = None, y_speed_mm_s: float = None, z_speed_mm_s: float = None) -> dict:
    """스테이지 이동 속도를 설정한다. 전달되지 않은 축의 속도는 기존 값을 유지한다."""
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
        
    try:
        # 1. 현재 장비에 설정된 속도를 읽어옵니다.
        current_vel = _stage.get_velocity()
        
        if not current_vel.get("ok"):
            return {"ok": False, "error": current_vel.get("error", "Failed to read current velocity")}
            
        # 2. 값이 안 들어왔으면(None) 기존 속도를 그대로 사용합니다.
        vx = x_speed_mm_s if x_speed_mm_s is not None else current_vel["x_speed_mm_s"]
        vy = y_speed_mm_s if y_speed_mm_s is not None else current_vel["y_speed_mm_s"]
        vz = z_speed_mm_s if z_speed_mm_s is not None else current_vel["z_speed_mm_s"]
        
        # 3. 조합된 속도로 컨트롤러에 명령을 내립니다. (va는 사용하지 않으므로 0.0)
        _stage.set_velocity(vx, vy, vz, 0.0)
        
        return {"ok": True, "x_speed_mm_s": vx, "y_speed_mm_s": vy, "z_speed_mm_s": vz}
        
    except Exception as e:
        return {"ok": False, "error": str(e)}

def move_stage(x: float, y: float, z: float = None) -> dict:
    """스테이지를 절대 좌표(mm)로 이동."""
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}

    # 범위 검증
    if not (0 <= x <= STAGE_MAX_X):
        return {"ok": False, "error": f"X out of range: {x} (allowed: 0-{STAGE_MAX_X})"}
    if not (0 <= y <= STAGE_MAX_Y):
        return {"ok": False, "error": f"Y out of range: {y} (allowed: 0-{STAGE_MAX_Y})"}
    if z is not None and not (STAGE_MIN_Z <= z <= STAGE_MAX_Z):
        return {"ok": False, "error": f"Z out of range: {z} (allowed: {STAGE_MIN_Z}-{STAGE_MAX_Z})"}

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
        return {"ok": False, "error": "Stage is not initialized."}
    try:
        pos = _stage.get_position()
        return {"ok": True, "x": pos[0], "y": pos[1], "z": pos[2]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def move_stage_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
    """스테이지를 현재 위치 기준 상대 이동(mm)."""
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
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
        return {"ok": False, "error": "Laser is not initialized."}
    try:
        _laser.laser_on()
        return {"ok": True, "status": "Laser ON"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def laser_off() -> dict:
    """레이저를 끈다."""
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    try:
        _laser.laser_off()
        return {"ok": True, "status": "Laser OFF"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_laser_power(percent: float) -> dict:
    """레이저 출력 설정. percent: ND 필터 투과율 0.004~100 (실수 허용)."""
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    try:
        percent = float(percent)
    except (TypeError, ValueError):
        return {"ok": False, "error": "percent must be a number."}
    lo = getattr(_laser, "ND_MIN_PCT", 0.004)
    hi = getattr(_laser, "ND_MAX_PCT", 100.0)
    if not (lo <= percent <= hi):
        return {"ok": False, "error": f"Valid range: {lo}-{hi} %"}
    try:
        _laser.set_power(percent)
        time.sleep(0.15)  # ND 필터 광학 settling (모터 정지 후 잔류 진동)
        return {"ok": True, "power_percent": percent}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 스펙트럼 수집 (Andor CCD)
# ──────────────────────────────────────────

def _persist_spectrum(result: dict, tag: str = "") -> dict:
    """측정 결과를 로컬(날짜/시간)에 저장하고 result['saved']에 경로·URL을 첨부한다.

    스테이지 좌표를 읽을 수 있으면 메타에 실어 제목/파일명이 좌표별로 붙게 한다
    (예: 10x10 스캔 → (x, y)). 저장 실패가 측정 결과 반환을 막지 않도록 방어적으로 처리.
    """
    meta: dict = {}
    if tag:
        meta["tag"] = tag
    try:
        if _stage is not None:
            pos = _stage.get_position()
            meta["x"], meta["y"] = round(float(pos[0]), 3), round(float(pos[1]), 3)
    except Exception:
        pass
    saved = _store_save_spectrum(result, meta or None)
    if saved.get("ok"):
        result["saved"] = saved
    else:
        result["saved_error"] = saved.get("error")
    return result


def acquire_spectrum(
    exposure: float = 0.2,
    power: float = 40,
    stabilize_sec: float = 0.5,
    acq_mode: str = 'single',
    num_accumulations: int = 1,
    kinetic_count: int = 1,
    kinetic_cycle_time: float = None,
    read_mode: str = 'fvb',
    hbin: int = 1,
    single_track_center: int = None,
    single_track_width: int = 1,
    trigger_mode: str = 'internal',
) -> dict:
    """
    라만 스펙트럼 수집 (Single / Accumulate / Kinetic 모드 지원).

    원자적 실행 흐름:
      1. 레이저 출력 설정 (ND filter motor 이동, 블로킹)
      2. 레이저 ON + 안정화 대기
      3. CCD 취득 모드 설정 (set_aq_* → set_ro_* 순서 필수)
      4. CCD 촬영 (StartAcquisition → 폴링 → GetAcquiredData)
      5. 레이저 OFF (성공/실패 무관하게 반드시 실행)

    Parameters
    ----------
    exposure : float
        CCD 노출 시간 [초]. 기본 0.2.
    power : float
        레이저 출력 [%]. 0.004 ~ 100 사이 임의 실수 (ND 필터 연속 조절). 기본 40.
    stabilize_sec : float
        레이저 ON 후 안정화 대기 [초]. 기본 0.5.
    acq_mode : str
        취득 모드. 'single' | 'accumulate' | 'kinetic'. 기본 'single'.
    num_accumulations : int
        Accumulate/Kinetic 모드에서 프레임당 누적 횟수. 기본 1.
    kinetic_count : int
        Kinetic 모드에서 수집할 총 프레임 수. 기본 1.
    kinetic_cycle_time : float or None
        Kinetic 프레임 간격 [초]. None이면 SDK가 자동 계산. 기본 None.
    read_mode : str
        CCD 읽기 모드. 'fvb' (Full Vertical Binning) | 'single_track'. 기본 'fvb'.
    hbin : int
        수평 비닝 픽셀 수. 기본 1.
    single_track_center : int or None
        read_mode='single_track' 시 중심 픽셀 행 번호 (필수).
    single_track_width : int
        read_mode='single_track' 시 트랙 폭 [픽셀]. 기본 1.
    trigger_mode : str
        트리거 모드. 'internal' | 'external' | 'external_start' |
        'external_exposure' | 'external_fvb_em' | 'software'. 기본 'internal'.

    Returns
    -------
    dict — Single / Accumulate 모드:
        ok, mode, length, max_intensity, sum_intensity, data,
        calibrated, exposure_time, laser_power_pct,
        [num_accumulations,] [raman_shift_cm-1, wavelength_nm, laser_nm]

    dict — Kinetic 모드:
        ok, mode, num_frames, kinetic_count, exposure_time, laser_power_pct,
        frames: list of {frame_index, intensity, length, max_intensity,
                         sum_intensity, calibrated, [raman_shift_cm-1, ...]}
    """
    if _ccd is None:
        return {"ok": False, "error": "The spectrometer (CCD) is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. Measurement becomes available automatically once stabilized."}
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}

    # 레이저 파워는 연속 조절(ND 필터, 0.004~100%). 드라이버 set_power가
    # 임의 %를 펄스 위치로 보간하며, 범위를 벗어나면 스스로 clamp한다.
    if not isinstance(power, (int, float)):
        return {"ok": False, "error": "power must be a number (%)."}
    if not (0.004 <= power <= 100):
        return {"ok": False, "error": "Valid power range: 0.004 - 100 (%)"}

    if acq_mode not in ('single', 'accumulate', 'kinetic'):
        return {"ok": False, "error": "acq_mode must be 'single' | 'accumulate' | 'kinetic'"}

    if read_mode not in ('fvb', 'single_track'):
        return {"ok": False, "error": "read_mode must be 'fvb' | 'single_track'"}

    if read_mode == 'single_track' and single_track_center is None:
        return {"ok": False, "error": "single_track_center is required when read_mode='single_track'"}

    raw = None
    try:
        # 1. 레이저 출력 설정 (ND filter motor 이동 — 블로킹)
        _laser.set_power(power)
        time.sleep(stabilize_sec)

        # 2. 레이저 ON + 안정화 대기
        _laser.laser_on()
        time.sleep(stabilize_sec)

        # 3-a. 취득 모드 설정 — 반드시 set_ro_*() 이전에 실행 (create_buffer가 aq_mode 의존)
        if acq_mode == 'single':
            _ccd.set_aq_single_scan()
            _ccd.set_exposure_time(exposure)
        elif acq_mode == 'accumulate':
            _ccd.set_aq_accumulate_scan(
                exposure_time=exposure,
                num_acc=num_accumulations,
            )
        elif acq_mode == 'kinetic':
            _ccd.set_aq_kinetic_scan(
                exp_time=exposure,
                num_kin=kinetic_count,
                num_acc=num_accumulations if num_accumulations > 1 else None,
                kin_time=kinetic_cycle_time,
            )

        # 3-b. 읽기 모드 설정 — create_buffer() 호출됨 (aq_mode 설정 후여야 함)
        if read_mode == 'fvb':
            _ccd.set_ro_full_vertical_binning(hbin=hbin)
        elif read_mode == 'single_track':
            _ccd.set_ro_single_track(
                center=single_track_center,
                width=single_track_width,
                hbin=hbin,
            )

        # 3-c. 트리거 및 셔터 모드 설정
        _ccd.set_trigger_mode(trigger_mode)
        _ccd.set_shutter_auto()   # 초기화 시 close로 닫힌 셔터를 취득 전 Auto로 복원

        # 3-d. 이전 취득 버퍼 해제 (파라미터 변경 시 SDK 내부 메모리 단편화 방지)
        _ccd.free_internal_memory()

        # 4. 촬영
        if acq_mode == 'kinetic':
            # Kinetic: 버퍼가 3D (num_kin, Ny_ro, Nx_ro) — start_acquisition_cycle() 직접 사용 불가
            _ccd.prepare_acquisition()        # 메모리 사전 할당 + 타이밍 초기화 (첫 프레임 지연 방지)
            _ccd.start_acquisition()
            if trigger_mode == 'software':    # software 트리거 발송 (없으면 ACQUIRING 상태에서 무한 대기)
                _ccd.send_software_trigger()

            # 외부 트리거 미도달 / 하드웨어 장애 시 무한 대기 방지
            _cyc = kinetic_cycle_time if kinetic_cycle_time else (exposure + 0.1)
            _timeout_s = _cyc * kinetic_count * max(num_accumulations, 1) * 2 + 15.0
            _deadline = time.time() + _timeout_s
            while _ccd.get_status() != 'IDLE':
                if time.time() > _deadline:
                    try:
                        _ccd.abort_acquisition()
                    except Exception:
                        pass
                    raise TimeoutError(
                        f"kinetic acquisition timeout: exceeded {_timeout_s:.1f} s "
                        f"(trigger={trigger_mode}, frames={kinetic_count})"
                    )
                time.sleep(0.05)
            raw = _ccd.get_acquired_data()
        else:
            # Single / Accumulate: trigger_mode를 전달하여 software trigger 지원
            # internal 트리거면 무한 대기 허용, 외부/소프트웨어면 deadline 부여
            _timeout_ms = (
                None if trigger_mode == 'internal'
                else int((exposure * max(num_accumulations, 1) * 2 + 15) * 1000)
            )
            raw = _ccd.start_acquisition_cycle(
                trigger_mode_str=trigger_mode,
                timeout_ms=_timeout_ms,
            )

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        # 5. 레이저 OFF (성공/실패 무관 — 안전 보장)
        try:
            _laser.laser_off()
        except Exception:
            pass

    if raw is None:
        return {"ok": False, "error": "CCD data acquisition failed"}

    # ── 결과 조립 ──
    if acq_mode == 'kinetic':
        # raw shape: (num_kin, Ny_ro, Nx_ro) — FVB이면 (num_kin, 1, Nx_ro)
        cal = getattr(_ccd, '_calibrator', None)
        frames = []
        for i in range(raw.shape[0]):
            frame_intensity = raw[i].flatten().tolist()
            frame = {
                "frame_index": i,
                "intensity": frame_intensity,
                "length": len(frame_intensity),
                "max_intensity": float(max(frame_intensity)),
                "sum_intensity": float(sum(frame_intensity)),
                "calibrated": False,
            }
            if cal is not None:
                pixels = range(len(frame_intensity))
                frame.update({
                    "calibrated": True,
                    "raman_shift_cm-1": [float(cal.pixel_to_raman_shift(p)) for p in pixels],
                    "wavelength_nm":    [float(cal.pixel_to_wavelength(p))   for p in pixels],
                    "laser_nm":         float(cal.laser_nm),
                })
            frames.append(frame)
        return _persist_spectrum({
            "ok": True,
            "mode": "kinetic",
            "num_frames": len(frames),
            "kinetic_count": kinetic_count,
            "exposure_time": exposure,
            "laser_power_pct": power,
            "frames": frames,
        })
    else:
        # Single / Accumulate: start_acquisition_cycle()이 calibration dict 반환
        data = raw
        if data.get("error"):
            return {"ok": False, "error": data["error"]}
        intensity = data["intensity"]
        result = {
            "ok": True,
            "mode": acq_mode,
            "length": len(intensity),
            "max_intensity": float(max(intensity)) if intensity else 0.0,
            "sum_intensity": float(sum(intensity)) if intensity else 0.0,
            "data": intensity,
            "calibrated": data.get("calibrated", False),
            "exposure_time": exposure,
            "laser_power_pct": power,
        }
        if acq_mode == 'accumulate':
            result["num_accumulations"] = num_accumulations
        if data.get("calibrated"):
            result["raman_shift_cm-1"] = data["raman_shift_cm-1"]
            result["wavelength_nm"]    = data["wavelength_nm"]
            result["laser_nm"]         = data["laser_nm"]
        return _persist_spectrum(result)

# ──────────────────────────────────────────
# CCD 파라미터 설정 툴
# ──────────────────────────────────────────

def get_ccd_info() -> dict:
    """현재 CCD 설정값 및 상태를 한 번에 조회한다."""
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    try:
        t           = _ccd.get_temperature()
        temp_status = _ccd.get_temperature_status()
        cam_status  = _ccd.get_status()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    def _attr(name):
        return getattr(_ccd, name, None)

    hs_conv = None
    if hasattr(_ccd, 'HSSpeeds_Conventional') and _ccd.HSSpeeds_Conventional:
        hs_conv = _ccd.HSSpeeds_Conventional[0]

    return {
        "ok":                      True,
        "camera_status":           cam_status,
        "temperature_C":           t,
        "temperature_status":      temp_status,
        "cooler_on":               _attr('cooler_on'),
        "exposure_time_s":         _attr('exposure_time'),
        "acquisition_mode":        _attr('aq_mode'),
        "read_mode":               _attr('ro_mode'),
        "num_accumulations":       _attr('num_acc'),
        "num_kinetics":            _attr('num_kin'),
        "em_mode":                 _attr('em_mode'),
        "em_gain":                 _attr('em_gain'),
        "output_amp":              _attr('output_amp'),
        "preamp_gain_index":       _attr('preamp_gain_i'),
        "preamp_gains_available":  _attr('preamp_gains'),
        "vs_speeds_us":            _attr('VSSpeeds'),
        "hs_speeds_conventional_mhz": hs_conv,
        "readout_pixels_Nx":       _attr('Nx_ro'),
        "readout_pixels_Ny":       _attr('Ny_ro'),
        "detector_Nx":             _attr('Nx'),
        "detector_Ny":             _attr('Ny'),
    }


def set_ccd_exposure(exposure_time: float) -> dict:
    """CCD 노출 시간(초)을 설정한다."""
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    if exposure_time <= 0:
        return {"ok": False, "error": "Exposure time must be greater than 0."}
    try:
        actual = _ccd.set_exposure_time(exposure_time)
        return {"ok": True, "exposure_time_s": actual}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_acquisition_mode(
    mode: str,
    num_accumulations: int = None,
    num_kinetics: int = None,
) -> dict:
    """
    CCD 취득 모드를 설정한다.

    mode: 'single' | 'accumulate' | 'kinetic' | 'run_till_abort'
    num_accumulations: accumulate/kinetic 모드에서 누적 횟수
    num_kinetics: kinetic 모드에서 총 프레임 수
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    valid = {'single', 'accumulate', 'kinetic', 'run_till_abort'}
    if mode not in valid:
        return {"ok": False, "error": f"Valid modes: {valid}"}
    try:
        _ccd.set_aq_mode(mode)
        if num_accumulations is not None:
            _ccd.set_num_accumulations(num_accumulations)
        if num_kinetics is not None and mode == 'kinetic':
            _ccd.set_num_kinetics(num_kinetics)
        return {
            "ok":                True,
            "acquisition_mode":  mode,
            "num_accumulations": num_accumulations,
            "num_kinetics":      num_kinetics,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_trigger_mode(mode: str) -> dict:
    """
    CCD 트리거 모드를 설정한다.

    mode: 'internal' | 'external' | 'external_start' |
          'external_exposure' | 'software'
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    valid = {'internal', 'external', 'external_start', 'external_exposure', 'software'}
    if mode not in valid:
        return {"ok": False, "error": f"Valid modes: {valid}"}
    try:
        _ccd.set_trigger_mode(mode)
        return {"ok": True, "trigger_mode": mode}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_read_mode(
    mode: str,
    hbin: int = 1,
    center: int = None,
    width: int = 1,
) -> dict:
    """
    CCD 읽기 모드(readout mode)를 설정한다.

    mode:
      'fvb'          — Full Vertical Binning (1D 스펙트럼, 기본)
      'single_track' — 특정 수직 행 하나만 읽음. center(행 번호) 필수
      'image'        — 2D 이미지 전체

    hbin:   수평 빈닝 계수 (기본 1)
    center: single_track 모드의 중심 행 번호 (1-based)
    width:  single_track 모드의 행 폭 (기본 1)
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    valid = {'fvb', 'single_track', 'image'}
    if mode not in valid:
        return {"ok": False, "error": f"Valid modes: {valid}"}
    try:
        if mode == 'fvb':
            _ccd.set_ro_full_vertical_binning(hbin=hbin)
        elif mode == 'single_track':
            if center is None:
                return {"ok": False, "error": "single_track mode requires the center parameter."}
            _ccd.set_ro_single_track(center=center, width=width, hbin=hbin)
        elif mode == 'image':
            _ccd.set_ro_image_mode(hbin=hbin)
        return {
            "ok":          True,
            "read_mode":   mode,
            "hbin":        hbin,
            "Nx_ro":       _ccd.Nx_ro,
            "Ny_ro":       _ccd.Ny_ro,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_preamp_gain(index: int) -> dict:
    """
    프리앰프(PreAmp) 이득 인덱스를 설정한다.
    사용 가능한 이득 목록은 get_ccd_info()의 preamp_gains_available 참조.
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    try:
        _ccd.set_preamp_gain(index)
        gain_val = _ccd.preamp_gains[index] if _ccd.preamp_gains else None
        return {"ok": True, "preamp_gain_index": index, "gain_value": gain_val}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_em_gain(gain: int) -> dict:
    """
    EM(Electron Multiplication) 이득을 설정한다.
    EM CCD 전용. get_ccd_info()의 em_gain_range 참조.
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    if not getattr(_ccd, 'em_mode', False):
        return {"ok": False, "error": "This camera is not an EM CCD."}
    try:
        _ccd.set_EMCCD_gain(gain)
        return {"ok": True, "em_gain": gain}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_mcp_gain(gain: int) -> dict:
    """
    iStar ICCD 카메라의 MCP(Micro-Channel Plate) 이득을 설정한다.
    허용 범위는 get_mcp_gain_range()로 확인.
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    try:
        low, high = _ccd.get_mcp_gain_range()
        if not (low <= gain <= high):
            return {"ok": False, "error": f"MCP gain out of range: {gain} (allowed: {low}-{high})"}
        _ccd.set_mcp_gain(gain)
        return {"ok": True, "mcp_gain": gain}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_mcp_gain_range() -> dict:
    """iStar ICCD 카메라의 MCP 이득 허용 범위(min, max)를 반환한다."""
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    try:
        low, high = _ccd.get_mcp_gain_range()
        return {"ok": True, "min": low, "max": high}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_output_amp(amp: int) -> dict:
    """
    출력 앰프를 선택한다.
    0 = EMCCD 앰프, 1 = 일반(Conventional) 앰프.
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    if amp not in (0, 1):
        return {"ok": False, "error": "amp must be 0 (EM) or 1 (Conventional)."}
    try:
        _ccd.set_output_amp(amp)
        return {"ok": True, "output_amp": amp, "mode": "EM" if amp == 0 else "Conventional"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_shift_speeds(vs_index: int = None, hs_index: int = None) -> dict:
    """
    수직(VS) 및 수평(HS) 시프트 속도 인덱스를 설정한다.
    사용 가능한 속도 목록은 get_ccd_info()의 vs_speeds_us / hs_speeds_conventional_mhz 참조.
    둘 중 하나만 지정해도 됩니다.
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    result = {"ok": True}
    try:
        if vs_index is not None:
            _ccd.set_vs_speed(vs_index)
            result["vs_speed_index"] = vs_index
            result["vs_speed_us"]    = (_ccd.VSSpeeds[vs_index]
                                        if vs_index < len(_ccd.VSSpeeds) else None)
        if hs_index is not None:
            _ccd.set_hs_speed_conventional(hs_index)
            result["hs_speed_index"] = hs_index
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return result


def set_ccd_temperature(temp: int) -> dict:
    """
    CCD 냉각 목표 온도(°C)를 설정한다.
    실제 안정화는 시간이 걸리며, 상태는 get_ccd_info()로 확인한다.
    일반적 범위: -80 ~ 20°C.
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    try:
        _ccd.set_temperature(temp)
        return {"ok": True, "target_temperature_C": temp}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_cooler(on: bool) -> dict:
    """CCD 냉각기를 켜거나(True) 끈다(False)."""
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    try:
        _ccd.set_cooler(on)
        return {"ok": True, "cooler": "ON" if on else "OFF"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_shutter(mode: str) -> dict:
    """
    셔터 모드를 설정한다.
    'auto'  — 취득 시 자동 열고 닫음 (기본)
    'open'  — 강제로 열어둠
    'close' — 강제로 닫아둠 (다크/배경 측정 시)
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    if mode not in ('auto', 'open', 'close'):
        return {"ok": False, "error": "mode must be one of 'auto', 'open', 'close'."}
    try:
        if mode == 'auto':
            _ccd.set_shutter_auto()
        elif mode == 'open':
            _ccd.set_shutter_open(True)
        elif mode == 'close':
            _ccd.set_shutter_close()
        return {"ok": True, "shutter": mode}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_image_flip(hflip: bool, vflip: bool) -> dict:
    """
    이미지 반전을 설정한다.
    hflip: 수평 좌우 반전 여부
    vflip: 수직 상하 반전 여부
    """
    if _ccd is None:
        return {"ok": False, "error": "CCD is not ready yet - it is cooling (stabilizing at -40 degC) or not connected. It becomes available automatically once stabilized."}
    try:
        _ccd.set_image_flip(hflip=hflip, vflip=vflip)
        return {"ok": True, "hflip": hflip, "vflip": vflip}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def start_camera_stream() -> dict:
    """
    카메라 실시간 스트리밍을 시작합니다.
    USE_camera_stream.py의 StreamingTUCam.start_stream()을 호출합니다.

    반환의 already_streaming: 이 호출 "이전에" 이미 스트리밍 중이었는지.
    호출자가 스트림 소유권을 판단하는 근거다 — 남이(예: 프론트 MJPEG 뷰) 켜 둔
    스트림을 내가 끄면 그쪽 화면이 죽는다. already_streaming=True면 끄지 말 것.
    """
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}

    try:
        # 이미 스트리밍 중인지 확인 (StreamingTUCam 내부 속성 활용)
        if getattr(_camera, 'is_streaming', False):
            return {"ok": True, "already_streaming": True,
                    "status": "Camera is already streaming."}

        _camera.start_stream()
        return {"ok": True, "already_streaming": False,
                "status": "Camera streaming started successfully."}

    except Exception as e:
        return {"ok": False, "error": f"Failed to start streaming: {str(e)}"}

def stop_camera_stream() -> dict:
    """
    카메라 실시간 스트리밍을 중지합니다.
    USE_camera_stream.py의 StreamingTUCam.stop_stream()을 호출합니다.
    """
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}

    try:
        if not getattr(_camera, 'is_streaming', False):
            return {"ok": True, "status": "Camera is not currently streaming."}

        _camera.stop_stream()
        return {"ok": True, "status": "Camera streaming stopped successfully."}

    except Exception as e:
        return {"ok": False, "error": f"Failed to stop streaming: {str(e)}"}


def set_camera_exposure(ms: float) -> dict:
    """카메라(TUCam) 노출 시간(ms)을 설정한다."""
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    if ms <= 0:
        return {"ok": False, "error": "Exposure time must be greater than 0."}
    try:
        _camera.set_exposure(ms)
        return {"ok": True, "exposure_ms": ms}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_camera_auto_exposure(enabled: bool) -> dict:
    """카메라 자동 노출을 활성화(True) 또는 비활성화(False)한다."""
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    try:
        from backend.TuCam.TUCam import TUCAM_Capa_SetValue, TUCAM_IDCAPA
        TUCAM_Capa_SetValue(
            _camera.TUCAMOPEN.hIdxTUCam,
            TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value,
            1 if enabled else 0,
        )
        return {"ok": True, "auto_exposure": enabled}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def capture_camera_frame() -> dict:
    """
    카메라에서 최신 프레임 1장을 캡처하여 통계 정보와 선명도 점수를 반환한다.
    스트리밍이 활성화되어 있어야 한다.
    선명도 점수(sharpness_score)는 라플라시안 분산으로 계산된다 — 오토포커스 시 활용 가능.
    """
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    try:
        import numpy as np
        import cv2
        frame = _camera.get_latest_frame()
        if frame is None:
            return {"ok": False, "error": "Failed to acquire frame (check whether streaming is active)"}

        gray = (frame.astype(np.float32)
                if frame.ndim == 2
                else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
        sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))

        return {
            "ok": True,
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "min_intensity": float(frame.min()),
            "max_intensity": float(frame.max()),
            "mean_intensity": float(frame.mean()),
            "sharpness_score": sharpness,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 레이저 — 가이드빔
# ──────────────────────────────────────────

def set_guide_beam_mode() -> dict:
    """
    레이저를 가이드빔 대기 상태로 전환한다.
    - 빔 스플리터(축04) → 대기 위치
    - ND 필터(축02) → 메인 빔 차단 위치
    측정 레이저 미사용 시 시편 정렬·초점 확인에 활용한다.
    """
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    try:
        _laser.set_guide_beam()
        return {"ok": True, "status": "Switched to guide-beam mode"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 오토포커스 (카메라 선명도 기반 Z 스윕)
# ──────────────────────────────────────────

def run_autofocus(
    initial_z: float = None,
    step_size: float = 0.030,
    min_step: float = 0.001,
    max_steps: int = 100,
) -> dict:
    """
    가이드빔 레이저 스팟 면적 최소화 기반 힐클라이밍 오토포커스.
    USE_autofocus_local.py의 AutoFocusLocal 알고리즘을 헤드리스로 실행한다.

    동작 원리:
      각 Z 위치에서 레이저 OFF → 배경 프레임 취득 → 레이저 ON → 레이저 프레임 취득
      → clip 차분 → GaussianBlur → Otsu threshold → 스팟 픽셀 수(면적) 계산
      → 면적이 작을수록(레이저 스팟이 날카로울수록) 초점이 맞음
      → 적응형 힐클라이밍: 개선되면 같은 방향 전진, 나빠지면 방향 반전 + 보폭 절반
      → 역대 최솟값(global_best_z) 위치로 최종 귀환

    Parameters
    ----------
    initial_z : float, optional
        탐색 시작 Z 위치(mm). None이면 현재 Z 유지.
    step_size : float
        초기 Z 이동 보폭(mm). 기본 0.030mm (30µm).
    min_step : float
        최소 보폭(mm) — 이 이하면 탐색 종료. 기본 0.001mm (1µm).
    max_steps : int
        최대 스텝 수 — 초과 시 강제 종료. 기본 100.

    Returns
    -------
    dict
        optimal_z, best_area_px, step_count, z_scores, current_position
    """
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}

    try:
        import numpy as np
        import cv2

        pos = _stage.get_position()
        cur_x, cur_y, cur_z = pos[0], pos[1], pos[2]
        cur_a = pos[3] if len(pos) > 3 else 0

        if initial_z is not None:
            cur_z = initial_z
            _stage.move_absolute(cur_x, cur_y, cur_z, cur_a)
            time.sleep(0.3)

        # 가이드빔 모드 + 카메라 스트리밍 보장
        _laser.set_guide_beam()
        if not getattr(_camera, 'is_streaming', False):
            _camera.start_stream()

        def _to_uint8(frame):
            img = frame.copy()
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img

        def _flush(n=3):
            for _ in range(n):
                _camera.get_latest_frame()

        def _capture_spot_area() -> int:
            """레이저 OFF/ON 차분 → Otsu → 스팟 픽셀 수 반환."""
            _laser.laser_off()
            _flush(3)
            refs = [_to_uint8(f) for _ in range(3) if (f := _camera.get_latest_frame()) is not None]
            ref = np.mean(refs, axis=0).astype(np.uint8) if refs else None

            _laser.laser_on()
            _flush(3)
            lfs = [_to_uint8(f) for _ in range(3) if (f := _camera.get_latest_frame()) is not None]
            laser_frame = np.mean(lfs, axis=0).astype(np.uint8) if lfs else None

            if ref is None or laser_frame is None:
                return 0
            diff = np.clip(laser_frame.astype(np.int16) - ref.astype(np.int16), 0, 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(diff, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return int(np.count_nonzero(binary))

        # 힐클라이밍 상태
        best_z = cur_z
        best_area = float('inf')
        direction = 1
        step_count = 0
        global_best_area = float('inf')
        global_best_z = cur_z
        z_scores: list = []
        sweep_state = 'init'

        while sweep_state != 'done':
            pos_now = _stage.get_position()
            z_now = pos_now[2] if pos_now else cur_z

            area = _capture_spot_area()
            z_scores.append({"z": round(z_now, 4), "area_px": area})

            if 0 < area < global_best_area:
                global_best_area = area
                global_best_z = z_now

            if sweep_state == 'init':
                best_area = area
                best_z = z_now
                _stage.move_absolute(cur_x, cur_y, best_z + direction * step_size, cur_a)
                time.sleep(0.3)
                sweep_state = 'check'

            elif sweep_state == 'check':
                step_count += 1
                if area < best_area:
                    best_area = area
                    best_z = z_now
                    _stage.move_absolute(cur_x, cur_y, best_z + direction * step_size, cur_a)
                    time.sleep(0.3)
                else:
                    direction *= -1
                    step_size /= 2.0
                    if step_size < min_step or step_count >= max_steps:
                        sweep_state = 'done'
                    else:
                        _stage.move_absolute(cur_x, cur_y, best_z + direction * step_size, cur_a)
                        time.sleep(0.3)

        # 역대 최솟값 위치로 최종 귀환
        _stage.move_absolute(cur_x, cur_y, global_best_z, cur_a)
        time.sleep(0.5)

        try:
            _laser.laser_off()
        except Exception:
            pass

        return {
            "ok": True,
            "optimal_z": global_best_z,
            "best_area_px": global_best_area,
            "step_count": step_count,
            "z_scores": z_scores,
            "current_position": {"x": cur_x, "y": cur_y, "z": global_best_z},
        }
    except Exception as e:
        try:
            _laser.laser_off()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 데이터 저장 / 로드
# ──────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ──────────────────────────────────────────
# 배경 제거 (IPBSA)
# ──────────────────────────────────────────

def _ipbsa(intensity, poly_order=5, max_iterations=100, threshold=0.001):
    import numpy as np
    y = np.array(intensity, dtype=np.float64)
    n = len(y)
    x = np.linspace(0.0, 1.0, n)
    working = y.copy()
    prev_bg = np.zeros(n, dtype=np.float64)
    converged = False
    for i in range(max_iterations):
        coeffs = np.polyfit(x, working, deg=poly_order)
        bg = np.polyval(coeffs, x)
        working = np.minimum(y, bg)
        denom = np.linalg.norm(prev_bg)
        if denom > 0 and np.linalg.norm(bg - prev_bg) / denom < threshold:
            converged = True
            break
        prev_bg = bg.copy()
    corrected = np.clip(y - bg, 0.0, None)
    return corrected.tolist(), bg.tolist(), i + 1, converged


def apply_background_subtraction(
    poly_order: int = 5,
    max_iterations: int = 100,
    threshold: float = 0.001,
    source: str = "last",
    version_label: str = "default",
    save_result: bool = False,
) -> dict:
    """IPBSA(반복 다항식 배경 제거)를 수행하고 결과를 _bg_versions에 저장한다."""
    global _bg_versions

    if not (2 <= poly_order <= 10):
        return {"ok": False, "error": f"poly_order must be 2-10 (got: {poly_order})"}
    if not (10 <= max_iterations <= 500):
        return {"ok": False, "error": f"max_iterations must be 10-500 (got: {max_iterations})"}
    if not (0.001 <= threshold <= 1.0):
        return {"ok": False, "error": f"threshold must be 0.001-1.0 (got: {threshold})"}

    intensity: list = []
    raman_shift = None

    if source == "last":
        if _last_spectrum is None:
            return {
                "ok": False,
                "error": "No saved spectrum. Call acquire_spectrum() first.",
            }
        if "data" not in _last_spectrum:
            return {
                "ok": False,
                "error": "The last spectrum is in Kinetic mode. This applies only to Single/Accumulate spectra.",
            }
        intensity = _last_spectrum["data"]
        raman_shift = _last_spectrum.get("raman_shift_cm-1")
    else:
        filepath = Path(source)
        if not filepath.is_absolute():
            filepath = _DATA_DIR / source
        if not filepath.exists():
            return {"ok": False, "error": f"File not found: {filepath}"}
        try:
            if filepath.suffix.lower() == ".json":
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if "data" in loaded:
                    intensity = loaded["data"]
                elif "corrected_data" in loaded:
                    intensity = loaded["corrected_data"]
                else:
                    return {"ok": False, "error": "The JSON file has no 'data' or 'corrected_data' key."}
                raman_shift = loaded.get("raman_shift_cm-1")
            elif filepath.suffix.lower() == ".csv":
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if not rows:
                    return {"ok": False, "error": "The CSV file is empty."}
                if "corrected_intensity" in rows[0]:
                    intensity = [float(r["corrected_intensity"]) for r in rows]
                elif "intensity" in rows[0]:
                    intensity = [float(r["intensity"]) for r in rows]
                else:
                    return {"ok": False, "error": "The CSV has no 'intensity' column."}
                if "raman_shift_cm-1" in rows[0]:
                    raman_shift = [float(r["raman_shift_cm-1"]) for r in rows]
            else:
                return {"ok": False, "error": "Unsupported file format (only JSON or CSV allowed)."}
        except Exception as e:
            return {"ok": False, "error": f"File load error: {e}"}

    if not intensity:
        return {"ok": False, "error": "The spectrum intensity array is empty."}
    if len(intensity) < poly_order + 1:
        return {
            "ok": False,
            "error": (
                f"Spectrum length ({len(intensity)}) is smaller than poly_order+1 ({poly_order + 1}). "
                "Lower the polynomial order."
            ),
        }

    try:
        corrected, background, iterations_run, converged = _ipbsa(
            intensity=intensity,
            poly_order=poly_order,
            max_iterations=max_iterations,
            threshold=threshold,
        )
    except Exception as e:
        return {"ok": False, "error": f"IPBSA algorithm error: {e}"}

    saved_path = None
    if save_result:
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            save_filepath = _DATA_DIR / f"bg_corrected_{version_label}.csv"
            with open(save_filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if raman_shift is not None:
                    writer.writerow(["pixel_index", "raman_shift_cm-1", "corrected_intensity", "background_intensity"])
                    for idx, (rs, ci, bi) in enumerate(zip(raman_shift, corrected, background)):
                        writer.writerow([idx, rs, ci, bi])
                else:
                    writer.writerow(["pixel_index", "corrected_intensity", "background_intensity"])
                    for idx, (ci, bi) in enumerate(zip(corrected, background)):
                        writer.writerow([idx, ci, bi])
            saved_path = str(save_filepath)
        except Exception:
            pass

    result = {
        "ok": True,
        "version_label": version_label,
        "poly_order": poly_order,
        "max_iterations": max_iterations,
        "threshold": threshold,
        "iterations_run": iterations_run,
        "converged": converged,
        "max_corrected_intensity": float(max(corrected)) if corrected else 0.0,
        "max_background_intensity": float(max(background)) if background else 0.0,
        "corrected_data": corrected,
        "background_data": background,
    }
    if raman_shift is not None:
        result["raman_shift_cm-1"] = raman_shift
    if saved_path is not None:
        result["saved_path"] = saved_path

    _bg_versions[version_label] = result.copy()
    return result


def list_bg_versions() -> dict:
    """저장된 모든 배경 제거 결과 버전의 목록과 주요 통계를 반환한다."""
    if not _bg_versions:
        return {
            "ok": True,
            "count": 0,
            "versions": [],
            "message": "No saved versions. Call apply_background_subtraction() first.",
        }
    summaries = []
    for label, v in _bg_versions.items():
        summaries.append({
            "version_label":            label,
            "poly_order":               v.get("poly_order"),
            "max_iterations":           v.get("max_iterations"),
            "threshold":                v.get("threshold"),
            "iterations_run":           v.get("iterations_run"),
            "converged":                v.get("converged"),
            "max_corrected_intensity":  v.get("max_corrected_intensity"),
            "max_background_intensity": v.get("max_background_intensity"),
            "has_raman_shift":          "raman_shift_cm-1" in v,
            "data_length":              len(v.get("corrected_data", [])),
        })
    return {"ok": True, "count": len(summaries), "versions": summaries}


def get_bg_version(version_label: str) -> dict:
    """특정 버전의 배경 제거 결과 전체 데이터를 반환한다."""
    if version_label not in _bg_versions:
        return {
            "ok": False,
            "error": f"Version '{version_label}' not found.",
            "available_versions": list(_bg_versions.keys()),
        }
    return {"ok": True, **_bg_versions[version_label]}


def save_spectrum(
    data: list,
    filename: str,
    raman_shift: list = None,
    wavelength_nm: list = None,
    metadata: dict = None,
) -> dict:
    """
    스펙트럼 강도 배열을 CSV 파일로 저장한다.

    Parameters
    ----------
    data : list[float]
        강도(intensity) 배열.
    filename : str
        저장 파일명 (.csv 확장자 없어도 됨).
    raman_shift : list[float], optional
        라만 시프트 축 [cm⁻¹].
    wavelength_nm : list[float], optional
        파장 축 [nm].
    metadata : dict, optional
        추가 메타데이터 — 같은 이름의 .json 파일로 저장.
    """
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not filename.endswith(".csv"):
            filename += ".csv"
        filepath = _DATA_DIR / filename

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if raman_shift is not None and wavelength_nm is not None:
                writer.writerow(["pixel_index", "raman_shift_cm-1", "wavelength_nm", "intensity"])
                for i, (rs, wl, v) in enumerate(zip(raman_shift, wavelength_nm, data)):
                    writer.writerow([i, rs, wl, v])
            elif raman_shift is not None:
                writer.writerow(["pixel_index", "raman_shift_cm-1", "intensity"])
                for i, (rs, v) in enumerate(zip(raman_shift, data)):
                    writer.writerow([i, rs, v])
            else:
                writer.writerow(["pixel_index", "intensity"])
                for i, v in enumerate(data):
                    writer.writerow([i, v])

        if metadata:
            with open(filepath.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        return {
            "ok": True,
            "filepath": str(filepath),
            "num_points": len(data),
            "has_calibration": raman_shift is not None,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def load_spectrum(filename: str) -> dict:
    """
    저장된 스펙트럼 CSV 파일을 로드한다.
    절대 경로 또는 data/ 디렉토리 상대 경로 모두 허용.
    """
    try:
        if not filename.endswith(".csv"):
            filename += ".csv"
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = _DATA_DIR / filename
        if not filepath.exists():
            return {"ok": False, "error": f"File not found: {filepath}"}

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = list(reader)

        intensity = [float(r["intensity"]) for r in rows]
        result: dict = {
            "ok": True,
            "filename": str(filepath),
            "num_points": len(intensity),
            "headers": headers,
            "intensity": intensity,
        }
        if "raman_shift_cm-1" in headers:
            result["raman_shift_cm-1"] = [float(r["raman_shift_cm-1"]) for r in rows]
        if "wavelength_nm" in headers:
            result["wavelength_nm"] = [float(r["wavelength_nm"]) for r in rows]
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 세션 / 포인트 데이터 관리
# ──────────────────────────────────────────

def create_session(session_id: str) -> dict:
    """
    새 실험 세션 디렉토리를 생성하고 메타데이터를 초기화한다.
    이후 save_point_data()로 포인트별 데이터를 저장할 수 있다.
    """
    try:
        session_dir = _DATA_DIR / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "session_id": session_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "points": [],
        }
        with open(session_dir / "session.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return {
            "ok": True,
            "session_id": session_id,
            "session_dir": str(session_dir),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def save_point_data(
    session_id: str,
    point_id: str,
    spectrum_data: list = None,
    raman_shift: list = None,
    position: dict = None,
) -> dict:
    """
    실험 세션의 특정 포인트 데이터(스펙트럼 + 위치)를 저장한다.

    Parameters
    ----------
    session_id : str
        create_session()으로 만든 세션 ID.
    point_id : str
        포인트 식별자 (예: 'P001').
    spectrum_data : list[float], optional
        강도 배열.
    raman_shift : list[float], optional
        라만 시프트 축 [cm⁻¹].
    position : dict, optional
        스테이지 위치 {'x': ..., 'y': ..., 'z': ...}.
    """
    try:
        session_dir = _DATA_DIR / "sessions" / session_id
        if not session_dir.exists():
            return {"ok": False, "error": f"No such session: {session_id}. Call create_session first."}

        saved_files: list[str] = []

        if spectrum_data is not None:
            spec_path = session_dir / f"{point_id}_spectrum.csv"
            with open(spec_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if raman_shift is not None:
                    writer.writerow(["pixel_index", "raman_shift_cm-1", "intensity"])
                    for i, (rs, v) in enumerate(zip(raman_shift, spectrum_data)):
                        writer.writerow([i, rs, v])
                else:
                    writer.writerow(["pixel_index", "intensity"])
                    for i, v in enumerate(spectrum_data):
                        writer.writerow([i, v])
            saved_files.append(str(spec_path))

        if position is not None:
            pos_path = session_dir / f"{point_id}_position.json"
            with open(pos_path, "w", encoding="utf-8") as f:
                json.dump({"point_id": point_id, "position": position}, f, ensure_ascii=False, indent=2)
            saved_files.append(str(pos_path))

        # 세션 메타 업데이트
        meta_path = session_dir / "session.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {"session_id": session_id, "points": []}

        entry = {
            "point_id": point_id,
            "has_spectrum": spectrum_data is not None,
            "has_position": position is not None,
            "saved_files": saved_files,
        }
        points: list = meta.get("points", [])
        idx = next((i for i, p in enumerate(points) if p["point_id"] == point_id), None)
        if idx is not None:
            points[idx] = entry
        else:
            points.append(entry)
        meta["points"] = points
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return {
            "ok": True,
            "session_id": session_id,
            "point_id": point_id,
            "saved_files": saved_files,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def capture_scene() -> dict:
    """현재 현미경(카메라) 화면을 저장한다 — run_analysis 가 이 위에 피크맵을 오버레이한다.

    스테이지 위치와 렌즈 FOV(µm)로 이미지의 스테이지 좌표 범위(extent, mm)를 계산해
    함께 저장하므로, 분석 코드에서 imshow(microscope_image, extent=image_extent) 후
    측정 (x,y)를 그 위에 정합해 찍을 수 있다. 카메라 스트리밍이 켜져 있어야 한다.
    """
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    import numpy as np
    import cv2
    frame = _camera.get_latest_frame()
    if frame is None:
        return {"ok": False, "error": "No camera frame. Start streaming first."}
    img = frame.copy()
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 3:                       # BGR → RGB (matplotlib 표시 기준)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    extent = None
    try:
        if _stage is not None:
            pos = _stage.get_position()
            cx, cy = float(pos[0]), float(pos[1])          # mm
            w_mm, h_mm = LENS_WIDTH_UM / 1000.0, LENS_HEIGHT_UM / 1000.0
            extent = [cx - w_mm / 2, cx + w_mm / 2, cy - h_mm / 2, cy + h_mm / 2]
    except Exception:
        pass

    saved = _store_save_scene(img, extent, {})
    if not saved.get("ok"):
        return saved
    return {"ok": True, "image_url": saved["image_url"], "extent": extent,
            "shape": list(img.shape),
            # saved 를 실으면 spectrum_event 배선을 타 캡처한 화면이 채팅에 바로 표시된다.
            "saved": {"title": "Microscope view capture", "image_url": saved["image_url"]},
            "note": "Usable later in run_analysis via microscope_image / image_extent."}


def analyze_microscope_image(question: str = "Find a specific object in the sample (e.g. a cell) and report its center-point pixel coordinates.") -> dict:
    """
    TuCam 현미경 카메라 화면을 PNG (Base64)로 캡처하여 반환.

    [왜 CAMERA_WIDTH×CAMERA_HEIGHT로 리사이즈하는가 — 두 가지 이유]

    1) 좌표계 일치 (기능 버그 수정)
       get_latest_frame()은 센서 네이티브 해상도를 그대로 준다(Config.ini의
       Width/Height는 뷰 기준 해상도일 뿐 프레임 크기가 아니다). 그런데 이 이미지를
       보고 vision LLM이 찍은 픽셀 좌표는 결국 move_to_pixel()로 들어가고,
       move_to_pixel은 이미지 중심을 (CAMERA_WIDTH/2, CAMERA_HEIGHT/2)로 가정해
       계산한다. 두 해상도가 다르면 스테이지가 엉뚱한 곳으로 이동한다.
       USE_scan.py도 같은 이유로 픽셀→스테이지 계산 전에 이 크기로 리사이즈한다.
       → 여기서 미리 맞춰 두면 이 함수의 출력 좌표계 = move_to_pixel의 입력 좌표계.

    2) API 이미지 제한
       Anthropic API는 이미지 1장당 base64 10MB, 긴 변 2576px가 상한이고 그보다 큰
       이미지는 어차피 서버에서 다운스케일된다. 네이티브 프레임을 무손실 PNG로 보내면
       12MB를 넘겨 요청 자체가 거부됐다(실제로 그랬다). 1060×800이면 ~2MB, 긴 변도
       상한 이하라 다운스케일이 없어 좌표가 보낸 그대로 유지된다.
    """
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    try:
        import base64
        import numpy as np
        import cv2
        frame = _camera.get_latest_frame()
        if frame is None:
            return {"ok": False, "error": "Failed to acquire frame (check whether streaming is active)"}

        if frame.dtype == np.uint16:
            frame = (frame >> 8).astype(np.uint8)

        if frame.ndim == 2:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_bgr = frame.copy()

        # 뷰 기준 해상도로 정규화 — 위 docstring의 (1)(2)
        if frame_bgr.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
            frame_bgr = cv2.resize(frame_bgr, (CAMERA_WIDTH, CAMERA_HEIGHT),
                                   interpolation=cv2.INTER_AREA)

        height, width = frame_bgr.shape[:2]
        ret, buf = cv2.imencode('.png', frame_bgr)
        enhanced_question = f"{question}\n\n[The attached image has an original resolution of {width}px wide by {height}px tall. When returning pixel coordinates, give exact pixel values based on this resolution.][Note: you return pixel coordinates, which are NOT stage coordinates. To move the stage to that location, you must use the move_to_pixel(pixel_x, pixel_y) function.]"

        if not ret:
            return {"ok": False, "error": "PNG encoding failed"}
        img_b64 = base64.b64encode(buf).decode('utf-8')

        return {
            "ok": True,
            "image_base64": img_b64,
            "question": enhanced_question,
            "width": width,
            "height": height,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


_UM_PER_PX_X = LENS_WIDTH_UM  / CAMERA_WIDTH   # µm/px  (305/1060)
_UM_PER_PX_Y = LENS_HEIGHT_UM / CAMERA_HEIGHT  # µm/px  (230/800)
_SIGN_X = -1  # pixel +X(right) → stage -X
_SIGN_Y = +1  # pixel +Y(down)  → stage +Y


def move_to_pixel(pixel_x: int, pixel_y: int) -> dict:
    """
    카메라 이미지의 픽셀 좌표를 스테이지 mm 좌표로 변환해 이동한다.
    이미지 중심(CAMERA_WIDTH/2, CAMERA_HEIGHT/2)이 현재 스테이지 위치에 대응한다.
    """
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    try:
        pos = _stage.get_position()
        if pos is None:
            return {"ok": False, "error": "Failed to query stage position"}
        
        dx_px = pixel_x - (CAMERA_WIDTH  / 2.0)
        dy_px = pixel_y - (CAMERA_HEIGHT / 2.0)
        dx_mm = dx_px * _UM_PER_PX_X * CALIB_FACTOR_X / 1000.0 * _SIGN_X
        dy_mm = dy_px * _UM_PER_PX_Y * CALIB_FACTOR_Y / 1000.0 * _SIGN_Y
        return move_stage(x=round(pos[0] + dx_mm, 4),
                          y=round(pos[1] + dy_mm, 4))
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────
# 그리드 매핑 — 5×5 같은 격자 스캔을 도구 1~2개로 압축
# ──────────────────────────────────────────
# [배경] 예전엔 에이전트가 25점 격자를 move→autofocus→acquire "직접" 75회로 돌렸다.
# 매 스텝 커지는 히스토리를 매번 통째로 재전송해 토큰이 사실상 제곱으로 불고(1턴 ~8.5k
# 토큰), 스텝마다 LLM 추론이 끼어 633초가 걸렸다. 아래 두 도구는 그 루프를 파이썬으로
# 내려 토큰·지연·재전송을 없앤다:
#   preview_grid_scan : 격자 위치를 '현재 카메라 화면'에 원으로 오버레이해 에이전트에게
#                       보여주기만 한다(이동·조사 없음). 에이전트가 눈으로 보고 승인/수정.
#   run_grid_scan     : 승인 후 실제 격자 스캔을 내부 루프로 수행, 압축 요약 1개만 반환.

# 오버레이 원 크기 — 프론트 카메라 뷰의 '레이저 조사점' 빨간 링(CameraView.tsx: w-4 ≈ 16px)과
# 대략 같은 크기로 그린다. 물리적 빔 지름이 아니라 조사점 표식이다(사용자 지시: 대충 그 크기).
_GRID_SPOT_RADIUS_PX = 14

# run_grid_scan 한 번이 낼 수 있는 누적 조사량 상한(mJ). 에이전트 층의 per-turn 회로차단기는
# 도구 이름이 acquire_spectrum일 때만 도는데 run_grid_scan은 내부에서 N번 조사하므로,
# 여기서 독립적으로 한 번 더 막는다(방어적 이중화). 에이전트 층에도 별도 누계 반영을 둔다.
_GRID_MAX_DOSE_MJ = 1000.0

# 격자 점 개수 안전 상한(폭주 방지).
_GRID_MAX_POINTS = 400

# mm/px 크기(양수). move_to_pixel과 동일 상수에서 유도 — 부호는 _SIGN_*로 따로 적용한다.
_MM_PER_PX_X = _UM_PER_PX_X * CALIB_FACTOR_X / 1000.0
_MM_PER_PX_Y = _UM_PER_PX_Y * CALIB_FACTOR_Y / 1000.0


def _grid_stage_coords(center_x: float, center_y: float,
                       rows: int, cols: int, spacing_mm: float) -> list:
    """(center_x,center_y) 중심 대칭 rows×cols 격자의 스테이지 좌표를 행 우선(raster)으로.
    반환 각 항목: (index, row, col, x_mm, y_mm)."""
    pts = []
    idx = 0
    for i in range(rows):
        dy = (i - (rows - 1) / 2.0) * spacing_mm
        for j in range(cols):
            dx = (j - (cols - 1) / 2.0) * spacing_mm
            pts.append((idx, i, j, round(center_x + dx, 4), round(center_y + dy, 4)))
            idx += 1
    return pts


def _mm_to_pixel(sx: float, sy: float, cx: float, cy: float):
    """스테이지 mm 좌표(sx,sy)를, 현재 위치(cx,cy)가 화면 중심에 오는 카메라
    이미지(CAMERA_WIDTH×CAMERA_HEIGHT) 픽셀로 투영한다. move_to_pixel의 역변환."""
    px = CAMERA_WIDTH  / 2.0 + (sx - cx) * _SIGN_X / _MM_PER_PX_X
    py = CAMERA_HEIGHT / 2.0 + (sy - cy) * _SIGN_Y / _MM_PER_PX_Y
    return px, py


def _validate_grid_args(rows, cols, spacing_mm):
    """preview/run 공통 인자 검증 — 실패 시 error dict, 통과 시 None."""
    if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
        return {"ok": False, "error": "rows and cols must be integers >= 1."}
    if rows * cols > _GRID_MAX_POINTS:
        return {"ok": False, "error": f"Too many points ({rows*cols}); max {_GRID_MAX_POINTS}."}
    try:
        if float(spacing_mm) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return {"ok": False, "error": "spacing_mm must be a number > 0."}
    return None


def preview_grid_scan(rows: int, cols: int, spacing_mm: float,
                      center_x: float = None, center_y: float = None) -> dict:
    """격자 스캔 '미리보기'. 스테이지 이동·레이저 조사 없이, rows×cols 격자 위치를 현재
    카메라 화면에 원으로 오버레이한 이미지를 반환한다(에이전트가 보고 승인/수정하도록).

    center_* 미지정 시 현재 스테이지 위치를 격자 중심으로 쓴다. 화면 중심 = 현재 스테이지
    위치이므로, 중심을 현재 위치로 두면 격자가 화면 중앙에 대칭으로 그려진다. 카메라 시야(FOV)는
    좁아(≈0.43×0.30mm) 격자가 넓으면 일부 점은 화면 밖이다 — 화면 안 점만 세어 함께 알려준다.
    """
    err = _validate_grid_args(rows, cols, spacing_mm)
    if err:
        return err
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    try:
        import base64
        import numpy as np
        import cv2

        spacing_mm = float(spacing_mm)
        pos = _stage.get_position()
        if pos is None:
            return {"ok": False, "error": "Failed to query stage position"}
        cur_x, cur_y = float(pos[0]), float(pos[1])          # 화면 중심 = 현재 위치
        cx = cur_x if center_x is None else float(center_x)  # 격자 중심
        cy = cur_y if center_y is None else float(center_y)

        frame = _camera.get_latest_frame()
        if frame is None:
            return {"ok": False, "error": "Failed to acquire frame (check whether streaming is active)"}
        if frame.dtype == np.uint16:
            frame = (frame >> 8).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        if frame_bgr.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
            frame_bgr = cv2.resize(frame_bgr, (CAMERA_WIDTH, CAMERA_HEIGHT),
                                   interpolation=cv2.INTER_AREA)

        pts = _grid_stage_coords(cx, cy, rows, cols, spacing_mm)
        n_in_view = 0
        for idx, i, j, sx, sy in pts:
            px, py = _mm_to_pixel(sx, sy, cur_x, cur_y)
            ipx, ipy = int(round(px)), int(round(py))
            if 0 <= ipx < CAMERA_WIDTH and 0 <= ipy < CAMERA_HEIGHT:
                n_in_view += 1
            # 화면 밖 점도 cv2.circle이 자동 클립하므로 그냥 그린다(가장자리 힌트).
            cv2.circle(frame_bgr, (ipx, ipy), _GRID_SPOT_RADIUS_PX, (0, 255, 0), 2)      # green: 스캔 점
        # (요청) 미리보기에는 스캔 예정점(초록 원)만 표시한다. 화면 중심의 '현재 조사점'
        # 빨간 링은 스캔 계획과 무관해 혼동을 줄 수 있어 그리지 않는다.

        ret, buf = cv2.imencode('.png', frame_bgr)
        if not ret:
            return {"ok": False, "error": "PNG encoding failed"}
        img_b64 = base64.b64encode(buf).decode('utf-8')
        # 같은 오버레이를 파일로도 저장해 image_url을 실으면, spectrum_event 배선을 타고
        # 이 미리보기가 채팅창에 그대로 인라인 표시된다(프론트 수정 불필요). 에이전트는 위의
        # image_base64로 '보고' 판단하고, 사람은 채팅에서 같은 그림을 본다.
        saved_img = _store_save_preview(buf.tobytes(), tag="grid_preview")

        fov_x, fov_y = CAMERA_WIDTH * _MM_PER_PX_X, CAMERA_HEIGHT * _MM_PER_PX_Y
        span_x, span_y = (cols - 1) * spacing_mm, (rows - 1) * spacing_mm
        question = (
            f"Grid scan PREVIEW (not executed yet): {rows}x{cols} = {rows*cols} points, "
            f"{spacing_mm} mm spacing, centered at stage (X={cx:.4f}, Y={cy:.4f}) mm. "
            f"Green circles mark the points to be scanned. "
            f"Grid span {span_x:.3f}x{span_y:.3f} mm vs camera view ~{fov_x:.3f}x{fov_y:.3f} mm; "
            f"{n_in_view}/{rows*cols} points fall within the current view (the rest are outside the frame "
            f"but will still be measured). If this layout looks right, call run_grid_scan with the SAME "
            f"parameters. If not, call preview_grid_scan again with adjusted rows/cols/spacing_mm/center."
        )
        out = {
            "ok": True,
            "rows": rows, "cols": cols, "spacing_mm": spacing_mm,
            "center": {"x": round(cx, 4), "y": round(cy, 4)},
            "n_points": rows * cols, "n_in_view": n_in_view,
            "fov_mm": {"x": round(fov_x, 4), "y": round(fov_y, 4)},
            "image_base64": img_b64,
            "question": question,
        }
        if saved_img.get("ok"):
            out["saved"] = {"title": f"Grid preview {rows}x{cols} @ {spacing_mm}mm",
                            "image_url": saved_img["image_url"]}
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_grid_scan(rows: int, cols: int, spacing_mm: float,
                  center_x: float = None, center_y: float = None,
                  autofocus: str = "each", exposure: float = 0.2, power: float = 40) -> dict:
    """격자 스캔 '실행'. rows×cols 격자를 내부 루프로 순회하며 각 점에서
    이동→(오토포커스)→스펙트럼 측정·자동저장하고, 압축 요약 1개만 반환한다.

    autofocus:
      "each"   — 매 점에서 오토포커스(가장 정확, 느림; 예전 수동 방식과 동일)
      "center" — 격자 중심에서 1회만 오토포커스 후 그 Z로 전체 측정(빠름, 평탄 시료용)
      "none"   — 오토포커스 없이 현재 Z로 측정
    레이저 조사가 실제로 일어나므로, 예상 누적 조사량이 상한을 넘으면 시작 전에 거부한다.
    """
    err = _validate_grid_args(rows, cols, spacing_mm)
    if err:
        return err
    if autofocus not in ("each", "center", "none"):
        return {"ok": False, "error": "autofocus must be one of: each, center, none."}
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    if _ccd is None:
        return {"ok": False, "error": "CCD is not initialized (cooling or not connected)."}
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    if autofocus != "none" and _camera is None:
        return {"ok": False, "error": "Camera is not initialized (required for autofocus). "
                                      "Use autofocus='none' to skip."}

    spacing_mm = float(spacing_mm)
    exposure, power = float(exposure), float(power)
    n = rows * cols
    dose_total = n * power * exposure * 0.01
    if dose_total > _GRID_MAX_DOSE_MJ:
        return {"ok": False, "error": (
            f"Safety block: estimated cumulative dose {dose_total:.1f} mJ exceeds the grid limit "
            f"({_GRID_MAX_DOSE_MJ} mJ). Reduce point count, power, or exposure.")}

    try:
        pos = _stage.get_position()
        if pos is None:
            return {"ok": False, "error": "Failed to query stage position"}
        cx = float(pos[0]) if center_x is None else float(center_x)
        cy = float(pos[1]) if center_y is None else float(center_y)

        pts = _grid_stage_coords(cx, cy, rows, cols, spacing_mm)
        # 범위 사전 검증 — 한 점이라도 벗어나면 시작조차 하지 않는다.
        for idx, i, j, sx, sy in pts:
            if not (0 <= sx <= STAGE_MAX_X and 0 <= sy <= STAGE_MAX_Y):
                return {"ok": False, "error": (
                    f"Point {idx} (row {i}, col {j}) at X={sx}, Y={sy} mm is outside the stage range "
                    f"(0..{STAGE_MAX_X} x 0..{STAGE_MAX_Y}). Adjust center/spacing/size.")}

        # center 모드: 격자 중심으로 이동 후 1회 오토포커스(이후 Z 유지).
        if autofocus == "center":
            mv = move_stage(x=cx, y=cy)
            if not mv.get("ok"):
                return {"ok": False, "error": f"Failed to move to grid center: {mv.get('error')}"}
            af = run_autofocus()
            if not af.get("ok"):
                return {"ok": False, "error": f"Autofocus at grid center failed: {af.get('error')}"}

        results = []
        n_ok = 0
        for idx, i, j, sx, sy in pts:
            mv = move_stage(x=sx, y=sy)
            if not mv.get("ok"):
                results.append({"i": idx, "row": i, "col": j, "x": sx, "y": sy,
                                "ok": False, "error": mv.get("error")})
                continue
            if autofocus == "each":
                run_autofocus()   # 실패해도 치명적이지 않다 — 현재 Z로 측정을 이어간다.
            res = _cache_and_return(acquire_spectrum(exposure=exposure, power=power))
            if res.get("ok"):
                n_ok += 1
                files = (res.get("saved") or {}).get("files") or {}
                ref = files.get("csv") or files.get("png") or ""
                fname = ref.replace("\\", "/").rsplit("/", 1)[-1]
                results.append({"i": idx, "x": sx, "y": sy,
                                "max_intensity": res.get("max_intensity"), "file": fname})
            else:
                results.append({"i": idx, "row": i, "col": j, "x": sx, "y": sy,
                                "ok": False, "error": res.get("error")})

        # 압축 반환 — 큰 격자에서 per-point 리스트가 에이전트 _slim(길이>32 리스트 폐기)에
        # 통째로 걸리지 않도록, 집계 통계는 항상 싣고 per-point는 32점 이하일 때만 인라인.
        oks = [r for r in results if "max_intensity" in r]
        fails = [r for r in results if r.get("ok") is False]
        inten = [r["max_intensity"] for r in oks if r.get("max_intensity") is not None]
        out = {
            "ok": True,
            "rows": rows, "cols": cols, "spacing_mm": spacing_mm,
            "center": {"x": round(cx, 4), "y": round(cy, 4)},
            "autofocus": autofocus, "exposure": exposure, "power": power,
            "n_points": n, "n_measured": n_ok, "n_failed": len(fails),
            "estimated_dose_mj": round(dose_total, 2),
            "intensity": ({"min": min(inten), "max": max(inten),
                           "mean": round(sum(inten) / len(inten), 1)} if inten else None),
            "note": ("Each point was auto-saved with its (x,y) tag. Use aggregate_spectra_csv / "
                     "combine_spectra / bundle_results / run_analysis to merge or inspect per-point data."),
        }
        if fails:
            out["failed_points"] = fails[:10]
        if n <= 32:
            out["points"] = oks
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────
# tool dispatch 테이블 (agent loop에서 사용)
# ──────────────────────────────────────────

TOOL_DISPATCH = {
    # ── 스테이지 ─────────────────────────────────────────────────────────────
    "move_stage":               lambda a: move_stage(**a),
    "get_stage_position":       lambda a: get_stage_position(),
    "move_stage_relative":      lambda a: move_stage_relative(**a),
    "get_stage_speed":          lambda a: get_stage_speed(),
    "set_stage_speed":          lambda a: set_stage_speed(**a),
    # ── 레이저 ──────────────────────────────────────────────────────────────
    "laser_on":                 lambda a: laser_on(),
    "laser_off":                lambda a: laser_off(),
    "set_laser_power":          lambda a: set_laser_power(**a),
    "set_guide_beam_mode":      lambda a: set_guide_beam_mode(),
    # ── 스펙트럼 수집 ────────────────────────────────────────────────────────
    "acquire_spectrum":         lambda a: _cache_and_return(acquire_spectrum(**a)),
    # ── 카메라 ──────────────────────────────────────────────────────────────
    "start_camera_stream":      lambda a: start_camera_stream(),
    "stop_camera_stream":       lambda a: stop_camera_stream(),
    "set_camera_exposure":      lambda a: set_camera_exposure(**a),
    "set_camera_auto_exposure": lambda a: set_camera_auto_exposure(**a),
    "capture_camera_frame":     lambda a: capture_camera_frame(),
    "analyze_microscope_image": lambda a: analyze_microscope_image(**a),
    "move_to_pixel":            lambda a: move_to_pixel(**a),
    "capture_scene":            lambda a: capture_scene(),
    # ── 오토포커스 ───────────────────────────────────────────────────────────
    "run_autofocus":            lambda a: run_autofocus(**a),
    # ── 그리드 매핑(미리보기 + 실행) ─────────────────────────────────────────
    "preview_grid_scan":        lambda a: preview_grid_scan(**a),
    "run_grid_scan":            lambda a: run_grid_scan(**a),
    # ── CCD 설정 ─────────────────────────────────────────────────────────────
    "get_ccd_info":             lambda a: get_ccd_info(),
    "set_ccd_exposure":         lambda a: set_ccd_exposure(**a),
    "set_ccd_acquisition_mode": lambda a: set_ccd_acquisition_mode(**a),
    "set_ccd_trigger_mode":     lambda a: set_ccd_trigger_mode(**a),
    "set_ccd_read_mode":        lambda a: set_ccd_read_mode(**a),
    "set_ccd_preamp_gain":      lambda a: set_ccd_preamp_gain(**a),
    "set_ccd_em_gain":          lambda a: set_ccd_em_gain(**a),
    "set_mcp_gain":             lambda a: set_mcp_gain(**a),
    "get_mcp_gain_range":       lambda a: get_mcp_gain_range(**a),
    "set_ccd_output_amp":       lambda a: set_ccd_output_amp(**a),
    "set_ccd_shift_speeds":     lambda a: set_ccd_shift_speeds(**a),
    "set_ccd_temperature":      lambda a: set_ccd_temperature(**a),
    "set_ccd_cooler":           lambda a: set_ccd_cooler(**a),
    "set_ccd_shutter":          lambda a: set_ccd_shutter(**a),
    "set_ccd_image_flip":       lambda a: set_ccd_image_flip(**a),
    # ── 데이터 저장 / 로드 ───────────────────────────────────────────────────
    "save_spectrum":            lambda a: save_spectrum(**a),
    "load_spectrum":            lambda a: load_spectrum(**a),
    # ── 측정 결과 정리(자동 저장분 대상) ─────────────────────────────────────
    "list_results":             lambda a: {"ok": True, "items": [
                                    {k: it[k] for k in ("base", "date", "title", "timestamp", "meta")}
                                    for it in _store_list_results(**a)]},
    "combine_spectra":          lambda a: _store_combine_spectra(**a),
    "aggregate_spectra_csv":    lambda a: _store_aggregate_csv(**a),
    "bundle_results":           lambda a: _store_bundle_results(**a),
    # ── 분석 전용 코드 샌드박스(하드웨어 미접근) ─────────────────────────────
    "run_analysis":             lambda a: _run_analysis(**a),
    # ── 외부 웹 검색(내부 지식에 없을 때) ────────────────────────────────────
    "web_search":               lambda a: _web_search(**a),
    # ── 세션 관리 ────────────────────────────────────────────────────────────
    "create_session":           lambda a: create_session(**a),
    "save_point_data":          lambda a: save_point_data(**a),
    # ── 배경 제거 (IPBSA) ────────────────────────────────────────────────────
    "apply_background_subtraction": lambda a: apply_background_subtraction(**a),
    "list_bg_versions":             lambda a: list_bg_versions(),
    "get_bg_version":               lambda a: get_bg_version(**a),
}