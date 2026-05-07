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


STAGE_MAX_X =  75.3169
STAGE_MAX_Y =  50.1879
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
        return {"ok": False, "error": "스테이지가 초기화되지 않았습니다."}
    try:
        speeds = _stage.get_velocity()
        return {"ok": True, "x_speed_mm_s": speeds[0], "y_speed_mm_s": speeds[1], "z_speed_mm_s": speeds[2]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def set_stage_speed(x_speed_mm_s: float, y_speed_mm_s: float, z_speed_mm_s: float = None) -> dict:
    """스테이지 이동 속도를 설정한다. x_speed_mm_s, y_speed_mm_s, z_speed_mm_s는 각 축의 이동 속도."""
    if _stage is None:
        return {"ok": False, "error": "스테이지가 초기화되지 않았습니다."}
    try:
        _stage.set_velocity(x_speed_mm_s, y_speed_mm_s, z_speed_mm_s)
        return {"ok": True, "x_speed_mm_s": x_speed_mm_s, "y_speed_mm_s": y_speed_mm_s, "z_speed_mm_s": z_speed_mm_s}
    except Exception as e:
        return {"ok": False, "error": str(e)}

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
    exposure: float = 0.2,
    power: int = 40,
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
    power : int
        레이저 출력 [%]. 20/40/60/80/100. 기본 40.
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
        return {"ok": False, "error": "분광기가 초기화되지 않았습니다."}
    if _laser is None:
        return {"ok": False, "error": "레이저가 초기화되지 않았습니다."}

    valid_powers = {20, 40, 60, 80, 100}
    if power not in valid_powers:
        return {"ok": False, "error": f"유효한 출력값: {valid_powers}"}

    if acq_mode not in ('single', 'accumulate', 'kinetic'):
        return {"ok": False, "error": "acq_mode는 'single' | 'accumulate' | 'kinetic'"}

    if read_mode not in ('fvb', 'single_track'):
        return {"ok": False, "error": "read_mode는 'fvb' | 'single_track'"}

    if read_mode == 'single_track' and single_track_center is None:
        return {"ok": False, "error": "read_mode='single_track' 사용 시 single_track_center 필요"}

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
                        f"kinetic 취득 타임아웃: {_timeout_s:.1f}초 초과 "
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
        return {"ok": False, "error": "CCD 데이터 수집 실패"}

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
        return {
            "ok": True,
            "mode": "kinetic",
            "num_frames": len(frames),
            "kinetic_count": kinetic_count,
            "exposure_time": exposure,
            "laser_power_pct": power,
            "frames": frames,
        }
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
        return result

# ──────────────────────────────────────────
# CCD 파라미터 설정 툴
# ──────────────────────────────────────────

def get_ccd_info() -> dict:
    """현재 CCD 설정값 및 상태를 한 번에 조회한다."""
    if _ccd is None:
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    if exposure_time <= 0:
        return {"ok": False, "error": "노출 시간은 0보다 커야 합니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    valid = {'single', 'accumulate', 'kinetic', 'run_till_abort'}
    if mode not in valid:
        return {"ok": False, "error": f"유효한 모드: {valid}"}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    valid = {'internal', 'external', 'external_start', 'external_exposure', 'software'}
    if mode not in valid:
        return {"ok": False, "error": f"유효한 모드: {valid}"}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    valid = {'fvb', 'single_track', 'image'}
    if mode not in valid:
        return {"ok": False, "error": f"유효한 모드: {valid}"}
    try:
        if mode == 'fvb':
            _ccd.set_ro_full_vertical_binning(hbin=hbin)
        elif mode == 'single_track':
            if center is None:
                return {"ok": False, "error": "single_track 모드는 center 파라미터가 필요합니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    if not getattr(_ccd, 'em_mode', False):
        return {"ok": False, "error": "이 카메라는 EM CCD가 아닙니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    try:
        low, high = _ccd.get_mcp_gain_range()
        if not (low <= gain <= high):
            return {"ok": False, "error": f"MCP 이득 범위 초과: {gain} (허용: {low}~{high})"}
        _ccd.set_mcp_gain(gain)
        return {"ok": True, "mcp_gain": gain}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_mcp_gain_range() -> dict:
    """iStar ICCD 카메라의 MCP 이득 허용 범위(min, max)를 반환한다."""
    if _ccd is None:
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    if amp not in (0, 1):
        return {"ok": False, "error": "amp는 0(EM) 또는 1(Conventional)이어야 합니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    try:
        _ccd.set_temperature(temp)
        return {"ok": True, "target_temperature_C": temp}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_ccd_cooler(on: bool) -> dict:
    """CCD 냉각기를 켜거나(True) 끈다(False)."""
    if _ccd is None:
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    if mode not in ('auto', 'open', 'close'):
        return {"ok": False, "error": "mode는 'auto', 'open', 'close' 중 하나여야 합니다."}
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
        return {"ok": False, "error": "CCD가 초기화되지 않았습니다."}
    try:
        _ccd.set_image_flip(hflip=hflip, vflip=vflip)
        return {"ok": True, "hflip": hflip, "vflip": vflip}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def start_camera_stream() -> dict:
    """
    카메라 실시간 스트리밍을 시작합니다.
    USE_camera_stream.py의 StreamingTUCam.start_stream()을 호출합니다.
    """
    if _camera is None:
        return {"ok": False, "error": "카메라가 초기화되지 않았습니다."}
    
    try:
        # 이미 스트리밍 중인지 확인 (StreamingTUCam 내부 속성 활용)
        if getattr(_camera, 'is_streaming', False):
            return {"ok": True, "status": "카메라는 이미 스트리밍 중입니다."}
            
        _camera.start_stream()
        return {"ok": True, "status": "카메라 스트리밍이 성공적으로 시작되었습니다."}
        
    except Exception as e:
        return {"ok": False, "error": f"스트리밍 시작 실패: {str(e)}"}

def stop_camera_stream() -> dict:
    """
    카메라 실시간 스트리밍을 중지합니다.
    USE_camera_stream.py의 StreamingTUCam.stop_stream()을 호출합니다.
    """
    if _camera is None:
        return {"ok": False, "error": "카메라가 초기화되지 않았습니다."}

    try:
        if not getattr(_camera, 'is_streaming', False):
            return {"ok": True, "status": "카메라는 현재 스트리밍 중이 아닙니다."}

        _camera.stop_stream()
        return {"ok": True, "status": "카메라 스트리밍이 성공적으로 중지되었습니다."}

    except Exception as e:
        return {"ok": False, "error": f"스트리밍 중지 실패: {str(e)}"}


def set_camera_exposure(ms: float) -> dict:
    """카메라(TUCam) 노출 시간(ms)을 설정한다."""
    if _camera is None:
        return {"ok": False, "error": "카메라가 초기화되지 않았습니다."}
    if ms <= 0:
        return {"ok": False, "error": "노출 시간은 0보다 커야 합니다."}
    try:
        _camera.set_exposure(ms)
        return {"ok": True, "exposure_ms": ms}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def set_camera_auto_exposure(enabled: bool) -> dict:
    """카메라 자동 노출을 활성화(True) 또는 비활성화(False)한다."""
    if _camera is None:
        return {"ok": False, "error": "카메라가 초기화되지 않았습니다."}
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
        return {"ok": False, "error": "카메라가 초기화되지 않았습니다."}
    try:
        import numpy as np
        import cv2
        frame = _camera.get_latest_frame()
        if frame is None:
            return {"ok": False, "error": "프레임 획득 실패 (스트리밍 중인지 확인)"}

        gray = (frame.astype(np.float32)
                if frame.ndim == 2
                else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
        sharpness = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))

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
        return {"ok": False, "error": "레이저가 초기화되지 않았습니다."}
    try:
        _laser.set_guide_beam()
        return {"ok": True, "status": "가이드빔 모드로 전환 완료"}
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
        return {"ok": False, "error": "스테이지가 초기화되지 않았습니다."}
    if _camera is None:
        return {"ok": False, "error": "카메라가 초기화되지 않았습니다."}
    if _laser is None:
        return {"ok": False, "error": "레이저가 초기화되지 않았습니다."}

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
        return {"ok": False, "error": f"poly_order는 2~10이어야 합니다 (입력: {poly_order})"}
    if not (10 <= max_iterations <= 500):
        return {"ok": False, "error": f"max_iterations는 10~500이어야 합니다 (입력: {max_iterations})"}
    if not (0.001 <= threshold <= 1.0):
        return {"ok": False, "error": f"threshold는 0.001~1.0이어야 합니다 (입력: {threshold})"}

    intensity: list = []
    raman_shift = None

    if source == "last":
        if _last_spectrum is None:
            return {
                "ok": False,
                "error": "저장된 스펙트럼이 없습니다. 먼저 acquire_spectrum()을 호출하세요.",
            }
        if "data" not in _last_spectrum:
            return {
                "ok": False,
                "error": "마지막 스펙트럼이 Kinetic 모드입니다. Single/Accumulate 스펙트럼에만 적용 가능합니다.",
            }
        intensity = _last_spectrum["data"]
        raman_shift = _last_spectrum.get("raman_shift_cm-1")
    else:
        filepath = Path(source)
        if not filepath.is_absolute():
            filepath = _DATA_DIR / source
        if not filepath.exists():
            return {"ok": False, "error": f"파일을 찾을 수 없습니다: {filepath}"}
        try:
            if filepath.suffix.lower() == ".json":
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if "data" in loaded:
                    intensity = loaded["data"]
                elif "corrected_data" in loaded:
                    intensity = loaded["corrected_data"]
                else:
                    return {"ok": False, "error": "JSON 파일에 'data' 또는 'corrected_data' 키가 없습니다."}
                raman_shift = loaded.get("raman_shift_cm-1")
            elif filepath.suffix.lower() == ".csv":
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if not rows:
                    return {"ok": False, "error": "CSV 파일이 비어 있습니다."}
                if "corrected_intensity" in rows[0]:
                    intensity = [float(r["corrected_intensity"]) for r in rows]
                elif "intensity" in rows[0]:
                    intensity = [float(r["intensity"]) for r in rows]
                else:
                    return {"ok": False, "error": "CSV에 'intensity' 컬럼이 없습니다."}
                if "raman_shift_cm-1" in rows[0]:
                    raman_shift = [float(r["raman_shift_cm-1"]) for r in rows]
            else:
                return {"ok": False, "error": "지원되지 않는 파일 형식입니다 (JSON 또는 CSV만 허용)."}
        except Exception as e:
            return {"ok": False, "error": f"파일 로드 오류: {e}"}

    if not intensity:
        return {"ok": False, "error": "스펙트럼 강도 배열이 비어 있습니다."}
    if len(intensity) < poly_order + 1:
        return {
            "ok": False,
            "error": (
                f"스펙트럼 길이({len(intensity)})가 poly_order+1({poly_order + 1})보다 작습니다. "
                "다항식 차수를 낮추세요."
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
        return {"ok": False, "error": f"IPBSA 알고리즘 오류: {e}"}

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
            "message": "저장된 버전이 없습니다. apply_background_subtraction()을 먼저 호출하세요.",
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
            "error": f"버전 '{version_label}'을 찾을 수 없습니다.",
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
            return {"ok": False, "error": f"파일을 찾을 수 없습니다: {filepath}"}

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
            return {"ok": False, "error": f"세션이 없습니다: {session_id}. create_session을 먼저 호출하세요."}

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
    # ── 오토포커스 ───────────────────────────────────────────────────────────
    "run_autofocus":            lambda a: run_autofocus(**a),
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
    # ── 세션 관리 ────────────────────────────────────────────────────────────
    "create_session":           lambda a: create_session(**a),
    "save_point_data":          lambda a: save_point_data(**a),
    # ── 배경 제거 (IPBSA) ────────────────────────────────────────────────────
    "apply_background_subtraction": lambda a: apply_background_subtraction(**a),
    "list_bg_versions":             lambda a: list_bg_versions(),
    "get_bg_version":               lambda a: get_bg_version(**a),
}