# -*- coding: utf-8 -*-
"""스펙트럼 수집·측정점 기록·격자 매핑 — 레이저가 실제로 시편에 나가는 도구들.

acquire_spectrum 이 laser ON → 파워 안정 → CCD 획득 → laser OFF 를 원자적으로 처리하고,
실패해도 레이저를 끈다. 격자 스캔은 그 루프를 파이썬으로 내린 것이고, 사람 승인 게이트
(preview 후 턴을 끊고 승인받아야 실행)가 코드 인터록으로 붙어 있다.

save_measurement_point 가 여기 있는 이유: 그 도구는 '직전 측정 + 직전 캡처 + 현재 좌표'를
읽으므로 세션 캐시와 스테이지 핸들에 기댄다. 측정과 같은 모듈에 두는 편이 의존을 늘리지
않는다.
"""
from __future__ import annotations

import json
import time

from backend.tools.hw_tools.config import CAMERA_HEIGHT, CAMERA_WIDTH, STAGE_MAX_X, STAGE_MAX_Y
from backend.service.vision import vision as _vis
from backend.service.store.spectrum_store import save_preview_png as _store_save_preview, save_spectrum as _store_save_spectrum
from backend.service.vision import optics_map as _om
from backend.tools.result import fail, ok
from backend.tools.schema import INTERNAL
from backend.service.safety.safety_limits import MAX_DOSE_MJ_PER_GRID as _GRID_MAX_DOSE_MJ, estimate_dose_mj
from pydantic import Field
from typing import Annotated, Literal, Optional
from backend.tools.hw_tools.hw_tools import hw_core as _hw
from backend.tools.hw_tools.hw_tools.camera_tools import run_autofocus
from backend.tools.hw_tools.hw_tools.hw_core import _ACQ_MODES_1D, _READ_MODES_1D, _SHUTTER_MODES, _TRIGGER_MODES, _apply_acq_mode, _apply_laser_power, _apply_read_mode, _apply_shutter, _apply_trigger_mode, _cache_and_return, _ccd_ready, _check_ccd_positive, _current_read_mode, _effective_shutter, _laser_off_quiet, _restore_guide_beam_quiet, _serialized, _sstate, _stage_unavailable
from backend.tools.hw_tools.hw_tools.stage_tools import move_stage


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
        if _hw._stage is not None:
            pos = _hw._stage.get_position()
            meta["x"], meta["y"] = round(float(pos[0]), 3), round(float(pos[1]), 3)
    except Exception:
        pass
    saved = _store_save_spectrum(result, meta or None)
    if saved.get("ok"):
        result["saved"] = saved
    else:
        result["saved_error"] = saved.get("error")
    return result


@_serialized("acquire_spectrum")
def acquire_spectrum(
    exposure: Annotated[Optional[float], Field(description="CCD exposure time (seconds). Omit to keep the CCD's current exposure (e.g. one set earlier by set_ccd_exposure).")] = None,
    power: Annotated[Optional[float], Field(ge=0.004, le=100, description="Laser power (transmittance %), a real value in 0.004-100. Omit ONLY to reuse a power you already set earlier in this session - if none has ever been set the call is REFUSED rather than defaulted, because choosing a dose for an unknown sample is your decision, not the tool's. Higher power gives more signal but photobleaches or burns fragile samples; when the sample's tolerance is unknown, start low and raise it after looking at the result.")] = None,
    stabilize_sec: Annotated[Optional[float], Field(description='Wait time for power stabilization after laser ON (seconds). Default 0.5')] = 0.5,
    acq_mode: Annotated[Optional[Literal['single', 'accumulate', 'kinetic']], Field(description="CCD acquisition mode. Omit to keep the CCD's current mode. 'single': single shot. 'accumulate': sum num_accumulations shots -> high-SNR spectrum. 'kinetic': acquire kinetic_count frames continuously -> time-series analysis.")] = None,
    num_accumulations: Annotated[Optional[int], Field(ge=1, description='Accumulations per frame in accumulate/kinetic mode. Omit to keep the current value.')] = None,
    kinetic_count: Annotated[Optional[int], Field(ge=1, description='Total number of frames to acquire in kinetic mode. Omit to keep the current value.')] = None,
    kinetic_cycle_time: Annotated[Optional[float], Field(description='Frame interval in kinetic mode (seconds). If omitted, the SDK auto-computes the minimum.')] = None,
    read_mode: Annotated[Optional[Literal['fvb', 'single_track']], Field(description="CCD readout mode. Omit to keep the CCD's current read mode. 'fvb': Full Vertical Binning - sum all rows, 1D spectrum. 'single_track': read only a specific track - single_track_center required.")] = None,
    hbin: Annotated[Optional[int], Field(ge=1, description='Horizontal binning pixel count. Omit to keep the current value.')] = None,
    single_track_center: Annotated[Optional[int], Field(description="Center pixel row number when read_mode='single_track'. Omit to reuse the currently configured track.")] = None,
    single_track_width: Annotated[Optional[int], Field(ge=1, description="Track width (pixels) when read_mode='single_track'. Omit to keep the current value.")] = None,
    trigger_mode: Annotated[Optional[Literal['internal', 'external', 'external_start', 'external_exposure', 'external_fvb_em', 'software']], Field(description="CCD trigger mode. Omit to keep the CCD's current trigger mode.")] = None,
    shutter: Annotated[Optional[Literal['auto', 'open', 'close']], Field(description="Shutter mode for this acquisition. Omit to KEEP the current setting - including one you set earlier with set_ccd_shutter. If nobody has ever set it, it opens to 'auto' (the CCD boots closed, so keeping that would silently hand you a dark frame). Use 'close' to acquire a DARK / background frame with no light reaching the detector - that is the supported way to measure a dark reference.")] = None,
    restore_guide_beam: Annotated[Optional[bool], Field(json_schema_extra=INTERNAL, description='Internal: restore the guide beam after acquisition. Consecutive callers (run_grid_scan) pass False to avoid toggling it on every point.')] = True,
) -> dict:
    """
    라만 스펙트럼 수집 (Single / Accumulate / Kinetic 모드 지원).

    [파라미터를 생략하면 '현재 장비 설정을 그대로 쓴다' — 2026-07-30 수정]
    예전에는 모든 파라미터에 기본값(0.2s / 40% / single / fvb / internal)이 박혀 있어,
    호출자가 생략하면 '유지'가 아니라 '기본값으로 리셋'이었다. 그래서 set_ccd_exposure,
    set_laser_power, set_ccd_acquisition_mode, set_ccd_read_mode, set_ccd_trigger_mode 로
    미리 맞춰 둔 설정이 측정 직전에 조용히 덮어써졌다(설정 툴 5개가 사실상 무효였다).
    "노출 1.0s 로 설정하고 측정" 같은 2단계 요청이 실제로는 0.2s 로 측정되던 원인이다.
    드라이버(andor_ccd_interface)는 이미 None=유지 규약을 갖고 있어 그대로 넘기면 된다.

    원자적 실행 흐름:
      1. 레이저 출력 설정 (ND filter motor 이동, 블로킹)
      2. 레이저 ON + 안정화 대기 (축04 빔스플리터 → 측정 위치)
      3. CCD 취득 모드 설정 (set_aq_* → set_ro_* 순서 필수)
      4. CCD 촬영 (StartAcquisition → 폴링 → GetAcquiredData)
      5. 레이저 OFF (성공/실패 무관하게 반드시 실행)
      6. 광학계를 가이드빔 위치로 복귀 (축04 → 대기 위치 = 카메라가 다시 보인다)

    Parameters
    ----------
    exposure : float or None
        CCD 노출 시간 [초]. None이면 현재 CCD 설정을 유지한다.
    power : float or None
        레이저 출력 [%], 0.004~100 (ND 필터 연속 조절).
        None이면 이 세션에서 마지막으로 설정된 파워를 재적용한다. **한 번도 설정된 적이
        없으면 임의값으로 쏘지 않고 거부한다** — 조사량 결정을 도구가 대신하지 않는다.
        레이저는 이 값을 항상 하드웨어에 적용한다 — 아래 주석 참고.
    stabilize_sec : float
        레이저 ON 후 안정화 대기 [초]. 기본 0.5.
    acq_mode : str or None
        'single' | 'accumulate' | 'kinetic'. None이면 현재 CCD 취득 모드를 유지한다.
    num_accumulations : int or None
        Accumulate/Kinetic 모드의 프레임당 누적 횟수. None이면 현재 값 유지.
    kinetic_count : int or None
        Kinetic 모드에서 수집할 총 프레임 수. None이면 현재 값 유지.
    kinetic_cycle_time : float or None
        Kinetic 프레임 간격 [초]. None이면 SDK가 자동 계산.
    read_mode : str or None
        'fvb' | 'single_track'. None이면 현재 CCD 읽기 모드를 유지한다.
        (CCD가 'image' 모드면 1D 스펙트럼을 조립할 수 없어 거부한다.)
    hbin : int or None
        수평 비닝 픽셀 수. None이면 현재 값 유지.
    single_track_center : int or None
        read_mode='single_track' 시 중심 픽셀 행. None이면 현재 설정된 트랙을 재사용.
    single_track_width : int or None
        read_mode='single_track' 시 트랙 폭 [픽셀]. None이면 현재 값 유지.
    trigger_mode : str or None
        'internal' | 'external' | 'external_start' | 'external_exposure' |
        'external_fvb_em' | 'software'. None이면 현재 트리거 모드를 유지한다.
    shutter : str or None
        'auto' | 'open' | 'close'. None이면 다른 파라미터와 같이 현재 설정을 유지한다 —
        단, 아무도 셔터를 지정한 적이 없으면 'auto'로 연다(CCD 초기화가 셔터를 close로
        두므로 그대로 두면 세션 첫 측정이 암전 프레임이 된다). 규약은 _effective_shutter
        참고. 다크/배경 프레임은 shutter='close'를 명시하거나 set_ccd_shutter('close')로
        걸어 둔다 — 후자는 이제 이후 측정에도 유지된다.

    Returns
    -------
    dict — Single / Accumulate 모드:
        ok, mode, length, max_intensity, sum_intensity, data,
        calibrated, exposure_time, laser_power_pct, shutter, trigger_mode,
        [num_accumulations,] [raman_shift_cm-1, wavelength_nm, laser_nm]
        exposure_time / laser_power_pct 등은 '요청값'이 아니라 장비에서 읽은 실제 적용값이다.

    dict — Kinetic 모드:
        ok, mode, num_frames, kinetic_count, exposure_time, laser_power_pct,
        frames: list of {frame_index, intensity, length, max_intensity,
                         sum_intensity, calibrated, [raman_shift_cm-1, ...]}
    """
    err = _ccd_ready()
    if err:
        return err
    if _hw._laser is None:
        return fail("Laser is not initialized.")

    # ── 파라미터 해석 (하드웨어를 만지기 전에 전부 검증한다) ──
    # 레이저 파워만은 None이어도 반드시 하드웨어에 적용한다. laser_on()이 측정빔을 쏘려면
    # 드라이버의 _power_set 플래그가 True여야 하고, 그 플래그는 set_power()만이 세운다
    # (가이드빔 모드나 오토포커스 직후엔 False다). 같은 위치로의 재이동은 무해하다.
    if power is None:
        # [왜 40% 폴백을 없앴는가 — 2026-07-31]
        # 예전에는 한 번도 파워를 정한 적이 없어도 조용히 40% 로 쏘았다. 그건 '유지'가
        # 아니라 이 도구가 조사량을 대신 결정한 것이고, 시료가 무엇인지 모르는 상태에서
        # 가장 하면 안 되는 결정이다(생체·고분자 시료는 40% 에서 태워 먹는다).
        # 이미 정해진 값이 있으면 그대로 재적용한다 — 가이드빔 모드로 갔다 와도
        # power_pct 는 남아 있으므로 '다시 무장'이 되어 의도대로 동작한다.
        last = getattr(_hw._laser, "power_pct", None)
        if last is None:
            return fail("No laser power has been set in this session, so there is nothing to 'keep' and this "
                        "tool will not pick one for you - firing an unknown sample at an arbitrary power can "
                        "photobleach or burn it. Decide the power yourself and pass acquire_spectrum(power=...), "
                        "or set it once with set_laser_power(percent) and then measure. If you do not know the "
                        "sample's tolerance, start low (a few percent), look at the signal, and raise it.")
        eff_power = float(last)
    else:
        eff_power = power                   # 검증은 _apply_laser_power 가 한다(단일 정책)

    eff_acq = acq_mode or getattr(_hw._ccd, 'aq_mode', None) or 'single'
    if eff_acq not in _ACQ_MODES_1D:
        return fail(f"acquire_spectrum supports {list(_ACQ_MODES_1D)}, but the CCD is "
                    f"currently in '{eff_acq}' mode. Pass acq_mode explicitly, or call "
                    f"set_ccd_acquisition_mode first.")

    # 현재 읽기 모드(드라이버 표기) → 이 함수의 인자 표기로 환산.
    eff_read = read_mode or _current_read_mode()
    if eff_read not in _READ_MODES_1D:
        return fail(f"acquire_spectrum assembles a 1D spectrum, but the CCD read mode is '{eff_read}'. "
                    f"Pass read_mode='fvb' (or 'single_track'), or call set_ccd_read_mode first.")

    eff_center = (single_track_center if single_track_center is not None
                  else getattr(_hw._ccd, 'ro_single_track_center', None))
    if eff_read == 'single_track' and eff_center is None:
        return fail("single_track_center is required when read_mode='single_track'")

    # ── 수치 파라미터도 레이저를 쏘기 전에 검증한다 — 2026-08-05 ──────────────
    # 이 값들이 실제로 SDK 로 들어가는 곳은 아래 try 블록의 _apply_acq_mode /
    # _apply_read_mode 이고, 그 시점에는 laser_on() 이 이미 끝나 있다. 검증을 거기에만
    # 맡기면 exposure=0 같은 값 하나에 시료가 조사된 뒤 DRV_P1INVALID 로 실패한다.
    # 셔터·트리거를 아래에서 미리 보는 것과 같은 이유로 여기서 끝낸다.
    # (같은 검사가 _apply_* 에도 있다 — 그쪽은 set_ccd_* 도구용이라 중복이 아니다.)
    for _n, _v, _i in (("exposure", exposure, False),
                       ("num_accumulations", num_accumulations, True),
                       ("kinetic_count", kinetic_count, True),
                       ("kinetic_cycle_time", kinetic_cycle_time, False),
                       ("hbin", hbin, True),
                       ("single_track_width", single_track_width, True),
                       # 캐시에서 온 값도 함께 본다 — 사용될 값 자체를 검사해야 한다.
                       ("single_track_center", eff_center if eff_read == 'single_track' else None, True)):
        err = _check_ccd_positive(_n, _v, integer=_i)
        if err:
            return err

    # 셔터·트리거는 레이저를 쏘기 전에 검증한다 — 잘못된 값으로 발사한 뒤 실패하면
    # 조사만 낭비되고 시료에는 이미 빔이 들어간 뒤다.
    eff_shutter = _effective_shutter(shutter)      # 규약은 _effective_shutter 단일 출처
    if eff_shutter not in _SHUTTER_MODES:
        return fail(f"shutter must be one of {list(_SHUTTER_MODES)}.")

    # 트리거를 생략하면 드라이버가 기억하는 현재 모드를 그대로 쓴다(SDK엔 getter가 없다).
    eff_trigger = trigger_mode or getattr(_hw._ccd, 'trigger_mode', None) or 'internal'
    if eff_trigger not in _TRIGGER_MODES:
        return fail(f"trigger_mode must be one of {list(_TRIGGER_MODES)}.")

    # 파워 검증도 발사 전에 끝낸다(_apply_laser_power 가 실제 적용까지 한다).
    err = _apply_laser_power(eff_power, settle_s=stabilize_sec)
    if err:
        return err

    raw = None
    try:
        # 1. (파워는 위에서 이미 적용됨 — 검증 실패 시 레이저를 켜지 않기 위해 앞으로 뺐다)
        # 2. 레이저 ON + 안정화 대기
        _hw._laser.laser_on()
        time.sleep(stabilize_sec)

        # 3-a. 취득 모드 설정 — 반드시 읽기 모드보다 먼저 (create_buffer가 aq_mode 의존).
        #      exposure/num_* 가 None이면 드라이버가 현재 값을 그대로 둔다.
        err = _apply_acq_mode(eff_acq, exposure=exposure,
                              num_accumulations=num_accumulations,
                              kinetic_count=kinetic_count,
                              kinetic_cycle_time=kinetic_cycle_time)
        if err:
            return err

        # 3-b. 읽기 모드 설정 — create_buffer()가 여기서 호출된다. aq_mode를 바꿨으면 버퍼
        #      모양도 다시 잡아야 하므로, read_mode를 생략했더라도 '현재 모드'로 한 번 다시
        #      적용한다(hbin/트랙 파라미터는 현재 값을 그대로 재사용).
        err = _apply_read_mode(eff_read, hbin=hbin, single_track_center=eff_center,
                               single_track_width=single_track_width)
        if err:
            return err

        # 3-c. 트리거 — 생략했으면 건드리지 않는다(현재 설정 유지).
        if trigger_mode is not None:
            err = _apply_trigger_mode(trigger_mode)
            if err:
                return err

        # 3-d. 셔터 — 생략 시 규약은 _effective_shutter 참고. 'close'면 다크 프레임이 된다.
        err = _apply_shutter(eff_shutter, explicit=(shutter is not None))
        if err:
            return err

        # 3-e. 이전 취득 버퍼 해제 (파라미터 변경 시 SDK 내부 메모리 단편화 방지)
        _hw._ccd.free_internal_memory()

        # 실제로 장비에 걸린 값을 읽어 둔다 — 타임아웃 계산과 결과 보고에 모두 쓴다.
        # 요청값이 아니라 실측값을 보고해야 '노출을 바꿨는데 왜 결과가 같은가'를 알 수 있다.
        try:
            eff_exposure = float(_hw._ccd.get_exposure_time())
        except Exception:
            eff_exposure = float(exposure) if exposure is not None else 0.0
        eff_num_acc = int(getattr(_hw._ccd, 'num_acc', 1) or 1)
        eff_num_kin = int(getattr(_hw._ccd, 'num_kin', 1) or 1)

        # 4. 촬영
        if eff_acq == 'kinetic':
            # Kinetic: 버퍼가 3D (num_kin, Ny_ro, Nx_ro) — start_acquisition_cycle() 직접 사용 불가
            _hw._ccd.prepare_acquisition()        # 메모리 사전 할당 + 타이밍 초기화 (첫 프레임 지연 방지)
            _hw._ccd.start_acquisition()
            if eff_trigger == 'software':     # software 트리거 발송 (없으면 ACQUIRING 상태에서 무한 대기)
                _hw._ccd.send_software_trigger()

            # 외부 트리거 미도달 / 하드웨어 장애 시 무한 대기 방지.
            # 실제 적용된 노출·프레임수로 계산한다(요청값은 None일 수 있다).
            _cyc = kinetic_cycle_time if kinetic_cycle_time else (eff_exposure + 0.1)
            _timeout_s = _cyc * max(eff_num_kin, 1) * max(eff_num_acc, 1) * 2 + 15.0
            _deadline = time.time() + _timeout_s
            while _hw._ccd.get_status() != 'IDLE':
                if time.time() > _deadline:
                    try:
                        _hw._ccd.abort_acquisition()
                    except Exception:
                        pass
                    raise TimeoutError(
                        f"kinetic acquisition timeout: exceeded {_timeout_s:.1f} s "
                        f"(trigger={eff_trigger}, frames={eff_num_kin})"
                    )
                time.sleep(0.05)
            raw = _hw._ccd.get_acquired_data()
        else:
            # Single / Accumulate: trigger_mode를 전달하여 software trigger 지원
            # internal 트리거면 무한 대기 허용, 외부/소프트웨어면 deadline 부여
            _timeout_ms = (
                None if eff_trigger == 'internal'
                else int((eff_exposure * max(eff_num_acc, 1) * 2 + 15) * 1000)
            )
            raw = _hw._ccd.start_acquisition_cycle(
                trigger_mode_str=eff_trigger,
                timeout_ms=_timeout_ms,
            )

    except Exception as e:
        return fail(str(e))

    finally:
        _laser_off_quiet()
        if restore_guide_beam:
            _restore_guide_beam_quiet()

    if raw is None:
        return fail("CCD data acquisition failed")

    # ── 결과 조립 ──
    # 보고하는 exposure_time / laser_power_pct / num_* 는 모두 '장비에 실제로 걸린 값'이다.
    if eff_acq == 'kinetic':
        # raw shape: (num_kin, Ny_ro, Nx_ro) — FVB이면 (num_kin, 1, Nx_ro)
        cal = getattr(_hw._ccd, '_calibrator', None)
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
        return _persist_spectrum(ok(mode="kinetic",
                                    num_frames=len(frames),
                                    kinetic_count=eff_num_kin,
                                    num_accumulations=eff_num_acc,
                                    exposure_time=eff_exposure,
                                    laser_power_pct=eff_power,
                                    trigger_mode=eff_trigger,
                                    shutter=eff_shutter,
                                    frames=frames))
    else:
        # Single / Accumulate: start_acquisition_cycle()이 calibration dict 반환
        data = raw
        if data.get("error"):
            return fail(data["error"])
        intensity = data["intensity"]
        result = ok(mode=eff_acq,
                    length=len(intensity),
                    max_intensity=float(max(intensity)) if intensity else 0.0,
                    sum_intensity=float(sum(intensity)) if intensity else 0.0,
                    data=intensity,
                    calibrated=data.get("calibrated", False),
                    exposure_time=eff_exposure,
                    laser_power_pct=eff_power,
                    trigger_mode=eff_trigger,
                    shutter=eff_shutter)
        if eff_acq == 'accumulate':
            result["num_accumulations"] = eff_num_acc
        if data.get("calibrated"):
            result["raman_shift_cm-1"] = data["raman_shift_cm-1"]
            result["wavelength_nm"]    = data["wavelength_nm"]
            result["laser_nm"]         = data["laser_nm"]
        return _persist_spectrum(result)


# ──────────────────────────────────────────
# 측정점 기록
# ──────────────────────────────────────────
# [create_session / save_point_data 를 대체한다 — 2026-07-30]
# 옛 두 툴은 data/sessions/<id>/ 라는 '세 번째 세션 체계'를 만들었다. run_store
# (data/runs/<label>/) 와 spectrum_store (data/results/<날짜>/<label>/) 가 이미 있는데
# 서로 참조하지 않아서, create_session("exp1") 을 만들어도 acquire_spectrum 결과는
# 거기 들어가지 않고 data/results 로 갔다 — "하나의 측정점 기록"이라는 요구와 정면으로
# 어긋났다. 게다가 save_point_data 는 spectrum_data=[...] 로 강도 배열을 인자로 받았는데,
# 관측 축약(_slim)이 길이 32 초과 리스트를 버리므로 에이전트는 그 배열을 애초에 갖고
# 있지 않다(= 지어내야 호출된다). 그리고 create_session 은 스테이지 드라이버의 DLL 세션
# 메서드(TangoController.create_session)와 이름까지 겹쳤다.
#
# 새 툴은 배열을 받지 않는다. 스펙트럼과 현미경 이미지는 이미 자동 저장되어 있으므로,
# '직전 측정 + 직전 캡처 + 현재 좌표'를 하나의 레코드로 묶어 run_store 세션에 남긴다.


def save_measurement_point(
    point_id: Annotated[str, Field(description="Short identifier for this point, e.g. 'P001'. Used in the filename.")],
    note: Annotated[Optional[str], Field(description='Optional note about this point (sample region, what you observed).')] = None,
) -> dict:
    """이 지점의 측정 결과를 '측정점 기록' 1건으로 묶어 저장한다.

    직전 acquire_spectrum 의 저장물과 직전 capture_scene 의 이미지, 그리고 현재 스테이지
    좌표를 한 JSON 레코드로 만들어 data/runs/<세션>/points/NN_<point_id>.json 에 남기고
    세션 manifest 에 인덱싱한다. 강도 배열을 인자로 넘길 필요가 없다 — 파일은 이미
    저장되어 있고, 이 툴은 그 포인터를 모을 뿐이다.

    Parameters
    ----------
    point_id : str
        포인트 식별자 (예: 'P001'). 파일명에 쓰이므로 짧게.
    note : str, optional
        이 지점에 대한 메모(시료 부위, 관찰 내용 등).

    Returns
    -------
    dict — ok, point_id, path(data/ 기준 상대경로), position,
           spectrum(참조한 측정 파일), image(참조한 현미경 이미지 URL)
    """
    try:
        from backend.service.store import run_store
    except Exception as e:
        return fail(f"run_store unavailable: {type(e).__name__}: {e}")

    pid = str(point_id or "").strip()
    if not pid:
        return fail("point_id is required (e.g. 'P001').")

    # ── 현재 좌표 ──
    position = None
    try:
        if _hw._stage is not None:
            p = _hw._stage.get_position()
            position = {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4),
                        "z": round(float(p[2]), 4)}
    except Exception:
        pass

    # ── 직전 측정의 저장물 ──
    spectrum = None
    _st = _sstate()
    _last_spectrum = _st["last_spectrum"]
    if _last_spectrum is not None:
        saved = _last_spectrum.get("saved") or {}
        files = saved.get("files") or {}
        spectrum = {
            "mode": _last_spectrum.get("mode"),
            "exposure_time": _last_spectrum.get("exposure_time"),
            "laser_power_pct": _last_spectrum.get("laser_power_pct"),
            "max_intensity": _last_spectrum.get("max_intensity"),
            "csv": files.get("csv"),
            "png": files.get("png"),
            "title": saved.get("title"),
        }
        if _last_spectrum.get("num_accumulations") is not None:
            spectrum["num_accumulations"] = _last_spectrum["num_accumulations"]

    # ── 직전 현미경 캡처 ──
    _last_scene = _st["last_scene"]
    image = dict(_last_scene) if _last_scene else None

    record = {
        "point_id": pid,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "position": position,
        "spectrum": spectrum,
        "image": image,
    }
    if note:
        record["note"] = str(note)

    # 무엇이 비었는지 명시해 준다 — "묶었다"고만 하고 실제로는 빈 레코드가 남는 것을 막는다.
    missing = [k for k in ("position", "spectrum", "image") if record.get(k) is None]

    try:
        filepath, rel = run_store.new_point_path(pid)
        filepath.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        run_store.record(run_store.KIND_POINT, rel, point_id=pid,
                         has_spectrum=spectrum is not None, has_image=image is not None)
    except Exception as e:
        return fail(f"Failed to write the point record: {type(e).__name__}: {e}")

    out = ok(point_id=pid, path=rel, position=position, spectrum=spectrum, image=image)
    if missing:
        out["missing"] = missing
        out["note_to_caller"] = (
            "This point record is missing: " + ", ".join(missing) + ". "
            "Acquire a spectrum (acquire_spectrum) and/or capture the view (capture_scene) at this "
            "position BEFORE calling save_measurement_point, so the record can reference them.")
    return out


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

# 오토포커스가 연속으로 이만큼 실패하면 격자 스캔을 중단한다. 한두 번은 그 점만의
# 문제(프레임 한 장을 놓쳤다)일 수 있지만, 연달아 실패하면 카메라·가이드빔·시료 높이 쪽
# 계통 문제다. 그 상태로 남은 점을 계속 쏘면 초점이 안 맞은 스펙트럼만 쌓이고 시료에는
# 조사량이 그대로 누적된다 — 되돌릴 수 없는 쪽(레이저)이 손해라 멈추는 편을 택한다.
_GRID_AF_ABORT_STREAK = 3

# run_grid_scan 한 번이 낼 수 있는 누적 조사량 상한(mJ)은 _GRID_MAX_DOSE_MJ 로 파일
# 상단에서 import 한다(backend.util.safety_limits). 에이전트 층의 per-turn 회로차단기는 도구
# 이름이 acquire_spectrum일 때만 도는데 run_grid_scan은 내부에서 N번 조사하므로, 여기서
# 독립적으로 한 번 더 막는다(방어적 이중화). 두 계층이 같은 상수·같은 공식을 쓰도록
# safety_limits 로 모았다 — 예전에는 1000.0 이 세 파일에 각각 박혀 있었다.

# 격자 점 개수 안전 상한(폭주 방지).
_GRID_MAX_POINTS = 400


# ── 그리드 스캔 사람-승인 게이트 (human-in-the-loop 하드 인터록) ──────────────────
# [왜 코드 게이트인가] 스키마/시스템 프롬프트로 "미리보기 후 멈추고 승인받아라"라고
# 지시해도 보장이 안 된다: ReAct(AILA)는 한 응답의 tool_call을 전부 실행하고, CoALA도
# preview와 run이 서로 다른 사이클이면 같은 턴 안에서 연속 실행될 수 있다. 레이저는
# 비가역이라, "실행 자체를 거부"하는 물리적 인터록이 프롬프트와 별개로 필요하다.
#
# [상태기계] state ∈ {"none","pending","armed"} + 승인된 격자 형상(geom).
#   preview_grid_scan 성공         → geom 저장, state="pending" (사람이 아직 못 봄)
#   grid_gate_begin_turn(대화)      → "pending"이면 "armed"(이번 턴 사람이 승인 가능),
#                                     "armed"인데 안 쓰였으면 만료 → "none"(재미리보기 필요)
#   run_grid_scan (enforce 시)      → "armed" + geom 일치일 때만 통과, 발사 직전 소비 → "none"
# 즉 미리보기와 실행 사이에 '사람 턴 경계'가 반드시 하나 끼도록 강제한다. 승인 창은 딱
# 한 턴(one-shot)이라, 취소하거나 무관한 요청을 한 뒤 stale 승인으로 실행되는 일이 없다.
#
# 벤치마크(run_experiment)는 사람이 없는 자율 평가이므로 grid_gate_begin_turn(interactive=
# False)로 게이트를 끈다(안 그러면 모든 벤치마크 격자 스캔이 승인 없이 거부된다).
#
# [왜 모듈 전역이 아니라 세션별인가 — 2026-07-31]
# 예전에는 "단일 장비·단일 사용자라 전역 dict로 충분하다"고 두었는데, 그러면 벤치마크
# 실행이 enforce를 끄는 순간 **동시에 열려 있던 대화 세션의 인터록까지 꺼졌다**.
# 즉 사람 승인 없이 레이저 격자 스캔이 나갈 수 있는 창이 열린다(테스트로 재현). 이제
# 세션 라벨로 키를 잡아, 한 세션이 게이트를 끄더라도 다른 세션은 영향받지 않는다.


def _gate() -> dict:
    """이 세션의 그리드 승인 게이트 상태."""
    return _sstate()["grid_gate"]


def _grid_gate_geom(rows, cols, spacing_mm, center_x, center_y) -> dict:
    """게이트 비교용 격자 형상 정규화 — 미리보기와 실행이 '같은 격자'인지 판정하는 기준.
    형상(rows/cols/spacing/center)만 본다. power/exposure/autofocus는 dose 회로차단기가
    따로 막으므로 승인 대상이 아니다(사람이 눈으로 승인한 것은 '어디에 몇 점을'이다).
    center는 모델이 넘긴 원값(None 포함)으로 비교한다 — 실측 위치로 해석하면 미리보기와
    실행 사이 스테이지 이동으로 값이 달라져 오탐이 난다."""
    return {
        "rows": int(rows), "cols": int(cols),
        "spacing_mm": round(float(spacing_mm), 4),
        "center_x": None if center_x is None else round(float(center_x), 4),
        "center_y": None if center_y is None else round(float(center_y), 4),
    }


def grid_gate_begin_turn(interactive: bool = True) -> None:
    """에이전트가 새 사용자 턴을 시작할 때 호출하는 훅(AILA/CoALA 공용).

    interactive=True (SSE 대화): 사람 승인 게이트 ON. 직전 턴에 만든 미리보기가 있으면
      '이제 사람이 보고 승인할 턴'이라는 뜻으로 pending→armed. 이미 armed였는데 실행에
      쓰이지 않았으면 승인 창이 만료된 것이라 none으로 되돌린다(재미리보기 강제).
    interactive=False (벤치마크 자율 실행): 게이트 OFF + 상태 초기화 — 사람이 없으므로
      미리보기 없이도 run_grid_scan을 허용한다."""
    _gate()["enforce"] = bool(interactive)
    if not interactive:
        _gate()["geom"] = None
        _gate()["resolved"] = None
        _gate()["state"] = "none"
        return
    if _gate()["state"] == "pending":
        _gate()["state"] = "armed"
    elif _gate()["state"] == "armed":
        # 지난 턴에 승인 가능(armed)했으나 실행하지 않았다 → 승인 창 만료.
        _gate()["state"] = "none"
        _gate()["geom"] = None
        _gate()["resolved"] = None


def _grid_gate_on_preview(geom: dict, resolved: tuple) -> None:
    """preview_grid_scan 성공 시 호출 — 승인 대기(pending) 상태로 만든다.

    geom 은 모델이 넘긴 원값(None 포함), resolved 는 미리보기가 실제로 그린 격자 중심(mm).
    둘 다 저장하는 이유 — 2026-08-01:
      center 를 생략한 미리보기는 geom 에 center=None 으로 남는다. 그런데 승인을 기다리는
      동안(또는 승인 턴에) 스테이지가 움직이면, run_grid_scan 이 center=None 을 다시
      '현재 위치'로 풀면서 **사람이 승인한 곳이 아닌 새 위치에 레이저를 쏜다**. geom 비교는
      None==None 이라 통과하므로 게이트가 이걸 못 잡았다.
      resolved 를 함께 남겨 두고 실행 때 그 좌표를 쓰면, 승인된 그림과 실제 발사 위치가
      항상 일치한다. geom 비교 규약(원값)은 그대로 둔다 — 스테이지 이동을 'Approval
      mismatch' 로 거부하면 모델이 왜 거부됐는지 알 수 없기 때문이다.
    """
    _gate()["geom"] = geom
    _gate()["resolved"] = {"x": round(float(resolved[0]), 4),
                           "y": round(float(resolved[1]), 4)}
    _gate()["state"] = "pending"


def _grid_gate_check(geom: dict):
    """run_grid_scan 진입 시 승인 여부 검사(부작용 없음). 통과면 None, 거부면 에러 dict를
    반환한다 — 에이전트는 이 관측을 읽고 '미리보기→턴 종료→대기'로 돌아가게 된다.
    실제 소비(승인 1회 사용)는 발사 직전 _grid_gate_consume()에서 한다 — 사전검증
    (범위/None) 실패로 승인이 헛되이 소모되지 않도록 검사와 소비를 분리한다."""
    if not _gate()["enforce"]:
        return None
    state = _gate()["state"]
    if state == "none" or _gate()["geom"] is None:
        return fail("Human approval required: no approved grid preview is on record. Call preview_grid_scan "
                    "first, show the user the preview, end your turn, and run only after the user approves.")
    if state == "pending":
        return fail("Human approval required: the grid preview has NOT been approved yet. Do not run in the "
                    "same turn as the preview - end your turn now, let the user see the preview, and call "
                    "run_grid_scan only after the user explicitly approves it in a new message.")
    if geom != _gate()["geom"]:
        return fail(f"Approval mismatch: the user approved grid {_gate()['geom']} but this run requests "
                    f"{geom}. Preview the exact grid again and get the user's approval before running.")
    return None


def _grid_gate_approved_center():
    """승인된 미리보기가 실제로 그린 격자 중심(x, y) — 없으면 None.

    게이트가 꺼져 있으면(벤치마크) None 을 준다: 승인 절차 자체가 없으므로 고정할
    '승인된 자리'도 없고, 호출자는 현재 스테이지 위치를 쓰면 된다.
    """
    if not _gate()["enforce"] or _gate()["state"] != "armed":
        return None
    r = _gate().get("resolved")
    if not r:
        return None
    return (float(r["x"]), float(r["y"]))


def _grid_gate_consume() -> None:
    """승인을 1회 소비한다 — 발사 직전(모든 사전검증 통과 후) 호출. 게이트가 꺼져 있으면
    (벤치마크) 아무 것도 하지 않는다."""
    if not _gate()["enforce"]:
        return
    _gate()["geom"] = None
    _gate()["resolved"] = None
    _gate()["state"] = "none"


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


def _validate_grid_args(rows, cols, spacing_mm):
    """preview/run 공통 인자 검증 — 실패 시 error dict, 통과 시 None."""
    if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
        return fail("rows and cols must be integers >= 1.")
    if rows * cols > _GRID_MAX_POINTS:
        return fail(f"Too many points ({rows*cols}); max {_GRID_MAX_POINTS}.")
    try:
        if float(spacing_mm) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return fail("spacing_mm must be a number > 0.")
    return None


def _validate_grid_range(pts):
    """격자 점이 모두 스테이지 가동범위 안에 있는지 — 실패 시 error dict, 통과 시 None.

    [왜 preview 와 run 이 같은 함수를 쓰는가 — 2026-08-01]
    예전에는 이 검증이 run_grid_scan 에만 있었다. 그래서 가동범위를 벗어나는 격자를
    preview_grid_scan 은 ok=True 로 통과시키고(화면 밖 점은 cv2 가 클립해 그려 준다),
    사람이 그 미리보기를 승인한 뒤 run_grid_scan 에서야 "범위 밖"으로 거부됐다.
    모델 입장에서는 '승인까지 받은 계획이 실행 단계에서만 거부되는' 모순이라 복구
    경로를 못 찾는다. 불가능한 격자는 미리보기 단계에서 막는다.
    """
    for idx, i, j, sx, sy in pts:
        if not (0 <= sx <= STAGE_MAX_X and 0 <= sy <= STAGE_MAX_Y):
            return fail(f"Point {idx} (row {i}, col {j}) at X={sx}, Y={sy} mm is outside the stage range "
                        f"(0..{STAGE_MAX_X} x 0..{STAGE_MAX_Y}). Adjust center/spacing/size.")
    return None


@_serialized("preview_grid_scan")
def preview_grid_scan(
    rows: Annotated[int, Field(description='Number of grid points stacked VERTICALLY = grid HEIGHT (stage Y axis), integer >= 1. e.g. rows=3 -> 3 points tall.')],
    cols: Annotated[int, Field(description='Number of grid points side-by-side HORIZONTALLY = grid WIDTH (stage X axis), integer >= 1. e.g. cols=2 -> 2 points wide.')],
    spacing_mm: Annotated[float, Field(description='Distance between adjacent points (mm), > 0')],
    center_x: Annotated[Optional[float], Field(description='Grid center X (mm). Optional; defaults to current stage X')] = None,
    center_y: Annotated[Optional[float], Field(description='Grid center Y (mm). Optional; defaults to current stage Y')] = None,
) -> dict:
    """격자 스캔 '미리보기'. 스테이지 이동·레이저 조사 없이, rows×cols 격자 위치를 현재
    카메라 화면에 원으로 오버레이한 이미지를 반환한다(에이전트가 보고 승인/수정하도록).

    center_* 미지정 시 현재 스테이지 위치를 격자 중심으로 쓴다. 화면 중심 = 현재 스테이지
    위치이므로, 중심을 현재 위치로 두면 격자가 화면 중앙에 대칭으로 그려진다. 카메라 시야(FOV)는
    좁아(≈0.43×0.30mm) 격자가 넓으면 일부 점은 화면 밖이다 — 화면 안 점만 세어 함께 알려준다.
    """
    err = _validate_grid_args(rows, cols, spacing_mm)
    if err:
        return err
    stage_err = _stage_unavailable()
    if stage_err:
        return stage_err
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    try:
        import base64
        import cv2

        spacing_mm = float(spacing_mm)
        pos = _hw._stage.get_position()
        if pos is None:
            return fail("Failed to query stage position")
        cur_x, cur_y = float(pos[0]), float(pos[1])          # 화면 중심 = 현재 위치
        cx = cur_x if center_x is None else float(center_x)  # 격자 중심
        cy = cur_y if center_y is None else float(center_y)

        frame = _hw._camera.get_latest_frame()
        if frame is None:
            return fail("Failed to acquire frame (check whether streaming is active)")
        frame_bgr = _vis.to_view_bgr(frame)     # 좌표계 정규화(다른 캡처 도구와 동일)

        pts = _grid_stage_coords(cx, cy, rows, cols, spacing_mm)
        # 가동범위 검증을 run_grid_scan 과 같은 함수로 여기서도 한다 — 실행 단계에서만
        # 거부되는 격자를 사람이 승인하게 두지 않는다(_validate_grid_range 주석 참고).
        range_err = _validate_grid_range(pts)
        if range_err:
            return range_err
        n_in_view = 0
        for idx, i, j, sx, sy in pts:
            # optics_map.stage_to_pixel = move_to_pixel 의 정확한 역변환(단일 출처)
            px, py = _om.stage_to_pixel(sx, sy, cur_x, cur_y)
            ipx, ipy = int(round(px)), int(round(py))
            if 0 <= ipx < CAMERA_WIDTH and 0 <= ipy < CAMERA_HEIGHT:
                n_in_view += 1
            # 화면 밖 점도 cv2.circle이 자동 클립하므로 그냥 그린다(가장자리 힌트).
            cv2.circle(frame_bgr, (ipx, ipy), _GRID_SPOT_RADIUS_PX, (0, 255, 0), 2)      # green: 스캔 점
        # (요청) 미리보기에는 스캔 예정점(초록 원)만 표시한다. 화면 중심의 '현재 조사점'
        # 빨간 링은 스캔 계획과 무관해 혼동을 줄 수 있어 그리지 않는다.

        ret, buf = cv2.imencode('.png', frame_bgr)
        if not ret:
            return fail("PNG encoding failed")
        img_b64 = base64.b64encode(buf).decode('utf-8')
        # 같은 오버레이를 파일로도 저장해 image_url을 실으면, spectrum_event 배선을 타고
        # 이 미리보기가 채팅창에 그대로 인라인 표시된다(프론트 수정 불필요). 에이전트는 위의
        # image_base64로 '보고' 판단하고, 사람은 채팅에서 같은 그림을 본다.
        saved_img = _store_save_preview(buf.tobytes(), tag="grid_preview")

        fov_x, fov_y = _om.fov_mm()             # capture_scene 의 extent 와 같은 출처
        span_x, span_y = (cols - 1) * spacing_mm, (rows - 1) * spacing_mm
        question = (
            f"Grid scan PREVIEW (not executed yet): {rows} rows x {cols} cols = {rows*cols} points "
            f"-> a {cols}-wide x {rows}-tall grid, {spacing_mm} mm spacing, "
            f"centered at stage (X={cx:.4f}, Y={cy:.4f}) mm. "
            f"Green circles mark the points to be scanned. "
            f"Grid span {span_x:.3f}x{span_y:.3f} mm vs camera view ~{fov_x:.3f}x{fov_y:.3f} mm; "
            f"{n_in_view}/{rows*cols} points fall within the current view (the rest are outside the frame "
            f"but will still be measured). "
            f"STOP HERE - do NOT run the scan yet. Show this preview to the user, confirm the "
            f"{cols}-wide x {rows}-tall orientation is what they asked for, then END YOUR TURN and WAIT "
            f"for their explicit approval. Only in a LATER turn, after the user approves, call run_grid_scan "
            f"with these SAME parameters. If the layout is wrong, call preview_grid_scan again with adjusted "
            f"rows/cols/spacing_mm/center."
        )
        out = ok(rows=rows,
                 cols=cols,
                 spacing_mm=spacing_mm,
                 center={"x": round(cx, 4), "y": round(cy, 4)},
                 n_points=rows * cols,
                 n_in_view=n_in_view,
                 fov_mm={"x": round(fov_x, 4), "y": round(fov_y, 4)},
                 image_base64=img_b64,
                 question=question)
        if saved_img.get("ok"):
            # image_file 은 모델용 — 승인 대기 중 턴이 넘어가 base64 가 히스토리에서
            # 빠져도 view_image(file_id) 로 이 미리보기를 다시 볼 수 있다.
            out["image_file"] = saved_img["file_id"]
            out["saved"] = {"title": f"Grid preview {rows}x{cols} @ {spacing_mm}mm",
                            "image_url": saved_img["image_url"]}
        # 사람-승인 게이트: 이 미리보기를 '승인 대기(pending)'로 등록한다. 이후 사용자
        # 턴 경계에서 armed로 올라가야 run_grid_scan이 통과한다(같은 턴 즉시 실행 차단).
        # resolved(cx, cy)를 함께 남긴다 — center 를 생략한 미리보기라도 실행이 '이 자리'에
        # 고정되도록(스테이지가 움직여도 승인된 그림과 발사 위치가 어긋나지 않게) 한다.
        _grid_gate_on_preview(_grid_gate_geom(rows, cols, spacing_mm, center_x, center_y),
                              (cx, cy))
        return out
    except Exception as e:
        return fail(str(e))


@_serialized("run_grid_scan")
def run_grid_scan(
    rows: Annotated[int, Field(description='Number of grid points stacked VERTICALLY = grid HEIGHT (stage Y axis), integer >= 1. e.g. rows=3 -> 3 points tall. Must match the approved preview.')],
    cols: Annotated[int, Field(description='Number of grid points side-by-side HORIZONTALLY = grid WIDTH (stage X axis), integer >= 1. e.g. cols=2 -> 2 points wide. Must match the approved preview.')],
    spacing_mm: Annotated[float, Field(description='Distance between adjacent points (mm), > 0')],
    exposure: Annotated[float, Field(description='Exposure time per point (s) - REQUIRED. Together with power this sets how much light each point receives, so choose it for THIS sample rather than reusing a number: too short buries the peaks in read noise, too long saturates the detector and multiplies the total run time by the point count.')],
    power: Annotated[float, Field(description="Laser power (%) per point - REQUIRED. This is the dose decision, and it is applied at EVERY point, so the sample sees it rows*cols times. Higher power raises signal but photobleaches or burns fragile samples (biological, polymer, thin film); if you are unsure of the sample's tolerance, start low and check one point before committing the whole grid. The cumulative dose is estimated up front and the scan is refused outright if it exceeds the limit.")],
    autofocus: Annotated[Literal['each', 'center', 'none'], Field(description="Autofocus strategy. 'each' = autofocus at every point (most accurate, slowest); 'center' = autofocus once at the grid center then reuse that Z (fast, for flat samples); 'none' = no autofocus, keep current Z. REQUIRED - this is a real trade-off, not a formality: 'each' costs an extra Z sweep (and guide-beam exposure) at every point, which on a large grid dominates the run time, while 'center' or 'none' will drift out of focus on a tilted or uneven sample and quietly return weak spectra. Decide from what you know about the sample's flatness.")],
    center_x: Annotated[Optional[float], Field(description='Grid center X (mm). Optional; defaults to current stage X')] = None,
    center_y: Annotated[Optional[float], Field(description='Grid center Y (mm). Optional; defaults to current stage Y')] = None,
) -> dict:
    """격자 스캔 '실행'. rows×cols 격자를 내부 루프로 순회하며 각 점에서
    이동→(오토포커스)→스펙트럼 측정·자동저장하고, 압축 요약 1개만 반환한다.

    autofocus:
      "each"   — 매 점에서 오토포커스(가장 정확, 느림; 예전 수동 방식과 동일)
      "center" — 격자 중심에서 1회만 오토포커스 후 그 Z로 전체 측정(빠름, 평탄 시료용)
      "none"   — 오토포커스 없이 현재 Z로 측정

    [exposure / power / autofocus 에 기본값이 없는 이유 — 2026-07-31]
    예전에는 0.2s / 40% / "each" 가 기본값이었다. 그런데 이 셋은 '실험 조건' 자체이고
    격자 전체에 rows*cols 번 반복 적용된다 — 생략하면 도구가 조사량을 대신 결정하는 셈이다.
    필수 인자로 바꿔 호출자가 반드시 값을 정하게 한다(스키마 required 와 일치시킨다).

    레이저 조사가 실제로 일어나므로, 예상 누적 조사량이 상한을 넘으면 시작 전에 거부한다.
    """
    err = _validate_grid_args(rows, cols, spacing_mm)
    if err:
        return err
    if autofocus not in ("each", "center", "none"):
        return fail("autofocus must be one of: each, center, none.")
    # 사람-승인 게이트(하드 인터록) — 하드웨어를 만지기 전에 먼저 막는다. 승인된 미리보기
    # 없이 레이저 격자 스캔을 실행하지 않는다. 실제 소비는 발사 직전(_grid_gate_consume).
    gate_err = _grid_gate_check(_grid_gate_geom(rows, cols, spacing_mm, center_x, center_y))
    if gate_err:
        return gate_err
    # 승인된 미리보기가 실제로 그렸던 중심. center 를 생략했을 때 '현재 위치'로 다시 푸는
    # 대신 이 값을 쓴다 — 승인 이후 스테이지가 움직여도 승인된 자리에 쏜다(#12).
    approved_center = _grid_gate_approved_center()
    stage_err = _stage_unavailable()
    if stage_err:
        return stage_err
    if _hw._ccd is None:
        return fail("CCD is not initialized (cooling or not connected).")
    if _hw._laser is None:
        return fail("Laser is not initialized.")
    if autofocus != "none" and _hw._camera is None:
        return fail("Camera is not initialized (required for autofocus). "
                    "Use autofocus='none' to skip.")

    spacing_mm = float(spacing_mm)
    exposure, power = float(exposure), float(power)
    n = rows * cols
    # 조사량 공식은 safety_limits 단일 출처 — 에이전트 계층의 턴 누계와 같은 척도여야
    # 두 한계값이 같은 의미를 갖는다.
    dose_total = estimate_dose_mj(power, exposure, n)
    if dose_total > _GRID_MAX_DOSE_MJ:
        return fail(f"Safety block: estimated cumulative dose {dose_total:.1f} mJ exceeds the grid limit "
                    f"({_GRID_MAX_DOSE_MJ} mJ). Reduce point count, power, or exposure.")

    # 점마다 광학계를 왕복시키면 ND/빔스플리터 모터 이동만 2n 번 늘어난다. 스캔 도중에는
    # 카메라를 볼 사람도 없으므로, 복귀는 아래 finally 에서 '실제로 쏜 경우 한 번만' 한다.
    # try 밖에서 정의한다 — 사전검증 도중 예외가 나도 finally 가 이 이름을 읽는다.
    fired = False

    try:
        pos = _hw._stage.get_position()
        if pos is None:
            return fail("Failed to query stage position")
        # 생략된 center 는 '승인된 미리보기가 그린 자리' → 없으면 현재 위치 순으로 푼다.
        fallback_x, fallback_y = (approved_center if approved_center
                                  else (float(pos[0]), float(pos[1])))
        cx = fallback_x if center_x is None else float(center_x)
        cy = fallback_y if center_y is None else float(center_y)

        pts = _grid_stage_coords(cx, cy, rows, cols, spacing_mm)
        # 범위 사전 검증 — 한 점이라도 벗어나면 시작조차 하지 않는다(preview 와 같은 함수).
        range_err = _validate_grid_range(pts)
        if range_err:
            return range_err

        # center 모드: 격자 중심으로 이동 후 1회 오토포커스(이후 Z 유지).
        if autofocus == "center":
            mv = move_stage(x=cx, y=cy)
            if not mv.get("ok"):
                return fail(f"Failed to move to grid center: {mv.get('error')}")
            af = run_autofocus()
            if not af.get("ok"):
                return fail(f"Autofocus at grid center failed: {af.get('error')}")

        # 모든 사전검증(범위/None/오토포커스) 통과 — 이제 레이저를 쏜다. 승인을 여기서
        # 소비한다(1회용). 이 지점 이후 재실행하려면 다시 미리보기·승인이 필요하다.
        _grid_gate_consume()

        results = []
        n_ok = 0
        af_failed, af_limit_hits, af_streak = 0, 0, 0
        aborted = None
        for idx, i, j, sx, sy in pts:
            mv = move_stage(x=sx, y=sy)
            if not mv.get("ok"):
                results.append(fail(mv.get("error"), i=idx, row=i, col=j, x=sx, y=sy))
                continue

            # ── 오토포커스 결과를 '읽는다' — 2026-07-31 ────────────────────────
            # 예전에는 run_autofocus() 를 부르고 반환을 통째로 버렸다("실패해도 치명적이지
            # 않다"). 그런데 초점이 안 맞은 Z 에서 찍은 스펙트럼은 신호가 약하거나 사실상
            # 빈 스펙트럼이고, 그걸 성공으로 세어 "25/25 측정 완료"라고 보고하면 호출자는
            # 데이터가 왜 이상한지 알 방법이 없다. 실패를 점별로 남기고 요약에 싣는다.
            af_ok = None
            if autofocus == "each":
                af = run_autofocus()
                af_ok = bool(af.get("ok"))
                if af_ok:
                    af_streak = 0
                    if af.get("z_limit_hits"):
                        af_limit_hits += 1
                else:
                    af_failed += 1
                    af_streak += 1
                    # 연속 실패는 이 점의 문제가 아니라 계통적 문제다(카메라 정지, 시료가
                    # Z 가동범위 밖, 가이드빔 미출력). 남은 점을 계속 쏘아 봐야 초점이 안 맞은
                    # 스펙트럼만 쌓이고 시료에는 조사량만 누적된다 — 여기서 멈춘다.
                    if af_streak >= _GRID_AF_ABORT_STREAK:
                        aborted = {
                            "at_point": idx, "row": i, "col": j,
                            "reason": (
                                f"Autofocus failed {af_streak} times in a row (last error: "
                                f"{af.get('error')}). That is a systematic problem, not a bad point - "
                                f"the camera may have stopped streaming, the guide beam may be off, or "
                                f"the sample may sit outside the Z travel range. The scan was STOPPED "
                                f"here instead of firing the laser at the remaining points out of focus."),
                        }
                        break

            fired = True
            res = _cache_and_return(acquire_spectrum(exposure=exposure, power=power,
                                                     restore_guide_beam=False))
            if res.get("ok"):
                n_ok += 1
                files = (res.get("saved") or {}).get("files") or {}
                ref = files.get("csv") or files.get("png") or ""
                fname = ref.replace("\\", "/").rsplit("/", 1)[-1]
                rec = {"i": idx, "x": sx, "y": sy,
                       "max_intensity": res.get("max_intensity"), "file": fname}
                if af_ok is False:
                    # 측정 자체는 됐지만 초점이 안 맞은 Z 에서 찍혔다 — 값을 그대로 믿으면 안 된다.
                    rec["autofocus_failed"] = True
                results.append(rec)
            else:
                results.append(fail(res.get("error"), i=idx, row=i, col=j, x=sx, y=sy))

        # 압축 반환 — 큰 격자에서 per-point 리스트가 에이전트 _slim(길이>32 리스트 폐기)에
        # 통째로 걸리지 않도록, 집계 통계는 항상 싣고 per-point는 32점 이하일 때만 인라인.
        oks = [r for r in results if "max_intensity" in r]
        fails = [r for r in results if r.get("ok") is False]
        inten = [r["max_intensity"] for r in oks if r.get("max_intensity") is not None]
        out = {
            # 중단됐으면 성공이 아니다 — 여기서 ok:True 를 주면 호출자가 "격자 스캔 완료"로
            # 보고해 버린다. 측정된 점들은 이미 자동저장돼 있으므로 데이터가 사라지지는 않는다.
            "ok": aborted is None,
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
        if aborted is not None:
            out["aborted"] = aborted
            out["error"] = aborted["reason"]
            out["note"] = (
                f"STOPPED EARLY at point {aborted['at_point']} of {n}. The {n_ok} point(s) measured "
                f"before that were auto-saved and are still usable, but the grid is INCOMPLETE - do not "
                f"report it as finished. Fix the focus problem (check the camera stream and the sample "
                f"height with run_autofocus on its own), then preview and re-run the grid.")
        if autofocus == "each" and af_failed:
            out["n_autofocus_failed"] = af_failed
            out.setdefault("warnings", []).append(
                f"Autofocus failed at {af_failed} of the points that were measured. Those spectra were "
                f"taken at whatever Z the stage happened to be at, so they may be out of focus and their "
                f"intensities are not comparable with the rest - the affected points are marked with "
                f"autofocus_failed. Do not read a weak signal there as a property of the sample.")
        if af_limit_hits:
            out["n_autofocus_z_limit"] = af_limit_hits
            out.setdefault("warnings", []).append(
                f"At {af_limit_hits} point(s) the focus search ran into the Z travel limit, meaning the "
                f"true focus is probably outside the reachable range. Repeating the scan will not help - "
                f"the sample or the objective needs to be repositioned.")
        if fails:
            out["failed_points"] = fails[:10]
        if n <= 32:
            out["points"] = oks
        return out
    except Exception as e:
        return fail(str(e))
    finally:
        # 스캔이 어떻게 끝나든(완주 / 중단 / 예외) 광학계를 카메라 위치로 되돌린다 —
        # 마지막 점의 acquire_spectrum 이 restore_guide_beam=False 로 돌았기 때문에
        # 여기서 안 되돌리면 스캔 후 카메라 화면이 캄캄한 채로 남는다.
        # 한 점도 쏘지 않았으면(사전검증 실패 등) 모터를 건드리지 않는다 — 미리 걸어 둔
        # 파워를 이유 없이 해제하지 않기 위해서다.
        if fired:
            _restore_guide_beam_quiet()
