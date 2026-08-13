# -*- coding: utf-8 -*-
"""CCD(Andor) 설정 도구.

설정은 지속된다. 같은 값을 acquire_spectrum 의 인자로도 넘길 수 있는데, 두 경로 모두
hw_core 의 _apply_* 를 통과하므로 허용값·순서·부작용이 구조적으로 같다.
"""
from __future__ import annotations

from backend.tools.result import fail, ok
from pydantic import Field
from typing import Annotated, Literal, Optional
from backend.tools.hw_tools.hw_tools import hw_core as _hw
from backend.tools.hw_tools.hw_tools.hw_core import _apply_acq_mode, _apply_read_mode, _apply_shutter, _apply_trigger_mode, _ccd_ready, _check_ccd_positive, _current_read_mode, _serialized


# ──────────────────────────────────────────
# CCD 파라미터 설정 툴
# ──────────────────────────────────────────

def get_ccd_info() -> dict:
    """현재 CCD 설정값 및 상태를 한 번에 조회한다."""
    err = _ccd_ready()
    if err:
        return err
    try:
        t           = _hw._ccd.get_temperature()
        temp_status = _hw._ccd.get_temperature_status()
        cam_status  = _hw._ccd.get_status()
    except Exception as e:
        return fail(str(e))

    def _attr(name):
        return getattr(_hw._ccd, name, None)

    hs_conv = None
    if hasattr(_hw._ccd, 'HSSpeeds_Conventional') and _hw._ccd.HSSpeeds_Conventional:
        hs_conv = _hw._ccd.HSSpeeds_Conventional[0]

    return ok(camera_status=cam_status,
              temperature_C=t,
              temperature_status=temp_status,
              cooler_on=_attr('cooler_on'),
              exposure_time_s=_attr('exposure_time'),
              acquisition_mode=_attr('aq_mode'),
              read_mode=_current_read_mode(),
              read_mode_driver=_attr('ro_mode'),
              trigger_mode=_attr('trigger_mode'),
              shutter_mode=_attr('shutter_mode'),
              num_accumulations=_attr('num_acc'),
              num_kinetics=_attr('num_kin'),
              em_mode=_attr('em_mode'),
              em_gain=_attr('em_gain'),
              output_amp=_attr('output_amp'),
              preamp_gain_index=_attr('preamp_gain_i'),
              preamp_gains_available=_attr('preamp_gains'),
              vs_speeds_us=_attr('VSSpeeds'),
              hs_speeds_conventional_mhz=hs_conv,
              readout_pixels_Nx=_attr('Nx_ro'),
              readout_pixels_Ny=_attr('Ny_ro'),
              detector_Nx=_attr('Nx'),
              detector_Ny=_attr('Ny'))


@_serialized("set_ccd_exposure")
def set_ccd_exposure(
    exposure_time: Annotated[float, Field(description='Exposure time [seconds]. e.g. 0.1, 0.5, 1.0')],
) -> dict:
    """CCD 노출 시간(초)을 설정한다."""
    err = _ccd_ready()
    if err:
        return err
    if exposure_time <= 0:
        return fail("Exposure time must be greater than 0.")
    try:
        actual = _hw._ccd.set_exposure_time(exposure_time)
        return ok(exposure_time_s=actual)
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_acquisition_mode")
def set_ccd_acquisition_mode(
    mode: Annotated[Literal['single', 'accumulate', 'kinetic', 'run_till_abort'], Field(description='Acquisition mode')],
    num_accumulations: Annotated[Optional[int], Field(ge=1, description='Number of accumulations (used in accumulate/kinetic mode). Omit to keep the value already on the CCD - 0 is not a way to switch accumulation off and is rejected. Note that accumulate mode with 1 accumulation is just a single shot - set this deliberately when you want averaging.')] = None,
    num_kinetics: Annotated[Optional[int], Field(ge=1, description='Total number of frames to acquire (used in kinetic mode). Omit to keep the value already on the CCD.')] = None,
) -> dict:
    err = _ccd_ready()
    if err:
        return err
    # 이 도구의 인자는 num_kinetics 인데 공용 경로(_apply_acq_mode)는 같은 값을
    # kinetic_count 로 부른다. 검증을 거기에만 맡기면 "kinetic_count must be at least 1"
    # 처럼 **이 도구에 존재하지 않는 인자명**으로 거절당해, 모델이 무엇을 고쳐야 할지
    # 알 수 없다(get_ccd_info 가 read_mode 를 도구 표기로 되돌려주는 것과 같은 이유).
    err = _check_ccd_positive("num_kinetics", num_kinetics, integer=True)
    if err:
        return err
    try:
        # 적용은 acquire_spectrum 과 완전히 같은 경로를 쓴다(허용값·순서 단일 출처).
        err = _apply_acq_mode(mode, num_accumulations=num_accumulations,
                              kinetic_count=num_kinetics)
        if err:
            return err
        out = ok(acquisition_mode=mode,
                 num_accumulations=getattr(_hw._ccd, 'num_acc', None),
                 num_kinetics=getattr(_hw._ccd, 'num_kin', None))
        if mode == 'run_till_abort':
            out["note"] = ("acquire_spectrum cannot run in 'run_till_abort'. Switch back with "
                           "set_ccd_acquisition_mode('single') before measuring, or pass "
                           "acq_mode explicitly to acquire_spectrum.")
        return out
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_trigger_mode")
def set_ccd_trigger_mode(
    mode: Annotated[Literal['internal', 'external', 'external_start', 'external_exposure', 'external_fvb_em', 'software'], Field(description='Trigger mode')],
) -> dict:
    """
    CCD 트리거 모드를 설정한다.

    mode: 'internal' | 'external' | 'external_start' |
          'external_exposure' | 'external_fvb_em' | 'software'

    acquire_spectrum(trigger_mode=) 와 **같은 코드**를 지나므로 허용값이 항상 일치한다
    (예전에는 이쪽에만 external_fvb_em 이 빠져 있어 같은 값이 한쪽에서만 통했다).
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        err = _apply_trigger_mode(mode)
        if err:
            return err
        return ok(trigger_mode=mode)
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_read_mode")
def set_ccd_read_mode(
    mode: Annotated[Literal['fvb', 'single_track', 'image'], Field(description='Readout mode')],
    hbin: Annotated[Optional[int], Field(description='Horizontal binning factor. Omit to keep the current value.')] = None,
    single_track_center: Annotated[Optional[int], Field(description='Center row number to read in single_track mode (1-based), e.g. 256. Required the first time you use single_track; omit to reuse the configured track.')] = None,
    single_track_width: Annotated[Optional[int], Field(description='Number of rows to read in single_track mode. Omit to keep the current value.')] = None,
) -> dict:
    """
    CCD 읽기 모드(readout mode)를 설정한다.

    mode:
      'fvb'          — Full Vertical Binning (1D 스펙트럼, 기본)
      'single_track' — 특정 수직 행 하나만 읽음. single_track_center(행 번호) 필수
      'image'        — 2D 이미지 전체 (acquire_spectrum 은 1D만 조립하므로 측정 전엔 부적합)

    hbin / single_track_center / single_track_width 는 acquire_spectrum 과 이름을 맞췄고,
    적용도 **같은 코드**를 지난다. None 이면 현재 설정값을 그대로 유지한다.

    'image' 는 이 도구로만 걸 수 있다 — acquire_spectrum 은 1D 스펙트럼을 조립하므로
    2D 이미지 모드로 두면 측정이 거부된다. 이미지 모드는 set_ccd_image_flip 처럼
    2D 전용 설정을 만질 때만 쓰고, 측정 전에 'fvb' 로 되돌린다.
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        err = _apply_read_mode(mode, hbin=hbin,
                               single_track_center=single_track_center,
                               single_track_width=single_track_width)
        if err:
            return err
        out = ok(read_mode=mode, hbin=_hw._ccd.get_current_hbin(), Nx_ro=_hw._ccd.Nx_ro, Ny_ro=_hw._ccd.Ny_ro)
        if mode == 'image':
            out["note"] = ("acquire_spectrum cannot assemble a 1D spectrum in 'image' mode. "
                           "Call set_ccd_read_mode('fvb') before measuring.")
        return out
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_preamp_gain")
def set_ccd_preamp_gain(
    index: Annotated[int, Field(description='Pre-amplifier gain index (0-based). Typically in the range 0-2')],
) -> dict:
    """
    프리앰프(PreAmp) 이득 인덱스를 설정한다.
    사용 가능한 이득 목록은 get_ccd_info()의 preamp_gains_available 참조.
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        _hw._ccd.set_preamp_gain(index)
        gain_val = _hw._ccd.preamp_gains[index] if _hw._ccd.preamp_gains else None
        return ok(preamp_gain_index=index, gain_value=gain_val)
    except Exception as e:
        return fail(str(e))


# [제거됨 — 이 장비가 지원하지 않는 이득 도구들]
#
# ── set_mcp_gain / get_mcp_gain_range (2026-07-30) ──
# MCP(Micro-Channel Plate) 이득 미지원. 실측 로그에서 SDK 가 GetMCPGainRange/SetMCPGain 에
# DRV_NOT_SUPPORTED(20991) 를 반환한다(iStar ICCD 전용 기능).
#
# ── set_ccd_em_gain / set_ccd_output_amp (2026-07-31) ──
# 이 카메라는 EM CCD 가 아니라 iDus 다 — Config.ini 가 명시한다:
#     /*CCDType : 0 (IDUS), 1 (EM)*/   →  [ANDOR_IDUS]
# 그래서 두 툴은 구조적으로 아무 일도 못 했다:
#   · set_ccd_em_gain   : em_mode 가 False 라 **항상** "This camera is not an EM CCD." 만 반환
#   · set_ccd_output_amp: 일반 CCD 는 출력 앰프가 하나뿐이라 고를 대상이 없다
#                         (hardware_manager._init_ccd 도 em_mode 일 때만 set_output_amp 를 부른다)
# 남겨 두면 MCP 때와 같은 함정이 된다 — "포화됐으니 게인을 낮춰라" 류의 과제에서 에이전트가
# 반드시 실패하는 경로로 유인되고, 실패 원인이 '툴이 없어서'가 아니라 '툴이 거짓말을 해서'가
# 되어 스스로 회복하지 못한다.
#
# 이 장비에서 실제로 이득을 조절하는 수단은 하나다:
#     set_ccd_preamp_gain(index)  +  get_ccd_info()의 preamp_gains_available
# 드라이버 메서드(AndorCCD.set_EMCCD_gain / set_output_amp)는 그대로 둔다 —
# hardware_manager 가 EM 장비로 교체될 경우를 위해 쓰고 있고, 여기서 없앤 것은 '에이전트 툴'이다.


@_serialized("set_ccd_shift_speeds")
def set_ccd_shift_speeds(
    vs_index: Annotated[Optional[int], Field(description='Vertical shift speed index')] = None,
    hs_index: Annotated[Optional[int], Field(description='Horizontal shift speed index')] = None,
) -> dict:
    """
    수직(VS) 및 수평(HS) 시프트 속도 인덱스를 설정한다.
    사용 가능한 속도 목록은 get_ccd_info()의 vs_speeds_us / hs_speeds_conventional_mhz 참조.
    둘 중 하나만 지정해도 됩니다.
    """
    err = _ccd_ready()
    if err:
        return err
    result = ok()
    try:
        if vs_index is not None:
            _hw._ccd.set_vs_speed(vs_index)
            result["vs_speed_index"] = vs_index
            result["vs_speed_us"]    = (_hw._ccd.VSSpeeds[vs_index]
                                        if vs_index < len(_hw._ccd.VSSpeeds) else None)
        if hs_index is not None:
            _hw._ccd.set_hs_speed_conventional(hs_index)
            result["hs_speed_index"] = hs_index
    except Exception as e:
        return fail(str(e))
    return result


@_serialized("set_ccd_temperature")
def set_ccd_temperature(
    temp: Annotated[int, Field(description='Target temperature [°C]. e.g. -40, -60, -80')],
) -> dict:
    """
    CCD 냉각 목표 온도(°C)를 설정한다.
    실제 안정화는 시간이 걸리며, 상태는 get_ccd_info()로 확인한다.
    일반적 범위: -80 ~ 20°C.
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        _hw._ccd.set_temperature(temp)
        return ok(target_temperature_C=temp)
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_cooler")
def set_ccd_cooler(
    on: Annotated[bool, Field(description='true = cooler ON, false = cooler OFF')],
) -> dict:
    """CCD 냉각기를 켜거나(True) 끈다(False)."""
    err = _ccd_ready()
    if err:
        return err
    try:
        _hw._ccd.set_cooler(on)
        return ok(cooler="ON" if on else "OFF")
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_shutter")
def set_ccd_shutter(
    mode: Annotated[Literal['auto', 'open', 'close'], Field(description='Shutter mode')],
) -> dict:
    """
    셔터 모드를 설정한다.
    'auto'  — 취득 시 자동 열고 닫음 (기본)
    'open'  — 강제로 열어둠
    'close' — 강제로 닫아둠 (다크/배경 측정 시)

    다른 CCD 설정 도구와 **같은 규약**이다(2026-07-31 수정): 여기서 한 번 걸어 두면
    이후 acquire_spectrum 이 shutter 인자를 생략해도 그 설정을 그대로 유지한다.
    따라서 다크/배경 프레임을 여러 장 찍을 때 set_ccd_shutter('close') 한 번이면 된다.

    예외는 '아무도 셔터를 지정한 적이 없는' 상태뿐이다 — CCD 초기화가 셔터를 close 로
    두기 때문에(hardware_manager._init_ccd, "촬영 직전까지 광원 차단"), 그 상태를 그대로
    유지하면 세션 첫 측정이 조용히 암전 프레임이 된다. 그래서 한 번도 지정되지 않았을
    때만 acquire_spectrum 이 'auto' 로 연다. 자세한 규약은 _effective_shutter 참고.
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        # 허용값·적용 코드는 acquire_spectrum 과 공용. explicit=True 로 '직접 지정했음'을
        # 남겨, 이후 측정이 이 설정을 덮어쓰지 않게 한다.
        err = _apply_shutter(mode, explicit=True)
        if err:
            return err
        return ok(shutter=mode,
                  note="This setting persists: later acquire_spectrum calls keep it unless you "
                       "pass their own shutter argument. Use 'close' for dark/background frames "
                       "and set it back to 'auto' before normal measurements.")
    except Exception as e:
        return fail(str(e))


@_serialized("set_ccd_image_flip")
def set_ccd_image_flip(
    hflip: Annotated[bool, Field(description='true = flip horizontally (left-right)')],
    vflip: Annotated[bool, Field(description='true = flip vertically (up-down)')],
) -> dict:
    """
    이미지 반전을 설정한다. read_mode='image' 에서만 허용한다.
    hflip: 수평 좌우 반전 여부
    vflip: 수직 상하 반전 여부

    [왜 image 모드로 제한하는가]
    이 분광기는 pixel 0 이 고파장쪽에 맺혀서, 초기화가 FVB 기준으로 hflip=True 를 걸어
    둔다(hardware_manager._init_ccd, Config.ini [ANDOR_IDUS] Reverse=True). 그 상태에서
    캘리브레이션(raman_shift_cm-1 / wavelength_nm)이 픽셀 순서와 맞춰져 있다.
    1D 스펙트럼 모드에서 이 값을 뒤집으면 세기 배열만 좌우로 뒤집히고 축 배열은 그대로여서,
    스펙트럼이 조용히 파장축과 어긋난다(에러도 안 난다 — 그래서 더 위험하다). vflip 은
    FVB 가 수직을 전부 합산하므로 애초에 의미가 없다.
    2D 이미지 모드에서는 축 정합 문제가 없어 그때만 노출한다.
    """
    err = _ccd_ready()
    if err:
        return err
    ro = getattr(_hw._ccd, 'ro_mode', None)
    if ro != 'IMG':
        return fail(f"Image flip is only allowed in the 'image' read mode (the CCD is currently in "
                    f"'{ro}'). In 1D spectrum modes (fvb / single_track) flipping would silently "
                    f"misalign the intensity array against the calibrated raman_shift_cm-1 / "
                    f"wavelength_nm axes, and the factory orientation is already set at startup. "
                    f"Call set_ccd_read_mode(mode='image') first if you really need a flipped 2D image.")
    try:
        _hw._ccd.set_image_flip(hflip=hflip, vflip=vflip)
        return ok(hflip=hflip, vflip=vflip, read_mode="image")
    except Exception as e:
        return fail(str(e))
