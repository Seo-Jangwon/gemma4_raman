# -*- coding: utf-8 -*-
"""가상 분광 CCD. `AndorCCD` 덕타입.

책임이 둘인데, 크기가 아주 다르다.

  (a) **장치 프로토콜 에뮬레이션** — 이 파일의 9할.
      읽기 모드·취득 모드·트리거·셔터·속도·이득·온도. 대부분은 값을 저장하고 조회 시
      그대로 돌려주는 setter 다. 시시해 보이지만 이것이 있어야 set_ccd_* 도구 12종과
      get_ccd_info 가 개발 PC 에서 실제로 실행된다 — 그게 이 대역을 만든 이유다.

  (b) **신호 내용** — _synthesize() 함수 하나. 지금은 스텁이다(§확장 지점).

[스펙트럼 내용이 아직 없는 이유]
물질별 피크·광표백·열손상·형광 배경은 다음 단계다. 지금 필요한 것은 '파라미터를 걸고
찍는 경로가 끝까지 도는가'이고, 그것은 평탄한 신호로도 전부 확인된다. 시료 물리를 여기
섞기 시작하면 (a) 와 (b) 가 한 덩어리가 되어 나중에 물리만 갈아 끼울 수 없게 된다.

[반환 계약 — 실물과 글자 단위로 같아야 한다]
  start_acquisition_cycle()  dict {intensity, calibrated, raman_shift_cm-1, wavelength_nm,
                                   laser_nm} 또는 {intensity: [], calibrated: False, error}
  get_acquired_data()        ndarray. single/accumulate (Ny_ro, Nx_ro), kinetic (num_kin, Ny_ro, Nx_ro)
acquire_tools 가 두 경로를 다르게 다루므로(전자는 dict, 후자는 배열) 한쪽만 맞추면 안 된다.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from backend.tools.hw_tools.SDKs.andor_codes import andor_ccd_consts as consts
from backend.tools.hw_tools.config import PIXEL_COUNT

#: 검출기 물리 크기. Nx 는 config(Config.ini 의 PixelCount)에서 온다 — 캘리브레이터가
#: 같은 값으로 파장축을 만들므로 여기서 따로 정하면 축과 배열 길이가 어긋난다.
_NX = int(PIXEL_COUNT)
_NY = 256

_FULL_WELL = 65535
_DARK_LEVEL = 500.0        # 조사가 없어도 깔리는 바닥(오프셋 + 암전류)
_READ_NOISE = 12.0

_AMBIENT_C = 20.0


# ══════════════════════════════════════════════════════════════════════════════
# ★ 확장 지점 — 시료가 내는 신호
# ══════════════════════════════════════════════════════════════════════════════

def _synthesize(axis_cm1, color_bgr, ctx) -> np.ndarray:
    """1D 스펙트럼(길이 len(axis_cm1)) 을 만든다. **여기에 색 → 피크를 넣는다.**

    지금은 스텁이다: 바닥값 + 노출·파워에 비례하는 광량 + 노이즈. 색을 쓰지 않는다.

    Parameters
    ----------
    axis_cm1 : ndarray
        픽셀별 라만 시프트(cm-1). _calibrator.pixel_to_raman_shift 로 만든 축이다.
        피크를 넣을 때 '몇 번 픽셀'이 아니라 '몇 cm-1'로 적을 수 있게 하려고 넘긴다.
    color_bgr : tuple[int, int, int]
        지금 레이저가 놓인 자리의 시료 색(scene.color_at). **이것이 물질 식별자다.**
        스텁은 쓰지 않지만, 다음 단계에서 색 → 피크 표를 여기에 건다.
    ctx : dict
        exposure_s, power_pct, num_acc, laser_armed(측정빔이 실제로 나갔는가).

    Returns
    -------
    ndarray  int32, 0..65535 로 잘린 값. 실물 get_acquired_data 와 같은 dtype·범위다.
    """
    n = len(axis_cm1)
    rng = np.random.default_rng()

    signal = np.full(n, _DARK_LEVEL, dtype=np.float64)
    if ctx["laser_armed"]:
        # 조사량에 비례해 바닥이 올라간다 — 파워 튜닝 도구가 '올렸더니 세졌다'를 볼 수
        # 있어야 배선이 도는지 확인된다. 물질 피크는 아직 없다.
        signal += 40.0 * ctx["power_pct"] * ctx["exposure_s"] * ctx["num_acc"]

    signal += rng.normal(0.0, _READ_NOISE, n)
    # 샷 노이즈: 세기의 제곱근에 비례. 노출을 늘리면 SNR 이 좋아지는 거동이 여기서 나온다.
    signal += rng.normal(0.0, 1.0, n) * np.sqrt(np.maximum(signal, 0.0))

    return np.clip(signal, 0, _FULL_WELL).astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# 장치
# ══════════════════════════════════════════════════════════════════════════════

class VirtualCCD:
    """Andor CCD 대역. 파라미터를 기억하고, 찍으라면 배열을 만들어 준다."""

    #: 촬영 대기를 실제 시간만큼 잰다. 0 으로 두면 즉시 끝난다(빠른 테스트용).
    #: 1.0 을 기본으로 두는 이유: 노출을 길게 준 요청이 정말 오래 걸려야 acquire_tools 의
    #: 타임아웃 계산과 kinetic 폴링 루프가 실제로 돌아 본다.
    TIME_SCALE = 1.0

    #: 냉각·승온 속도(°C/s). 실물은 수 분 걸리지만, 그걸 그대로 흉내내면 서버 기동마다
    #: 몇 분을 기다리게 된다. 중요한 것은 '안정화 전에는 STABILIZED 가 아니다'는 상태
    #: 전이가 실제로 일어나는 것이다.
    TEMP_RATE_C_PER_S = 20.0

    def __init__(self, debug: bool = False, initialize_to_defaults: bool = True,
                 mgr=None, scene=None):
        """mgr / scene 은 신호를 만들 때 '어디를 보고 있는가'를 알기 위한 것이다.
        없으면 스텁이 색을 (0,0,0) 으로 본다 — 지금은 색을 쓰지 않으므로 무해하다."""
        self._mgr = mgr
        self._scene = scene
        self._lock = threading.RLock()

        # ── 검출기 ──
        self.Nx, self.Ny = _NX, _NY
        self.Nx_ro, self.Ny_ro = _NX, 1
        self.hbin = 1
        self.ro_mode = 'FULL_VERTICAL_BINNING'   # hw_core._current_read_mode 가 읽는 이름
        self.em_mode = False

        # ── 취득 ──
        self.exposure_time = 0.1
        self.num_acc = 1
        self.num_kin = 1
        self.aq_mode = 'single'
        self.trigger_mode = 'internal'
        self.shutter_explicit = False
        self._shutter = 'close'

        # ── 속도·이득 (get_ccd_info 가 인덱싱한다 — 비워 두면 IndexError) ──
        self.numVSSpeeds = 4
        self.VSSpeeds = [4.25, 8.25, 16.25, 32.25]
        self.numHSSpeeds_Conventional = [4]
        self.HSSpeeds_Conventional = [[3.0, 1.0, 0.05, 0.01]]
        self.preamp_gains = [1.0, 2.0, 4.0]
        self.vs_speed_index = 0
        self.hs_speed_index = 0
        self.preamp_gain_index = 0
        self.ad_channel = 0
        self.output_amp = 1
        self.cosmic_ray_filter = False
        self.hflip, self.vflip = False, False

        # ── 온도 ──
        self.temperature_setpoint = -40
        self.cooler_on = False
        self._temp_c = _AMBIENT_C
        self._temp_t0 = time.time()
        self.temperature_status_num = consts.DRV_TEMP_OFF

        # ── 상태 ──
        self._acq_deadline = 0.0
        self._calibrator = None

    # ── 온도 ────────────────────────────────────────────────────────────────
    def _tick_temp(self) -> None:
        """마지막 조회 이후 흐른 시간만큼 목표(또는 상온)로 다가간다."""
        now = time.time()
        dt, self._temp_t0 = now - self._temp_t0, now
        target = float(self.temperature_setpoint) if self.cooler_on else _AMBIENT_C
        step = self.TEMP_RATE_C_PER_S * max(dt, 0.0)
        if abs(target - self._temp_c) <= step:
            self._temp_c = target
        else:
            self._temp_c += step * (1.0 if target > self._temp_c else -1.0)

        if not self.cooler_on:
            self.temperature_status_num = consts.DRV_TEMP_OFF
        elif abs(self._temp_c - target) < 0.5:
            self.temperature_status_num = consts.DRV_TEMP_STABILIZED
        else:
            self.temperature_status_num = consts.DRV_TEMP_NOT_REACHED

    def get_temperature(self) -> int:
        """실물이 int 를 준다 — hardware_manager 가 f"{t:5d}" 로 찍으므로 float 이면 죽는다."""
        with self._lock:
            self._tick_temp()
            return int(round(self._temp_c))

    def get_temperature_status(self) -> str:
        with self._lock:
            self._tick_temp()
            return {consts.DRV_TEMP_OFF: 'OFF',
                    consts.DRV_TEMP_NOT_STABILIZED: 'NOT_STABILIZED',
                    consts.DRV_TEMP_STABILIZED: 'STABILIZED',
                    consts.DRV_TEMP_NOT_REACHED: 'NOT_REACHED',
                    consts.DRV_TEMP_DRIFT: 'DRIFT'}.get(self.temperature_status_num, 'UNKNOWN')

    def set_temperature(self, t):
        with self._lock:
            self._tick_temp()
            self.temperature_setpoint = int(t)
        return self.temperature_setpoint

    def set_cooler(self, on: bool):
        with self._lock:
            self._tick_temp()
            self.cooler_on = bool(on)
        return self.cooler_on

    # ── 검출기 설정 ─────────────────────────────────────────────────────────
    def set_ad_channel(self, chan_i: int = 0):
        self.ad_channel = int(chan_i)

    def get_fastest_recommended_vs_speed(self):
        return (0, self.VSSpeeds[0])

    def set_vs_speed(self, index: int):
        self.vs_speed_index = int(index)

    def set_hs_speed_conventional(self, index: int):
        self.hs_speed_index = int(index)

    def set_preamp_gain(self, index: int):
        self.preamp_gain_index = int(index)

    def set_output_amp(self, amp: int):
        self.output_amp = int(amp)

    def has_em_ccd(self) -> bool:
        return False

    def set_cosmic_ray_filter(self, on: bool):
        self.cosmic_ray_filter = bool(on)

    def set_image_flip(self, hflip: bool = False, vflip: bool = False):
        self.hflip, self.vflip = bool(hflip), bool(vflip)

    # ── 셔터 ────────────────────────────────────────────────────────────────
    def set_shutter_open(self, *a, **kw):
        self._shutter = 'open'

    def set_shutter_close(self, *a, **kw):
        self._shutter = 'close'

    def set_shutter_auto(self, *a, **kw):
        self._shutter = 'auto'

    # ── 읽기 모드 ───────────────────────────────────────────────────────────
    def set_ro_full_vertical_binning(self, hbin: int = 1):
        self.ro_mode, self.hbin = 'FULL_VERTICAL_BINNING', int(hbin)
        self.Nx_ro, self.Ny_ro = int(self.Nx / self.hbin), 1

    def set_ro_single_track(self, center, width: int = 1, hbin: int = 1):
        self.ro_mode, self.hbin = 'SINGLE_TRACK', int(hbin)
        self.single_track_center, self.single_track_width = int(center), int(width)
        self.Nx_ro, self.Ny_ro = int(self.Nx / self.hbin), 1

    def set_ro_image_mode(self, hbin: int = 1, vbin: int = 1, **kw):
        self.ro_mode, self.hbin = 'IMG', int(hbin)
        self.Nx_ro = int(self.Nx / self.hbin)
        self.Ny_ro = int(self.Ny / max(int(vbin), 1))

    def get_current_hbin(self) -> int:
        return self.hbin

    # ── 취득 모드 ───────────────────────────────────────────────────────────
    def set_exposure_time(self, dt):
        self.exposure_time = float(dt)
        return self.exposure_time

    def get_exposure_time(self):
        return self.exposure_time

    def set_num_accumulations(self, num):
        self.num_acc = max(1, int(num))

    def set_num_kinetics(self, num):
        self.num_kin = max(1, int(num))

    def set_aq_single_scan(self, exposure=None):
        self.aq_mode = 'single'
        if exposure is not None:
            self.set_exposure_time(exposure)

    def set_aq_accumulate_scan(self, exposure_time=None, num_acc=None, cycle_time=None):
        self.aq_mode = 'accumulate'
        if exposure_time is not None:
            self.set_exposure_time(exposure_time)
        if num_acc is not None:
            self.set_num_accumulations(num_acc)

    def set_aq_kinetic_scan(self, exp_time=None, num_acc=None, acc_time=None,
                            num_kin=None, kin_time=None):
        self.aq_mode = 'kinetic'
        if exp_time is not None:
            self.set_exposure_time(exp_time)
        if num_acc is not None:
            self.set_num_accumulations(num_acc)
        if num_kin is not None:
            self.set_num_kinetics(num_kin)

    def set_aq_run_till_abort_scan(self):
        self.aq_mode = 'run_till_abort'

    def set_trigger_mode(self, mode: str):
        self.trigger_mode = str(mode)

    # ── 촬영 ────────────────────────────────────────────────────────────────
    def _acq_seconds(self) -> float:
        frames = self.num_kin if self.aq_mode == 'kinetic' else 1
        return self.exposure_time * max(self.num_acc, 1) * max(frames, 1) * self.TIME_SCALE

    def get_status(self) -> str:
        return 'ACQUIRING' if time.time() < self._acq_deadline else 'IDLE'

    def prepare_acquisition(self):
        return None

    def start_acquisition(self):
        """비동기로 시작만 한다 — kinetic 경로가 get_status() 를 폴링해 끝을 기다린다."""
        self._acq_deadline = time.time() + self._acq_seconds()

    def send_software_trigger(self):
        return None

    def abort_acquisition(self):
        self._acq_deadline = 0.0

    def free_internal_memory(self):
        return None

    def create_buffer(self):
        return None

    # ── 데이터 ──────────────────────────────────────────────────────────────
    def _axis_cm1(self) -> np.ndarray:
        """픽셀 → cm-1 축. 캘리브레이터가 없으면 픽셀 번호를 그대로 축으로 쓴다."""
        cal = self._calibrator
        if cal is None:
            return np.arange(self.Nx_ro, dtype=np.float64)
        return np.array([float(cal.pixel_to_raman_shift(p)) for p in range(self.Nx_ro)])

    def _ctx(self) -> dict:
        """지금 무엇이 시료에 닿고 있는가. _synthesize 의 입력이다."""
        laser = getattr(self._mgr, "laser", None)
        armed = bool(getattr(laser, "_power_set", False)) and bool(getattr(laser, "is_on", False))
        # 셔터가 닫혀 있으면 빛이 검출기에 못 온다 — 다크 프레임이 되는 경로다.
        if self._shutter == 'close':
            armed = False
        return {"exposure_s": float(self.exposure_time),
                "power_pct": float(getattr(laser, "power_pct", 0.0) or 0.0),
                "num_acc": max(int(self.num_acc), 1),
                "laser_armed": armed}

    def _color(self) -> tuple[int, int, int]:
        stage = getattr(self._mgr, "stage", None)
        if self._scene is None or stage is None:
            return (0, 0, 0)
        pos = stage.get_position()
        return (0, 0, 0) if pos is None else self._scene.color_at(pos[0], pos[1])

    def _one_frame(self) -> np.ndarray:
        """(Ny_ro, Nx_ro). 세로 방향은 같은 스펙트럼을 반복한다 — 세로가 1 이 아닌 것은
        image 모드뿐이고, 그때도 이 대역이 재현할 대상은 '가로 축의 신호'다."""
        line = _synthesize(self._axis_cm1(), self._color(), self._ctx())
        return np.tile(line, (max(self.Ny_ro, 1), 1))

    def get_acquired_data(self) -> np.ndarray:
        with self._lock:
            if self.aq_mode == 'kinetic':
                return np.stack([self._one_frame() for _ in range(max(self.num_kin, 1))])
            return self._one_frame()

    def start_acquisition_cycle(self, trigger_mode_str: str = 'internal',
                                timeout_ms=None) -> dict:
        """찍고 캘리브레이션까지 붙여 dict 로 돌려준다(single/accumulate 경로).

        실물은 여기서 WaitForAcquisition 이 막힌다. 대역도 같은 시간만큼 기다린다 —
        그래야 acquire_tools 가 계산한 타임아웃이 실제로 의미를 갖는다.
        """
        with self._lock:
            wait_s = self._acq_seconds()
            if timeout_ms is not None and wait_s * 1000.0 > float(timeout_ms):
                return {"intensity": [], "calibrated": False,
                        "error": "WaitForAcquisition 실패 (타임아웃 또는 트리거 없음)"}
            if wait_s > 0:
                time.sleep(wait_s)
            self._acq_deadline = 0.0

            intensity = self._one_frame().flatten().tolist()
            result = {"intensity": intensity, "calibrated": False}
            cal = self._calibrator
            if cal is not None:
                pixels = range(len(intensity))
                result.update({
                    "calibrated": True,
                    "raman_shift_cm-1": [float(cal.pixel_to_raman_shift(p)) for p in pixels],
                    "wavelength_nm": [float(cal.pixel_to_wavelength(p)) for p in pixels],
                    "laser_nm": float(cal.laser_nm),
                })
            return result

    def close(self):
        self.cooler_on = False
        self._acq_deadline = 0.0

    def __repr__(self) -> str:
        return (f"<VirtualCCD {self.aq_mode}/{self.ro_mode} {self.Ny_ro}x{self.Nx_ro} "
                f"exp={self.exposure_time}s T={int(round(self._temp_c))}C>")
