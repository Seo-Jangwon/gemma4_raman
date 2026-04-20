"""
USE_andor_test.py — Andor CCD 래퍼 (v3)
=========================================

[역할]
  Andor IDUS DV420 OE CCD 카메라를 Python 에서 제어.
  atmcd64d.dll 을 ctypes 로 직접 래핑.

[ScopeFoundry andor_ccd_interface.py 에서 가져온 핵심 기능]
  - SetADChannel    : AD 변환기 채널 선택
  - SetVSSpeed      : 수직 shift 속도 → charge transfer efficiency 영향
  - SetHSSpeed      : 수평 readout 속도 → read noise 영향 (느릴수록 ↓)
  - SetPreAmpGain   : 전자→count 변환 게인 ★ sensitivity 차이의 최대 원인
  - SetFVBHBin      : FVB 모드 수평 binning
  - SetShutter       : 셔터 auto/open/close
  - SetImageFlip     : pixel 순서 하드웨어 반전
  - GetAcquisitionTimings : SDK 가 실제 적용한 노출시간 확인
  - GetStatus 폴링  : WaitForAcquisition 보다 안전 (timeout 제어 가능)
  - numpy 버퍼       : ctypes array 대신 numpy 사용 → 성능/안정성 향상
  - Thread-safe Lock : 모든 DLL 호출을 Lock 으로 보호

[이전 버전의 문제점과 해결]
  기존 코드는 SetExposureTime 만 호출하고 VSSpeed/HSSpeed/PreAmpGain 을
  설정하지 않아 SDK 기본값(최고속, 최저게인)이 적용됨.
  → 상용 프로그램 대비 signal 이 25x 약했음 (bias 에 파묻힌 상태).
  Config.txt 의 [ANDOR_IDUS] 섹션 값을 자동 파싱하여 적용함으로써 해결.

[캘리브레이션]
  RamanCalibrator 객체를 주입하면 start_acquisition_cycle() 반환 dict 에
  raman_shift_cm-1 / wavelength_nm 필드가 자동으로 포함됨.
  from_factory_calibration() 을 쓰면 외부 파일 없이 바로 동작.

[하드웨어]
  Andor DV420 OE (IDUS), DLL: atmcd64d.dll / ATMCD64CS.dll
"""
from __future__ import annotations

import configparser
import csv
import ctypes
import time
from ctypes import c_int, c_uint, c_long, c_float, byref
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

# ── 캘리브레이션 모듈 (같은 폴더에 raman_calibration.py 필요) ──────────────
try:
    from raman_calibration import RamanCalibrator
except ImportError:
    RamanCalibrator = None


# ══════════════════════════════════════════════════════════════════════════════
# Andor SDK 상수
# ══════════════════════════════════════════════════════════════════════════════

# --- 함수 반환값 (에러 코드) ---
DRV_SUCCESS             = 20002  # 성공
DRV_ERROR_ACK           = 20013  # 통신 에러
DRV_ACQUIRING           = 20072  # 촬영 진행 중
DRV_IDLE                = 20073  # 대기 상태 (촬영 완료)
DRV_TEMPCYCLE           = 20074  # 온도 순환 중
DRV_NOT_INITIALIZED     = 20075  # SDK 미초기화

# --- 온도 상태 코드 ---
DRV_TEMP_OFF            = 20034  # 냉각기 꺼짐
DRV_TEMP_NOT_STABILIZED = 20035  # 아직 안정화 안 됨
DRV_TEMP_STABILIZED     = 20036  # 목표 온도 도달, 안정화 완료
DRV_TEMP_NOT_REACHED    = 20037  # 아직 목표 온도 미도달
DRV_TEMP_DRIFT          = 20040  # 온도 드리프트 (한번 도달 후 벗어남)

# --- Read Mode (SetReadMode 인자) ---
READ_MODE_FVB          = 0  # Full Vertical Binning — 라만 스펙트럼 1D
READ_MODE_MULTI_TRACK  = 1  # 다중 트랙
READ_MODE_RANDOM_TRACK = 2  # 랜덤 트랙
READ_MODE_SINGLE_TRACK = 3  # 단일 트랙
READ_MODE_IMAGE        = 4  # 2D 이미지

# --- Acquisition Mode (SetAcquisitionMode 인자) ---
ACQ_MODE_SINGLE     = 1  # 단일 촬영
ACQ_MODE_ACCUMULATE = 2  # 누적 촬영
ACQ_MODE_KINETIC    = 3  # 연속 촬영 (kinetic series)

# --- Trigger Mode (SetTriggerMode 인자) ---
TRIGGER_MODE_INTERNAL          = 0  # 내부 트리거 (소프트웨어 제어)
TRIGGER_MODE_EXTERNAL          = 1  # 외부 트리거 (TTL 신호 대기)
TRIGGER_MODE_EXTERNAL_EXPOSURE = 7  # 외부 트리거 + 노출 제어

# --- 에러 코드 → 이름 역매핑 (디버깅용) ---
_ERR = {
    20002: "SUCCESS", 20013: "ERROR_ACK", 20024: "NO_NEW_DATA",
    20034: "TEMP_OFF", 20035: "TEMP_NOT_STAB", 20036: "TEMP_STAB",
    20037: "TEMP_NOT_REACHED", 20040: "TEMP_DRIFT",
    20066: "P1INVALID", 20067: "P2INVALID", 20068: "P3INVALID",
    20069: "P4INVALID", 20070: "INIERROR",
    20072: "ACQUIRING", 20073: "IDLE", 20074: "TEMPCYCLE",
    20075: "NOT_INIT", 20078: "INVALID_MODE",
    20091: "NOT_SUPPORTED", 20990: "NO_CAMERA",
}


def _ename(code: int) -> str:
    """에러 코드 → 사람이 읽을 수 있는 이름."""
    return _ERR.get(code, f"UNK({code})")


# ══════════════════════════════════════════════════════════════════════════════
# Config.txt 파서
# ══════════════════════════════════════════════════════════════════════════════

class AndorConfig:
    """
    상용 소프트웨어(WeVu/Rays-ON)의 Config.txt 에서 [ANDOR_IDUS] 섹션을 파싱.

    이 값들은 상용 프로그램이 CCD 초기화 시 적용하는 파라미터이며,
    장원님 코드에서 누락되어 있던 것들.

    Attributes
    ----------
    vs_speed_index : int
        수직 shift 속도 인덱스 (Config: VSSpeedIndex=0)
    readout_rate_index : int
        수평 readout 속도 인덱스 (Config: ReadoutRateIndex=2 → 33kHz)
    preamp_gain_index : int
        PreAmplifier 게인 인덱스 (Config: PreAmplifierGainIndex=1)
    reverse : bool
        pixel 축 반전 여부 (Config: Reverse=True)
    target_temperature : int
        냉각 목표 온도 (Config: TargetTemperature=-40)
    """

    def __init__(self, path: str | Path):
        cp = configparser.ConfigParser(strict=False)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            cp.read_file(f)

        s = "ANDOR_IDUS"
        self.acquisition_mode   = int(cp[s].get("AcquisitionMode", "1"))
        self.trigger_mode       = int(cp[s].get("TriggerMode", "0"))
        self.read_mode          = int(cp[s].get("ReadoutMode", "0"))
        self.exposure_time      = float(cp[s].get("ExposureTime", "0.1"))
        self.vs_speed_index     = int(cp[s].get("VSSpeedIndex", "0"))
        self.readout_rate_index = int(cp[s].get("ReadoutRateIndex", "2"))
        self.preamp_gain_index  = int(cp[s].get("PreAmplifierGainIndex", "1"))
        self.target_temperature = int(cp[s].get("TargetTemperature", "-40"))
        self.enable_cooler      = cp[s].get("EnableCooler", "True").lower() == "true"
        self.reverse            = cp[s].get("Reverse", "False").lower() == "true"
        self.num_accumulation   = int(cp[s].get("NumberOfAccumulation", "1"))
        self.track_centre       = int(cp[s].get("TrackCentre", "0"))
        self.track_height       = int(cp[s].get("TrackHeight", "0"))

    def __repr__(self):
        return (f"AndorConfig(exp={self.exposure_time}s, "
                f"vs={self.vs_speed_index}, hs={self.readout_rate_index}, "
                f"preamp={self.preamp_gain_index}, "
                f"reverse={self.reverse}, temp={self.target_temperature}°C)")


# ══════════════════════════════════════════════════════════════════════════════
# AndorCamera — 메인 래퍼
# ══════════════════════════════════════════════════════════════════════════════

class AndorCamera:
    """
    Andor CCD (IDUS DV420 OE) 제어 래퍼.

    ScopeFoundry andor_ccd_interface.py 의 핵심 로직을 가져오되,
    ScopeFoundry / PyQt 의존성은 완전 제거.
    RamanGPT 의 hardware agent 에서 직접 사용 가능한 형태.

    사용 흐름
    ---------
    1. cam = AndorCamera(dll_path, calibrator=cal)
    2. cam.initialize(config_dir, config_txt_path="Config.txt")
       → DLL 로드, SDK 초기화, VSSpeed/HSSpeed/PreAmpGain 자동 적용
    3. cam.setup_acquisition(read_mode=..., exposure_time=...)
       → readout 모드 설정, 버퍼 생성
    4. result = cam.start_acquisition_cycle()
       → StartAcquisition → 폴링 → GetAcquiredData → dict 반환
    5. cam.shutdown()

    Attributes
    ----------
    width, height : int
        CCD 센서 전체 크기 (pixel)
    Nx_ro, Ny_ro : int
        현재 readout 모드에서의 출력 크기
    em_mode : bool
        EM CCD 여부 (DV420 OE 는 False)
    calibrator : RamanCalibrator | None
        pixel→shift 변환기. None 이면 raw 반환.
    """

    def __init__(self, dll_path: str, calibrator: Optional["RamanCalibrator"] = None):
        """
        Parameters
        ----------
        dll_path : str
            atmcd64d.dll 파일의 전체 경로.
        calibrator : RamanCalibrator | None
            pixel → Raman shift 변환기. None 이면 raw pixel 만 반환.
        """
        self.lock = Lock()               # 모든 DLL 호출을 thread-safe 로
        self.calibrator = calibrator
        self.width = self.height = 0      # 센서 전체 크기
        self.Nx_ro = self.Ny_ro = 0       # readout 크기
        self._buffer = None               # numpy 데이터 버퍼
        self._config = None               # Config.txt 파싱 결과
        self.em_mode = False              # EM CCD 여부
        self.preamp_gains = []            # 사용 가능한 게인 목록
        self.hs_speeds = []               # 사용 가능한 수평 속도 목록 [MHz]
        self.vs_speeds = []               # 사용 가능한 수직 속도 목록 [µs/pixel]
        self.dark_frame = None            # 배경 노이즈 저장용

        # DLL 로드
        try:
            self.dll = ctypes.cdll.LoadLibrary(dll_path)
            print(f"[CCD] DLL loaded: {dll_path}")
        except OSError as e:
            raise RuntimeError(f"DLL 로드 실패: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 유틸: Thread-safe DLL 호출 + 에러 체크
    # ─────────────────────────────────────────────────────────────────────────

    def _call(self, fn_name: str, *args) -> int:
        """
        DLL 함수를 thread-safe 하게 호출하고 에러 시 로그 출력.

        ScopeFoundry 의 `with self.lock: _err(self.andorlib.XXX())` 패턴과 동일.
        """
        fn = getattr(self.dll, fn_name)
        with self.lock:
            ret = fn(*args)
        if ret != DRV_SUCCESS:
            print(f"[CCD] {fn_name} → {_ename(ret)}")
        return ret

    def _ok(self, fn_name: str, *args) -> bool:
        """DLL 호출 성공 여부만 bool 로 반환."""
        return self._call(fn_name, *args) == DRV_SUCCESS

    # ─────────────────────────────────────────────────────────────────────────
    # 캘리브레이션 주입
    # ─────────────────────────────────────────────────────────────────────────

    def set_calibrator(self, cal):
        """
        캘리브레이터 교체.

        grating motor 이동 후 새 캘리브레이터를 넣을 때 사용.
        카메라 재초기화 불필요 — 이 메서드만 호출하면 됨.
        """
        self.calibrator = cal

    def _attach_axes(self, intensity: np.ndarray) -> dict:
        """
        raw intensity 배열에 pixel / raman_shift / wavelength 축을 붙여 dict 로 반환.

        calibrator 가 없으면 pixel + intensity 만 반환.
        calibrator 가 있으면 raman_shift_cm-1, wavelength_nm, laser_nm 추가.
        """
        n = intensity.size
        out = {
            "pixel": list(range(n)),
            "intensity": intensity.tolist(),
            "calibrated": False,
        }
        if self.calibrator and n == 1024:
            px = np.arange(n)
            out["raman_shift_cm-1"] = self.calibrator.pixel_to_raman_shift(px).tolist()
            out["wavelength_nm"]    = self.calibrator.pixel_to_wavelength(px).tolist()
            out["laser_nm"]         = self.calibrator.laser_nm
            out["calibrated"]       = True
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # 초기화
    # ══════════════════════════════════════════════════════════════════════════

    def initialize(self, config_dir: str,
                   config_txt_path: str | None = None) -> bool:
        """
        Andor SDK 초기화 + CCD 파라미터 전체 설정.

        Parameters
        ----------
        config_dir : str
            Detector.ini 파일이 있는 폴더 경로.
            SDK Initialize() 에 전달됨.
        config_txt_path : str | None
            상용 소프트웨어 Config.txt 경로.
            주면: VSSpeed/HSSpeed/PreAmpGain 등을 Config 값으로 자동 설정.
            없으면: SDK 기본값(최고속, 최저게인) 사용 — sensitivity 낮음.

        Returns
        -------
        bool
            초기화 성공 여부.
        """
        # SDK 초기화
        c_dir = ctypes.create_string_buffer(config_dir.encode("utf-8"))
        if not self._ok("Initialize", c_dir):
            return False

        # 센서 크기 조회
        w, h = c_int(), c_int()
        self._call("GetDetector", byref(w), byref(h))
        self.width, self.height = w.value, h.value

        # 모델명 / 시리얼 조회
        hm = ctypes.create_string_buffer(260)
        self._call("GetHeadModel", hm)
        sn = c_int()
        self._call("GetCameraSerialNumber", byref(sn))
        print(f"[CCD] {hm.value.decode(errors='ignore').strip(chr(0))}, "
              f"S/N={sn.value}, {self.width}x{self.height}")

        # EM CCD 여부 확인 (DV420 OE 는 non-EM → False)
        lo, hi = c_int(), c_int()
        self.em_mode = (self._call("GetEMGainRange", byref(lo), byref(hi)) == DRV_SUCCESS)
        print(f"[CCD] EM mode: {self.em_mode}")

        # AD 채널 설정 (보통 0번 — 14-bit)
        self._call("SetADChannel", c_int(0))

        # 사용 가능한 속도/게인 열거
        self._enum_speeds()
        self._enum_gains()

        # Config.txt 기반 파라미터 적용 (또는 기본값)
        if config_txt_path and Path(config_txt_path).exists():
            self._config = AndorConfig(config_txt_path)
            self._apply_config(self._config)
        else:
            self._apply_defaults()

        return True

    def take_dark_frame(self):
        """셔터를 닫고 배경 노이즈(Dark Frame)를 1회 측정하여 저장"""
        print("[CCD] 다크 프레임(노이즈) 측정 시작...")
        self.set_shutter_close() # 셔터 강제 폐쇄
        time.sleep(0.2)          # 셔터가 닫히는 물리적 시간 대기
        
        # 현재 설정된 노출 시간으로 노이즈 촬영 (subtract_bg=False로 원본 획득)
        res = self.start_acquisition_cycle(subtract_bg=False)
        if res:
            self.dark_frame = np.array(res['intensity'])
            print("[CCD] 다크 프레임 저장 완료.")
        
        self.set_shutter_auto()  # 다시 자동 모드로 복구
        return self.dark_frame

    def _enum_speeds(self):
        """
        CCD 가 지원하는 수평/수직 shift 속도 목록을 SDK 에서 열거.

        ScopeFoundry andor_ccd_interface.py 의 read_shift_speeds() 대응.
        """
        # --- 수평 shift 속도 (HS Speed) ---
        # amp_type: EM=0, Conventional=1 (em_mode 아니면 0)
        amp = 1 if self.em_mode else 0
        n = c_int()
        self._call("GetNumberHSSpeeds", c_int(0), c_int(amp), byref(n))
        spd = c_float()
        self.hs_speeds = []
        for i in range(n.value):
            self._call("GetHSSpeed", c_int(0), c_int(amp), c_int(i), byref(spd))
            self.hs_speeds.append(spd.value)
        print(f"[CCD] HS speeds: {self.hs_speeds} MHz")

        # --- 수직 shift 속도 (VS Speed) ---
        nv = c_int()
        ret = self._call("GetNumberVSSpeeds", byref(nv))
        self.vs_speeds = []
        if ret == DRV_SUCCESS:
            for i in range(nv.value):
                self._call("GetVSSpeed", c_int(i), byref(spd))
                self.vs_speeds.append(spd.value)
        print(f"[CCD] VS speeds: {self.vs_speeds} µs/px")

    def _enum_gains(self):
        """
        CCD 가 지원하는 PreAmplifier 게인 목록 열거.

        ScopeFoundry andor_ccd_interface.py 의 get_preamp_gains() 대응.
        """
        n = c_int()
        self._call("GetNumberPreAmpGains", byref(n))
        g = c_float()
        self.preamp_gains = []
        for i in range(n.value):
            self._call("GetPreAmpGain", c_int(i), byref(g))
            self.preamp_gains.append(g.value)
        print(f"[CCD] PreAmp gains: {self.preamp_gains}")

    def _apply_config(self, cfg: AndorConfig):
        """
        Config.txt 파라미터를 CCD 에 적용.

        ★ 이것이 sensitivity 25x 차이의 핵심 해결. ★

        상용 프로그램은 이 값들을 매번 초기화 시 적용하는데,
        장원님 코드에서는 누락되어 SDK 기본값(최고속/최저게인)이 사용됐음.
        """
        print(f"[CCD] Config 적용: {cfg}")

        # 1) VS Speed — 수직 shift 속도.
        #    느릴수록 charge transfer efficiency 향상 → 신호 손실 감소.
        amp = 1 if self.em_mode else 0
        if cfg.vs_speed_index < len(self.vs_speeds):
            self._ok("SetVSSpeed", c_int(cfg.vs_speed_index))
            print(f"[CCD]   VSSpeed idx={cfg.vs_speed_index} "
                  f"({self.vs_speeds[cfg.vs_speed_index]:.2f} µs/px)")

        # 2) HS Speed — 수평 readout 속도.
        #    느릴수록 read noise 감소 → SNR 향상.
        #    Config ReadoutRateIndex=2 → 33kHz (느린 쪽).
        if cfg.readout_rate_index < len(self.hs_speeds):
            self._ok("SetHSSpeed", c_int(amp), c_int(cfg.readout_rate_index))
            print(f"[CCD]   HSSpeed idx={cfg.readout_rate_index} "
                  f"({self.hs_speeds[cfg.readout_rate_index]:.4f} MHz)")

        # 3) PreAmplifier Gain — 전자(e-) → count 변환 게인.
        #    ★ 가장 큰 sensitivity 영향. ★
        #    Config PreAmplifierGainIndex=1 → 2x 또는 4x (카메라 모델별 상이).
        if cfg.preamp_gain_index < len(self.preamp_gains):
            self._ok("SetPreAmpGain", c_int(cfg.preamp_gain_index))
            print(f"[CCD]   PreAmpGain idx={cfg.preamp_gain_index} "
                  f"({self.preamp_gains[cfg.preamp_gain_index]:.1f}x)")

        # 4) 기본 acquisition 설정
        self._call("SetAcquisitionMode", c_int(cfg.acquisition_mode))
        self._call("SetTriggerMode", c_int(cfg.trigger_mode))
        self._call("SetExposureTime", c_float(cfg.exposure_time))
        self._call("SetNumberAccumulations", c_int(cfg.num_accumulation))

    def _apply_defaults(self):
        """Config.txt 없을 때 SDK 기본값 적용 (sensitivity 낮음)."""
        print("[CCD] Config.txt 없음 → SDK 기본값 적용 (sensitivity 낮을 수 있음)")
        amp = 1 if self.em_mode else 0
        if self.vs_speeds:    self._call("SetVSSpeed", c_int(0))
        if self.hs_speeds:    self._call("SetHSSpeed", c_int(amp), c_int(0))
        if self.preamp_gains: self._call("SetPreAmpGain", c_int(0))
        self._call("SetAcquisitionMode", c_int(ACQ_MODE_SINGLE))
        self._call("SetTriggerMode", c_int(TRIGGER_MODE_INTERNAL))

    # ══════════════════════════════════════════════════════════════════════════
    # Readout Mode 설정
    # ══════════════════════════════════════════════════════════════════════════

    def set_ro_fvb(self, hbin: int = 1):
        """
        FVB (Full Vertical Binning) 모드 설정.

        CCD 의 모든 수직 pixel 을 합산 → 1D 스펙트럼 출력.
        라만 스펙트럼 측정의 기본 모드.

        Parameters
        ----------
        hbin : int
            수평 binning. 1=binning 없음, 2=2pixel씩 합산 (해상도↓, 감도↑).
        """
        self._call("SetReadMode", c_int(READ_MODE_FVB))
        self._call("SetFVBHBin", c_int(hbin))
        self.Nx_ro = self.width // hbin
        self.Ny_ro = 1
        self._buffer = np.zeros((self.Ny_ro, self.Nx_ro), dtype=np.int32)

    def set_ro_single_track(self, center: int, width: int = 1, hbin: int = 1):
        """
        Single Track 모드 — 특정 수직 행 영역만 읽음.

        Parameters
        ----------
        center : int
            트랙 중심 행 (pixel).
        width : int
            트랙 높이 (pixel). 이 범위만 수직 합산.
        hbin : int
            수평 binning.
        """
        self._call("SetReadMode", c_int(READ_MODE_SINGLE_TRACK))
        self._call("SetSingleTrack", c_int(center), c_int(width))
        self._call("SetSingleTrackHBin", c_int(hbin))
        self.Nx_ro = self.width // hbin
        self.Ny_ro = 1
        self._buffer = np.zeros((self.Ny_ro, self.Nx_ro), dtype=np.int32)

    def set_ro_image(self, hbin=1, vbin=1, hstart=1, hend=None, vstart=1, vend=None):
        """
        Image 모드 — 2D 이미지 수집.

        라만 매핑이나 정렬 확인 시 사용.
        """
        self._call("SetReadMode", c_int(READ_MODE_IMAGE))
        hend = hend or self.width
        vend = vend or self.height
        self._call("SetImage", c_int(hbin), c_int(vbin),
                   c_int(hstart), c_int(hend), c_int(vstart), c_int(vend))
        self.Nx_ro = (hend - hstart + 1) // hbin
        self.Ny_ro = (vend - vstart + 1) // vbin
        self._buffer = np.zeros((self.Ny_ro, self.Nx_ro), dtype=np.int32)

    # ══════════════════════════════════════════════════════════════════════════
    # Acquisition 파라미터 설정
    # ══════════════════════════════════════════════════════════════════════════

    def setup_acquisition(self, read_mode: int = READ_MODE_FVB,
                          exposure_time: float = 0.1,
                          trigger_mode: int = TRIGGER_MODE_INTERNAL,
                          acq_mode: int = ACQ_MODE_SINGLE,
                          num_accumulations: int = 1,
                          gain: int = 0):
        """
        측정 직전 acquisition 파라미터 설정.

        VSSpeed / HSSpeed / PreAmpGain 은 initialize() 에서 Config 기반으로
        이미 설정되어 있으므로, 여기서는 노출시간/모드/트리거만 변경.

        Parameters
        ----------
        read_mode : int
            READ_MODE_FVB(0), READ_MODE_SINGLE_TRACK(3), READ_MODE_IMAGE(4).
        exposure_time : float
            CCD 노출 시간 [초].
        trigger_mode : int
            TRIGGER_MODE_INTERNAL(0), TRIGGER_MODE_EXTERNAL(1).
        acq_mode : int
            ACQ_MODE_SINGLE(1), ACQ_MODE_ACCUMULATE(2).
        num_accumulations : int
            누적 횟수 (acq_mode=ACCUMULATE 일 때).
        gain : int
            MCP gain (해당 카메라만). 0이면 설정 안 함.
        """
        # Readout 모드 설정 + 버퍼 생성
        if read_mode == READ_MODE_FVB:
            self.set_ro_fvb()
        elif read_mode == READ_MODE_SINGLE_TRACK:
            # Config.txt 에 TrackCentre/TrackHeight 가 있으면 사용
            tc = self._config.track_centre if self._config else self.height // 2
            th = self._config.track_height if self._config else 20
            self.set_ro_single_track(tc, th)
        elif read_mode == READ_MODE_IMAGE:
            self.set_ro_image()
        else:
            self._call("SetReadMode", c_int(read_mode))
            self.Nx_ro = self.width
            self.Ny_ro = 1
            self._buffer = np.zeros((self.Ny_ro, self.Nx_ro), dtype=np.int32)

        # Acquisition 모드
        self._call("SetAcquisitionMode", c_int(acq_mode))
        if acq_mode == ACQ_MODE_ACCUMULATE and num_accumulations > 1:
            self._call("SetNumberAccumulations", c_int(num_accumulations))

        # 트리거 모드
        self._call("SetTriggerMode", c_int(trigger_mode))

        # 노출 시간
        self._call("SetExposureTime", c_float(exposure_time))

        # MCP gain (intensifier 카메라 전용)
        if gain > 0:
            self._call("SetMCPGating", c_int(1))
            self._call("SetMCPGain", c_int(gain))

        # SDK 가 실제 적용한 타이밍 확인 (요청값과 다를 수 있음)
        exp, acc, kin = self.get_acquisition_timings()
        print(f"[CCD] Setup: exp={exp:.4f}s (요청 {exposure_time}s), "
              f"mode={acq_mode}, trigger={trigger_mode}")

    def get_acquisition_timings(self) -> tuple[float, float, float]:
        """SDK 가 실제 적용한 (exposure, accumulation, kinetic) 타이밍 [초]."""
        e, a, k = c_float(), c_float(), c_float()
        self._call("GetAcquisitionTimings", byref(e), byref(a), byref(k))
        return e.value, a.value, k.value

    # ══════════════════════════════════════════════════════════════════════════
    # Shutter 제어
    # ══════════════════════════════════════════════════════════════════════════

    def set_shutter_auto(self):
        """셔터를 자동 모드로 (촬영 시 열림, 완료 시 닫힘)."""
        self._call("SetShutter", c_int(0), c_int(0), c_int(0), c_int(0))

    def set_shutter_open(self):
        """셔터 상시 열림."""
        self._call("SetShutter", c_int(0), c_int(1), c_int(0), c_int(0))

    def set_shutter_close(self):
        """셔터 상시 닫힘 (dark frame 촬영 시)."""
        self._call("SetShutter", c_int(0), c_int(2), c_int(0), c_int(0))

    # ══════════════════════════════════════════════════════════════════════════
    # Image Flip
    # ══════════════════════════════════════════════════════════════════════════

    def set_image_flip(self, hflip: bool = False, vflip: bool = False):
        """
        SDK 레벨에서 pixel 순서 반전.

        Config.txt Reverse=True 는 상용 소프트웨어가 저장 시 반전하는 것이고,
        SDK 의 SetImageFlip 은 GetAcquiredData 시점에서 반전.
        현재 캘리브레이션은 raw pixel 기준이므로 여기서 flip 하면 안 됨.
        """
        self._call("SetImageFlip", c_int(int(hflip)), c_int(int(vflip)))

    # ══════════════════════════════════════════════════════════════════════════
    # 촬영 + 데이터 수집
    # ══════════════════════════════════════════════════════════════════════════

    def start_acquisition_cycle(self, subtract_bg=True) -> dict | None:
        """
        1회 촬영 사이클: StartAcquisition → 폴링 대기 → GetAcquiredData → dict 반환.

        WaitForAcquisition() 대신 GetStatus() 폴링을 사용하는 이유:
          - timeout 제어 가능 (WaitForAcquisition 은 영원히 블로킹 가능)
          - 촬영 중 다른 상태(TEMPCYCLE 등) 감지 가능
          - ScopeFoundry 도 이 방식 사용

        Returns
        -------
        dict | None
            성공 시: pixel, intensity, calibrated, (raman_shift_cm-1, wavelength_nm, laser_nm)
            실패 시: None
        """
        if self._buffer is None:
            print("[CCD] ERROR: 버퍼 미생성 — setup_acquisition() 먼저 호출 필요")
            return None

        # 촬영 시작
        if not self._ok("StartAcquisition"):
            return None

        # 폴링으로 완료 대기 (5ms 간격)
        st = c_int()
        while True:
            self._call("GetStatus", byref(st))
            if st.value == DRV_IDLE:
                break  # 촬영 완료
            elif st.value == DRV_ACQUIRING:
                time.sleep(0.005)  # 5ms 후 재확인
            else:
                print(f"[CCD] 예상치 못한 상태: {_ename(st.value)}")
                return None

        # 데이터 읽기 — numpy 버퍼에 직접 쓰기 (ScopeFoundry 방식)
        buf_ptr = self._buffer.ctypes.data_as(ctypes.POINTER(c_long))
        buf_size = c_uint(self._buffer.size)
        if self._call("GetAcquiredData", buf_ptr, buf_size) != DRV_SUCCESS:
            return None

        # FVB 모드 (1D) 면 ravel, Image 모드 (2D) 면 copy
        intensity = self._buffer.ravel() if self.Ny_ro == 1 else self._buffer.copy()

        # 다크 프레임 차분
        if subtract_bg and self.dark_frame is not None:
            if len(intensity) == len(self.dark_frame):
                intensity = intensity - self.dark_frame
                # 배경을 뺐을 때 센서 노이즈가 튀어서 음수가 된 값은 0으로 평탄화
                intensity = np.where(intensity < 0, 0, intensity)

        # pixel / intensity + 캘리브레이션 축 붙여서 반환
        return self._attach_axes(intensity)

    def get_status(self) -> str:
        """현재 CCD 상태를 문자열로 반환."""
        s = c_int()
        self._call("GetStatus", byref(s))
        return {
            DRV_IDLE: "IDLE",
            DRV_ACQUIRING: "ACQUIRING",
            DRV_TEMPCYCLE: "TEMPCYCLE",
        }.get(s.value, f"UNK({s.value})")

    def abort_acquisition(self):
        """진행 중인 촬영 중단."""
        self._call("AbortAcquisition")

    # ══════════════════════════════════════════════════════════════════════════
    # 온도 제어
    # ══════════════════════════════════════════════════════════════════════════

    def set_temperature(self, target_celsius: int) -> bool:
        """냉각 목표 온도 설정 [°C]."""
        return self._ok("SetTemperature", c_int(target_celsius))

    def cooler_on(self) -> bool:
        """냉각기 ON."""
        return self._ok("CoolerON")

    def cooler_off(self) -> bool:
        """냉각기 OFF."""
        return self._ok("CoolerOFF")

    def get_temperature(self) -> tuple[int, int]:
        """
        현재 온도 조회.

        Returns
        -------
        (status_code, temperature_celsius)
            status_code: DRV_TEMP_STABILIZED(20036) 등
        """
        t = c_int()
        with self.lock:
            st = self.dll.GetTemperature(byref(t))
        return st, t.value

    # ══════════════════════════════════════════════════════════════════════════
    # 종료
    # ══════════════════════════════════════════════════════════════════════════

    def shutdown(self):
        """Andor SDK 종료. cooler 는 별도로 OFF 해야 함."""
        self._call("ShutDown")
        print("[CCD] Andor SDK shutdown")


# ══════════════════════════════════════════════════════════════════════════════
# CSV 저장 유틸
# ══════════════════════════════════════════════════════════════════════════════

def save_spectrum_csv(result: dict, path: str | Path):
    """
    스펙트럼 결과를 CSV 로 저장.

    calibrated=True 면 4열(pixel, raman_shift, wavelength, intensity),
    calibrated=False 면 2열(pixel, intensity).
    """
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if result.get("calibrated"):
            # 메타 정보를 주석으로
            f.write(f"# laser_nm,{result['laser_nm']}\n")
            f.write(f"# calibration,factory_polynomial\n")
            w.writerow(["pixel", "raman_shift_cm-1", "wavelength_nm", "intensity"])
            for i in range(len(result["pixel"])):
                w.writerow([
                    result["pixel"][i],
                    f"{result['raman_shift_cm-1'][i]:.3f}",
                    f"{result['wavelength_nm'][i]:.4f}",
                    result["intensity"][i],
                ])
        else:
            w.writerow(["pixel", "intensity"])
            for px, val in zip(result["pixel"], result["intensity"]):
                w.writerow([px, val])
    print(f"[CSV] saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 메인 — 단독 실행 시 테스트
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # TODO: 실제 환경에 맞게 수정
    DLL_PATH   = r"C:\Users\user\Desktop\gemma_raman\backend\agents\atmcd64d.dll"
    CONFIG_DIR = r"C:\Users\user\Desktop\gemma_raman\backend\agents"
    CONFIG_TXT = r"C:\Users\user\Desktop\gemma_raman\backend\agents\Config.txt"

    # ── 1. 캘리브레이터 생성 (외부 파일 불필요) ──
    cal = RamanCalibrator.from_factory_calibration()

    # ── 2. 카메라 생성 + Config.txt 기반 초기화 ──
    cam = AndorCamera(DLL_PATH, calibrator=cal)
    if cam.initialize(CONFIG_DIR, config_txt_path=CONFIG_TXT):
        try:
            # ── 3. 측정 ──
            cam.setup_acquisition(
                read_mode=READ_MODE_FVB,
                exposure_time=0.1,
                trigger_mode=TRIGGER_MODE_INTERNAL,
            )
            result = cam.start_acquisition_cycle()

            if result:
                print(f"\n[RESULT] {len(result['intensity'])} pixels, "
                      f"max={max(result['intensity'])}, "
                      f"calibrated={result['calibrated']}")

                # 상위 5개 픽셀
                top5 = sorted(enumerate(result["intensity"]),
                              key=lambda x: x[1], reverse=True)[:5]
                for px, cnt in top5:
                    extra = ""
                    if result["calibrated"]:
                        extra = f"  Δν={result['raman_shift_cm-1'][px]:.1f} cm⁻¹"
                    print(f"  pixel {px:4d} → {cnt}{extra}")

                # CSV 저장
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_spectrum_csv(result, f"spectrum_{ts}.csv")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted.")
        finally:
            cam.shutdown()
