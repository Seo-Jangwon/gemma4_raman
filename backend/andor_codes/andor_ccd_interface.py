"""
andor_ccd_interface.py — Andor CCD 카메라 저수준 ctypes 래퍼
=============================================================

Andor SDK DLL(atmcd64d.dll / atmcd32d.dll)을 ctypes 로 직접 호출하는 순수 하드웨어 제어 클래스. 

[사용 흐름]
  cam = AndorCCD(debug=False, initialize_to_defaults=False)
  cam.set_vs_speed(0)
  cam.set_hs_speed_conventional(2)
  cam.set_preamp_gain(1)
  cam.set_ro_full_vertical_binning(hbin=1)   # FVB: 1D 스펙트럼
  cam.set_aq_single_scan(exposure=0.2)
  cam.set_trigger_mode('internal')
  cam.set_temperature(-40)
  cam.set_cooler(True)
  # ... 온도 안정화 대기 ...
  cam.start_acquisition()
  while cam.get_status() != 'IDLE':
      time.sleep(0.05)
  data = cam.get_acquired_data()  # numpy int32 (1, Nx_ro)
  cam.close()

[DLL 탐색 순서] (64bit)
  1. C:\\Program Files\\Andor Driver Pack 2\\atmcd64d.dll
  2. C:\\Program Files\\Andor SOLIS\\atmcd64d_legacy.dll
  3. 이 파일과 같은 폴더의 atmcd64d.dll (32bit: atmcd32d.dll)

[스레드 안전성]
  모든 DLL 호출은 threading.Lock 으로 보호됨.
  여러 스레드에서 동시에 호출해도 안전.
"""

from __future__ import absolute_import, print_function
import ctypes
from ctypes import c_int, c_uint, c_byte, c_ubyte, c_short, c_double, c_float, c_long
from ctypes import pointer, byref, windll, cdll
import time
import numpy as np
import os
from threading import Lock

import platform
import logging

from enum import Enum

from . import andor_ccd_consts as consts  # 상수 정의 파일


logger = logging.getLogger(__name__)


# 기본값 상수
DEFAULT_TEMPERATURE = -40   # 초기화 시 기본 냉각 목표 온도 [°C]
DEFAULT_EM_GAIN     = 1    # 초기화 시 기본 EM 증폭 이득
DEFAULT_OUTPUT_AMP  = 0     # 0 = EM 앰프, 1 = 일반(conventional) 앰프


# Andor SDK SetReadMode() 에 넘기는 읽기 모드 번호
class AndorReadMode(Enum):
    FullVerticalBinning = 0  # FVB: 수직 전체 합산 → 1D 스펙트럼
    # MultiTrack          = 1  # 다중 트랙 (미구현)
    # RandomTrack         = 2  # 랜덤 트랙 (미구현)
    SingleTrack         = 1  # 단일 트랙: 특정 행만 읽기
    Image               = 2  # 2D 이미지 전체 (ROI 설정 가능)


def _err(retval):
    """SDK 반환값이 DRV_SUCCESS 가 아니면 IOError 발생."""
    if retval == consts.DRV_SUCCESS:
        return retval
    else:
        err_name = consts.consts_by_num.get(retval)
        raise IOError("Andor DRV Failure {}: {}".format(retval, err_name))


class AndorCCD(object):
    """
    Andor CCD 카메라 저수준 제어 클래스.

    Parameters
    ----------
    debug : bool
        True 이면 상세 로그 출력.
    initialize_to_defaults : bool
        True 이면 생성 시 기본값(온도 -80°C, EM gain 10 등) 자동 적용.
        False 이면 생성 후 직접 설정해야 함.
    """

    def __init__(self, debug=False, initialize_to_defaults=True):

        self.debug = debug
        self.lock = Lock()  # 모든 DLL 호출을 직렬화하는 스레드 잠금

        # ── DLL 경로 탐색 ─────────────────────────────────────────────────
        if platform.architecture()[0] == '64bit':
            andorlibpath = None
            # Andor SOLIS 또는 Driver Pack 설치 경로 먼저 확인
            for path in [r"C:\Program Files\Andor Driver Pack 2\atmcd64d.dll",
                         r"C:\Program Files\Andor SOLIS\atmcd64d_legacy.dll"]:
                if os.path.exists(path):
                    andorlibpath = path
                    break

            # 설치 경로에 없으면 이 파일과 같은 폴더에서 찾음
            if andorlibpath is None:
                andorlibpath = str(os.path.join(os.path.dirname(__file__), "atmcd64d.dll"))
        else:
            # 32bit 환경: 같은 폴더의 32bit DLL 사용
            andorlibpath = str(os.path.join(os.path.dirname(__file__), "atmcd32d.dll"))

        self.andorlib = windll.LoadLibrary(andorlibpath)  # DLL 로드

        if self.debug:
            logger.debug("AndorCCD initializing")

        # SDK 초기화: 카메라 탐색 및 연결 (빈 문자열 → 기본 검색 경로)
        with self.lock:
            _err(self.andorlib.Initialize(''))
        if self.debug:
            logger.debug("Andor CCD Library Initialization Successful")

        # 카메라 기본 정보 읽기
        self.get_head_model()          # 헤드 모델명 (예: "DU420A-OE")
        self.get_serial_number()       # 시리얼 번호
        self.get_hardware_version()    # 하드웨어 버전 6개 정수
        self.get_software_version()    # 소프트웨어 버전 6개 정수
        self.get_detector_shape()      # 픽셀 크기 (Nx, Ny) 저장
        self.get_num_ad_channels()     # AD 채널 수 저장
        self.get_num_output_amplifiers()  # 출력 앰프 수 저장

        # 기본값 적용 (요청 시만)
        self._calibrator = None  # RamanCalibrator 인스턴스 (외부에서 주입)

        if initialize_to_defaults:
            self.set_ad_channel()          # 기본 AD 채널 설정
            self.set_aq_single_scan()      # 취득 모드: 단일 스캔
            self.set_num_accumulations(1)  # 누적 횟수: 1
            self.set_num_kinetics(1)       # kinetic 프레임 수: 1

        # EM CCD 여부 확인 및 기본 EM 이득 설정
        self.em_mode = self.has_em_ccd()
        if self.em_mode:
            self.get_EM_gain_range()   # EM 이득 허용 범위 저장
            self.get_EMCCD_gain()      # 현재 EM 이득 저장

        # 시프트 속도 목록 읽기 및 기본값 설정
        self.read_shift_speeds()
        if initialize_to_defaults:
            if self.em_mode:
                self.set_hs_speed_em()        # EM 수평 시프트 속도 기본값
            self.set_hs_speed_conventional()  # 일반 수평 시프트 속도 기본값
            self.set_vs_speed()               # 수직 시프트 속도 기본값

        # 프리앰프 이득 목록 읽기 및 기본값 설정
        self.get_preamp_gains()
        if initialize_to_defaults:
            self.set_preamp_gain()

        # 온도 범위 확인 및 현재 온도 읽기
        self.get_temperature_range()
        self.get_temperature()

        if initialize_to_defaults:
            self.set_temperature(DEFAULT_TEMPERATURE)  # 목표 온도 설정
            self.set_cooler_on()                       # 냉각기 켜기
            self.set_shutter_open(False)               # 셔터 닫기
            self.set_output_amp(DEFAULT_OUTPUT_AMP)    # 출력 앰프 설정
            if self.em_mode:
                self.set_EMCCD_gain(DEFAULT_EM_GAIN)   # EM 이득 기본값 설정


    # ══════════════════════════════════════════════════════════════════
    # 초기화 / 정보 조회 함수
    # ══════════════════════════════════════════════════════════════════

    def get_head_model(self):
        """카메라 헤드 모델명 문자열 반환 (예: 'DU420A-OE')."""
        headModel = ctypes.create_string_buffer(consts.MAX_PATH)
        with self.lock:
            _err(self.andorlib.GetHeadModel(headModel))
        self.headModel = headModel.raw.decode().strip('\x00')
        print(self.headModel)
        if self.debug:
            logger.debug("Head model: " + repr(self.headModel))
        return self.headModel

    def get_serial_number(self):
        """카메라 시리얼 번호(정수) 반환."""
        serialNumber = c_int(-1)
        with self.lock:
            _err(self.andorlib.GetCameraSerialNumber(byref(serialNumber)))
        self.serialNumber = serialNumber.value
        if self.debug:
            logger.debug('Serial Number: %g' % self.serialNumber)
        return serialNumber.value

    def get_hardware_version(self):
        """하드웨어 버전 정보 6-tuple 반환."""
        HW = [c_int(i) for i in range(6)]
        with self.lock:
            _err(self.andorlib.GetHardwareVersion(*[byref(h) for h in HW]))
        self.hardware_version = tuple([h.value for h in HW])
        if self.debug:
            logger.debug('Hardware information: {}'.format(repr(self.hardware_version)))
        return self.hardware_version

    def get_software_version(self):
        """소프트웨어(SDK) 버전 정보 6-tuple 반환."""
        SW = [c_int(i) for i in range(6)]
        with self.lock:
            _err(self.andorlib.GetSoftwareVersion(*[byref(s) for s in SW]))
        self.software_version = tuple([s.value for s in SW])
        if self.debug:
            logger.debug('Software information: %s' % repr(self.software_version))
        return self.software_version

    def get_detector_shape(self):
        """
        검출기 픽셀 크기 반환.

        Returns
        -------
        (Nx, Ny) : (int, int)
            Nx = 수평 픽셀 수, Ny = 수직 픽셀 수.
            self.Nx, self.Ny 에도 저장됨.
        """
        pixelsX = c_int(1)
        pixelsY = c_int(1)
        with self.lock:
            _err(self.andorlib.GetDetector(byref(pixelsX), byref(pixelsY)))
        self.Nx = pixelsX.value  # 수평 픽셀 수 (예: 1024)
        self.Ny = pixelsY.value  # 수직 픽셀 수 (예: 256)
        if self.debug:
            logger.debug("Dimensions: {} {}".format(self.Nx, self.Ny))
        return self.Nx, self.Ny

    def get_num_ad_channels(self):
        """AD 변환기 채널 수 반환 (보통 1)."""
        numADChan = c_int(-1)
        retval = self.andorlib.GetNumberADChannels(byref(numADChan))
        assert retval == consts.DRV_SUCCESS, "Andor DRV Failure %i" % retval
        self.numADChan = numADChan.value
        if self.debug:
            logger.debug('# of AD channels [expecting one]: %g' % self.numADChan)
        return self.numADChan

    def get_num_output_amplifiers(self):
        """출력 앰프 수 반환 (EMCCD: 2, IDUS: 1)."""
        ampNum = c_int(-1)
        retval = self.andorlib.GetNumberAmp(byref(ampNum))
        assert retval == consts.DRV_SUCCESS, "Andor DRV Failure %i" % retval
        self.ampNum = ampNum.value
        if self.debug:
            logger.debug('Number of output amplifiers: %g' % self.ampNum)
        return self.ampNum

    def get_preamp_gains(self):
        """
        사용 가능한 프리앰프 이득 목록 반환.

        Returns
        -------
        list[float]
            각 인덱스에 해당하는 이득 배율 (예: [1.0, 2.4, 4.9]).
            self.preamp_gains 에도 저장됨.
        """
        numGains = c_int(-1)
        with self.lock:
            _err(self.andorlib.GetNumberPreAmpGains(pointer(numGains)))
        if self.debug:
            logger.debug('# of gains: %g ' % numGains.value)
        self.numGains = numGains.value
        self.preamp_gains = []
        gain = c_float(-1)
        for i in range(numGains.value):
            with self.lock:
                _err(self.andorlib.GetPreAmpGain(i, byref(gain)))
            self.preamp_gains.append(gain.value)
        if self.debug:
            logger.debug('Preamp gains available: %s' % self.preamp_gains)
        return self.preamp_gains

    def has_em_ccd(self):
        """
        이 카메라가 EM(Electron Multiplication) 기능을 지원하는지 확인.

        Returns
        -------
        bool
            True: EMCCD 지원, False: 일반 CCD (예: IDUS).
        """
        try:
            self.get_EM_gain_range()
            return True
        except IOError:
            return False


    # ══════════════════════════════════════════════════════════════════
    # AD 채널 / 버퍼 설정
    # ══════════════════════════════════════════════════════════════════

    def set_ad_channel(self, chan_i=0):
        """
        AD 변환기 채널 선택.

        Parameters
        ----------
        chan_i : int
            채널 인덱스 (0 ~ numADChan-1).
        """
        assert chan_i in range(0, self.numADChan)
        with self.lock:
            _err(self.andorlib.SetADChannel(int(chan_i)))
        self.ad_chan = chan_i
        return self.ad_chan

    def create_buffer(self):
        """
        현재 읽기 모드와 취득 모드에 맞는 numpy 버퍼 생성.

        버퍼 형태:
          - single/accumulate/run_till_abort → (Ny_ro, Nx_ro)
          - kinetic                           → (num_kin, Ny_ro, Nx_ro)
        int32 형식. GetAcquiredData() 결과를 저장할 공간.
        """
        if self.aq_mode in ('single', 'accumulate', 'run_till_abort'):
            self.buffer = np.zeros(shape=(self.Ny_ro, self.Nx_ro), dtype=np.int32)
        elif self.aq_mode == 'kinetic':
            self.get_num_kinetics()
            self.buffer = np.zeros(shape=(self.num_kin, self.Ny_ro, self.Nx_ro), dtype=np.int32)
        else:
            raise ValueError("Andor Unknown acq mode {}".format(self.aq_mode))
        print(self.buffer.shape)
        return self.buffer


    # ══════════════════════════════════════════════════════════════════
    # 읽기 모드 설정 (ReadOut Mode)
    # ══════════════════════════════════════════════════════════════════

    def set_readout_mode(self, ro_mode):
        """AndorReadMode Enum 값으로 읽기 모드를 설정."""
        if ro_mode == AndorReadMode.FullVerticalBinning:
            self.set_ro_full_vertical_binning()
        elif ro_mode == AndorReadMode.Image:
            self.set_ro_image_mode()
        elif ro_mode == AndorReadMode.MultiTrack:
            raise NotImplementedError()
        elif ro_mode == AndorReadMode.RandomTrack:
            raise NotImplementedError()
        elif ro_mode == AndorReadMode.SingleTrack:
            self.set_ro_single_track(256, 20)

    def set_read_mode_by_name(self, name):
        """이름 문자열로 읽기 모드 설정 ('FullVerticalBinning', 'Image', 'SingleTrack' 등)."""
        read_mode_dict = dict(
            FullVerticalBinning=0,
            # MultiTrack=1,
            # RandomTrack=2,
            SingleTrack=1,
            Image=2)
        readout_mode_id = read_mode_dict[name]
        self.set_readmode(readout_mode_id)

    def set_read_mode(self, mode_id):
        """SetReadMode() 직접 호출 (정수 mode_id: 0=FVB, 1=SingleTrack, 2=Image)."""
        with self.lock:
            _err(self.andorlib.SetReadMode(mode_id))

    def set_ro_full_vertical_binning(self, hbin=1):
        """
        FVB (Full Vertical Binning) 읽기 모드 설정.
        수직 방향을 전부 합산해 1D 스펙트럼을 얻는다.

        Parameters
        ----------
        hbin : int
            수평 빈닝 계수 (기본 1 = 빈닝 없음).

        결과:
          self.Nx_ro = Nx / hbin,  self.Ny_ro = 1
        """
        self.ro_mode = 'FULL_VERTICAL_BINNING'
        with self.lock:
            _err(self.andorlib.SetReadMode(0))  # 0 = FVB
        self.ro_fvb_hbin = hbin
        with self.lock:
            _err(self.andorlib.SetFVBHBin(self.ro_fvb_hbin))
        self.Nx_ro = int(self.Nx / hbin)  # 수평 픽셀 수 (빈닝 후)
        self.Ny_ro = 1                     # FVB는 항상 1행
        self.create_buffer()

    def set_ro_single_track(self, center, width=1, hbin=1):
        """
        Single Track 읽기 모드 설정.
        특정 수직 행(center ± width/2)만 읽는다.

        Parameters
        ----------
        center : int
            읽을 행의 중심 픽셀 번호 (1-based).
        width : int
            읽을 행 수 (기본 1).
        hbin : int
            수평 빈닝 계수.
        """
        self.ro_mode = 'SINGLE_TRACK'
        with self.lock:
            _err(self.andorlib.SetReadMode(3))
        with self.lock:
            _err(self.andorlib.SetSingleTrack(c_int(center), c_int(width)))
        with self.lock:
            _err(self.andorlib.SetSingleTrackHBin(c_int(hbin)))
        self.ro_st_hbin = hbin
        self.Nx_ro = int(self.Nx / hbin)
        self.Ny_ro = 1
        self.ro_single_track_center = center
        self.ro_single_track_width = width
        self.create_buffer()

    def set_ro_multi_track(self, number, height, offset):
        """Multi Track 모드 — 미구현."""
        raise NotImplementedError

    def set_ro_random_track(self, positions):
        """Random Track 모드 — 미구현."""
        raise NotImplementedError

    def set_ro_image_mode(self, hbin=1, vbin=1, hstart=1, hend=None, vstart=1, vend=None):
        """
        Image 모드 설정 (2D 전체 또는 ROI).

        Parameters
        ----------
        hbin, vbin : int
            수평/수직 빈닝 계수.
        hstart, hend : int
            읽을 수평 픽셀 범위 (1-based, 포함). None이면 전체.
        vstart, vend : int
            읽을 수직 픽셀 범위 (1-based, 포함). None이면 전체.

        결과:
          self.Nx_ro = (hend-hstart+1) / hbin
          self.Ny_ro = (vend-vstart+1) / vbin
        """
        self.ro_mode = 'IMG'
        with self.lock:
            _err(self.andorlib.SetReadMode(4))  # 4 = Image

        if hend is None:
            hend = self.Nx
        if vend is None:
            vend = self.Ny

        assert hend > hstart
        assert vend > vstart

        self.hbin = hbin
        self.vbin = vbin
        self.hstart = hstart
        self.hend = hend
        self.vstart = vstart
        self.vend = vend

        with self.lock:
            _err(self.andorlib.SetImage(c_int(hbin), c_int(vbin),
                                        c_int(hstart), c_int(hend),
                                        c_int(vstart), c_int(vend)))

        self.Nx_ro = int((self.hend - self.hstart + 1) / self.hbin)
        self.Ny_ro = int((self.vend - self.vstart + 1) / self.vbin)

        logger.debug("self.Nx_ro: {}, self.Ny_ro: {}".format(self.Nx_ro, self.Ny_ro))

        self.create_buffer()

    def get_current_hbin(self):
        """현재 읽기 모드의 수평 빈닝 계수 반환."""
        if self.ro_mode == 'IMG':
            return self.hbin
        elif self.ro_mode == 'SINGLE_TRACK':
            return self.ro_st_hbin
        elif self.ro_mode == 'FULL_VERTICAL_BINNING':
            return self.ro_fvb_hbin


    # ══════════════════════════════════════════════════════════════════
    # 취득 모드 설정 (Acquisition Mode)
    # ══════════════════════════════════════════════════════════════════

    def set_aq_mode(self, mode):
        """
        취득 모드 문자열로 설정.

        Parameters
        ----------
        mode : str
            'single' | 'accumulate' | 'kinetic' | 'run_till_abort'
        """
        print('set_aq_mode', mode)
        assert mode in ('single', 'accumulate', 'kinetic', 'run_till_abort')
        if mode == 'single':        return self.set_aq_single_scan()
        if mode == 'accumulate':    return self.set_aq_accumulate_scan()
        if mode == 'kinetic':       return self.set_aq_kinetic_scan()
        if mode == 'run_till_abort': return self.set_aq_run_till_abort_scan()

    def get_aq_mode(self):
        """현재 취득 모드 문자열 반환."""
        return self.aq_mode

    def set_aq_single_scan(self, exposure=None):
        """
        단일 프레임 취득 모드 설정.

        Parameters
        ----------
        exposure : float, optional
            노출 시간 [초]. None 이면 변경 안 함.
        """
        self.aq_mode = 'single'
        with self.lock:
            _err(self.andorlib.SetAcquisitionMode(1))  # SDK: 1 = Single
        if exposure is not None:
            with self.lock:
                _err(self.andorlib.SetExposureTime(c_float(exposure)))

    def set_aq_accumulate_scan(self, exposure_time=None, num_acc=None, cycle_time=None):
        """
        Accumulate 취득 모드 설정.
        동일 노출을 num_acc 회 반복하여 더한다.

        Parameters
        ----------
        exposure_time : float, optional
            단일 노출 시간 [초].
        num_acc : int, optional
            누적 횟수.
        cycle_time : float, optional
            누적 사이클 시간 [초] (내부 트리거 시만 유효).
        """
        self.aq_mode = 'accumulate'
        with self.lock:
            _err(self.andorlib.SetAcquisitionMode(2))  # SDK: 2 = Accumulate
        if exposure_time is not None:
            with self.lock:
                _err(self.andorlib.SetExposureTime(c_float(exposure_time)))
        if num_acc is not None:
            with self.lock:
                _err(self.andorlib.SetNumberAccumulations(num_acc))
        if cycle_time is not None:
            with self.lock:
                _err(self.andorlib.SetAccumulationCycleTime(cycle_time))

    def set_aq_kinetic_scan(self, exp_time=None, num_acc=None, acc_time=None,
                            num_kin=None, kin_time=None):
        """
        Kinetic Series 취득 모드 설정.
        연속 프레임을 빠르게 취득한다.

        Parameters
        ----------
        exp_time : float, optional
            단일 프레임 노출 시간 [초].
        num_acc : int, optional
            프레임당 누적 횟수.
        acc_time : float, optional
            누적 사이클 시간 [초].
        num_kin : int, optional
            취득할 총 프레임 수.
        kin_time : float, optional
            프레임 간 사이클 시간 [초] (내부 트리거 시만 유효).
        """
        self.aq_mode = 'kinetic'
        with self.lock:
            _err(self.andorlib.SetAcquisitionMode(3))  # SDK: 3 = Kinetic
        if exp_time is not None:
            with self.lock:
                _err(self.andorlib.SetExposureTime(c_float(exp_time)))
        if num_acc is not None:
            with self.lock:
                _err(self.andorlib.SetNumberAccumulations(num_acc))
        if acc_time is not None:
            with self.lock:
                _err(self.andorlib.SetAccumulationCycleTime(acc_time))
        if num_kin is not None:
            with self.lock:
                _err(self.andorlib.SetNumberKinetics(num_kin))
        if kin_time is not None:
            with self.lock:
                _err(self.andorlib.SetKineticCycleTime(kin_time))
        print('kinetic')

    def set_aq_run_till_abort_scan(self):
        """Run Till Abort 모드: abort_acquisition() 호출 전까지 연속 취득."""
        self.aq_mode = 'run_till_abort'
        print('set_aq_run_till_abort_scan')
        with self.lock:
            _err(self.andorlib.SetAcquisitionMode(5))  # SDK: 5 = Run Till Abort
        print('set_aq_run_till_abort_scan')

    def set_aq_fast_kinetic_scan(self):
        """Fast Kinetic 모드 — 미구현."""
        raise NotImplementedError()

    def set_aq_frame_transfer_scan(self):
        """Frame Transfer 모드 — 미구현."""
        raise NotImplementedError()


    # ══════════════════════════════════════════════════════════════════
    # 트리거 모드 설정
    # ══════════════════════════════════════════════════════════════════

    # SetTriggerMode() 에 넘기는 정수값
    trigger_modes = dict(
        internal          = 0,   # 내부 트리거 (소프트웨어 제어)
        external          = 1,   # 외부 TTL 트리거
        external_start    = 6,   # 외부 트리거로 시작 후 내부 타이밍
        external_exposure = 7,   # 외부 TTL HIGH 동안 노출 (ExternalExposure)
        external_fvb_em   = 9,   # 외부 트리거 + FVB EM
        software          = 10,  # 소프트웨어 SendSoftwareTrigger() 사용
    )

    def set_trigger_mode(self, mode='internal'):
        """
        트리거 모드 설정.

        Parameters
        ----------
        mode : str
            'internal' | 'external' | 'external_start' |
            'external_exposure' | 'external_fvb_em' | 'software'
        """
        mode = mode.lower()
        retval = self.andorlib.SetTriggerMode(self.trigger_modes[mode])
        assert retval == consts.DRV_SUCCESS, "Andor DRV Failure %i" % retval


    # ══════════════════════════════════════════════════════════════════
    # 시프트 속도 설정 (Shift Speed & Gain)
    # ══════════════════════════════════════════════════════════════════

    def read_shift_speeds(self):
        """
        카메라에서 사용 가능한 수평/수직 시프트 속도 목록 읽기.

        읽은 후 다음 속성이 채워진다:
          self.numHSSpeeds_EM[chan]         — EM 채널별 수평 속도 수
          self.numHSSpeeds_Conventional[chan] — 일반 채널별 수평 속도 수
          self.HSSpeeds_EM[chan][i]         — EM 속도 [MHz]
          self.HSSpeeds_Conventional[chan][i] — 일반 속도 [MHz]
          self.numVSSpeeds                  — 수직 속도 수
          self.VSSpeeds[i]                  — 수직 속도 [μs/pixel]
        """
        numHSSpeeds = c_int(-1)
        self.numHSSpeeds_EM = []
        self.numHSSpeeds_Conventional = []

        # 채널별 수평 속도 수 조회
        for chan_i in range(self.numADChan):
            if self.em_mode:
                retval = self.andorlib.GetNumberHSSpeeds(chan_i, 0, byref(numHSSpeeds))  # 0 = EM 모드
                assert retval == consts.DRV_SUCCESS
                self.numHSSpeeds_EM.append(numHSSpeeds.value)

                retval = self.andorlib.GetNumberHSSpeeds(chan_i, 1, byref(numHSSpeeds))  # 1 = 일반 모드
                assert retval == consts.DRV_SUCCESS
                self.numHSSpeeds_Conventional.append(numHSSpeeds.value)
            else:
                retval = self.andorlib.GetNumberHSSpeeds(chan_i, 0, byref(numHSSpeeds))
                assert retval == consts.DRV_SUCCESS
                self.numHSSpeeds_Conventional.append(numHSSpeeds.value)

        logger.debug('# of horizontal speeds EM: {}'.format(self.numHSSpeeds_EM))
        logger.debug('# of horizontal speeds Conventional: {}'.format(self.numHSSpeeds_Conventional))

        # 수평 속도 값 읽기
        self.HSSpeeds_EM = []
        self.HSSpeeds_Conventional = []
        speed = c_float(0)
        for chan_i in range(self.numADChan):
            self.HSSpeeds_EM.append([])
            if self.em_mode:
                hsspeeds = self.HSSpeeds_EM[chan_i]
                for i in range(self.numHSSpeeds_EM[chan_i]):
                    retval = self.andorlib.GetHSSpeed(chan_i, 0, i, byref(speed))
                    assert retval == consts.DRV_SUCCESS
                    hsspeeds.append(speed.value)
                conventional_index = 1
            else:
                conventional_index = 0

            self.HSSpeeds_Conventional.append([])
            hsspeeds = self.HSSpeeds_Conventional[chan_i]
            for i in range(self.numHSSpeeds_Conventional[chan_i]):
                retval = self.andorlib.GetHSSpeed(chan_i, conventional_index, i, byref(speed))
                assert retval == consts.DRV_SUCCESS
                hsspeeds.append(speed.value)

        logger.debug('EM Horizontal speeds: {} MHz'.format(self.HSSpeeds_EM))
        logger.debug('Conventional Horizontal speeds: {} MHz'.format(self.HSSpeeds_Conventional))

        # 수직 시프트 속도 읽기
        numVSSpeeds = c_int(-1)
        retval = self.andorlib.GetNumberVSSpeeds(byref(numVSSpeeds))
        if retval == 20991:  # DRV_NOT_SUPPORTED: iDus IR InGaAs 같은 단선 검출기
            self.numVSSpeeds = 0
        else:
            assert retval == consts.DRV_SUCCESS
            self.numVSSpeeds = numVSSpeeds.value

        self.VSSpeeds = []
        speed = c_float(0)
        for i in range(self.numVSSpeeds):
            retval = self.andorlib.GetVSSpeed(i, byref(speed))
            assert retval == consts.DRV_SUCCESS
            self.VSSpeeds.append(speed.value)
        if self.debug:
            logger.debug('Vertical speeds [microseconds per pixel shift]: %s' % self.VSSpeeds)

    def get_hs_speed_val_conventional(self, speed_index):
        """일반 모드 수평 시프트 속도 값 반환 (미구현)."""
        pass

    def set_hs_speed_em(self, speed_index=0):
        """
        EM 모드 수평 시프트(readout) 속도 설정.

        Parameters
        ----------
        speed_index : int
            HSSpeeds_EM 인덱스. 0이 가장 빠름.
        """
        logger.debug("set_hs_speed_em {}".format(speed_index))
        assert 0 <= speed_index < self.numHSSpeeds_EM[self.ad_chan]
        with self.lock:
            _err(self.andorlib.SetHSSpeed(0, speed_index))  # 0 = EM 모드

    def set_hs_speed_conventional(self, speed_index=0):
        """
        일반(Conventional) 모드 수평 시프트 속도 설정.
        IDUS 같이 EM이 없는 카메라에서도 사용.

        Parameters
        ----------
        speed_index : int
            HSSpeeds_Conventional 인덱스.
        """
        print("set_hs_speed_conventional", speed_index, self.ad_chan)
        assert 0 <= speed_index < self.numHSSpeeds_Conventional[self.ad_chan]
        if self.em_mode:
            with self.lock:
                _err(self.andorlib.SetHSSpeed(1, speed_index))  # 1 = 일반 모드
        else:
            with self.lock:
                _err(self.andorlib.SetHSSpeed(0, speed_index))  # EM 없는 카메라는 0

    def set_vs_speed(self, speed_index=0):
        """
        수직 시프트 속도 설정.

        Parameters
        ----------
        speed_index : int
            VSSpeeds 인덱스. 클수록 느리지만 노이즈 적음.
        """
        assert 0 <= speed_index < self.numVSSpeeds
        with self.lock:
            _err(self.andorlib.SetVSSpeed(speed_index))

    def set_preamp_gain(self, gain_i=0):
        """
        프리앰프 이득 설정.

        Parameters
        ----------
        gain_i : int
            preamp_gains 인덱스 (0-based).
        """
        with self.lock:
            _err(self.andorlib.SetPreAmpGain(gain_i))
        self.preamp_gain_i = gain_i


    # ══════════════════════════════════════════════════════════════════
    # 이미지 회전 / 반전 (SDK 소프트웨어 처리, 카메라 내부 아님)
    # ══════════════════════════════════════════════════════════════════

    def get_image_flip(self):
        """현재 이미지 반전 상태 (hflip, vflip) 반환."""
        hflip, vflip = c_int(-1), c_int(-1)
        with self.lock:
            _err(self.andorlib.GetImageFlip(byref(hflip), byref(vflip)))
        self.hflip = bool(hflip.value)
        self.vflip = bool(vflip.value)
        return self.hflip, self.vflip

    def set_image_flip(self, hflip=True, vflip=False):
        """
        이미지 반전 설정.

        Parameters
        ----------
        hflip : bool
            True이면 수평 좌우 반전. Config.txt Reverse=True 에 대응.
        vflip : bool
            True이면 수직 상하 반전.
        """
        with self.lock:
            _err(self.andorlib.SetImageFlip(c_int(bool(hflip)), c_int(bool(vflip))))

    def get_image_hflip(self):
        """수평 반전 상태 반환."""
        return self.get_image_flip()[0]

    def set_image_hflip(self, hflipNew):
        """수평 반전만 변경 (수직 반전 상태 유지)."""
        hflipOld, vflipOld = self.get_image_flip()
        self.set_image_flip(hflipNew, vflipOld)
        logger.debug("set_image_hflip: {}".format(hflipNew))

    def get_image_vflip(self):
        """수직 반전 상태 반환."""
        return self.get_image_flip()[1]

    def set_image_vflip(self, vflipNew):
        """수직 반전만 변경 (수평 반전 상태 유지)."""
        hflipOld, vflipOld = self.get_image_flip()
        self.set_image_flip(hflipOld, vflipNew)
        logger.debug("set_image_vflip: {}".format(vflipNew))

    def set_image_rotate(self, rotate=0):
        """
        이미지 회전 설정.

        Parameters
        ----------
        rotate : int
            0 = 회전 없음, 1 = 90° 시계방향, 2 = 90° 반시계방향.
        """
        assert rotate in [0, 1, 2]
        with self.lock:
            _err(self.andorlib.SetImageRotation(c_int(rotate)))


    # ══════════════════════════════════════════════════════════════════
    # 셔터 제어
    # ══════════════════════════════════════════════════════════════════

    def set_shutter_auto(self):
        """셔터 자동 모드: 취득 시 자동 열리고 닫힘."""
        with self.lock:
            _err(self.andorlib.SetShutter(0, 0, 0, 0))  # mode 0 = Auto

    def set_shutter_open(self, open=True):
        """
        셔터 열기/닫기.

        Parameters
        ----------
        open : bool
            True = 강제 열기, False = 강제 닫기.
        """
        if open:
            with self.lock:
                _err(self.andorlib.SetShutter(0, 1, 0, 0))  # mode 1 = Open
        else:
            self.set_shutter_close()

    def set_shutter_close(self):
        """셔터 강제 닫기."""
        with self.lock:
            _err(self.andorlib.SetShutter(0, 2, 0, 0))  # mode 2 = Close


    # ══════════════════════════════════════════════════════════════════
    # 온도 / 냉각기 제어
    # ══════════════════════════════════════════════════════════════════

    def set_cooler_on(self):
        """냉각기 켜기 (CoolerON)."""
        with self.lock:
            _err(self.andorlib.CoolerON())
        self.cooler_on = True

    def set_cooler_off(self):
        """냉각기 끄기 (CoolerOFF). 종료 전 반드시 호출."""
        with self.lock:
            _err(self.andorlib.CoolerOFF())
        self.cooler_on = False

    def set_cooler(self, coolerOn):
        """
        냉각기 켜기/끄기.

        Parameters
        ----------
        coolerOn : bool
        """
        if coolerOn:
            self.set_cooler_on()
        else:
            self.set_cooler_off()

    def get_cooler(self):
        """현재 냉각기 상태 반환 (True = ON)."""
        return self.cooler_on

    def get_temperature_range(self):
        """
        이 카메라의 허용 온도 범위 반환.

        Returns
        -------
        (min_temp, max_temp) : (int, int)  [°C]
        """
        min_t, max_t = c_int(0), c_int(0)
        with self.lock:
            _err(self.andorlib.GetTemperatureRange(byref(min_t), byref(max_t)))
        self.min_temp = min_t.value
        self.max_temp = max_t.value
        return self.min_temp, self.max_temp

    def set_temperature(self, new_temp):
        """
        목표 냉각 온도 설정.

        Parameters
        ----------
        new_temp : int
            목표 온도 [°C]. 허용 범위: get_temperature_range() 참고.
        """
        with self.lock:
            _err(self.andorlib.SetTemperature(c_int(new_temp)))
        self.get_temperature()

    def get_temperature(self):
        """
        현재 CCD 온도 반환.

        Returns
        -------
        int
            현재 온도 [°C]. self.temperature 에도 저장됨.
            온도 상태 코드(DRV_TEMP_*)는 self.temperature_status_num 에 저장됨.
        """
        lastTemp = c_int(0)
        with self.lock:
            retval = self.andorlib.GetTemperature(byref(lastTemp))
        if retval == consts.DRV_ACQUIRING:
            raise IOError("Camera busy acquiring")
        elif retval in (consts.DRV_NOT_INITIALIZED, consts.DRV_ERROR_ACK):
            _err(retval)
        else:
            self.temperature = lastTemp.value
            self.temperature_status_num = retval  # DRV_TEMP_* 코드
            return self.temperature

    # 온도 상태 코드 → 문자열 매핑
    temp_status_dict = {
        consts.DRV_TEMP_OFF:            'OFF',
        consts.DRV_TEMP_NOT_STABILIZED: 'NOT_STABILIZED',
        consts.DRV_TEMP_STABILIZED:     'STABILIZED',
        consts.DRV_TEMP_NOT_REACHED:    'NOT_REACHED',
        consts.DRV_TEMP_NOT_SUPPORTED:  'NOT_SUPPORTED',
        consts.DRV_TEMP_DRIFT:          'DRIFT',
    }

    def get_temperature_status(self):
        """
        온도 상태 문자열 반환.

        Returns
        -------
        str
            'OFF' | 'NOT_STABILIZED' | 'STABILIZED' | 'NOT_REACHED' | 'DRIFT'
            ★ 'STABILIZED' 가 되면 촬영 가능 상태.
        """
        self.get_temperature()
        return self.temp_status_dict[self.temperature_status_num]


    # ══════════════════════════════════════════════════════════════════
    # 취득 제어 및 데이터 읽기
    # 기본 흐름: start_acquisition() → (폴링) → get_acquired_data()
    # ══════════════════════════════════════════════════════════════════

    def start_acquisition(self):
        """취득 시작. 비블로킹: get_status() 로 완료 여부 확인해야 함."""
        # todo: 레이저 켜기
        with self.lock:
            _err(self.andorlib.StartAcquisition())

    def abort_acquisition(self):
        """진행 중인 취득 강제 중단."""
        with self.lock:
            _err(self.andorlib.AbortAcquisition())
        # todo: 레이저 끄기

    # GetStatus() 반환 코드 → 문자열 매핑
    _status_name_dict = {
        consts.DRV_IDLE:                'IDLE',         # 유휴 (취득 완료)
        consts.DRV_TEMPCYCLE:           'TEMPCYCLE',    # 온도 사이클 중
        consts.DRV_ACQUIRING:           'ACQUIRING',    # 취득 중
        consts.DRV_ACCUM_TIME_NOT_MET:  'ACCUM_TIME_NOT_MET',
        consts.DRV_KINETIC_TIME_NOT_MET:'KINETIC_TIME_NOT_MET',
        consts.DRV_ERROR_ACK:           'ERROR_ACK',
        consts.DRV_ACQ_BUFFER:          'ACQ_BUFFER',
        consts.DRV_SPOOLERROR:          'SPOOLERROR',
    }

    def get_status(self):
        """
        카메라 현재 상태 반환.

        Returns
        -------
        str
            'IDLE' | 'ACQUIRING' | 'TEMPCYCLE' | ...
            ★ 'IDLE' 이 되면 get_acquired_data() 호출 가능.
        """
        status = c_int(-1)
        with self.lock:
            _err(self.andorlib.GetStatus(byref(status)))
        self.status_id = status.value
        self.status_name = self._status_name_dict[self.status_id]
        return self.status_name

    def get_acquired_data(self):
        """
        취득 완료된 데이터를 self.buffer 로 복사 후 반환.

        Returns
        -------
        np.ndarray (int32)
            FVB/Single: (1, Nx_ro) | Image: (Ny_ro, Nx_ro) | Kinetic: (num_kin, Ny_ro, Nx_ro)
        """
        with self.lock:
            _err(self.andorlib.GetAcquiredData(
                self.buffer.ctypes.data_as(ctypes.POINTER(c_long)),
                c_uint(self.buffer.size)))
        return self.buffer


    # ══════════════════════════════════════════════════════════════════
    # 취득 타이밍 설정
    # ══════════════════════════════════════════════════════════════════

    def get_acquisition_timings(self):
        """
        실제 적용된 타이밍 값 조회.

        Returns
        -------
        (exposure_time, accumulation_time, kinetic_cycle_time) : (float, float, float) [초]
        """
        exposure = c_float(-1)
        accum    = c_float(-1)
        kinetic  = c_float(-1)
        with self.lock:
            _err(self.andorlib.GetAcquisitionTimings(byref(exposure), byref(accum), byref(kinetic)))
        self.exposure_time      = exposure.value
        self.accumulation_time  = accum.value
        self.kinetic_cycle_time = kinetic.value
        return self.exposure_time, self.accumulation_time, self.kinetic_cycle_time

    def set_exposure_time(self, dt):
        """
        노출 시간 설정.

        Parameters
        ----------
        dt : float
            노출 시간 [초].

        Returns
        -------
        float
            SDK가 실제로 적용한 노출 시간 [초] (요청값과 미세하게 다를 수 있음).
        """
        with self.lock:
            _err(self.andorlib.SetExposureTime(c_float(dt)))
        self.get_acquisition_timings()
        if self.debug:
            logger.debug('set exposure to: {}'.format(self.exposure_time))
        return self.exposure_time

    def get_exposure_time(self):
        """현재 노출 시간 [초] 반환."""
        return self.get_acquisition_timings()[0]

    def set_num_accumulations(self, num):
        """누적 횟수 설정 (Accumulate / Kinetic 모드용)."""
        with self.lock:
            _err(self.andorlib.SetNumberAccumulations(num))
        self.num_acc = num

    def get_num_accumulations(self):
        """현재 누적 횟수 반환."""
        return self.num_acc

    def set_num_kinetics(self, num):
        """
        Kinetic Series 프레임 수 설정.

        Parameters
        ----------
        num : int
            한 번의 StartAcquisition() 에서 취득할 총 프레임 수.
        """
        print('set_num_kinetics', num)
        with self.lock:
            _err(self.andorlib.SetNumberKinetics(int(num)))
        self.num_kin = num

    def get_num_kinetics(self):
        """현재 Kinetic 프레임 수 반환."""
        return self.num_kin

    def set_accumulation_cycle_time(self, acc_time):
        """누적 사이클 시간 설정 [초] (내부 트리거 시만 유효)."""
        with self.lock:
            _err(self.andorlib.SetAccumulationCycleTime(c_float(acc_time)))

    def set_kinetic_cycle_time(self, kin_time):
        """Kinetic 프레임 간격 설정 [초] (내부 트리거 시만 유효)."""
        with self.lock:
            _err(self.andorlib.SetKineticCycleTime(c_float(kin_time)))


    # ══════════════════════════════════════════════════════════════════
    # EM (Electron Multiplication) 이득 제어
    # ══════════════════════════════════════════════════════════════════

    def set_EM_advanced(self, state=True):
        """고급 EM 이득 모드 활성화 (고이득 설정 가능)."""
        with self.lock:
            _err(self.andorlib.SetEMGainRange(c_int(state)))

    def get_EM_gain_range(self):
        """
        EM 이득 허용 범위 반환.

        Returns
        -------
        (low, high) : (int, int)
        """
        low, high = c_int(-1), c_int(-1)
        with self.lock:
            _err(self.andorlib.GetEMGainRange(byref(low), byref(high)))
        self.em_gain_range = (low.value, high.value)
        return self.em_gain_range

    def get_EMCCD_gain(self):
        """현재 EM CCD 이득값 반환."""
        gain = c_int(-1)
        with self.lock:
            _err(self.andorlib.GetEMCCDGain(byref(gain)))
        self.em_gain = gain.value
        return self.em_gain

    def set_EMCCD_gain(self, gain):
        """
        EM CCD 이득 설정.

        Parameters
        ----------
        gain : int
            설정할 이득값 (em_gain_range 범위 내).
        """
        low, high = self.em_gain_range
        assert low <= gain <= high
        with self.lock:
            _err(self.andorlib.SetEMCCDGain(c_int(gain)))


    # ══════════════════════════════════════════════════════════════════
    # 출력 앰프 설정
    # ══════════════════════════════════════════════════════════════════

    def set_output_amp(self, amp):
        """
        출력 앰프 선택.

        Parameters
        ----------
        amp : int
            0 = EMCCD 앰프, 1 = 일반(Conventional) 앰프.
        """
        with self.lock:
            _err(self.andorlib.SetOutputAmplifier(c_int(amp)))
        self.output_amp = amp

    def get_output_amp(self):
        """현재 출력 앰프 인덱스 반환."""
        return self.output_amp


    # ══════════════════════════════════════════════════════════════════
    # raman_tools 호환 고수준 API
    # ══════════════════════════════════════════════════════════════════

    def setup_acquisition(self, read_mode: int, exposure_time: float, trigger_mode: int):
        """
        raman_tools.acquire_spectrum() 호환 설정 API.

        Parameters
        ----------
        read_mode : int
            0 = FVB (Full Vertical Binning, 1D 스펙트럼)
        exposure_time : float
            노출 시간 [초].
        trigger_mode : int
            0 = 내부 트리거.
        """
        # aq_mode 를 먼저 설정해야 set_ro_*() 내부의 create_buffer() 가 동작함
        self.set_aq_single_scan()

        if read_mode == 0:
            self.set_ro_full_vertical_binning()

        _trig_str = {v: k for k, v in self.trigger_modes.items()}.get(trigger_mode, 'internal')
        self.set_trigger_mode(_trig_str)
        self.set_exposure_time(exposure_time)

    def start_acquisition_cycle(self) -> dict:
        """
        단일 촬영 사이클 실행 후 raman_tools 호환 dict 반환.

        반환 dict:
          intensity : list[int]          — 픽셀별 강도 (1D)
          calibrated : bool
          raman_shift_cm-1 : list[float] — _calibrator 주입 시
          wavelength_nm    : list[float] — _calibrator 주입 시
          laser_nm         : float       — _calibrator 주입 시
        """
        self.start_acquisition()
        while self.get_status() != 'IDLE':
            time.sleep(0.05)

        raw = self.get_acquired_data()        # ndarray (Ny_ro, Nx_ro)
        intensity = raw.flatten().tolist()    # FVB → (1, Nx) → 1D list

        result = {"intensity": intensity, "calibrated": False}

        cal = self._calibrator
        if cal is not None:
            pixels = range(len(intensity))
            result.update({
                "calibrated":       True,
                "raman_shift_cm-1": [float(cal.pixel_to_raman_shift(p)) for p in pixels],
                "wavelength_nm":    [float(cal.pixel_to_wavelength(p))   for p in pixels],
                "laser_nm":         float(cal.laser_nm),
            })
        return result


    # ══════════════════════════════════════════════════════════════════
    # 종료
    # ══════════════════════════════════════════════════════════════════

    def close(self):
        """
        Andor SDK 종료. 반드시 프로그램 종료 전에 호출.
        ★ 냉각기를 켠 경우 set_cooler(False) 후 온도가 -5°C 이상이 된 뒤 호출할 것.
        """
        with self.lock:
            _err(self.andorlib.ShutDown())


    # ══════════════════════════════════════════════════════════════════
    # 원형 버퍼 / 순환 이미지 관리 (Kinetic / Run-till-abort 모드용)
    # ══════════════════════════════════════════════════════════════════

    def get_total_number_images_acquired(self):
        """원형 버퍼에 지금까지 취득된 총 이미지 수 반환."""
        num = c_long(0)
        with self.lock:
            _err(self.andorlib.GetTotalNumberImagesAcquired(byref(num)))
        return num.value

    def get_number_new_images(self):
        """
        아직 읽지 않은 새 이미지의 (first, last) 인덱스 반환.

        원형 버퍼가 덮어써진 이미지는 포함하지 않음.
        GetImages() 와 함께 사용.
        """
        first = c_long(0)
        last  = c_long(0)
        with self.lock:
            _err(self.andorlib.GetNumberNewImages(byref(first), byref(last)))
        return first.value, last.value

    def get_number_available_images(self):
        """현재 버퍼에서 읽을 수 있는 이미지의 (first, last) 인덱스 반환."""
        first = c_long(0)
        last  = c_long(0)
        with self.lock:
            _err(self.andorlib.GetNumberAvailableImages(byref(first), byref(last)))
        return first.value, last.value

    def get_images(self, first, last, buf):
        """
        원형 버퍼에서 first~last 범위 이미지를 buf 에 복사.

        Returns
        -------
        (validfirst, validlast, buf)
        """
        validfirst = c_long(0)
        validlast  = c_long(0)
        with self.lock:
            _err(self.andorlib.GetImages(
                c_long(first), c_long(last),
                buf.ctypes.data_as(ctypes.POINTER(c_long)),
                c_uint(buf.size),
                byref(validfirst), byref(validlast)))
        return validfirst, validlast, buf

    def get_oldest_image(self, arr=None):
        """
        원형 버퍼에서 가장 오래된 미읽은 이미지 반환.

        Returns
        -------
        np.ndarray or None
            새 데이터 없으면 None 반환.
        """
        if arr is None:
            arr = np.zeros((self.Ny_ro, self.Nx_ro), dtype=np.int32)

        arr_ptr  = arr.ctypes.data_as(ctypes.POINTER(c_long))
        arr_size = c_uint(arr.size)
        with self.lock:
            retval = self.andorlib.GetOldestImage(arr_ptr, arr_size)

        if retval == consts.DRV_NO_NEW_DATA:  # 새 데이터 없음
            return None
        else:
            _err(retval)
        return arr


# ══════════════════════════════════════════════════════════════════════
# 단독 실행 테스트
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import time

    cam = AndorCCD(debug=True)

    cam.set_ro_image_mode()
    cam.set_trigger_mode('internal')
    cam.set_exposure_time(1.0)
    cam.andorlib.SetOutputAmplifier(0)   # EMCCD 앰프
    cam.read_shift_speeds()
    cam.andorlib.SetOutputAmplifier(1)   # 일반 앰프
    cam.andorlib.SetEMGainMode(1)
    print("EM_gain_range", cam.get_EM_gain_range())
    cam.start_acquisition()
    stat = "ACQUIRING",
    while stat != "IDLE":
        time.sleep(0.1)
        stati, stat = cam.get_status()
    cam.get_acquired_data()
    cam.set_shutter_close()

    import pylab as pl
    pl.imshow(cam.buffer, interpolation='nearest', origin='lower')
    pl.show()
