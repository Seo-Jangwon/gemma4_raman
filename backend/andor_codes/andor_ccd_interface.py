"""
andor_ccd_interface.py — Andor CCD 카메라 제어 클래스 (pyAndorSDK2 기반)
==========================================================================

공식 Andor Python SDK(pyAndorSDK2)를 사용하는 스레드 안전 카메라 제어 클래스.

[사용 흐름]
  cam = AndorCCD(debug=False, initialize_to_defaults=False)
  cam.set_vs_speed(0)
  cam.set_hs_speed_conventional(0)
  cam.set_preamp_gain(0)
  cam.set_ro_full_vertical_binning(hbin=1)   # FVB: 1D 스펙트럼
  cam.set_aq_single_scan(exposure=0.2)
  cam.set_trigger_mode('internal')
  cam.set_temperature(-40)
  cam.set_cooler(True)
  # ... 온도 안정화 대기 ...
  cam.start_acquisition()
  cam.wait_for_acquisition()        # 이벤트 기반 대기 (폴링 불필요)
  data = cam.get_acquired_data()    # numpy int32 (Ny_ro, Nx_ro)
  cam.close()

[DLL 탐색]
  pyAndorSDK2 패키지에 내장된 DLL 자동 사용:
  backend/Andor SDK/Python/pyAndorSDK2/pyAndorSDK2/libs/Windows/64/atmcd64d.dll

[스레드 안전성]
  모든 SDK 호출은 threading.Lock 으로 보호됨.
  단, WaitForAcquisition 은 락 없이 호출 (다른 스레드에서 AbortAcquisition 가능).
"""

from __future__ import absolute_import, print_function
import sys
import os
import threading
import logging
import numpy as np

# pyAndorSDK2 패키지 경로 추가 (pip 미설치 환경 대비)
_SDK_CANDIDATES = [
    r"C:\Program Files\Andor SDK\Python\pyAndorSDK2",
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Andor SDK', 'Python', 'pyAndorSDK2')),
]
for _SDK_PATH in _SDK_CANDIDATES:
    if _SDK_PATH not in sys.path:
        sys.path.insert(0, _SDK_PATH)

# atmcd64d.dll 위치 — ctypes.util.find_library()가 PATH에서 탐색하도록 전달
_DLL_DIR = r"C:\Program Files\Andor SDK"

from pyAndorSDK2 import atmcd
from pyAndorSDK2.atmcd_errors import Error_Codes

from . import andor_ccd_consts as consts

logger = logging.getLogger(__name__)

# 기본값 상수
DEFAULT_TEMPERATURE = -40   # 초기화 기본 냉각 목표 [°C]
DEFAULT_EM_GAIN     = 1     # 초기화 기본 EM 이득
DEFAULT_OUTPUT_AMP  = 0     # 0 = EMCCD 앰프, 1 = Conventional 앰프


def _check(ret, func_name=""):
    """DRV_SUCCESS 가 아니면 IOError 발생."""
    if int(ret) != consts.DRV_SUCCESS:
        try:
            name = Error_Codes(int(ret)).name
        except ValueError:
            name = str(ret)
        raise IOError(f"Andor SDK [{func_name}]: {name} ({int(ret)})")


class AndorCCD(object):
    """
    Andor CCD 카메라 제어 클래스 (pyAndorSDK2 기반).

    Parameters
    ----------
    debug : bool
        True 이면 상세 로그 출력.
    initialize_to_defaults : bool
        True 이면 생성 시 기본값 자동 적용.
        False 이면 생성 후 직접 설정.
    """

    def __init__(self, debug=False, initialize_to_defaults=True):
        self.debug = debug
        self.lock = threading.Lock()
        self._calibrator = None  # RamanCalibrator 인스턴스 (외부에서 주입)

        # SDK 객체 생성 — DLL 경로 명시 전달 (find_library가 PATH에서 탐색)
        self.sdk = atmcd(userPath=_DLL_DIR)

        # SDK 초기화
        with self.lock:
            _check(self.sdk.Initialize(""), "Initialize")

        if self.debug:
            logger.debug("AndorCCD Library Initialization Successful")

        # 카메라 기본 정보
        self.get_head_model()
        self.get_serial_number()
        self.get_hardware_version()
        self.get_software_version()
        self.get_detector_shape()
        self.get_num_ad_channels()
        self.get_num_output_amplifiers()

        # 내부 상태 초기화
        self.aq_mode = 'single'
        self.num_kin = 1
        self.num_acc = 1
        self.Nx_ro = self.Nx
        self.Ny_ro = 1
        self.buffer = np.zeros((1, self.Nx), dtype=np.int32)
        self.ro_mode = 'FULL_VERTICAL_BINNING'
        self.ro_fvb_hbin = 1
        # 현재 셔터 모드 캐시('auto'|'open'|'close'). Andor SDK는 SetShutter만 있고
        # 읽기(GetShutter)가 없어, ro_mode처럼 set 시점에 캐시해야 상태 조회가 가능하다.
        self.shutter_mode = 'auto'
        self.ad_chan = 0
        self.output_amp = DEFAULT_OUTPUT_AMP
        self.cooler_on = False
        self.temperature_status_num = consts.DRV_TEMP_OFF

        if initialize_to_defaults:
            self.set_ad_channel()
            self.set_aq_single_scan()
            self.set_num_accumulations(1)
            self.set_num_kinetics(1)

        # EM CCD 확인
        self.em_mode = self.has_em_ccd()
        if self.em_mode:
            self.get_EM_gain_range()
            self.get_EMCCD_gain()
        else:
            self.em_gain_range = (0, 0)
            self.em_gain = 0

        # 시프트 속도 목록
        self.read_shift_speeds()
        if initialize_to_defaults:
            if self.em_mode:
                self.set_hs_speed_em()
            self.set_hs_speed_conventional()
            if self.numVSSpeeds > 0:
                self.set_vs_speed()

        # 프리앰프 이득
        self.get_preamp_gains()
        if initialize_to_defaults:
            self.set_preamp_gain()

        # 온도
        self.get_temperature_range()
        self.get_temperature()

        if initialize_to_defaults:
            self.set_temperature(DEFAULT_TEMPERATURE)
            self.set_cooler_on()
            self.set_shutter_open(False)
            self.set_output_amp(DEFAULT_OUTPUT_AMP)
            if self.em_mode:
                self.set_EMCCD_gain(DEFAULT_EM_GAIN)


    # ══════════════════════════════════════════════════════════════════
    # 초기화 / 카메라 정보 조회
    # ══════════════════════════════════════════════════════════════════

    def get_head_model(self):
        """카메라 헤드 모델명 반환 (예: 'DU420A-OE')."""
        with self.lock:
            (ret, name) = self.sdk.GetHeadModel()
        _check(ret, "GetHeadModel")
        self.headModel = name
        logger.debug("Head model: " + repr(self.headModel))
        if self.debug:
            logger.debug("Head model: " + repr(self.headModel))
        return self.headModel

    def get_serial_number(self):
        """카메라 시리얼 번호(정수) 반환."""
        with self.lock:
            (ret, number) = self.sdk.GetCameraSerialNumber()
        _check(ret, "GetCameraSerialNumber")
        self.serialNumber = number
        if self.debug:
            logger.debug('Serial Number: %g' % self.serialNumber)
        return self.serialNumber

    def get_hardware_version(self):
        """하드웨어 버전 정보 6-tuple 반환."""
        with self.lock:
            (ret, PCB, Decode, dummy1, dummy2, FirmVer, FirmBuild) = self.sdk.GetHardwareVersion()
        _check(ret, "GetHardwareVersion")
        self.hardware_version = (PCB, Decode, dummy1, dummy2, FirmVer, FirmBuild)
        if self.debug:
            logger.debug('Hardware information: {}'.format(repr(self.hardware_version)))
        return self.hardware_version

    def get_software_version(self):
        """소프트웨어(SDK) 버전 정보 6-tuple 반환."""
        with self.lock:
            (ret, eprom, coffile, vxdrev, vxdver, dllrev, dllver) = self.sdk.GetSoftwareVersion()
        _check(ret, "GetSoftwareVersion")
        self.software_version = (eprom, coffile, vxdrev, vxdver, dllrev, dllver)
        if self.debug:
            logger.debug('Software information: %s' % repr(self.software_version))
        return self.software_version

    def get_detector_shape(self):
        """
        검출기 픽셀 크기 반환.

        Returns
        -------
        (Nx, Ny) : (int, int)
        """
        with self.lock:
            (ret, xpixels, ypixels) = self.sdk.GetDetector()
        _check(ret, "GetDetector")
        self.Nx = int(xpixels)
        self.Ny = int(ypixels)
        if self.debug:
            logger.debug("Dimensions: {} {}".format(self.Nx, self.Ny))
        return self.Nx, self.Ny

    def get_num_ad_channels(self):
        """AD 변환기 채널 수 반환."""
        with self.lock:
            (ret, channels) = self.sdk.GetNumberADChannels()
        _check(ret, "GetNumberADChannels")
        self.numADChan = channels
        if self.debug:
            logger.debug('# of AD channels: %g' % self.numADChan)
        return self.numADChan

    def get_num_output_amplifiers(self):
        """출력 앰프 수 반환 (EMCCD: 2, IDUS: 1)."""
        with self.lock:
            (ret, amp) = self.sdk.GetNumberAmp()
        _check(ret, "GetNumberAmp")
        self.ampNum = amp
        if self.debug:
            logger.debug('Number of output amplifiers: %g' % self.ampNum)
        return self.ampNum

    def get_preamp_gains(self):
        """
        사용 가능한 프리앰프 이득 목록 반환.

        Returns
        -------
        list[float]
        """
        with self.lock:
            (ret, noGains) = self.sdk.GetNumberPreAmpGains()
        _check(ret, "GetNumberPreAmpGains")
        self.numGains = noGains
        self.preamp_gains = []
        for i in range(noGains):
            with self.lock:
                (ret, gain) = self.sdk.GetPreAmpGain(i)
            _check(ret, "GetPreAmpGain")
            self.preamp_gains.append(gain)
        if self.debug:
            logger.debug('Preamp gains available: %s' % self.preamp_gains)
        return self.preamp_gains

    def has_em_ccd(self):
        """EM 기능 지원 여부 반환."""
        try:
            self.get_EM_gain_range()
            return True
        except IOError:
            return False

    def get_capabilities(self):
        """카메라 기능(AndorCapabilities) 구조체 반환."""
        with self.lock:
            (ret, caps) = self.sdk.GetCapabilities()
        _check(ret, "GetCapabilities")
        return caps

    def get_pixel_size(self):
        """픽셀 물리 크기 반환 (x_um, y_um) [마이크로미터]."""
        with self.lock:
            (ret, xpix, ypix) = self.sdk.GetPixelSize()
        _check(ret, "GetPixelSize")
        return float(xpix), float(ypix)

    def get_bit_depth(self, channel=0):
        """ADC 비트 심도 반환."""
        with self.lock:
            (ret, depth) = self.sdk.GetBitDepth(channel)
        _check(ret, "GetBitDepth")
        return int(depth)


    # ══════════════════════════════════════════════════════════════════
    # AD 채널 / 버퍼
    # ══════════════════════════════════════════════════════════════════

    def set_ad_channel(self, chan_i=0):
        """AD 채널 선택."""
        assert chan_i in range(0, self.numADChan)
        with self.lock:
            _check(self.sdk.SetADChannel(int(chan_i)), "SetADChannel")
        self.ad_chan = chan_i
        return self.ad_chan

    def create_buffer(self):
        """
        현재 읽기/취득 모드에 맞는 numpy 버퍼 생성.

        버퍼 형태:
          - single/accumulate/run_till_abort → (Ny_ro, Nx_ro)
          - kinetic                           → (num_kin, Ny_ro, Nx_ro)
        """
        if self.aq_mode in ('single', 'accumulate', 'run_till_abort'):
            self.buffer = np.zeros(shape=(self.Ny_ro, self.Nx_ro), dtype=np.int32)
        elif self.aq_mode == 'kinetic':
            self.get_num_kinetics()
            self.buffer = np.zeros(shape=(self.num_kin, self.Ny_ro, self.Nx_ro), dtype=np.int32)
        elif self.aq_mode == 'fast_kinetic':
            self.buffer = np.zeros(shape=(self.num_fast_kin, self.Ny_ro, self.Nx_ro), dtype=np.int32)
        else:
            raise ValueError("Andor Unknown acq mode {}".format(self.aq_mode))
        logger.debug("buffer shape: {}".format(self.buffer.shape))
        return self.buffer


    # ══════════════════════════════════════════════════════════════════
    # 읽기 모드 설정 (ReadOut Mode)
    # ══════════════════════════════════════════════════════════════════

    def set_read_mode(self, mode_id):
        """SetReadMode() 직접 호출 (0=FVB, 1=MultiTrack, 2=RandomTrack, 3=SingleTrack, 4=Image)."""
        with self.lock:
            _check(self.sdk.SetReadMode(int(mode_id)), "SetReadMode")

    def set_ro_full_vertical_binning(self, hbin=1):
        """
        FVB (Full Vertical Binning) 읽기 모드.
        수직 전체 합산 → 1D 스펙트럼.

        Parameters
        ----------
        hbin : int  수평 빈닝 계수 (기본 1).
        """
        self.ro_mode = 'FULL_VERTICAL_BINNING'
        with self.lock:
            _check(self.sdk.SetReadMode(int(consts.Read_Mode.FULL_VERTICAL_BINNING)), "SetReadMode")
        self.ro_fvb_hbin = hbin
        with self.lock:
            _check(self.sdk.SetFVBHBin(int(hbin)), "SetFVBHBin")
        self.Nx_ro = int(self.Nx / hbin)
        self.Ny_ro = 1
        self.create_buffer()

    def set_ro_single_track(self, center, width=1, hbin=1):
        """
        Single Track 읽기 모드.
        특정 수직 행(center ± width/2)만 읽기.

        Parameters
        ----------
        center : int  중심 픽셀 번호 (1-based).
        width  : int  읽을 행 수.
        hbin   : int  수평 빈닝 계수.
        """
        self.ro_mode = 'SINGLE_TRACK'
        with self.lock:
            _check(self.sdk.SetReadMode(int(consts.Read_Mode.SINGLE_TRACK)), "SetReadMode")
        with self.lock:
            _check(self.sdk.SetSingleTrack(int(center), int(width)), "SetSingleTrack")
        with self.lock:
            _check(self.sdk.SetSingleTrackHBin(int(hbin)), "SetSingleTrackHBin")
        self.ro_st_hbin = hbin
        self.Nx_ro = int(self.Nx / hbin)
        self.Ny_ro = 1
        self.ro_single_track_center = center
        self.ro_single_track_width = width
        self.create_buffer()

    def set_ro_multi_track(self, number, height, offset, hbin=1):
        """
        Multi-Track 읽기 모드.
        여러 평행 트랙을 동시에 읽기.

        Parameters
        ----------
        number : int   트랙 수.
        height : int   각 트랙 높이(픽셀).
        offset : int   첫 번째 트랙 오프셋.
        hbin   : int   수평 빈닝 계수.
        """
        self.ro_mode = 'MULTI_TRACK'
        with self.lock:
            _check(self.sdk.SetReadMode(int(consts.Read_Mode.MULTI_TRACK)), "SetReadMode")
        with self.lock:
            (ret, bottom, gap) = self.sdk.SetMultiTrack(int(number), int(height), int(offset))
        _check(ret, "SetMultiTrack")
        with self.lock:
            _check(self.sdk.SetMultiTrackHBin(int(hbin)), "SetMultiTrackHBin")
        self.Nx_ro = int(self.Nx / hbin)
        self.Ny_ro = number
        self.ro_multi_track_number = number
        self.ro_multi_track_bottom = bottom
        self.ro_multi_track_gap = gap
        self.create_buffer()

    def set_ro_random_track(self, areas, hbin=1):
        """
        Random-Track 읽기 모드.
        임의 위치의 여러 트랙 읽기.

        Parameters
        ----------
        areas : list of (start, end) tuples  각 트랙의 픽셀 범위 (1-based).
        hbin  : int  수평 빈닝 계수.
        """
        self.ro_mode = 'RANDOM_TRACK'
        with self.lock:
            _check(self.sdk.SetReadMode(int(consts.Read_Mode.RANDOM_TRACK)), "SetReadMode")
        flat = [v for pair in areas for v in pair]
        with self.lock:
            _check(self.sdk.SetRandomTracks(int(len(areas)), flat), "SetRandomTracks")
        self.Nx_ro = self.Nx  # SDK는 RandomTrack 수평 빈닝 미지원
        self.Ny_ro = len(areas)
        self.create_buffer()

    def set_ro_image_mode(self, hbin=1, vbin=1, hstart=1, hend=None, vstart=1, vend=None):
        """
        Image 모드 (2D 전체 또는 ROI).

        Parameters
        ----------
        hbin, vbin         : int  수평/수직 빈닝.
        hstart, hend       : int  수평 픽셀 범위 (1-based). None = 전체.
        vstart, vend       : int  수직 픽셀 범위 (1-based). None = 전체.
        """
        self.ro_mode = 'IMG'
        with self.lock:
            _check(self.sdk.SetReadMode(int(consts.Read_Mode.IMAGE)), "SetReadMode")

        if hend is None:
            hend = self.Nx
        if vend is None:
            vend = self.Ny

        assert hend > hstart
        assert vend > vstart

        self.hbin   = hbin
        self.vbin   = vbin
        self.hstart = hstart
        self.hend   = hend
        self.vstart = vstart
        self.vend   = vend

        with self.lock:
            _check(self.sdk.SetImage(int(hbin), int(vbin),
                                     int(hstart), int(hend),
                                     int(vstart), int(vend)), "SetImage")

        self.Nx_ro = int((self.hend - self.hstart + 1) / self.hbin)
        self.Ny_ro = int((self.vend - self.vstart + 1) / self.vbin)
        logger.debug("Nx_ro: {}, Ny_ro: {}".format(self.Nx_ro, self.Ny_ro))
        self.create_buffer()

    def get_current_hbin(self):
        """현재 읽기 모드의 수평 빈닝 계수 반환."""
        if self.ro_mode == 'IMG':
            return self.hbin
        elif self.ro_mode == 'SINGLE_TRACK':
            return self.ro_st_hbin
        elif self.ro_mode == 'FULL_VERTICAL_BINNING':
            return self.ro_fvb_hbin
        return 1

    def get_maximum_binning(self, read_mode, horz_vert):
        """
        최대 빈닝 계수 조회.

        Parameters
        ----------
        read_mode  : int  ReadMode 값 (0~4).
        horz_vert  : int  0=수평, 1=수직.
        """
        with self.lock:
            (ret, max_bin) = self.sdk.GetMaximumBinning(int(read_mode), int(horz_vert))
        _check(ret, "GetMaximumBinning")
        return int(max_bin)


    # ══════════════════════════════════════════════════════════════════
    # 취득 모드 설정 (Acquisition Mode)
    # ══════════════════════════════════════════════════════════════════

    def set_aq_mode(self, mode):
        """
        취득 모드 문자열로 설정.

        Parameters
        ----------
        mode : str  'single' | 'accumulate' | 'kinetic' | 'run_till_abort'
        """
        logger.debug('set_aq_mode: %s', mode)
        assert mode in ('single', 'accumulate', 'kinetic', 'run_till_abort')
        if mode == 'single':
            return self.set_aq_single_scan()
        if mode == 'accumulate':
            return self.set_aq_accumulate_scan()
        if mode == 'kinetic':
            return self.set_aq_kinetic_scan()
        if mode == 'run_till_abort':
            return self.set_aq_run_till_abort_scan()

    def get_aq_mode(self):
        """현재 취득 모드 문자열 반환."""
        return self.aq_mode

    def set_aq_single_scan(self, exposure=None):
        """단일 프레임 취득 모드."""
        self.aq_mode = 'single'
        with self.lock:
            _check(self.sdk.SetAcquisitionMode(int(consts.Acquisition_Mode.SINGLE_SCAN)),
                   "SetAcquisitionMode")
        if exposure is not None:
            with self.lock:
                _check(self.sdk.SetExposureTime(float(exposure)), "SetExposureTime")

    def set_aq_accumulate_scan(self, exposure_time=None, num_acc=None, cycle_time=None):
        """
        Accumulate 취득 모드.
        동일 노출을 num_acc 회 반복하여 더한다.
        """
        self.aq_mode = 'accumulate'
        with self.lock:
            _check(self.sdk.SetAcquisitionMode(int(consts.Acquisition_Mode.ACCUMULATE)),
                   "SetAcquisitionMode")
        if exposure_time is not None:
            with self.lock:
                _check(self.sdk.SetExposureTime(float(exposure_time)), "SetExposureTime")
        if num_acc is not None:
            with self.lock:
                _check(self.sdk.SetNumberAccumulations(int(num_acc)), "SetNumberAccumulations")
        if cycle_time is not None:
            with self.lock:
                _check(self.sdk.SetAccumulationCycleTime(float(cycle_time)),
                       "SetAccumulationCycleTime")

    def set_aq_kinetic_scan(self, exp_time=None, num_acc=None, acc_time=None,
                            num_kin=None, kin_time=None):
        """
        Kinetic Series 취득 모드.
        연속 프레임 빠르게 취득.
        """
        self.aq_mode = 'kinetic'
        with self.lock:
            _check(self.sdk.SetAcquisitionMode(int(consts.Acquisition_Mode.KINETICS)),
                   "SetAcquisitionMode")
        if exp_time is not None:
            with self.lock:
                _check(self.sdk.SetExposureTime(float(exp_time)), "SetExposureTime")
        if num_acc is not None:
            with self.lock:
                _check(self.sdk.SetNumberAccumulations(int(num_acc)), "SetNumberAccumulations")
        if acc_time is not None:
            with self.lock:
                _check(self.sdk.SetAccumulationCycleTime(float(acc_time)),
                       "SetAccumulationCycleTime")
        if num_kin is not None:
            with self.lock:
                _check(self.sdk.SetNumberKinetics(int(num_kin)), "SetNumberKinetics")
        if kin_time is not None:
            with self.lock:
                _check(self.sdk.SetKineticCycleTime(float(kin_time)), "SetKineticCycleTime")
        logger.debug('set_aq_kinetic_scan')

    def set_aq_run_till_abort_scan(self):
        """Run Till Abort 모드: abort_acquisition() 호출 전까지 연속 취득."""
        self.aq_mode = 'run_till_abort'
        logger.debug('set_aq_run_till_abort_scan')
        with self.lock:
            _check(self.sdk.SetAcquisitionMode(int(consts.Acquisition_Mode.RUN_TILL_ABORT)),
                   "SetAcquisitionMode")

    def set_aq_fast_kinetic_scan(self, exp_time=None, series=None, mode=None,
                                  hbin=1, vbin=1, offset=1):
        """Fast Kinetics 취득 모드."""
        self.aq_mode = 'fast_kinetic'
        with self.lock:
            _check(self.sdk.SetAcquisitionMode(int(consts.Acquisition_Mode.FAST_KINETICS)),
                   "SetAcquisitionMode")
        if exp_time is not None and series is not None and mode is not None:
            self.num_fast_kin = int(series)
            with self.lock:
                _check(self.sdk.SetFastKinetics(int(self.Ny_ro), int(series),
                                                float(exp_time), int(mode),
                                                int(hbin), int(vbin)), "SetFastKinetics")


    # ══════════════════════════════════════════════════════════════════
    # 트리거 모드 설정
    # ══════════════════════════════════════════════════════════════════

    trigger_modes = dict(
        internal          = int(consts.Trigger_Mode.INTERNAL),
        external          = int(consts.Trigger_Mode.EXTERNAL),
        external_start    = int(consts.Trigger_Mode.EXTERNAL_START),
        external_exposure = int(consts.Trigger_Mode.EXTERNAL_EXPOSURE_BULB),
        external_fvb_em   = int(consts.Trigger_Mode.EXTERNAL_FVB_EM),
        software          = int(consts.Trigger_Mode.SOFTWARE_TRIGGER),
    )

    def set_trigger_mode(self, mode='internal'):
        """
        트리거 모드 설정.

        Parameters
        ----------
        mode : str  'internal' | 'external' | 'external_start' |
                    'external_exposure' | 'external_fvb_em' | 'software'
        """
        mode = mode.lower()
        with self.lock:
            _check(self.sdk.SetTriggerMode(self.trigger_modes[mode]), "SetTriggerMode")

    def send_software_trigger(self):
        """소프트웨어 트리거 발송 (software 트리거 모드 시 사용)."""
        with self.lock:
            _check(self.sdk.SendSoftwareTrigger(), "SendSoftwareTrigger")


    # ══════════════════════════════════════════════════════════════════
    # 시프트 속도 설정 (Shift Speed & Gain)
    # ══════════════════════════════════════════════════════════════════

    def read_shift_speeds(self):
        """
        사용 가능한 수평/수직 시프트 속도 목록 조회.

        결과 속성:
          self.numHSSpeeds_EM[chan]           — EM 채널별 수평 속도 수
          self.numHSSpeeds_Conventional[chan] — 일반 채널별 수평 속도 수
          self.HSSpeeds_EM[chan][i]           — EM 속도 [MHz]
          self.HSSpeeds_Conventional[chan][i] — 일반 속도 [MHz]
          self.numVSSpeeds                    — 수직 속도 수
          self.VSSpeeds[i]                    — 수직 속도 [μs/pixel]
        """
        self.numHSSpeeds_EM = []
        self.numHSSpeeds_Conventional = []

        for chan_i in range(self.numADChan):
            if self.em_mode:
                with self.lock:
                    (ret, n_em) = self.sdk.GetNumberHSSpeeds(chan_i, 0)   # 0 = EM
                _check(ret, "GetNumberHSSpeeds")
                self.numHSSpeeds_EM.append(n_em)

                with self.lock:
                    (ret, n_conv) = self.sdk.GetNumberHSSpeeds(chan_i, 1) # 1 = Conventional
                _check(ret, "GetNumberHSSpeeds")
                self.numHSSpeeds_Conventional.append(n_conv)
            else:
                with self.lock:
                    (ret, n_conv) = self.sdk.GetNumberHSSpeeds(chan_i, 0)
                _check(ret, "GetNumberHSSpeeds")
                self.numHSSpeeds_Conventional.append(n_conv)

        logger.debug('HS Speeds EM counts: {}'.format(self.numHSSpeeds_EM))
        logger.debug('HS Speeds Conventional counts: {}'.format(self.numHSSpeeds_Conventional))

        self.HSSpeeds_EM = []
        self.HSSpeeds_Conventional = []
        for chan_i in range(self.numADChan):
            self.HSSpeeds_EM.append([])
            if self.em_mode:
                for i in range(self.numHSSpeeds_EM[chan_i]):
                    with self.lock:
                        (ret, spd) = self.sdk.GetHSSpeed(chan_i, 0, i)
                    _check(ret, "GetHSSpeed")
                    self.HSSpeeds_EM[chan_i].append(spd)
                conventional_typ = 1
            else:
                conventional_typ = 0

            self.HSSpeeds_Conventional.append([])
            for i in range(self.numHSSpeeds_Conventional[chan_i]):
                with self.lock:
                    (ret, spd) = self.sdk.GetHSSpeed(chan_i, conventional_typ, i)
                _check(ret, "GetHSSpeed")
                self.HSSpeeds_Conventional[chan_i].append(spd)

        logger.debug('EM HS speeds: {} MHz'.format(self.HSSpeeds_EM))
        logger.debug('Conventional HS speeds: {} MHz'.format(self.HSSpeeds_Conventional))

        # 수직 시프트 속도
        with self.lock:
            (ret, nvs) = self.sdk.GetNumberVSSpeeds()
        if int(ret) == consts.DRV_NOT_SUPPORTED:
            self.numVSSpeeds = 0
        else:
            _check(ret, "GetNumberVSSpeeds")
            self.numVSSpeeds = nvs

        self.VSSpeeds = []
        for i in range(self.numVSSpeeds):
            with self.lock:
                (ret, spd) = self.sdk.GetVSSpeed(i)
            _check(ret, "GetVSSpeed")
            self.VSSpeeds.append(spd)

        if self.debug:
            logger.debug('VS speeds [us/pixel]: %s' % self.VSSpeeds)

    def set_hs_speed_em(self, speed_index=0):
        """EM 모드 수평 시프트 속도 설정."""
        logger.debug("set_hs_speed_em {}".format(speed_index))
        assert 0 <= speed_index < self.numHSSpeeds_EM[self.ad_chan]
        with self.lock:
            _check(self.sdk.SetHSSpeed(0, int(speed_index)), "SetHSSpeed")

    def set_hs_speed_conventional(self, speed_index=0):
        """일반(Conventional) 모드 수평 시프트 속도 설정."""
        print("set_hs_speed_conventional", speed_index, self.ad_chan)
        assert 0 <= speed_index < self.numHSSpeeds_Conventional[self.ad_chan]
        typ = 1 if self.em_mode else 0
        with self.lock:
            _check(self.sdk.SetHSSpeed(typ, int(speed_index)), "SetHSSpeed")

    def set_vs_speed(self, speed_index=0):
        """수직 시프트 속도 설정."""
        assert 0 <= speed_index < self.numVSSpeeds
        with self.lock:
            _check(self.sdk.SetVSSpeed(int(speed_index)), "SetVSSpeed")

    def get_fastest_recommended_vs_speed(self):
        """
        SDK 권장 최대 수직 시프트 속도 조회.

        Returns
        -------
        (index, speed_us) : (int, float)
        """
        with self.lock:
            (ret, index, speed) = self.sdk.GetFastestRecommendedVSSpeed()
        _check(ret, "GetFastestRecommendedVSSpeed")
        return int(index), float(speed)

    def set_preamp_gain(self, gain_i=0):
        """프리앰프 이득 설정."""
        with self.lock:
            _check(self.sdk.SetPreAmpGain(int(gain_i)), "SetPreAmpGain")
        self.preamp_gain_i = gain_i

    def is_preamp_gain_available(self, channel, amplifier, index, pa):
        """지정 채널/앰프/속도/이득 조합이 유효한지 확인."""
        with self.lock:
            (ret, status) = self.sdk.IsPreAmpGainAvailable(int(channel), int(amplifier),
                                                            int(index), int(pa))
        _check(ret, "IsPreAmpGainAvailable")
        return bool(status)


    # ══════════════════════════════════════════════════════════════════
    # 이미지 회전 / 반전
    # ══════════════════════════════════════════════════════════════════

    def get_image_flip(self):
        """현재 이미지 반전 상태 (hflip, vflip) 반환."""
        with self.lock:
            (ret, hflip, vflip) = self.sdk.GetImageFlip()
        _check(ret, "GetImageFlip")
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)
        return self.hflip, self.vflip

    def set_image_flip(self, hflip=True, vflip=False):
        """이미지 반전 설정."""
        with self.lock:
            _check(self.sdk.SetImageFlip(int(bool(hflip)), int(bool(vflip))), "SetImageFlip")

    def get_image_hflip(self):
        return self.get_image_flip()[0]

    def set_image_hflip(self, hflipNew):
        hflipOld, vflipOld = self.get_image_flip()
        self.set_image_flip(hflipNew, vflipOld)
        logger.debug("set_image_hflip: {}".format(hflipNew))

    def get_image_vflip(self):
        return self.get_image_flip()[1]

    def set_image_vflip(self, vflipNew):
        hflipOld, vflipOld = self.get_image_flip()
        self.set_image_flip(hflipOld, vflipNew)
        logger.debug("set_image_vflip: {}".format(vflipNew))

    def set_image_rotate(self, rotate=0):
        """이미지 회전 (0=없음, 1=90° CW, 2=90° CCW)."""
        assert rotate in [0, 1, 2]
        with self.lock:
            _check(self.sdk.SetImageRotate(int(rotate)), "SetImageRotate")


    # ══════════════════════════════════════════════════════════════════
    # 셔터 제어
    # ══════════════════════════════════════════════════════════════════

    def set_shutter_auto(self):
        """셔터 자동 모드 (취득 시 자동 개폐)."""
        with self.lock:
            _check(self.sdk.SetShutter(0, int(consts.Shutter_Mode.FULLY_AUTO), 0, 0), "SetShutter")
        self.shutter_mode = 'auto'

    def set_shutter_open(self, open=True):
        """셔터 강제 열기(True) / 닫기(False)."""
        if open:
            with self.lock:
                _check(self.sdk.SetShutter(0, int(consts.Shutter_Mode.PERMANENTLY_OPEN), 0, 0),
                       "SetShutter")
            self.shutter_mode = 'open'
        else:
            self.set_shutter_close()

    def set_shutter_close(self):
        """셔터 강제 닫기."""
        with self.lock:
            _check(self.sdk.SetShutter(0, int(consts.Shutter_Mode.PERMANENTLY_CLOSED), 0, 0),
                   "SetShutter")
        self.shutter_mode = 'close'

    def set_cosmic_ray_filter(self, enabled: bool = True):
        """우주선(Cosmic Ray) 필터 모드 설정. accumulate 모드에서만 유효."""
        mode = 2 if enabled else 0
        with self.lock:
            _check(self.sdk.SetFilterMode(mode), "SetFilterMode")

    def get_shutter_min_times(self):
        """셔터 최소 개폐 시간 반환 (closing_ms, opening_ms)."""
        with self.lock:
            (ret, closing, opening) = self.sdk.GetShutterMinTimes()
        _check(ret, "GetShutterMinTimes")
        return int(closing), int(opening)

    def is_internal_mechanical_shutter(self):
        """내부 기계식 셔터 장착 여부 반환 (iXon 전용)."""
        with self.lock:
            (ret, is_int) = self.sdk.IsInternalMechanicalShutter()
        _check(ret, "IsInternalMechanicalShutter")
        return bool(is_int)


    # ══════════════════════════════════════════════════════════════════
    # 온도 / 냉각기 제어
    # ══════════════════════════════════════════════════════════════════

    def set_cooler_on(self):
        """냉각기 켜기."""
        with self.lock:
            _check(self.sdk.CoolerON(), "CoolerON")
        self.cooler_on = True

    def set_cooler_off(self):
        """냉각기 끄기. 종료 전 반드시 호출."""
        with self.lock:
            _check(self.sdk.CoolerOFF(), "CoolerOFF")
        self.cooler_on = False

    def set_cooler(self, coolerOn):
        """냉각기 켜기/끄기."""
        if coolerOn:
            self.set_cooler_on()
        else:
            self.set_cooler_off()

    def get_cooler(self):
        """현재 냉각기 상태 (True = ON)."""
        return self.cooler_on

    def is_cooler_on(self):
        """SDK에서 직접 냉각기 상태 조회."""
        with self.lock:
            (ret, status) = self.sdk.IsCoolerOn()
        _check(ret, "IsCoolerOn")
        return bool(status)

    def get_temperature_range(self):
        """허용 온도 범위 반환 (min_temp, max_temp) [°C]."""
        with self.lock:
            (ret, mintemp, maxtemp) = self.sdk.GetTemperatureRange()
        _check(ret, "GetTemperatureRange")
        self.min_temp = mintemp
        self.max_temp = maxtemp
        return self.min_temp, self.max_temp

    def set_temperature(self, new_temp):
        """목표 냉각 온도 설정 [°C]."""
        with self.lock:
            _check(self.sdk.SetTemperature(int(new_temp)), "SetTemperature")
        self.get_temperature()

    def get_temperature(self):
        """
        현재 CCD 온도 반환 [°C].

        Returns
        -------
        int  현재 온도. 온도 상태 코드는 self.temperature_status_num.
        """
        with self.lock:
            (ret, temp) = self.sdk.GetTemperature()
        status = int(ret)
        if status == consts.DRV_ACQUIRING:
            raise IOError("Camera busy acquiring")
        elif status in (consts.DRV_NOT_INITIALIZED, consts.DRV_ERROR_ACK):
            _check(ret, "GetTemperature")
        else:
            self.temperature = int(temp)
            self.temperature_status_num = status
            return self.temperature

    def get_temperature_f(self):
        """현재 CCD 온도 반환 [°C] (float 정밀도)."""
        with self.lock:
            (ret, temp) = self.sdk.GetTemperatureF()
        self.temperature_status_num = int(ret)
        return float(temp)

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
        """온도 상태 문자열 반환 ('STABILIZED' 되면 촬영 가능)."""
        self.get_temperature()
        return self.temp_status_dict.get(self.temperature_status_num, 'UNKNOWN')


    # ══════════════════════════════════════════════════════════════════
    # 취득 제어 및 데이터 읽기
    # ══════════════════════════════════════════════════════════════════

    def prepare_acquisition(self):
        """
        취득 전 메모리 사전 할당 (StartAcquisition 지연 최소화).
        Kinetic 시리즈 등 긴 취득에서 특히 효과적.
        """
        with self.lock:
            _check(self.sdk.PrepareAcquisition(), "PrepareAcquisition")

    def start_acquisition(self):
        """취득 시작 (비블로킹). wait_for_acquisition() 또는 get_status() 로 완료 확인."""
        with self.lock:
            _check(self.sdk.StartAcquisition(), "StartAcquisition")

    def abort_acquisition(self):
        """진행 중인 취득 강제 중단."""
        with self.lock:
            _check(self.sdk.AbortAcquisition(), "AbortAcquisition")

    def wait_for_acquisition(self, timeout_ms=None):
        """
        취득 완료까지 블로킹 대기 (이벤트 기반, 폴링보다 효율적).
        락 없이 호출 → 다른 스레드에서 abort_acquisition() 가능.

        Parameters
        ----------
        timeout_ms : int, optional  타임아웃 [밀리초]. None = 무한 대기.

        Returns
        -------
        bool  True = 취득 완료, False = 타임아웃 또는 취소됨.
        """
        if timeout_ms is None:
            ret = self.sdk.WaitForAcquisition()
        else:
            ret = self.sdk.WaitForAcquisitionTimeOut(int(timeout_ms))
        if int(ret) not in (consts.DRV_SUCCESS, consts.DRV_NO_NEW_DATA):
            _check(ret, "WaitForAcquisition")
        return int(ret) == consts.DRV_SUCCESS

    def cancel_wait(self):
        """다른 스레드에서 wait_for_acquisition() 을 강제 반환시킴."""
        with self.lock:
            self.sdk.CancelWait()

    def set_driver_event(self, event_handle):
        """
        Win32 이벤트 핸들 등록. 취득 완료 시 이벤트 시그널.
        WindowsEvents.py 예제 패턴.

        Parameters
        ----------
        event_handle : HANDLE  win32event.CreateEvent() 로 얻은 핸들.
                                None 전달 시 이벤트 연결 해제.
        """
        with self.lock:
            _check(self.sdk.SetDriverEvent(event_handle), "SetDriverEvent")

    # GetStatus 반환 코드 → 문자열 매핑
    _status_name_dict = {
        consts.DRV_IDLE:                'IDLE',
        consts.DRV_TEMPCYCLE:           'TEMPCYCLE',
        consts.DRV_ACQUIRING:           'ACQUIRING',
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
        str  'IDLE' | 'ACQUIRING' | 'TEMPCYCLE' | ...
        """
        with self.lock:
            (ret, status) = self.sdk.GetStatus()
        _check(ret, "GetStatus")
        self.status_id = int(status)
        self.status_name = self._status_name_dict.get(self.status_id, str(self.status_id))
        return self.status_name

    def get_acquisition_progress(self):
        """
        현재 취득 진행 상황 반환 (Kinetic/Accumulate 모드).

        Returns
        -------
        (acc, series) : (int, int)  현재 누적/시리즈 카운터
        """
        with self.lock:
            (ret, acc, series) = self.sdk.GetAcquisitionProgress()
        _check(ret, "GetAcquisitionProgress")
        return int(acc), int(series)

    def get_acquired_data(self):
        """
        취득 완료된 데이터 반환 (int32).

        GetMostRecentImage (단일/누적) 또는 GetImages (kinetic) 사용.
        누적 모드에서 32-bit 범위 필요하므로 int32 유지.

        Returns
        -------
        np.ndarray (int32)
            FVB/Single: (Ny_ro, Nx_ro) | Kinetic: (num_kin, Ny_ro, Nx_ro)
        """
        if self.aq_mode == 'kinetic':
            with self.lock:
                (ret_idx, first, last) = self.sdk.GetNumberNewImages()
            _check(ret_idx, "GetNumberNewImages")
            count = last - first + 1
            total = count * self.Ny_ro * self.Nx_ro
            with self.lock:
                (ret, arr, *_) = self.sdk.GetImages(int(first), int(last), int(total))
            _check(ret, "GetImages")
            self.buffer = np.array(arr, dtype=np.int32).reshape(
                count, self.Ny_ro, self.Nx_ro)
        elif self.aq_mode == 'fast_kinetic':
            total = self.num_fast_kin * self.Ny_ro * self.Nx_ro
            with self.lock:
                (ret, arr) = self.sdk.GetAcquiredData(int(total))
            _check(ret, "GetAcquiredData")
            self.buffer = np.array(arr, dtype=np.int32).reshape(
                self.num_fast_kin, self.Ny_ro, self.Nx_ro)
        else:
            size = self.Ny_ro * self.Nx_ro
            with self.lock:
                (ret, arr) = self.sdk.GetMostRecentImage(int(size))
            _check(ret, "GetMostRecentImage")
            self.buffer = np.array(arr, dtype=np.int32).reshape(self.Ny_ro, self.Nx_ro)
        return self.buffer

    def get_acquired_data16(self):
        """
        취득 완료된 데이터 반환 (uint16, 16-bit 단일 프레임용).

        Returns
        -------
        np.ndarray (uint16)
        """
        if self.aq_mode == 'kinetic':
            total = self.num_kin * self.Ny_ro * self.Nx_ro
            with self.lock:
                (ret, arr, *_) = self.sdk.GetImages16(1, int(self.num_kin), int(total))
            _check(ret, "GetImages16")
            return np.array(arr, dtype=np.uint16).reshape(
                self.num_kin, self.Ny_ro, self.Nx_ro)
        else:
            size = self.Ny_ro * self.Nx_ro
            with self.lock:
                (ret, arr) = self.sdk.GetMostRecentImage16(int(size))
            _check(ret, "GetMostRecentImage16")
            return np.array(arr, dtype=np.uint16).reshape(self.Ny_ro, self.Nx_ro)

    def get_most_recent_image16(self):
        """가장 최근 프레임 16-bit 반환 (Run-till-abort 실시간 표시용)."""
        size = self.Ny_ro * self.Nx_ro
        with self.lock:
            (ret, arr) = self.sdk.GetMostRecentImage16(int(size))
        _check(ret, "GetMostRecentImage16")
        return np.array(arr, dtype=np.uint16).reshape(self.Ny_ro, self.Nx_ro)

    def get_images16(self, first, last):
        """
        원형 버퍼에서 first~last 범위 16-bit 이미지 반환.

        Returns
        -------
        np.ndarray (uint16) shape: (last-first+1, Ny_ro, Nx_ro)
        """
        count = last - first + 1
        size = count * self.Ny_ro * self.Nx_ro
        with self.lock:
            (ret, arr, *_) = self.sdk.GetImages16(int(first), int(last), int(size))
        _check(ret, "GetImages16")
        return np.array(arr, dtype=np.uint16).reshape(count, self.Ny_ro, self.Nx_ro)

    def free_internal_memory(self):
        """내부 취득 버퍼 해제 (취득 사이 메모리 회수)."""
        with self.lock:
            self.sdk.FreeInternalMemory()


    # ══════════════════════════════════════════════════════════════════
    # 취득 타이밍 설정
    # ══════════════════════════════════════════════════════════════════

    def get_acquisition_timings(self):
        """
        실제 적용된 타이밍 조회.

        Returns
        -------
        (exposure_time, accumulation_time, kinetic_cycle_time) : (float, float, float) [초]
        """
        with self.lock:
            (ret, exposure, accumulate, kinetic) = self.sdk.GetAcquisitionTimings()
        _check(ret, "GetAcquisitionTimings")
        self.exposure_time      = float(exposure)
        self.accumulation_time  = float(accumulate)
        self.kinetic_cycle_time = float(kinetic)
        return self.exposure_time, self.accumulation_time, self.kinetic_cycle_time

    def set_exposure_time(self, dt):
        """노출 시간 설정 [초]."""
        with self.lock:
            _check(self.sdk.SetExposureTime(float(dt)), "SetExposureTime")
        self.get_acquisition_timings()
        if self.debug:
            logger.debug('set exposure to: {}'.format(self.exposure_time))
        return self.exposure_time

    def get_exposure_time(self):
        """현재 노출 시간 [초] 반환."""
        return self.get_acquisition_timings()[0]

    def set_num_accumulations(self, num):
        """누적 횟수 설정."""
        with self.lock:
            _check(self.sdk.SetNumberAccumulations(int(num)), "SetNumberAccumulations")
        self.num_acc = int(num)

    def get_num_accumulations(self):
        return self.num_acc

    def set_num_kinetics(self, num):
        """Kinetic Series 프레임 수 설정."""
        logger.debug('set_num_kinetics: %s', num)
        with self.lock:
            _check(self.sdk.SetNumberKinetics(int(num)), "SetNumberKinetics")
        self.num_kin = int(num)

    def get_num_kinetics(self):
        return self.num_kin

    def set_accumulation_cycle_time(self, acc_time):
        """누적 사이클 시간 설정 [초]."""
        with self.lock:
            _check(self.sdk.SetAccumulationCycleTime(float(acc_time)),
                   "SetAccumulationCycleTime")

    def set_kinetic_cycle_time(self, kin_time):
        """Kinetic 프레임 간격 설정 [초]."""
        with self.lock:
            _check(self.sdk.SetKineticCycleTime(float(kin_time)), "SetKineticCycleTime")


    # ══════════════════════════════════════════════════════════════════
    # EM (Electron Multiplication) 이득 제어
    # ══════════════════════════════════════════════════════════════════

    def set_EM_advanced(self, state=True):
        """고급 EM 이득 모드 활성화 (x300 초과 이득 접근 가능)."""
        with self.lock:
            _check(self.sdk.SetEMAdvanced(int(bool(state))), "SetEMAdvanced")

    def set_EM_gain_mode(self, mode):
        """
        EM 이득 모드 선택.
        0: DAC 0-255 (기본), 1: DAC 0-4095, 2: Linear, 3: Real EM gain
        """
        with self.lock:
            _check(self.sdk.SetEMGainMode(int(mode)), "SetEMGainMode")

    def get_EM_gain_range(self):
        """EM 이득 허용 범위 반환 (low, high)."""
        with self.lock:
            (ret, low, high) = self.sdk.GetEMGainRange()
        _check(ret, "GetEMGainRange")
        self.em_gain_range = (int(low), int(high))
        return self.em_gain_range

    def get_EMCCD_gain(self):
        """현재 EM 이득값 반환."""
        with self.lock:
            (ret, gain) = self.sdk.GetEMCCDGain()
        _check(ret, "GetEMCCDGain")
        self.em_gain = int(gain)
        return self.em_gain

    def set_EMCCD_gain(self, gain):
        """EM 이득 설정 (em_gain_range 범위 내)."""
        low, high = self.em_gain_range
        assert low <= gain <= high
        with self.lock:
            _check(self.sdk.SetEMCCDGain(int(gain)), "SetEMCCDGain")


    # ══════════════════════════════════════════════════════════════════
    # 출력 앰프 설정
    # ══════════════════════════════════════════════════════════════════

    def set_output_amp(self, amp):
        """출력 앰프 선택 (0=EMCCD, 1=Conventional)."""
        with self.lock:
            _check(self.sdk.SetOutputAmplifier(int(amp)), "SetOutputAmplifier")
        self.output_amp = amp

    def get_output_amp(self):
        return self.output_amp


    # ══════════════════════════════════════════════════════════════════
    # 파일 저장 (SaveAsSIF.py, Spooling.py 패턴)
    # ══════════════════════════════════════════════════════════════════

    def save_as_sif(self, path):
        """마지막 취득 데이터를 Andor SIF 형식으로 저장."""
        with self.lock:
            _check(self.sdk.SaveAsSif(str(path)), "SaveAsSif")

    def save_as_calibrated_sif(self, path, data_type, unit, coeff, rayleigh_wave):
        """
        교정 정보 포함 SIF 저장.

        Parameters
        ----------
        path         : str    파일 경로.
        data_type    : int    X축 레이블 (0=pixel, 1=wavelength, 2=wave number, ...).
        unit         : int    X축 단위.
        coeff        : list[float]  3차 다항식 계수 4개.
        rayleigh_wave: float  레일리 파장 [nm].
        """
        with self.lock:
            _check(self.sdk.SaveAsCalibratedSif(str(path), int(data_type), int(unit),
                                                coeff, float(rayleigh_wave)),
                   "SaveAsCalibratedSif")

    def set_spool(self, active, method, path, frame_buffer_size):
        """
        취득 중 디스크 자동 저장(Spooling) 설정.

        Parameters
        ----------
        active            : int  1=활성, 0=비활성.
        method            : int  Spool_Mode 열거형 값.
        path              : str  저장 경로.
        frame_buffer_size : int  내부 프레임 버퍼 크기.
        """
        with self.lock:
            _check(self.sdk.SetSpool(int(active), int(method),
                                     str(path), int(frame_buffer_size)), "SetSpool")

    def get_spool_progress(self):
        """현재 스풀 진행 상황 (저장된 이미지 수) 반환."""
        with self.lock:
            (ret, index) = self.sdk.GetSpoolProgress()
        _check(ret, "GetSpoolProgress")
        return int(index)


    # ══════════════════════════════════════════════════════════════════
    # DDG / iStar 게이팅 제어 (USBiStar.py 패턴)
    # ══════════════════════════════════════════════════════════════════

    def set_gate_mode(self, mode):
        """
        게이팅 모드 설정 (Gate_Mode 열거형 값).
        Gate_Mode.GATE_USING_DDG = 5 (DDG 제어 게이팅).
        """
        with self.lock:
            _check(self.sdk.SetGateMode(int(mode)), "SetGateMode")

    def get_gate_mode(self):
        """현재 게이팅 모드 반환."""
        with self.lock:
            (ret, mode) = self.sdk.GetGateMode()
        _check(ret, "GetGateMode")
        return int(mode)

    def set_ddg_gate_time(self, delay_ps, width_ps):
        """
        DDG 게이트 타이밍 설정.

        Parameters
        ----------
        delay_ps : int  게이트 지연 [피코초].
        width_ps : int  게이트 폭 [피코초].
        """
        with self.lock:
            _check(self.sdk.SetDDGGateTime(int(delay_ps), int(width_ps)), "SetDDGGateTime")

    def get_ddg_gate_time(self):
        """현재 DDG 게이트 타이밍 반환 (delay_ps, width_ps)."""
        with self.lock:
            (ret, delay, width) = self.sdk.GetDDGGateTime()
        _check(ret, "GetDDGGateTime")
        return int(delay), int(width)

    def set_ddg_external_output_time(self, index, delay_ps, width_ps):
        """DDG 외부 출력 펄스 타이밍 설정."""
        with self.lock:
            _check(self.sdk.SetDDGExternalOutputTime(int(index),
                                                     int(delay_ps), int(width_ps)),
                   "SetDDGExternalOutputTime")

    def set_mcp_gain(self, gain):
        """MCP(microchannel plate) 이득 설정 (iStar 전용)."""
        with self.lock:
            _check(self.sdk.SetMCPGain(int(gain)), "SetMCPGain")

    def get_mcp_gain(self):
        """현재 MCP 이득 반환."""
        with self.lock:
            (ret, gain) = self.sdk.GetMCPGain()
        _check(ret, "GetMCPGain")
        return int(gain)

    def get_mcp_gain_range(self):
        """MCP 이득 허용 범위 반환 (min, max)."""
        with self.lock:
            (ret, low, high) = self.sdk.GetMCPGainRange()
        _check(ret, "GetMCPGainRange")
        return int(low), int(high)


    # ══════════════════════════════════════════════════════════════════
    # raman_tools 호환 고수준 API
    # ══════════════════════════════════════════════════════════════════

    def start_acquisition_cycle(self, trigger_mode_str: str = 'internal',
                                timeout_ms: int | None = None) -> dict:
        """
        단일 촬영 사이클 실행 후 raman_tools 호환 dict 반환.

        흐름: PrepareAcquisition → StartAcquisition → [SendSoftwareTrigger]
              → WaitForAcquisition (이벤트 기반) → GetMostRecentImage → 캘리브레이션 적용

        Parameters
        ----------
        trigger_mode_str : str
            현재 트리거 모드 문자열. 'software'이면 SendSoftwareTrigger() 자동 호출.
        timeout_ms : int | None
            WaitForAcquisition 타임아웃 [밀리초]. None = 무한 대기 (internal 트리거에 적합).
            external/software 트리거 사용 시 반드시 설정하여 트리거 미도달 시 무한 블로킹 방지.

        Returns
        -------
        dict
            intensity        : list[int]   픽셀별 강도 (1D).
            calibrated       : bool
            error            : str         취득 실패 시에만 포함.
            raman_shift_cm-1 : list[float] (_calibrator 주입 시)
            wavelength_nm    : list[float] (_calibrator 주입 시)
            laser_nm         : float       (_calibrator 주입 시)
        """
        self.prepare_acquisition()
        self.start_acquisition()

        if trigger_mode_str == 'software':
            self.send_software_trigger()

        success = self.wait_for_acquisition(timeout_ms=timeout_ms)
        if not success:
            self.sdk.AbortAcquisition()
            return {
                "intensity": [],
                "calibrated": False,
                "error": "WaitForAcquisition 실패 (타임아웃 또는 트리거 없음)",
            }

        raw = self.get_acquired_data()
        intensity = raw.flatten().tolist()

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
        냉각 중인 경우 set_cooler(False) 후 온도 -5°C 이상 된 뒤 호출.
        """
        with self.lock:
            _check(self.sdk.ShutDown(), "ShutDown")


    # ══════════════════════════════════════════════════════════════════
    # 원형 버퍼 / 순환 이미지 관리 (Kinetic / Run-till-abort)
    # ══════════════════════════════════════════════════════════════════

    def get_total_number_images_acquired(self):
        """취득된 총 이미지 수 반환."""
        with self.lock:
            (ret, index) = self.sdk.GetTotalNumberImagesAcquired()
        _check(ret, "GetTotalNumberImagesAcquired")
        return int(index)

    def get_number_new_images(self):
        """아직 읽지 않은 새 이미지 (first, last) 인덱스 반환."""
        with self.lock:
            (ret, first, last) = self.sdk.GetNumberNewImages()
        _check(ret, "GetNumberNewImages")
        return int(first), int(last)

    def get_number_available_images(self):
        """현재 버퍼에서 읽을 수 있는 (first, last) 인덱스 반환."""
        with self.lock:
            (ret, first, last) = self.sdk.GetNumberAvailableImages()
        _check(ret, "GetNumberAvailableImages")
        return int(first), int(last)

    def get_images(self, first, last, buf):
        """
        원형 버퍼에서 first~last 범위 이미지를 buf 에 복사 (32-bit).

        Returns
        -------
        (validfirst, validlast, buf)
        """
        size = buf.size
        with self.lock:
            (ret, arr, validfirst, validlast) = self.sdk.GetImages(
                int(first), int(last), int(size))
        _check(ret, "GetImages")
        np.copyto(buf, np.array(arr, dtype=np.int32).reshape(buf.shape))
        return int(validfirst), int(validlast), buf

    def get_oldest_image(self, arr=None):
        """
        원형 버퍼에서 가장 오래된 미읽은 이미지 반환.

        Returns
        -------
        np.ndarray or None  새 데이터 없으면 None.
        """
        size = self.Ny_ro * self.Nx_ro
        with self.lock:
            (ret, raw) = self.sdk.GetOldestImage(int(size))
        if int(ret) == consts.DRV_NO_NEW_DATA:
            return None
        _check(ret, "GetOldestImage")
        result = np.array(raw, dtype=np.int32).reshape(self.Ny_ro, self.Nx_ro)
        if arr is not None:
            np.copyto(arr, result)
            return arr
        return result


# ══════════════════════════════════════════════════════════════════════
# 단독 실행 테스트
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    cam = AndorCCD(debug=True)

    cam.set_ro_full_vertical_binning()
    cam.set_aq_single_scan(exposure=0.1)
    cam.set_trigger_mode('internal')

    print("Head model:", cam.headModel)
    print("Serial:", cam.serialNumber)
    print("Detector:", cam.Nx, "x", cam.Ny)
    print("Fastest VS speed:", cam.get_fastest_recommended_vs_speed())
    print("Capabilities:", cam.get_capabilities())

    cam.prepare_acquisition()
    cam.start_acquisition()
    done = cam.wait_for_acquisition(timeout_ms=5000)
    if done:
        data = cam.get_acquired_data()
        print("Data shape:", data.shape, "Max:", data.max())
        cam.save_as_sif("test_output.sif")
        print("Saved to test_output.sif")

    cam.set_shutter_close()
    cam.close()
