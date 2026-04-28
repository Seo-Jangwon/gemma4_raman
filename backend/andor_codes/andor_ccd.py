"""
andor_ccd.py — ScopeFoundry HardwareComponent 래퍼
===================================================

AndorCCD(andor_ccd_interface.py) 를 ScopeFoundry 의
HardwareComponent 로 감싼 클래스.

ScopeFoundry GUI 앱에서만 사용. 독립 실행 불가.

[역할]
  - GUI 위젯 ↔ 하드웨어 설정값 양방향 동기화 (LoggedQuantity)
  - 온도 / 취득 모드 / ROI / 시프트 속도 등을 GUI에서 제어 가능
  - 배경 이미지 저장 및 차감 지원

[사용 예 (ScopeFoundry 앱 내부)]
  app.add_hardware(AndorCCDHW(app))
  andor_hw = app.hardware['andor_ccd']
  andor_hw.settings['exposure_time'] = 0.5
  andor_hw.ccd_dev.start_acquisition()
"""

from __future__ import absolute_import, print_function
from ScopeFoundry import HardwareComponent
from collections import OrderedDict

try:
    # 패키지 내에서 사용 시 상대 import, 단독 실행 시 절대 import 시도
    from andor_ccd_interface import AndorCCD, AndorReadMode, DEFAULT_TEMPERATURE
except Exception as err:
    print("Could not load modules needed for AndorCCD:", err)


class AndorCCDHW(HardwareComponent):
    """
    ScopeFoundry HardwareComponent for Andor CCD.

    connect() 호출 시 AndorCCD 인스턴스 생성 및 하드웨어 연결.
    disconnect() 호출 시 SDK 종료.

    Attributes (connect 후 접근 가능)
    ----------
    ccd_dev : AndorCCD
        저수준 DLL 래퍼 객체.
    background : np.ndarray or None
        배경 이미지 (차감 기능용).
    """

    def setup(self):
        """ScopeFoundry 초기화: LoggedQuantity(설정 항목) 등록."""
        self.name = "andor_ccd"
        self.debug = True
        self.background = None  # 배경 차감용 이미지 (None = 미설정)

        # ── 상태 표시 ───────────────────────────────────────────────
        self.status = self.add_logged_quantity(
            name='ccd_status', dtype=str, initial="?", fmt="%s", ro=True)
        # CCD 현재 동작 상태 ('IDLE' / 'ACQUIRING' / ...)

        # ── 온도 제어 ────────────────────────────────────────────────
        self.temperature = self.add_logged_quantity(
            name="temperature", dtype=int,
            ro=True, unit="C", vmin=-300, vmax=300, si=False)
        # 현재 CCD 온도 [°C] (읽기 전용)

        self.settings.New('temp_setpoint', dtype=int, unit="C",
                          vmin=-300, vmax=300, initial=-10, si=False)
        # 목표 냉각 온도 설정값 [°C]

        self.settings.New('temp_status', dtype=str, ro=True, initial="?")
        # 온도 상태 문자열 ('STABILIZED' 등)

        self.cooler_on = self.add_logged_quantity(
            name="cooler_on", dtype=bool, ro=False, initial=True)
        # 냉각기 ON/OFF

        # ── 노출 / EM 이득 ──────────────────────────────────────────
        self.exposure_time = self.add_logged_quantity(
            name="exposure_time",
            dtype=float, spinbox_decimals=4,
            fmt="%e", ro=False,
            unit="sec", si=True,
            vmin=1e-3, vmax=1000)
        # 노출 시간 [초]

        self.settings.New("has_em_ccd", dtype=bool, ro=True, initial=False)
        # EM CCD 지원 여부 (연결 시 자동 감지)

        self.em_gain = self.add_logged_quantity(
            "em_gain", dtype=int, ro=False, si=False, vmin=1, vmax=4096)
        # EM 전자 증폭 이득

        # ── 취득 모드 ────────────────────────────────────────────────
        self.acq_mode = self.add_logged_quantity(
            'acq_mode', dtype=str,
            initial='single',
            choices=('single', 'accumulate', 'kinetic', 'run_till_abort'))
        # 취득 모드 선택

        self.acc_time = self.add_logged_quantity('acc_time', dtype=float, unit='s', initial=0.1, si=True)
        # Accumulate 사이클 시간 [초]
        self.kin_time = self.add_logged_quantity('kin_time', dtype=float, unit='s', initial=0.1, si=True)
        # Kinetic 사이클 시간 [초]
        self.num_acc  = self.add_logged_quantity('num_acc', dtype=int, initial=1, vmin=1)
        # 누적 횟수
        self.num_kin  = self.add_logged_quantity('num_kin', dtype=int, initial=1, vmin=1)
        # Kinetic 프레임 수

        # ── 출력 앰프 / AD 채널 / 시프트 속도 ──────────────────────
        self.output_amp = self.add_logged_quantity(
            "output_amp", dtype=int, ro=False,
            choices=[("0: EMCCD / Default", 0), ("1: Conventional", 1)])
        # 출력 앰프: 0=EMCCD, 1=일반

        self.ad_chan = self.add_logged_quantity(
            "ad_chan", dtype=int, choices=[('', 0)], initial=0)
        # AD 변환기 채널 인덱스

        self.hs_speed_em = self.add_logged_quantity(
            "hs_speed_em", dtype=int, choices=[('', 0)], initial=0)
        # EM 수평 시프트 속도 인덱스

        self.hs_speed_conventional = self.add_logged_quantity(
            "hs_chan_conventional", dtype=int, choices=[('', 0)], initial=0)
        # 일반 수평 시프트 속도 인덱스

        self.vs_speed = self.add_logged_quantity(
            "vertical_shift_speed", dtype=int, choices=[('', 0)], initial=0)
        # 수직 시프트 속도 인덱스

        # ── 셔터 / 트리거 ────────────────────────────────────────────
        self.shutter_open = self.add_logged_quantity(
            "shutter_open", dtype=bool, ro=False, initial=False)
        # 셔터 열기/닫기

        self.trigger_mode = self.add_logged_quantity(
            "trigger_mode", dtype=str, initial='internal',
            choices=("internal", "external", "external_start",
                     "external_exposure", "external_fvb_em", "software"))
        # 트리거 모드

        # ── 읽기 모드 ────────────────────────────────────────────────
        self.readout_mode = self.add_logged_quantity(
            name="readout_mode", dtype=str, ro=False,
            initial='Image',
            choices=("Image", "FullVerticalBinning", "SingleTrack"))
        # 읽기 모드: Image=2D 전체, FullVerticalBinning=1D 스펙트럼, SingleTrack=단일 행

        # ── ROI: Image 모드 ──────────────────────────────────────────
        self.roi_img_hstart = self.add_logged_quantity("roi_img_hstart", dtype=int, unit='px', ro=False, initial=1)
        self.roi_img_hend   = self.add_logged_quantity("roi_img_hend",   dtype=int, unit='px', ro=False, initial=512)
        self.roi_img_hbin   = self.add_logged_quantity("roi_img_hbin",   dtype=int, unit='px', ro=False, initial=1)
        self.roi_img_vstart = self.add_logged_quantity("roi_img_vstart", dtype=int, unit='px', initial=1,   ro=False)
        self.roi_img_vend   = self.add_logged_quantity("roi_img_vend",   dtype=int, unit='px', initial=512, ro=False)
        self.roi_img_vbin   = self.add_logged_quantity("roi_img_vbin",   dtype=int, unit='px', initial=1,   ro=False)
        # Image 모드 ROI 범위 및 빈닝 설정

        # ── ROI: Single Track 모드 ───────────────────────────────────
        self.roi_st_center = self.add_logged_quantity("roi_st_center", dtype=int, unit='px', ro=False, initial=256)
        # 단일 트랙 중심 행
        self.roi_st_width  = self.add_logged_quantity("roi_st_width",  dtype=int, unit='px', ro=False, initial=10)
        # 단일 트랙 읽을 행 수
        self.roi_st_hbin   = self.add_logged_quantity("roi_st_hbin",   dtype=int, unit='px', ro=False, initial=1)
        # 단일 트랙 수평 빈닝

        # ── ROI: FVB 모드 ────────────────────────────────────────────
        self.roi_fvb_hbin = self.add_logged_quantity(
            "roi_fvb_hbin", dtype=int, unit='px', ro=False, initial=1)
        # FVB 수평 빈닝

        self.settings.New('ccd_shape',     dtype=int, array=True, ro=True)
        # 전체 검출기 크기 [height, width]
        self.settings.New('readout_shape', dtype=int, array=True, ro=True)
        # 현재 읽기 설정 후 실제 출력 크기 [Ny_ro, Nx_ro]

        # ── 이미지 반전 ──────────────────────────────────────────────
        self.hflip = self.add_logged_quantity("hflip", dtype=bool, initial=True)
        # 수평 좌우 반전 (Config.txt Reverse=True 에 대응)
        self.vflip = self.add_logged_quantity("vflip", dtype=bool, initial=False)
        # 수직 상하 반전

        # ── 편의 오퍼레이션 버튼 ────────────────────────────────────
        self.add_operation("set_readout",   self.set_readout)
        # GUI 버튼: ROI 설정 즉시 적용
        self.add_operation("set_full_image", self.set_full_image)
        # GUI 버튼: ROI를 전체 이미지로 초기화


    def connect(self):
        """하드웨어 연결: AndorCCD 인스턴스 생성 및 설정값 동기화."""
        if self.debug:
            self.log.debug("Connecting to Andor EMCCD")

        # AndorCCD 인스턴스 생성 (기본값 자동 적용 안 함 — GUI에서 설정)
        self.ccd_dev = AndorCCD(debug=self.debug, initialize_to_defaults=False)

        # ── LoggedQuantity ↔ 하드웨어 함수 연결 ─────────────────────
        self.status.hardware_read_func = self.ccd_dev.get_status

        # 온도
        self.temperature.hardware_read_func = self.ccd_dev.get_temperature
        self.settings.temp_setpoint.connect_to_hardware(write_func=self.ccd_dev.set_temperature)
        self.settings.temp_setpoint.write_to_hardware()
        self.settings.temp_status.connect_to_hardware(self.ccd_dev.get_temperature_status)
        self.cooler_on.connect_to_hardware(write_func=self.ccd_dev.set_cooler)
        self.cooler_on.write_to_hardware()

        # 노출 시간
        self.exposure_time.hardware_set_func  = self.ccd_dev.set_exposure_time
        self.exposure_time.hardware_read_func = self.ccd_dev.get_exposure_time
        self.exposure_time.write_to_hardware()

        # EM 이득 (EM CCD인 경우만)
        self.settings['has_em_ccd'] = self.ccd_dev.has_em_ccd()
        if self.settings['has_em_ccd']:
            self.em_gain.hardware_read_func = self.ccd_dev.get_EMCCD_gain
            self.em_gain.hardware_set_func  = self.ccd_dev.set_EMCCD_gain
            self.em_gain.write_to_hardware()
        else:
            self.em_gain.change_readonly(True)  # EM 없으면 비활성화

        # 출력 앰프 / AD 채널
        self.output_amp.hardware_set_func = self.ccd_dev.set_output_amp
        self.output_amp.write_to_hardware()
        self.ad_chan.hardware_set_func = self.ccd_dev.set_ad_channel
        self.ad_chan.write_to_hardware()

        # 시프트 속도
        if self.settings['has_em_ccd']:
            self.hs_speed_em.hardware_set_func = self.ccd_dev.set_hs_speed_em
        else:
            self.hs_speed_em.change_readonly(True)  # EM 없으면 비활성화
        self.vs_speed.hardware_set_func = self.ccd_dev.set_vs_speed
        self.hs_speed_conventional.hardware_set_func = self.ccd_dev.set_hs_speed_conventional

        # 셔터 / 트리거
        self.shutter_open.hardware_set_func = self.ccd_dev.set_shutter_open
        self.shutter_open.write_to_hardware()
        self.trigger_mode.hardware_set_func = self.ccd_dev.set_trigger_mode
        self.trigger_mode.write_to_hardware()

        # 이미지 반전
        self.hflip.hardware_set_func = self.ccd_dev.set_image_hflip
        self.hflip.write_to_hardware()
        self.vflip.hardware_set_func = self.ccd_dev.set_image_vflip
        self.vflip.write_to_hardware()

        # 취득 모드 / 누적 / kinetic 설정
        self.acq_mode.connect_to_hardware(write_func=self.ccd_dev.set_aq_mode)
        self.acq_mode.write_to_hardware()
        self.num_acc.connect_to_hardware(write_func=self.ccd_dev.set_num_accumulations)
        self.num_acc.write_to_hardware()
        self.num_kin.connect_to_hardware(write_func=self.ccd_dev.set_num_kinetics)
        try:
            self.num_kin.write_to_hardware()
        except Exception as err:
            self.log.error("set_num_kinetics failed {}".format(err))

        self.acc_time.connect_to_hardware(write_func=self.ccd_dev.set_accumulation_cycle_time)
        try:
            self.acc_time.write_to_hardware()
        except Exception as err:
            self.log.error("set_accumulation_cycle_time failed {}".format(err))

        self.kin_time.connect_to_hardware(write_func=self.ccd_dev.set_kinetic_cycle_time)
        self.kin_time.write_to_hardware()

        # ── ROI 범위 한계 갱신 (검출기 실제 크기 기반) ──────────────
        width, height = self.ccd_dev.get_detector_shape()
        self.settings['ccd_shape'] = height, width
        self.roi_fvb_hbin.change_min_max(1, width)
        self.roi_img_hbin.change_min_max(1, width)
        self.roi_img_hend.change_min_max(1, width)
        self.roi_img_hstart.change_min_max(1, width)
        self.roi_img_vbin.change_min_max(1, height)
        self.roi_img_vend.change_min_max(1, height)
        self.roi_img_vstart.change_min_max(1, height)
        self.roi_st_center.change_min_max(1, height)
        self.roi_st_hbin.change_min_max(1, width)
        self.roi_st_width.change_min_max(1, height)

        # ── EM 수평 속도 선택 목록 생성 ─────────────────────────────
        if self.settings['has_em_ccd']:
            shift_speed_names = OrderedDict()
            for chan_i in range(self.ccd_dev.numADChan):
                for speed_i, speed in enumerate(self.ccd_dev.HSSpeeds_EM[chan_i]):
                    shift_speed_names[speed_i] = (
                        shift_speed_names.get(speed_i, "")
                        + " AD{}-{:.2f}MHz".format(chan_i, speed))
            choices = [(name, num) for num, name in shift_speed_names.items()]
            self.hs_speed_em.change_choice_list(choices)

        # ── 일반 수평 속도 선택 목록 생성 ───────────────────────────
        shift_speed_names = OrderedDict()
        for chan_i in range(self.ccd_dev.numADChan):
            for speed_i, speed in enumerate(self.ccd_dev.HSSpeeds_Conventional[chan_i]):
                shift_speed_names[speed_i] = (
                    shift_speed_names.get(speed_i, "")
                    + " AD{}-{:.2f}MHz".format(chan_i, speed))
        choices = [(name, num) for num, name in shift_speed_names.items()]
        self.hs_speed_conventional.change_choice_list(choices)

        # ── 수직 속도 선택 목록 생성 ────────────────────────────────
        choices = []
        for speed_i in range(self.ccd_dev.numVSSpeeds):
            choices.append((
                "Speed {} - {:.2f} us".format(speed_i, self.ccd_dev.VSSpeeds[speed_i]),
                speed_i))
        self.vs_speed.change_choice_list(choices)

        # ── AD 채널 선택 목록 생성 ──────────────────────────────────
        choices = []
        for chan_i in range(self.ccd_dev.numADChan):
            choices.append(("AD{}".format(chan_i), chan_i))
        self.ad_chan.change_choice_list(choices)

        # 하드웨어에서 모든 설정값 읽어 GUI 동기화
        self.read_from_hardware()
        self.set_readout()  # 현재 ROI 설정 카메라에 적용


    def disconnect(self):
        """하드웨어 연결 해제 및 SDK 종료."""
        self.settings.disconnect_all_from_hardware()
        if hasattr(self, 'ccd_dev'):
            self.ccd_dev.close()
            del self.ccd_dev
        self.is_connected = False


    def is_background_valid(self):
        """
        저장된 배경 이미지가 현재 버퍼와 크기가 맞는지 확인.

        Returns
        -------
        bool
        """
        bg = self.background
        if bg is not None:
            if bg.shape == self.ccd_dev.buffer.shape:
                return True
            else:
                self.log.debug("Background not the correct shape {} {}"
                               .format(self.ccd_dev.buffer.shape, bg.shape))
        else:
            self.log.info("No Background available, raw data shown")
        return False


    def interrupt_acquisition(self):
        """취득 중이면 강제 중단 (IDLE 상태가 아닐 때만)."""
        stat = self.settings.ccd_status.read_from_hardware()
        if stat != 'IDLE':
            self.ccd_dev.abort_acquisition()
        stat = self.settings.ccd_status.read_from_hardware()


    def set_readout(self):
        """
        현재 LoggedQuantity 값에 따라 카메라에 ROI/반전 설정 적용.
        GUI에서 읽기 모드나 ROI를 변경할 때마다 호출.
        """
        self.ccd_dev.set_image_flip(self.hflip.val, self.vflip.val)

        ro_mode = self.readout_mode.val
        if ro_mode == 'FullVerticalBinning':
            # FVB: 1D 스펙트럼 모드
            self.ccd_dev.set_ro_full_vertical_binning(self.roi_fvb_hbin.val)
        elif ro_mode == 'Image':
            # 2D 이미지 (ROI 포함)
            self.ccd_dev.set_ro_image_mode(
                self.roi_img_hbin.val,
                self.roi_img_vbin.val,
                self.roi_img_hstart.val,
                self.roi_img_hend.val,
                self.roi_img_vstart.val,
                self.roi_img_vend.val)
        elif ro_mode == 'SingleTrack':
            # 단일 행 읽기
            self.ccd_dev.set_ro_single_track(
                self.roi_st_center.val,
                self.roi_st_width.val,
                self.roi_st_hbin.val)
        else:
            raise NotImplementedError("ro mode not implemented %s", ro_mode)

        self.settings['readout_shape'] = [self.ccd_dev.Ny_ro, self.ccd_dev.Nx_ro]


    def read_temp_op(self):
        """온도 정보 로그 출력 (디버그용)."""
        self.log.debug("get_temperature_range: {}".format(self.ccd_dev.get_temperature_range()))
        self.log.debug("get_temperature: {}".format(self.ccd_dev.get_temperature()))
        self.log.debug("get_cooler: {}".format(self.ccd_dev.get_cooler()))


    def set_full_image(self):
        """ROI를 전체 검출기 크기로 초기화하고 Image 모드로 설정."""
        width, height = self.ccd_dev.get_detector_shape()
        self.readout_mode.update_value('Image')
        self.roi_img_hstart.update_value(1)
        self.roi_img_hend.update_value(width)
        self.roi_img_hbin.update_value(1)
        self.roi_img_vstart.update_value(1)
        self.roi_img_vend.update_value(height)
        self.roi_img_vbin.update_value(1)
        self.roi_st_center.update_value(height / 2)
        self.roi_st_width.update_value(height / 10)
        self.roi_st_hbin.update_value(1)
        self.roi_fvb_hbin.update_value(1)
        self.set_readout()


    def get_acquired_data(self):
        """
        취득 완료된 이미지 데이터 반환.

        일반(Conventional) 앰프 사용 시 수평 반전 보정 적용.

        Returns
        -------
        np.ndarray (int32)
        """
        buffer_ = self.ccd_dev.get_acquired_data()
        # 일반 앰프(output_amp=1)를 쓰면 이미지가 수평 반전되므로 보정
        if self.settings['output_amp'] == 1:
            buffer_ = buffer_[:, ::-1]
        return buffer_
