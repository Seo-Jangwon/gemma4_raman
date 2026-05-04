from enum import IntEnum


class readmodes(IntEnum):
    """AndorCapabilities 구조체의 ulReadModes 필드에서 사용하는 읽기 모드 비트 플래그."""
    AC_READMODE_FULLIMAGE = 1       # 전체 이미지 읽기 모드 지원
    AC_READMODE_SUBIMAGE = 2        # 서브 이미지(관심 영역) 읽기 모드 지원
    AC_READMODE_SINGLETRACK = 4     # 단일 트랙 읽기 모드 지원
    AC_READMODE_FVB = 8             # 전체 수직 비닝(FVB) 읽기 모드 지원
    AC_READMODE_MULTITRACK = 16     # 멀티 트랙 읽기 모드 지원
    AC_READMODE_RANDOMTRACK = 32    # 랜덤 트랙 읽기 모드 지원
    AC_READMODE_MULTITRACKSCAN = 64 # 멀티 트랙 스캔 읽기 모드 지원


class stepmodes(IntEnum):
    """키네틱 시리즈에서 스텝 간격 변화 방식."""
    AT_STEPMODE_CONSTANT = 0     # 일정한 간격으로 스텝
    AT_STEPMODE_EXPONENTIAL = 1  # 지수적으로 증가하는 간격
    AT_STEPMODE_LOGARITHMIC = 2  # 로그 스케일 간격
    AT_STEPMODE_LINEAR = 3       # 선형적으로 증가하는 간격
    AT_STEPMODE_OFF = 100        # 스텝 모드 비활성


class gatemodes(IntEnum):
    """AndorCapabilities 구조체의 게이트 모드 비트 플래그 - ICCD 카메라 전용."""
    AT_GATEMODE_FIRE_AND_GATE = 0  # FIRE 신호와 게이트 입력 AND 조합
    AT_GATEMODE_FIRE_ONLY = 1      # FIRE 펄스로만 게이팅 제어
    AT_GATEMODE_GATE_ONLY = 2      # 게이트 입력으로만 게이팅 제어
    AT_GATEMODE_CW_ON = 3          # 연속파 ON (게이팅 항상 활성)
    AT_GATEMODE_CW_OFF = 4         # 연속파 OFF (게이팅 항상 비활성)
    AT_GATEMODE_DDG = 5            # DDG(디지털 지연 발생기)를 이용한 게이팅


class triggermodes(IntEnum):
    """AndorCapabilities 구조체의 ulTriggerModes 필드에서 사용하는 트리거 모드 비트 플래그."""
    AC_TRIGGERMODE_INTERNAL = 1                  # 내부 트리거 모드 지원
    AC_TRIGGERMODE_EXTERNAL = 2                  # 외부 트리거 모드 지원
    AC_TRIGGERMODE_EXTERNAL_FVB_EM = 4           # 외부 FVB EM 트리거 지원
    AC_TRIGGERMODE_CONTINUOUS = 8                # 연속 트리거 모드 지원
    AC_TRIGGERMODE_EXTERNALSTART = 16            # 외부 시작 트리거 지원
    AC_TRIGGERMODE_EXTERNALEXPOSURE = 32         # 외부 노출(벌브) 트리거 지원
    AC_TRIGGERMODE_INVERTED = 0x40               # 반전된 트리거 극성 지원
    AC_TRIGGERMODE_EXTERNAL_CHARGESHIFTING = 0x80  # 외부 전하 이동 트리거 지원
    AC_TRIGGERMODE_BULB = 32                     # 벌브 트리거 지원 (EXTERNALEXPOSURE와 동일 값)


class acquistionModes(IntEnum):
    """AndorCapabilities 구조체의 ulAcqModes 필드에서 사용하는 획득 모드 비트 플래그."""
    AC_ACQMODE_SINGLE = 1          # 단일 스캔 모드 지원
    AC_ACQMODE_VIDEO = 2           # 비디오(연속) 모드 지원
    AC_ACQMODE_ACCUMULATE = 4      # 누적 모드 지원
    AC_ACQMODE_KINETIC = 8         # 키네틱 시리즈 모드 지원
    AC_ACQMODE_FRAMETRANSFER = 16  # 프레임 전송 모드 지원
    AC_ACQMODE_FASTKINETICS = 32   # 고속 키네틱 모드 지원
    AC_ACQMODE_OVERLAP = 64        # 오버랩 획득 모드 지원
    AC_ACQMODE_TDI = 128           # TDI(시간 지연 적분) 모드 지원


class cameratype(IntEnum):
    """AndorCapabilities 구조체의 ulCameraType 필드에서 반환되는 카메라 모델 식별자."""
    AC_CAMERATYPE_PDA = 0           # PDA(포토다이오드 어레이) 카메라
    AC_CAMERATYPE_IXON = 1          # iXon EMCCD 카메라
    AC_CAMERATYPE_ICCD = 2          # ICCD 이미지 증배관 카메라
    AC_CAMERATYPE_EMCCD = 3         # EMCCD 전자 증배 CCD 카메라
    AC_CAMERATYPE_CCD = 4           # 일반 CCD 카메라
    AC_CAMERATYPE_ISTAR = 5         # iStar ICCD 카메라
    AC_CAMERATYPE_VIDEO = 6         # 비디오 카메라
    AC_CAMERATYPE_IDUS = 7          # iDus 분광기용 CCD 카메라
    AC_CAMERATYPE_NEWTON = 8        # Newton 분광기용 CCD 카메라
    AC_CAMERATYPE_SURCAM = 9        # 서피스 카메라
    AC_CAMERATYPE_USBICCD = 10      # USB ICCD 카메라
    AC_CAMERATYPE_LUCA = 11         # Luca EMCCD 카메라
    AC_CAMERATYPE_RESERVED = 12     # 예약됨
    AC_CAMERATYPE_IKON = 13         # iKon CCD 카메라
    AC_CAMERATYPE_INGAAS = 14       # InGaAs 적외선 카메라
    AC_CAMERATYPE_IVAC = 15         # iVac CCD 카메라
    AC_CAMERATYPE_UNPROGRAMMED = 16 # 미프로그램 카메라
    AC_CAMERATYPE_CLARA = 17        # Clara CCD 카메라
    AC_CAMERATYPE_USBISTAR = 18     # USB iStar ICCD 카메라
    AC_CAMERATYPE_SIMCAM = 19       # 시뮬레이션 카메라
    AC_CAMERATYPE_NEO = 20          # Neo sCMOS 카메라
    AC_CAMERATYPE_IXONULTRA = 21    # iXon Ultra EMCCD 카메라
    AC_CAMERATYPE_VOLMOS = 22       # Volmos sCMOS 카메라
    AC_CAMERATYPE_IVAC_CCD = 23     # iVac CCD 모델
    AC_CAMERATYPE_ASPEN = 24        # Aspen CCD 카메라
    AC_CAMERATYPE_ASCENT = 25       # Ascent CCD 카메라
    AC_CAMERATYPE_ALTA = 26         # Alta CCD 카메라
    AC_CAMERATYPE_ALTAF = 27        # Alta-F CCD 카메라
    AC_CAMERATYPE_IKONXL = 28       # iKon-XL CCD 카메라
    AC_CAMERATYPE_RES1 = 29         # 예약 모델 1
    AC_CAMERATYPE_ISTAR_SCMOS = 30  # iStar sCMOS 카메라
    AC_CAMERATYPE_IKONLR = 31       # iKon-L R 시리즈 CCD 카메라
    AC_PIXELMODE_8BIT = 1           # 8비트 픽셀 모드


class SetFunctions(IntEnum):
    """AndorCapabilities 구조체의 ulSetFunctions 필드 - 설정 가능한 기능 비트 플래그."""
    AC_SETFUNCTION_VREADOUT = 0x01              # 수직 읽기 속도 설정 가능
    AC_SETFUNCTION_HREADOUT = 0x02              # 수평 읽기 속도 설정 가능
    AC_SETFUNCTION_TEMPERATURE = 0x04           # 냉각 온도 설정 가능
    AC_SETFUNCTION_MCPGAIN = 0x08               # MCP 이득 설정 가능
    AC_SETFUNCTION_EMCCDGAIN = 0x10             # EMCCD 이득 설정 가능
    AC_SETFUNCTION_BASELINECLAMP = 0x20         # 기준선 클램프 설정 가능
    AC_SETFUNCTION_VSAMPLITUDE = 0x40           # 수직 클럭 전압 진폭 설정 가능
    AC_SETFUNCTION_HIGHCAPACITY = 0x80          # 고용량 모드 설정 가능
    AC_SETFUNCTION_BASELINEOFFSET = 0x0100      # 기준선 오프셋 설정 가능
    AC_SETFUNCTION_PREAMPGAIN = 0x0200          # 프리앰프 이득 설정 가능
    AC_SETFUNCTION_CROPMODE = 0x0400            # 크롭 모드 설정 가능
    AC_SETFUNCTION_DMAPARAMETERS = 0x0800       # DMA 파라미터 설정 가능
    AC_SETFUNCTION_HORIZONTALBIN = 0x1000       # 수평 비닝 설정 가능
    AC_SETFUNCTION_MULTITRACKHRANGE = 0x2000    # 멀티 트랙 수평 범위 설정 가능
    AC_SETFUNCTION_RANDOMTRACKNOGAPS = 0x4000   # 랜덤 트랙 간격 없음 설정 가능
    AC_SETFUNCTION_EMADVANCED = 0x8000          # EM 고급 설정 가능
    AC_SETFUNCTION_GATEMODE = 0x010000          # 게이트 모드 설정 가능
    AC_SETFUNCTION_DDGTIMES = 0x020000          # DDG 타이밍 설정 가능
    AC_SETFUNCTION_IOC = 0x040000               # IOC(입출력 제어) 설정 가능
    AC_SETFUNCTION_INTELLIGATE = 0x080000       # 인텔리게이트 설정 가능
    AC_SETFUNCTION_INSERTION_DELAY = 0x100000   # 삽입 지연 설정 가능
    AC_SETFUNCTION_GATESTEP = 0x200000          # 게이트 스텝 설정 가능
    AC_SETFUNCTION_GATEDELAYSTEP = 0x200000     # 게이트 지연 스텝 설정 가능 (GATESTEP과 동일 값)
    AC_SETFUNCTION_TRIGGERTERMINATION = 0x400000   # 트리거 종단 저항 설정 가능
    AC_SETFUNCTION_EXTENDEDNIR = 0x800000          # 확장 NIR 모드 설정 가능
    AC_SETFUNCTION_SPOOLTHREADCOUNT = 0x1000000    # 스풀 스레드 수 설정 가능
    AC_SETFUNCTION_REGISTERPACK = 0x2000000        # 레지스터 패킹 설정 가능
    AC_SETFUNCTION_PRESCANS = 0x4000000            # 프리스캔 횟수 설정 가능
    AC_SETFUNCTION_GATEWIDTHSTEP = 0x8000000       # 게이트 폭 스텝 설정 가능
    AC_SETFUNCTION_EXTENDED_CROP_MODE = 0x10000000 # 확장 크롭 모드 설정 가능
    AC_SETFUNCTION_SUPERKINETICS = 0x20000000      # 슈퍼 키네틱 모드 설정 가능
    AC_SETFUNCTION_TIMESCAN = 0x40000000           # 타임 스캔 설정 가능
    AC_SETFUNCTION_CROPMODETYPE = 0x80000000       # 크롭 모드 타입 설정 가능
    AC_SETFUNCTION_GAIN = 8                        # 이득 설정 가능 (MCPGAIN과 동일 값)
    AC_SETFUNCTION_ICCDGAIN = 8                    # ICCD 이득 설정 가능 (MCPGAIN과 동일 값)


class GetFunctions(IntEnum):
    """AndorCapabilities 구조체의 ulGetFunctions 필드 - 조회 가능한 기능 비트 플래그."""
    AC_GETFUNCTION_TEMPERATURE = 0x01           # 현재 온도 조회 가능
    AC_GETFUNCTION_TARGETTEMPERATURE = 0x02     # 목표 온도 조회 가능
    AC_GETFUNCTION_TEMPERATURERANGE = 0x04      # 온도 범위 조회 가능
    AC_GETFUNCTION_DETECTORSIZE = 0x08          # 검출기 크기 조회 가능
    AC_GETFUNCTION_MCPGAIN = 0x10               # MCP 이득 조회 가능
    AC_GETFUNCTION_EMCCDGAIN = 0x20             # EMCCD 이득 조회 가능
    AC_GETFUNCTION_HVFLAG = 0x40                # 고전압 플래그 조회 가능
    AC_GETFUNCTION_GATEMODE = 0x80              # 게이트 모드 조회 가능
    AC_GETFUNCTION_DDGTIMES = 0x0100            # DDG 타이밍 조회 가능
    AC_GETFUNCTION_IOC = 0x0200                 # IOC 조회 가능
    AC_GETFUNCTION_INTELLIGATE = 0x0400         # 인텔리게이트 조회 가능
    AC_GETFUNCTION_INSERTION_DELAY = 0x0800     # 삽입 지연 조회 가능
    AC_GETFUNCTION_GATESTEP = 0x1000            # 게이트 스텝 조회 가능
    AC_GETFUNCTION_GATEDELAYSTEP = 0x1000       # 게이트 지연 스텝 조회 가능 (GATESTEP과 동일 값)
    AC_GETFUNCTION_PHOSPHORSTATUS = 0x2000      # 형광체 상태 조회 가능
    AC_GETFUNCTION_MCPGAINTABLE = 0x4000        # MCP 이득 테이블 조회 가능
    AC_GETFUNCTION_BASELINECLAMP = 0x8000       # 기준선 클램프 조회 가능
    AC_GETFUNCTION_GATEWIDTHSTEP = 0x10000      # 게이트 폭 스텝 조회 가능
    AC_GETFUNCTION_GAIN = 0x10                  # 이득 조회 가능 (MCPGAIN과 동일 값)
    AC_GETFUNCTION_ICCDGAIN = 0x10              # ICCD 이득 조회 가능 (MCPGAIN과 동일 값)


class Features(IntEnum):
    """AndorCapabilities 구조체의 ulFeatures 필드 - 카메라 지원 기능 비트 플래그."""
    AC_FEATURES_POLLING = 1                           # 폴링 방식 상태 확인 지원
    AC_FEATURES_EVENTS = 2                            # 이벤트 기반 알림 지원
    AC_FEATURES_SPOOLING = 4                          # 스풀링(대용량 연속 저장) 지원
    AC_FEATURES_SHUTTER = 8                           # 내부 셔터 지원
    AC_FEATURES_SHUTTEREX = 16                        # 외부 셔터 제어 지원
    AC_FEATURES_EXTERNAL_I2C = 32                     # 외부 I2C 통신 지원
    AC_FEATURES_SATURATIONEVENT = 64                  # 포화(saturation) 이벤트 알림 지원
    AC_FEATURES_FANCONTROL = 128                      # 팬 속도 제어 지원
    AC_FEATURES_MIDFANCONTROL = 256                   # 중간 속도 팬 제어 지원
    AC_FEATURES_TEMPERATUREDURINGACQUISITION = 512    # 획득 중 온도 모니터링 지원
    AC_FEATURES_KEEPCLEANCONTROL = 1024               # 클린 유지 제어 지원
    AC_FEATURES_DDGLITE = 0x0800                      # DDGLite(간소화 DDG) 지원
    AC_FEATURES_FTEXTERNALEXPOSURE = 0x1000           # 프레임 전송 외부 노출 지원
    AC_FEATURES_KINETICEXTERNALEXPOSURE = 0x2000      # 키네틱 외부 노출 지원
    AC_FEATURES_DACCONTROL = 0x4000                   # DAC 직접 제어 지원
    AC_FEATURES_METADATA = 0x8000                     # 메타데이터 첨부 지원
    AC_FEATURES_IOCONTROL = 0x10000                   # I/O 포트 제어 지원
    AC_FEATURES_PHOTONCOUNTING = 0x20000              # 광자 계수 모드 지원
    AC_FEATURES_COUNTCONVERT = 0x40000                # 계수 변환 지원
    AC_FEATURES_DUALMODE = 0x80000                    # 듀얼 모드 지원
    AC_FEATURES_OPTACQUIRE = 0x100000                 # 최적화 획득 지원
    AC_FEATURES_REALTIMESPURIOUSNOISEFILTER = 0x200000    # 실시간 노이즈 필터 지원
    AC_FEATURES_POSTPROCESSSPURIOUSNOISEFILTER = 0x400000 # 후처리 노이즈 필터 지원
    AC_FEATURES_DUALPREAMPGAIN = 0x800000             # 듀얼 프리앰프 이득 지원
    AC_FEATURES_DEFECT_CORRECTION = 0x1000000         # 결함 픽셀 보정 지원
    AC_FEATURES_STARTOFEXPOSURE_EVENT = 0x2000000     # 노출 시작 이벤트 알림 지원
    AC_FEATURES_ENDOFEXPOSURE_EVENT = 0x4000000       # 노출 종료 이벤트 알림 지원
    AC_FEATURES_CAMERALINK = 0x8000000                # CameraLink 인터페이스 지원
    AC_FEATURES_FIFOFULL_EVENT = 0x10000000           # FIFO 가득 참 이벤트 지원
    AC_FEATURES_SENSOR_PORT_CONFIGURATION = 0x20000000  # 센서 포트 구성 지원
    AC_FEATURES_SENSOR_COMPENSATION = 0x40000000      # 센서 보정 지원
    AC_FEATURES_IRIG_SUPPORT = 0x80000000             # IRIG 타임코드 지원


class PixelModes(IntEnum):
    """AndorCapabilities 구조체의 ulPixelMode 필드 - 픽셀 비트 깊이 및 색상 모드 비트 플래그."""
    AC_PIXELMODE_14BIT = 2       # 14비트 픽셀 깊이
    AC_PIXELMODE_16BIT = 4       # 16비트 픽셀 깊이
    AC_PIXELMODE_32BIT = 8       # 32비트 픽셀 깊이
    AC_PIXELMODE_MONO = 0x000000 # 모노크롬(흑백) 색상 모드
    AC_PIXELMODE_RGB = 0x010000  # RGB 컬러 모드
    AC_PIXELMODE_CMY = 0x020000  # CMY 컬러 모드


class EmGainModes(IntEnum):
    """AndorCapabilities 구조체의 ulEMGainCapability 필드 - EM 이득 범위 비트 플래그."""
    AC_EMGAIN_8BIT = 1      # 8비트(0~255) EM 이득 범위
    AC_EMGAIN_12BIT = 2     # 12비트(0~4095) EM 이득 범위
    AC_EMGAIN_LINEAR12 = 4  # 선형 12비트 EM 이득 범위
    AC_EMGAIN_REAL12 = 8    # 실제 EM 이득 값 12비트 범위


class Features2(IntEnum):
    """AndorCapabilities 구조체의 ulFeatures2 필드 - 추가 기능 비트 플래그."""
    AC_FEATURES2_ESD_EVENTS = 1                 # ESD(정전기 방전) 이벤트 알림 지원
    AC_FEATURES2_DUAL_PORT_CONFIGURATION = 2    # 듀얼 포트 구성 지원


class CameraCapabilities(IntEnum):
    """DDGLite 및 기타 카메라 하드웨어 기능 제어용 상수."""
    AT_NoOfVersionInfoIds = 2              # 버전 정보 항목 수
    AT_VERSION_INFO_LEN = 80               # 버전 정보 문자열 최대 길이
    AT_CONTROLLER_CARD_MODEL_LEN = 80      # 컨트롤러 카드 모델명 최대 길이
    AT_DDGLite_ControlBit_GlobalEnable = 0x01    # DDGLite 전체 채널 활성화 비트
    AT_DDGLite_ControlBit_ChannelEnable = 0x01   # DDGLite 개별 채널 활성화 비트
    AT_DDGLite_ControlBit_FreeRun = 0x02         # DDGLite 자유 실행 모드 비트
    AT_DDGLite_ControlBit_DisableOnFrame = 0x04  # DDGLite 프레임 종료 시 비활성 비트
    AT_DDGLite_ControlBit_RestartOnFire = 0x08   # DDGLite FIRE 신호 시 재시작 비트
    AT_DDGLite_ControlBit_Invert = 0x10          # DDGLite 출력 신호 반전 비트
    AT_DDGLite_ControlBit_EnableOnFire = 0x20    # DDGLite FIRE 신호 시 활성화 비트
    AT_DDG_POLARITY_POSITIVE = 0                 # DDG 양의 극성
    AT_DDG_POLARITY_NEGATIVE = 1                 # DDG 음의 극성
    AT_DDG_TERMINATION_50OHMS = 0                # DDG 50Ω 종단
    AT_DDG_TERMINATION_HIGHZ = 1                 # DDG 고임피던스 종단
