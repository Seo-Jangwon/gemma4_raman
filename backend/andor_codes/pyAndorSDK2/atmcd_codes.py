from enum import IntEnum, unique


@unique
class Read_Mode(IntEnum):
    """읽기 모드 옵션 (Read mode options)
    """
    FULL_VERTICAL_BINNING = 0  # 전체 수직 비닝: 모든 픽셀을 수직으로 합산하여 1D 스펙트럼 취득
    MULTI_TRACK = 1            # 멀티 트랙: 여러 행 범위를 동시에 읽기
    RANDOM_TRACK = 2           # 랜덤 트랙: 임의 위치의 행들을 선택적으로 읽기
    SINGLE_TRACK = 3           # 단일 트랙: 지정된 단일 행만 읽기
    IMAGE = 4                  # 이미지: 전체 2D 이미지 읽기


@unique
class Trigger_Mode(IntEnum):
    """트리거 모드 옵션 (Trigger mode options)
    """
    INTERNAL = 0                    # 내부 트리거: 카메라 내부 타이밍으로 획득 시작
    EXTERNAL = 1                    # 외부 트리거: 외부 전기 신호로 각 획득 시작
    EXTERNAL_START = 6              # 외부 시작 트리거: 외부 신호로 시작 후 내부 타이밍 사용
    EXTERNAL_EXPOSURE_BULB = 7      # 외부 노출(벌브): 외부 신호 지속 시간 동안 노출 유지
    EXTERNAL_FVB_EM = 9             # 외부 FVB EM 트리거: FVB EM 모드용 외부 트리거
    SOFTWARE_TRIGGER = 10           # 소프트웨어 트리거: SendSoftwareTrigger 명령으로 획득 시작
    EXTERNAL_CHARGE_SHIFTING = 12   # 외부 전하 이동 트리거


@unique
class Acquisition_Mode(IntEnum):
    """획득 모드 옵션 (Acquisition mode options)
    """
    SINGLE_SCAN = 1       # 단일 스캔: 1회 획득 후 종료
    ACCUMULATE = 2        # 누적: 여러 프레임을 합산하여 단일 데이터로 저장
    KINETICS = 3          # 키네틱 시리즈: 연속 프레임을 고속으로 순차 저장
    FAST_KINETICS = 4     # 고속 키네틱: CCD 내부에서 매우 빠른 연속 획득
    RUN_TILL_ABORT = 5    # 중단 시까지 실행: AbortAcquisition 호출 전까지 연속 획득


@unique
class Spool_Mode(IntEnum):
    """스풀 모드 옵션 (Spool mode options)
    """
    FILE_32_BIT_SEQUENCE = 0  # 32비트 정수 파일 시퀀스로 스풀
    # 누적 획득 여부에 따라 포맷 결정: 누적 시 32비트, 아닐 경우 16비트 정수
    DATA_DEPENDENT_FORMAT = 1
    FILE_16_BIT_SEQUENCE = 2              # 16비트 정수 파일 시퀀스로 스풀
    MULTIPLE_DIRECTORY_STRUCTURE = 3      # 다중 디렉터리 구조로 스풀
    SPOOL_TO_RAM = 4                      # RAM으로 스풀 (메모리 직접 저장)
    SPOOL_TO_16_BIT_FITS = 5              # 16비트 FITS 천문 파일 형식으로 스풀
    SPOOL_TO_SIF = 6                      # Andor SIF 파일 형식으로 스풀
    SPOOL_TO_16_BIT_TIFF = 7             # 16비트 TIFF 이미지 파일로 스풀
    COMPRESSED_MULTIPLE_DIRECTORY_STRUCTURE = 8  # 압축된 다중 디렉터리 구조로 스풀


@unique
class Gate_Mode(IntEnum):
    """게이트 모드 옵션 (Gate mode options) - iStar ICCD 카메라 전용
    """
    FIRE_ANDED_WITH_THE_GATE_INPUT = 0           # FIRE 신호와 게이트 입력 신호를 AND 조합
    GATING_CONTROLLED_FROM_FIRE_PULSE_ONLY = 1   # FIRE 펄스로만 게이팅 제어
    GATING_CONTROLLED_FROM_SMB_GATE_INPUT_ONLY = 2  # SMB 게이트 입력으로만 게이팅 제어
    GATING_ON_CONTINUOUSLY = 3                   # 게이팅 항상 ON (MCP 상시 활성)
    GATING_OFF_CONTINUOUSLY = 4                  # 게이팅 항상 OFF (MCP 상시 비활성)
    GATE_USING_DDG = 5                           # DDG(디지털 지연 발생기)를 이용한 게이팅


@unique
class Shutter_Mode(IntEnum):
    """셔터 모드 옵션 (Shutter mode options)
    """
    FULLY_AUTO = 0             # 완전 자동: 획득 중 셔터 자동 개폐
    PERMANENTLY_OPEN = 1       # 항상 열림: 셔터 고정 개방
    PERMANENTLY_CLOSED = 2     # 항상 닫힘: 셔터 고정 폐쇄 (암전류 측정 등에 사용)
    OPEN_FOR_FVB_SERIES = 4    # FVB 시리즈 동안만 열림
    OPEN_FOR_ANY_SERIES = 5    # 모든 시리즈 획득 동안 열림
