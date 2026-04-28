"""
andor_ccd_consts.py — Andor CCD SDK 상수 정의
===================================================

Andor SDK 함수 호출 시 반환되는 에러 코드와
GetCapabilities 쿼리용 비트마스크 상수 정의

[주의]
  AC_ACQMODE_*, AC_READMODE_*, AC_TRIGGERMODE_* 는
  GetCapabilities() 의 비트마스크 결과 해석용 상수이며,
  SetAcquisitionMode() 같은 직접 제어 함수의 인수로 쓰는 값과 다름. 
  직접 제어 함수에는 andor_ccd_interface.py 의 메서드를 사용할 것.
"""

MAX_PATH = 260  # Windows MAX_PATH: 파일 경로 최대 길이

# ══════════════════════════════════════════════════════
# 일반 오류 코드
# ══════════════════════════════════════════════════════
DRV_ERROR_CODES               = 20001  # 에러 코드 범위 시작
DRV_SUCCESS                   = 20002  # 성공 (모든 SDK 함수의 정상 반환값)
DRV_VXDNOTINSTALLED           = 20003  # VXD 드라이버 미설치
DRV_ERROR_SCAN                = 20004  # 스캔 오류
DRV_ERROR_CHECK_SUM           = 20005  # 체크섬 불일치
DRV_ERROR_FILELOAD            = 20006  # 파일 로드 실패
DRV_UNKNOWN_FUNCTION          = 20007  # 알 수 없는 함수
DRV_ERROR_VXD_INIT            = 20008  # VXD 초기화 오류
DRV_ERROR_ADDRESS             = 20009  # 잘못된 주소
DRV_ERROR_PAGELOCK            = 20010  # 페이지 잠금 오류
DRV_ERROR_PAGEUNLOCK          = 20011  # 페이지 잠금 해제 오류
DRV_ERROR_BOARDTEST           = 20012  # 보드 테스트 실패
DRV_ERROR_ACK                 = 20013  # 카메라 응답 없음 (통신 오류)
DRV_ERROR_UP_FIFO             = 20014  # FIFO 업로드 오류
DRV_ERROR_PATTERN             = 20015  # 패턴 오류

# ══════════════════════════════════════════════════════
# 데이터 수집 오류 코드
# ══════════════════════════════════════════════════════
DRV_ACQUISITION_ERRORS        = 20017  # 수집 오류 범위 시작
DRV_ACQ_BUFFER                = 20018  # 수집 버퍼 오류
DRV_ACQ_DOWNFIFO_FULL         = 20019  # 다운 FIFO 꽉 참
DRV_PROC_UNKONWN_INSTRUCTION  = 20020  # 알 수 없는 명령어
DRV_ILLEGAL_OP_CODE           = 20021  # 잘못된 op 코드
DRV_KINETIC_TIME_NOT_MET      = 20022  # kinetic 사이클 시간 부족
DRV_ACCUM_TIME_NOT_MET        = 20023  # accumulate 사이클 시간 부족
DRV_NO_NEW_DATA               = 20024  # 새로운 이미지 데이터 없음
DRV_PCI_DMA_FAIL              = 20025  # PCI DMA 전송 실패
DRV_SPOOLERROR                = 20026  # 스풀 오류
DRV_SPOOLSETUPERROR           = 20027  # 스풀 설정 오류
DRV_FILESIZELIMITERROR        = 20028  # 파일 크기 한계 초과
DRV_ERROR_FILESAVE            = 20029  # 파일 저장 실패

# ══════════════════════════════════════════════════════
# 온도 상태 코드 (GetTemperature 반환값)
# GetTemperature() 함수는 온도 int 가 아니라 이 상태값을 반환한다.
# ══════════════════════════════════════════════════════
DRV_TEMPERATURE_CODES         = 20033  # 온도 코드 범위 시작
DRV_TEMPERATURE_OFF           = 20034  # 냉각기 꺼짐
DRV_TEMPERATURE_NOT_STABILIZED= 20035  # 냉각 중 (목표 온도 미달, 하강 중)
DRV_TEMPERATURE_STABILIZED    = 20036  # ★ 목표 온도 안정화 완료
DRV_TEMPERATURE_NOT_REACHED   = 20037  # 목표 온도에 아직 도달 못함
DRV_TEMPERATURE_OUT_RANGE     = 20038  # 온도 범위 벗어남
DRV_TEMPERATURE_NOT_SUPPORTED = 20039  # 온도 제어 미지원 모델
DRV_TEMPERATURE_DRIFT         = 20040  # 온도 드리프트 발생

# 위와 동일한 값 — 짧은 이름 버전 (코드에서 더 많이 쓰임)
DRV_TEMP_CODES                = 20033
DRV_TEMP_OFF                  = 20034  # 냉각기 OFF
DRV_TEMP_NOT_STABILIZED       = 20035  # 냉각 중
DRV_TEMP_STABILIZED           = 20036  # 안정화 완료
DRV_TEMP_NOT_REACHED          = 20037  # 목표 미달
DRV_TEMP_OUT_RANGE            = 20038  # 범위 벗어남
DRV_TEMP_NOT_SUPPORTED        = 20039  # 미지원
DRV_TEMP_DRIFT                = 20040  # 드리프트

# ══════════════════════════════════════════════════════
# 일반/드라이버 오류 코드
# ══════════════════════════════════════════════════════
DRV_GENERAL_ERRORS            = 20049
DRV_INVALID_AUX               = 20050  # 잘못된 AUX 포트
DRV_COF_NOTLOADED             = 20051  # COF 파일 미로드
DRV_FPGAPROG                  = 20052  # FPGA 프로그래밍 오류
DRV_FLEXERROR                 = 20053  # Flex 오류
DRV_GPIBERROR                 = 20054  # GPIB 통신 오류
DRV_EEPROMVERSIONERROR        = 20055  # EEPROM 버전 불일치

DRV_DATATYPE                  = 20064  # 데이터 타입 오류
DRV_DRIVER_ERRORS             = 20065  # 드라이버 오류 범위 시작
DRV_P1INVALID                 = 20066  # 1번 파라미터 범위 초과
DRV_P2INVALID                 = 20067  # 2번 파라미터 범위 초과
DRV_P3INVALID                 = 20068  # 3번 파라미터 범위 초과
DRV_P4INVALID                 = 20069  # 4번 파라미터 범위 초과
DRV_INIERROR                  = 20070  # INI 파일 오류
DRV_COFERROR                  = 20071  # COF 파일 오류
DRV_ACQUIRING                 = 20072  # ★ 현재 데이터 수집 중
DRV_IDLE                      = 20073  # ★ 유휴 상태 (수집 완료)
DRV_TEMPCYCLE                 = 20074  # 온도 사이클 진행 중
DRV_NOT_INITIALIZED           = 20075  # SDK 초기화 안 됨
DRV_P5INVALID                 = 20076  # 5번 파라미터 범위 초과
DRV_P6INVALID                 = 20077  # 6번 파라미터 범위 초과
DRV_INVALID_MODE              = 20078  # 잘못된 모드
DRV_INVALID_FILTER            = 20079  # 잘못된 필터

DRV_I2CERRORS                 = 20080  # I2C 오류 범위 시작
DRV_I2CDEVNOTFOUND            = 20081  # I2C 장치 없음
DRV_I2CTIMEOUT                = 20082  # I2C 통신 타임아웃
DRV_P7INVALID                 = 20083  # 7번 파라미터 범위 초과
DRV_P8INVALID                 = 20084  # 8번 파라미터 범위 초과
DRV_P9INVALID                 = 20085  # 9번 파라미터 범위 초과
DRV_P10INVALID                = 20086  # 10번 파라미터 범위 초과
DRV_P11INVALID                = 20087  # 11번 파라미터 범위 초과

DRV_USBERROR                  = 20089  # USB 오류
DRV_IOCERROR                  = 20090  # I/O 제어 오류
DRV_VRMVERSIONERROR           = 20091  # VRM 버전 불일치
DRV_GATESTEPERROR             = 20092  # Gate step 오류
DRV_USB_INTERRUPT_ENDPOINT_ERROR = 20093  # USB 인터럽트 엔드포인트 오류
DRV_RANDOM_TRACK_ERROR        = 20094  # Random track 오류
DRV_INVALID_TRIGGER_MODE      = 20095  # 잘못된 트리거 모드
DRV_LOAD_FIRMWARE_ERROR       = 20096  # 펌웨어 로드 실패
DRV_DIVIDE_BY_ZERO_ERROR      = 20097  # 0 나누기 오류
DRV_INVALID_RINGEXPOSURES     = 20098  # 잘못된 링 노출 설정
DRV_BINNING_ERROR             = 20099  # 빈닝 설정 오류
DRV_INVALID_AMPLIFIER         = 20100  # 잘못된 앰프 선택
DRV_INVALID_COUNTCONVERT_MODE = 20101  # 잘못된 카운트 변환 모드

DRV_ERROR_NOCAMERA            = 20990  # ★ 카메라 없음 (연결 안 됨)
DRV_NOT_SUPPORTED             = 20991  # 해당 카메라에서 미지원 기능
DRV_NOT_AVAILABLE             = 20992  # 현재 사용 불가 (다른 작업 중 등)

DRV_ERROR_MAP                 = 20115  # 메모리 맵 오류
DRV_ERROR_UNMAP               = 20116  # 메모리 맵 해제 오류
DRV_ERROR_MDL                 = 20117  # MDL 오류
DRV_ERROR_UNMDL               = 20118  # MDL 해제 오류
DRV_ERROR_BUFFSIZE            = 20119  # 버퍼 크기 오류
DRV_ERROR_NOHANDLE            = 20121  # 핸들 없음

DRV_GATING_NOT_AVAILABLE      = 20130  # 게이팅 미지원
DRV_FPGA_VOLTAGE_ERROR        = 20131  # FPGA 전압 오류

DRV_OW_CMD_FAIL               = 20150  # One-wire 명령 실패
DRV_OWMEMORY_BAD_ADDR         = 20151  # One-wire 잘못된 주소
DRV_OWCMD_NOT_AVAILABLE       = 20152  # One-wire 명령 불가
DRV_OW_NO_SLAVES              = 20153  # One-wire 슬레이브 없음
DRV_OW_NOT_INITIALIZED        = 20154  # One-wire 미초기화
DRV_OW_ERROR_SLAVE_NUM        = 20155  # One-wire 슬레이브 번호 오류
DRV_MSTIMINGS_ERROR           = 20156  # MS 타이밍 오류

# ══════════════════════════════════════════════════════
# Optional Advanced (OA) 오류 코드
# ══════════════════════════════════════════════════════
DRV_OA_NULL_ERROR                       = 20173
DRV_OA_PARSE_DTD_ERROR                  = 20174
DRV_OA_DTD_VALIDATE_ERROR               = 20175
DRV_OA_FILE_ACCESS_ERROR                = 20176
DRV_OA_FILE_DOES_NOT_EXIST              = 20177
DRV_OA_XML_INVALID_OR_NOT_FOUND_ERROR   = 20178
DRV_OA_PRESET_FILE_NOT_LOADED           = 20179
DRV_OA_USER_FILE_NOT_LOADED             = 20180
DRV_OA_PRESET_AND_USER_FILE_NOT_LOADED  = 20181
DRV_OA_INVALID_FILE                     = 20182
DRV_OA_FILE_HAS_BEEN_MODIFIED           = 20183
DRV_OA_BUFFER_FULL                      = 20184
DRV_OA_INVALID_STRING_LENGTH            = 20185
DRV_OA_INVALID_CHARS_IN_NAME            = 20186
DRV_OA_INVALID_NAMING                   = 20187
DRV_OA_GET_CAMERA_ERROR                 = 20188
DRV_OA_MODE_ALREADY_EXISTS              = 20189
DRV_OA_STRINGS_NOT_EQUAL                = 20190
DRV_OA_NO_USER_DATA                     = 20191
DRV_OA_VALUE_NOT_SUPPORTED              = 20192
DRV_OA_MODE_DOES_NOT_EXIST              = 20193
DRV_OA_CAMERA_NOT_SUPPORTED             = 20194
DRV_OA_FAILED_TO_GET_MODE               = 20195

DRV_PROCESSING_FAILED                   = 20211  # 이미지 처리 실패

# ══════════════════════════════════════════════════════
# GetCapabilities 비트마스크 — 취득 모드 지원 여부 조회용
# SetAcquisitionMode() 의 인수값이 아님에 주의.
# ══════════════════════════════════════════════════════
AC_ACQMODE_SINGLE        = 1    # 단일 프레임 수집 지원
AC_ACQMODE_VIDEO         = 2    # 비디오(Run till abort) 지원
AC_ACQMODE_ACCUMULATE    = 4    # Accumulate 모드 지원
AC_ACQMODE_KINETIC       = 8    # Kinetic series 지원
AC_ACQMODE_FRAMETRANSFER = 16   # Frame transfer 지원
AC_ACQMODE_FASTKINETICS  = 32   # Fast kinetics 지원
AC_ACQMODE_OVERLAP       = 64   # Overlap 모드 지원

# ══════════════════════════════════════════════════════
# GetCapabilities 비트마스크 — 읽기 모드 지원 여부 조회용
# SetReadMode() 의 인수값(0~4)과 다름에 주의.
# ══════════════════════════════════════════════════════
AC_READMODE_FULLIMAGE      = 1   # 전체 이미지 읽기 지원
AC_READMODE_SUBIMAGE       = 2   # 서브이미지(ROI) 지원
AC_READMODE_SINGLETRACK    = 4   # 단일 트랙 지원
AC_READMODE_FVB            = 8   # Full Vertical Binning 지원
AC_READMODE_MULTITRACK     = 16  # 멀티트랙 지원
AC_READMODE_RANDOMTRACK    = 32  # 랜덤트랙 지원
AC_READMODE_MULTITRACKSCAN = 64  # 멀티트랙 스캔 지원

# ══════════════════════════════════════════════════════
# GetCapabilities 비트마스크 — 트리거 모드 지원 여부 조회용
# SetTriggerMode() 의 인수값(0, 1, 6, 7, 9, 10)과 다름에 주의.
# ══════════════════════════════════════════════════════
AC_TRIGGERMODE_INTERNAL              = 1     # 내부 트리거 지원
AC_TRIGGERMODE_EXTERNAL              = 2     # 외부 트리거 지원
AC_TRIGGERMODE_EXTERNAL_FVB_EM       = 4     # 외부 FVB EM 트리거 지원
AC_TRIGGERMODE_CONTINUOUS            = 8     # 연속 트리거 지원
AC_TRIGGERMODE_EXTERNALSTART         = 16    # 외부 시작 트리거 지원
AC_TRIGGERMODE_EXTERNALEXPOSURE      = 32    # 외부 노출 트리거 지원
AC_TRIGGERMODE_INVERTED              = 0x40  # 반전 트리거 지원
AC_TRIGGERMODE_EXTERNAL_CHARGESHIFTING = 0x80  # 외부 전하 이동 트리거 지원

# ══════════════════════════════════════════════════════
# 숫자 → 상수 이름 역방향 조회 딕셔너리
# 오류 코드를 사람이 읽을 수 있는 이름으로 변환할 때 사용.
# 예: consts_by_num[20002] → 'DRV_SUCCESS'
# ══════════════════════════════════════════════════════
consts_by_num = dict()
for name, num in list(locals().items()):
    if name.startswith("DRV_") or name.startswith('AC_'):
        consts_by_num[num] = name
