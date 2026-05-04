from enum import IntEnum


class Error_Codes (IntEnum):
    """Andor SDK2 드라이버 함수의 반환 코드 (Error codes for sdk2).

    모든 SDK 함수는 unsigned int 형태의 반환 코드를 반환한다.
    DRV_SUCCESS(20002)가 정상 완료를 의미하며, 그 외는 오류 상태를 나타낸다.
    """
    DRV_ERROR_CODES = 20001              # 에러 코드 기준값
    DRV_SUCCESS = 20002                  # 함수 정상 완료
    DRV_VXDNOTINSTALLED = 20003          # VxD 드라이버 미설치
    DRV_ERROR_SCAN = 20004               # 스캔 오류
    DRV_ERROR_CHECK_SUM = 20005          # 체크섬 오류
    DRV_ERROR_FILELOAD = 20006           # 파일 로드 오류
    DRV_UNKNOWN_FUNCTION = 20007         # 알 수 없는 함수 호출
    DRV_ERROR_VXD_INIT = 20008           # VxD 초기화 오류
    DRV_ERROR_ADDRESS = 20009            # 주소 오류
    DRV_ERROR_PAGELOCK = 20010           # 페이지 잠금 오류
    DRV_ERROR_PAGEUNLOCK = 20011         # 페이지 잠금 해제 오류
    DRV_ERROR_BOARDTEST = 20012          # 보드 테스트 오류
    DRV_ERROR_ACK = 20013                # 응답(ACK) 오류
    DRV_ERROR_UP_FIFO = 20014            # 업스트림 FIFO 오류
    DRV_ERROR_PATTERN = 20015            # 패턴 오류

    DRV_ACQUISITION_ERRORS = 20017       # 획득 오류 기준값
    DRV_ACQ_BUFFER = 20018               # 획득 버퍼 오류
    DRV_ACQ_DOWNFIFO_FULL = 20019        # 다운스트림 FIFO 가득 참
    DRV_PROC_UNKONWN_INSTRUCTION = 20020 # 알 수 없는 프로세서 명령
    DRV_ILLEGAL_OP_CODE = 20021          # 잘못된 연산 코드
    DRV_KINETIC_TIME_NOT_MET = 20022     # 키네틱 시간 조건 미충족
    DRV_ACCUM_TIME_NOT_MET = 20023       # 누적 시간 조건 미충족
    DRV_NO_NEW_DATA = 20024              # 새로운 데이터 없음
    DRV_PCI_DMA_FAIL = 20025             # PCI DMA 전송 실패
    DRV_SPOOLERROR = 20026               # 스풀 오류
    DRV_SPOOLSETUPERROR = 20027          # 스풀 설정 오류
    DRV_FILESIZELIMITERROR = 20028       # 파일 크기 한도 초과
    DRV_ERROR_FILESAVE = 20029           # 파일 저장 오류

    DRV_TEMPERATURE_CODES = 20033        # 온도 코드 기준값
    DRV_TEMPERATURE_OFF = 20034          # 냉각 꺼짐
    DRV_TEMPERATURE_NOT_STABILIZED = 20035  # 온도 안정화 중
    DRV_TEMPERATURE_STABILIZED = 20036   # 온도 안정화 완료
    DRV_TEMPERATURE_NOT_REACHED = 20037  # 목표 온도 미도달
    DRV_TEMPERATURE_OUT_RANGE = 20038    # 온도 범위 초과
    DRV_TEMPERATURE_NOT_SUPPORTED = 20039  # 온도 제어 미지원
    DRV_TEMPERATURE_DRIFT = 20040        # 온도 드리프트 발생

    # DRV_TEMP_* 는 DRV_TEMPERATURE_* 의 단축 별칭
    DRV_TEMP_CODES = 20033
    DRV_TEMP_OFF = 20034
    DRV_TEMP_NOT_STABILIZED = 20035
    DRV_TEMP_STABILIZED = 20036
    DRV_TEMP_NOT_REACHED = 20037
    DRV_TEMP_OUT_RANGE = 20038
    DRV_TEMP_NOT_SUPPORTED = 20039
    DRV_TEMP_DRIFT = 20040

    DRV_GENERAL_ERRORS = 20049           # 일반 오류 기준값
    DRV_INVALID_AUX = 20050              # 잘못된 보조 파라미터
    DRV_COF_NOTLOADED = 20051            # COF 파일 미로드
    DRV_FPGAPROG = 20052                 # FPGA 프로그래밍 오류
    DRV_FLEXERROR = 20053                # Flex 오류
    DRV_GPIBERROR = 20054                # GPIB 통신 오류
    DRV_EEPROMVERSIONERROR = 20055       # EEPROM 버전 오류
    DRV_DATATYPE = 20064                 # 데이터 타입 오류

    DRV_DRIVER_ERRORS = 20065            # 드라이버 오류 기준값
    DRV_P1INVALID = 20066                # 첫 번째 파라미터 오류
    DRV_P2INVALID = 20067                # 두 번째 파라미터 오류
    DRV_P3INVALID = 20068                # 세 번째 파라미터 오류
    DRV_P4INVALID = 20069                # 네 번째 파라미터 오류
    DRV_INIERROR = 20070                 # INI 파일 오류
    DRV_COFERROR = 20071                 # COF 파일 오류
    DRV_ACQUIRING = 20072                # 획득 진행 중 (다른 작업 불가)
    DRV_IDLE = 20073                     # 카메라 유휴 상태
    DRV_TEMPCYCLE = 20074                # 온도 사이클 진행 중
    DRV_NOT_INITIALIZED = 20075          # 드라이버 초기화 안 됨
    DRV_P5INVALID = 20076                # 다섯 번째 파라미터 오류
    DRV_P6INVALID = 20077                # 여섯 번째 파라미터 오류
    DRV_INVALID_MODE = 20078             # 잘못된 모드
    DRV_INVALID_FILTER = 20079           # 잘못된 필터
    DRV_I2CERRORS = 20080                # I2C 통신 오류
    DRV_I2CDEVNOTFOUND = 20081           # I2C 장치 없음
    DRV_I2CTIMEOUT = 20082               # I2C 타임아웃
    DRV_P7INVALID = 20083                # 일곱 번째 파라미터 오류
    DRV_P8INVALID = 20084                # 여덟 번째 파라미터 오류
    DRV_P9INVALID = 20085                # 아홉 번째 파라미터 오류
    DRV_P10INVALID = 20086               # 열 번째 파라미터 오류
    DRV_P11INVALID = 20087               # 열한 번째 파라미터 오류
    DRV_USBERROR = 20089                 # USB 통신 오류
    DRV_IOCERROR = 20090                 # IOC 오류
    DRV_VRMVERSIONERROR = 20091          # VRM 버전 오류
    DRV_GATESTEPERROR = 20092            # 게이트 스텝 오류
    DRV_USB_INTERRUPT_ENDPOINT_ERROR = 20093  # USB 인터럽트 엔드포인트 오류
    DRV_RANDOM_TRACK_ERROR = 20094       # 랜덤 트랙 오류
    DRV_INVALID_TRIGGER_MODE = 20095     # 잘못된 트리거 모드
    DRV_LOAD_FIRMWARE_ERROR = 20096      # 펌웨어 로드 오류
    DRV_DIVIDE_BY_ZERO_ERROR = 20097     # 0 나누기 오류
    DRV_INVALID_RINGEXPOSURES = 20098    # 잘못된 링 노출 설정
    DRV_BINNING_ERROR = 20099            # 비닝 설정 오류
    DRV_INVALID_AMPLIFIER = 20100        # 잘못된 증폭기 선택
    DRV_INVALID_COUNTCONVERT_MODE = 20101  # 잘못된 계수 변환 모드
    DRV_USB_INTERRUPT_ENDPOINT_TIMEOUT = 20102  # USB 인터럽트 엔드포인트 타임아웃

    DRV_ERROR_NOCAMERA = 20990           # 카메라 없음 또는 연결 안 됨
    DRV_NOT_SUPPORTED = 20991            # 현재 카메라에서 지원하지 않는 기능
    DRV_NOT_AVAILABLE = 20992            # 현재 상태에서 사용 불가

    DRV_ERROR_MAP = 20115                # 메모리 매핑 오류
    DRV_ERROR_UNMAP = 20116              # 메모리 매핑 해제 오류
    DRV_ERROR_MDL = 20117                # MDL 오류
    DRV_ERROR_UNMDL = 20118              # MDL 해제 오류
    DRV_ERROR_BUFFSIZE = 20119           # 버퍼 크기 오류
    DRV_ERROR_NOHANDLE = 20121           # 유효한 핸들 없음

    DRV_GATING_NOT_AVAILABLE = 20130     # 게이팅 기능 미지원
    DRV_FPGA_VOLTAGE_ERROR = 20131       # FPGA 전압 오류

    DRV_OW_CMD_FAIL = 20150              # 1-Wire 명령 실패
    DRV_OWMEMORY_BAD_ADDR = 20151        # 1-Wire 메모리 잘못된 주소
    DRV_OWCMD_NOT_AVAILABLE = 20152      # 1-Wire 명령 사용 불가
    DRV_OW_NO_SLAVES = 20153             # 1-Wire 슬레이브 장치 없음
    DRV_OW_NOT_INITIALIZED = 20154       # 1-Wire 초기화 안 됨
    DRV_OW_ERROR_SLAVE_NUM = 20155       # 1-Wire 슬레이브 번호 오류
    DRV_MSTIMINGS_ERROR = 20156          # 밀리초 타이밍 오류

    DRV_OA_NULL_ERROR = 20173            # OptAcquire NULL 오류
    DRV_OA_PARSE_DTD_ERROR = 20174       # OptAcquire DTD 파싱 오류
    DRV_OA_DTD_VALIDATE_ERROR = 20175    # OptAcquire DTD 검증 오류
    DRV_OA_FILE_ACCESS_ERROR = 20176     # OptAcquire 파일 접근 오류
    DRV_OA_FILE_DOES_NOT_EXIST = 20177   # OptAcquire 파일 없음
    DRV_OA_XML_INVALID_OR_NOT_FOUND_ERROR = 20178  # OptAcquire XML 오류
    DRV_OA_PRESET_FILE_NOT_LOADED = 20179   # OptAcquire 프리셋 파일 미로드
    DRV_OA_USER_FILE_NOT_LOADED = 20180     # OptAcquire 사용자 파일 미로드
    DRV_OA_PRESET_AND_USER_FILE_NOT_LOADED = 20181  # OptAcquire 프리셋/사용자 파일 모두 미로드
    DRV_OA_INVALID_FILE = 20182          # OptAcquire 잘못된 파일
    DRV_OA_FILE_HAS_BEEN_MODIFIED = 20183  # OptAcquire 파일이 수정됨
    DRV_OA_BUFFER_FULL = 20184           # OptAcquire 버퍼 가득 참
    DRV_OA_INVALID_STRING_LENGTH = 20185 # OptAcquire 잘못된 문자열 길이
    DRV_OA_INVALID_CHARS_IN_NAME = 20186 # OptAcquire 이름에 잘못된 문자 포함
    DRV_OA_INVALID_NAMING = 20187        # OptAcquire 잘못된 이름 형식
    DRV_OA_GET_CAMERA_ERROR = 20188      # OptAcquire 카메라 정보 조회 오류
    DRV_OA_MODE_ALREADY_EXISTS = 20189   # OptAcquire 모드 이름 중복
    DRV_OA_STRINGS_NOT_EQUAL = 20190     # OptAcquire 문자열 불일치
    DRV_OA_NO_USER_DATA = 20191          # OptAcquire 사용자 데이터 없음
    DRV_OA_VALUE_NOT_SUPPORTED = 20192   # OptAcquire 지원하지 않는 값
    DRV_OA_MODE_DOES_NOT_EXIST = 20193   # OptAcquire 모드 없음
    DRV_OA_CAMERA_NOT_SUPPORTED = 20194  # OptAcquire 카메라 미지원
    DRV_OA_FAILED_TO_GET_MODE = 20195    # OptAcquire 모드 조회 실패
    DRV_OA_CAMERA_NOT_AVAILABLE = 20196  # OptAcquire 카메라 사용 불가

    DRV_PROCESSING_FAILED = 20211        # 데이터 처리 실패
