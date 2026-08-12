from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors, CameraCapabilities

# --- 라이브러리 및 카메라 초기화 ---
sdk = atmcd()  # atmcd DLL 로드 및 SDK 객체 생성
codes = atmcd_codes

ret = sdk.Initialize("")  # 카메라 드라이버 초기화
helper = CameraCapabilities.CapabilityHelper(sdk)  # 카메라 기능 조회 헬퍼 생성
print("Function Initialize returned {}".format(ret))

if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:

    # --- 카메라 시리얼 번호 조회 ---
    (ret, iSerialNumber) = sdk.GetCameraSerialNumber()
    print("Function GetCameraSerialNumber returned {} Serial No: {}".format(
        ret, iSerialNumber))

    # --- 카메라 지원 기능 출력 ---
    helper.print_acquisition_modes()   # 지원하는 획득 모드 목록 출력
    helper.print_get_functions()       # 조회 가능한 기능 목록 출력
    helper.print_read_modes()          # 지원하는 읽기 모드 목록 출력
    helper.print_FTRead_modes()        # 프레임 전송 호환 읽기 모드 출력

    # --- 정리: 카메라 드라이버 종료 ---
    ret = sdk.ShutDown()
    print("Function Shutdown returned {}".format(ret))

else:
    print("Cannot continue, could not initialise camera")
