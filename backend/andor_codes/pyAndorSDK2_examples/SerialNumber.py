from pyAndorSDK2 import atmcd, atmcd_errors

# --- 라이브러리 및 카메라 초기화 ---
sdk = atmcd("")  # atmcd DLL 로드 및 SDK 객체 생성 (경로 인자로 DLL 위치 지정 가능)

ret = sdk.Initialize("")  # 카메라 드라이버 초기화
print("Function Initialize returned {}".format(ret))

if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:

    # --- 카메라 시리얼 번호 조회 및 출력 ---
    (ret, iSerialNumber) = sdk.GetCameraSerialNumber()
    print("Function GetCameraSerialNumber returned {} Serial No: {}".format(
        ret, iSerialNumber))

    # --- 정리: 카메라 드라이버 종료 ---
    ret = sdk.ShutDown()
    print("Function Shutdown returned {}".format(ret))

else:
    print("Cannot continue, could not initialise camera")
