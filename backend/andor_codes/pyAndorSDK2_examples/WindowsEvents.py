import win32event  # pip install pyWin32 로 설치
from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors

# --- 라이브러리 및 카메라 초기화 ---
sdk = atmcd()  # atmcd DLL 로드 및 SDK 객체 생성
codes = atmcd_codes

(ret) = sdk.Initialize("")  # 카메라 드라이버 초기화
print("Function Initialize returned {}".format(ret))

if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:

    # --- 카메라 시리얼 번호 조회 ---
    (ret, iSerialNumber) = sdk.GetCameraSerialNumber()
    print("Function GetCameraSerialNumber returned {} Serial No: {}".format(
        ret, iSerialNumber))

    # --- 획득 파라미터 설정 ---
    ret = sdk.CoolerON()  # 열전냉각 소자(TEC) 작동 시작
    print("Function CoolerON returned {}".format(ret))

    ret = sdk.SetAcquisitionMode(codes.Acquisition_Mode.SINGLE_SCAN)  # 단일 스캔 모드
    print("Function SetAcquisitionMode returned {} mode = Single Scan".format(ret))

    ret = sdk.SetReadMode(codes.Read_Mode.IMAGE)  # 전체 2D 이미지 읽기 모드
    print("Function SetReadMode returned {} mode = Image".format(ret))

    ret = sdk.SetTriggerMode(codes.Trigger_Mode.INTERNAL)  # 내부 트리거 사용
    print("Function SetTriggerMode returned {} mode = Internal".format(ret))

    (ret, xpixels, ypixels) = sdk.GetDetector()  # 검출기 픽셀 수 조회
    print("Function GetDetector returned {} xpixels = {} ypixels = {}".format(
        ret, xpixels, ypixels))

    # 이미지 영역 설정: 비닝 없이 전체 검출기 사용
    ret = sdk.SetImage(1, 1, 1, xpixels, 1, ypixels)
    print("Function SetImage returned {} hbin = 1 vbin = 1 hstart = 1 hend = {} vstart = 1 vend = {}".format(
        ret, xpixels, ypixels))

    ret = sdk.SetExposureTime(2)  # 노출 시간 2초 설정
    print("Function SetExposureTime returned {} time = 2s".format(ret))

    (ret, fminExposure, fAccumulate, fKinetic) = sdk.GetAcquisitionTimings()
    print("Function GetAcquisitionTimings returned {} exposure = {} accumulate = {} kinetic = {}".format(
        ret, fminExposure, fAccumulate, fKinetic))

    ret = sdk.PrepareAcquisition()  # 획득 전 카메라 사전 준비
    print("Function PrepareAcquisition returned {}".format(ret))

    # --- Windows 이벤트 기반 획득 완료 감지 설정 ---
    event = win32event.CreateEvent(None, 0, 0, None)  # 자동 리셋 이벤트 핸들 생성
    ret = sdk.SetDriverEvent(event)  # 드라이버에 이벤트 핸들 등록 (획득 완료 시 신호 발생)
    print("Function SetDriverEvent returned {}".format(ret))

    # --- 이미지 획득 시작 ---
    ret = sdk.StartAcquisition()
    print("Function StartAcquisition returned {}".format(ret))

    # WaitForSingleObject: 이벤트 신호가 올 때까지 무한 대기 (폴링보다 CPU 효율적)
    print("Enter WaitForSingleObject")
    ret = win32event.WaitForSingleObject(event, win32event.INFINITE)
    print("Function WaitForSingleObject returned {}".format(ret))

    # --- 획득된 이미지 데이터 읽기 ---
    imageSize = xpixels * ypixels
    (ret, arr, validfirst, validlast) = sdk.GetImages16(1, 1, imageSize)
    print("Function GetImages16 returned {} first pixel = {} size = {}".format(
        ret, arr[0], imageSize))

    # --- 정리: 카메라 드라이버 종료 ---
    (ret) = sdk.ShutDown()
    print("Function Shutdown returned {}".format(ret))

else:
    print("Cannot continue, could not initialise camera")
