import os
import time
from backend.tools.hw_tools.SDKs.andor_codes.pyAndorSDK2 import atmcd_codes, atmcd_errors
from pyAndorSDK2 import atmcd

# --- 라이브러리 및 카메라 초기화 ---
sdk = atmcd()  # atmcd DLL 로드 및 SDK 객체 생성
codes = atmcd_codes
ret = sdk.Initialize("")  # 카메라 드라이버 초기화
print("Function Initialize returned {}".format(ret))

if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
    NUMKIN = 10  # 취득할 키네틱 프레임 수

    # --- 획득 모드 설정 (키네틱 시리즈 + 스풀링) ---
    ret = sdk.SetAcquisitionMode(codes.Acquisition_Mode.KINETICS)  # 키네틱 시리즈 모드
    print("Function SetAcquisitionMode returned {} mode = Kinetics".format(ret))

    ret = sdk.SetKineticCycleTime(0.5)  # 키네틱 사이클 간격 0.5초 설정
    print("Function SetKineticCycleTime returned {} cycle time = 0.5 seconds".format(ret))

    ret = sdk.SetNumberKinetics(NUMKIN)  # 총 키네틱 프레임 수(10) 설정
    print("Function SetNumberKinetics returned {}".format(ret))

    ret = sdk.SetTriggerMode(codes.Trigger_Mode.SOFTWARE_TRIGGER)  # 소프트웨어 트리거 사용
    print("Function SetTriggerMode returned {} mode = Software trigger".format(ret))

    ret = sdk.SetReadMode(codes.Read_Mode.IMAGE)  # 전체 2D 이미지 읽기 모드
    print("Function SetReadMode returned {} mode = Image".format(ret))

    # --- 스풀 설정: 데이터를 실시간으로 파일에 저장 ---
    directory = os.getcwd()  # 현재 작업 디렉터리 (원하는 경로로 변경 가능)
    filename = "{}-{}".format(directory, time.strftime("%Y-%m-%d-%H-%M"))
    # 스풀 활성(1), 16비트 FITS 파일 형식, 파일 경로, 버퍼 크기(10프레임)
    ret = sdk.SetSpool(1, codes.Spool_Mode.SPOOL_TO_16_BIT_FITS, filename, 10)
    print("Function SetSpool returned {} ".format(ret))

    (ret, xpixels, ypixels) = sdk.GetDetector()  # 검출기 픽셀 수 조회
    print("Function GetDetector returned {} xpixels = {} ypixels = {}".format(
        ret, xpixels, ypixels))

    # 이미지 영역 설정: 비닝 없이 전체 검출기 사용
    ret = sdk.SetImage(1, 1, 1, xpixels, 1, ypixels)
    print("Function SetImage returned {} hbin = 1 vbin = 1 hstart = 1 hend = {} vstart = 1 vend = {}".format(
        ret, xpixels, ypixels))

    ret = sdk.SetExposureTime(0.5)  # 노출 시간 0.5초 설정
    print("Function SetExposureTime returned {} time = 0.5s".format(ret))

    # --- 키네틱 시리즈 획득 시작 ---
    ret = sdk.StartAcquisition()
    print("Function StartAcquisition returned {} ".format(ret))

    # --- 모든 프레임이 획득될 때까지 폴링 대기 ---
    index = 0
    while index < NUMKIN:
        (ret, index) = sdk.GetTotalNumberImagesAcquired()  # 지금까지 획득된 프레임 수 조회
        print("Function current count {} ".format(index), end="\r")
    print("")

    # --- 정리: 카메라 드라이버 종료 ---
    ret = sdk.ShutDown()
    print("Function Shutdown returned {}".format(ret))

else:
    print("Cannot continue, could not initialise camera")
