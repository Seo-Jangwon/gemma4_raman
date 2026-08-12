import time
import numpy as np
import matplotlib.pyplot as plt
from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors

# --- 라이브러리 및 카메라 초기화 ---
sdk = atmcd()  # atmcd DLL 로드 및 SDK 객체 생성
codes = atmcd_codes
ret = sdk.Initialize("")  # 카메라 드라이버 초기화
print("Function Initialize returned {}".format(ret))

if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
    # --- 냉각 설정 및 온도 안정화 대기 ---
    ret = sdk.SetTemperature(-60)  # 목표 온도 -60°C 설정
    print("Function SetTemperature returned {} target temperature -60".format(ret))

    ret = sdk.CoolerON()  # 열전냉각 소자(TEC) 작동 시작
    print("Function CoolerOn returned {}".format(ret))

    # DRV_TEMP_STABILIZED 반환 시까지 5초 간격으로 온도 폴링
    while ret != atmcd_errors.Error_Codes.DRV_TEMP_STABILIZED:
        time.sleep(5)
        (ret, temperature) = sdk.GetTemperature()
        print("Function GetTemperature returned {} current temperature = {}".format(
            ret, temperature), end="\r")

    print("")
    print("Temperature stabilized")

    # --- 획득 파라미터 설정 (Run Till Abort 모드) ---
    ret = sdk.SetAcquisitionMode(codes.Acquisition_Mode.RUN_TILL_ABORT)  # 중단 시까지 연속 획득
    print("Function SetAcquisitionMode returned {} mode = Run Till Abort".format(ret))

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

    ret = sdk.SetExposureTime(0.2)  # 노출 시간 0.2초 설정
    print("Function SetExposureTime returned {} time = 0.2s".format(ret))

    # --- 연속 획득 시작 ---
    ret = sdk.StartAcquisition()  # 획득 시작 (Run Till Abort: 명시적 AbortAcquisition 전까지 지속)
    print("Function StartAcquisition returned {}".format(ret))

    imageSize = xpixels * ypixels
    fig, imgwindow = plt.subplots()  # matplotlib 이미지 표시 창 생성

    # --- 실시간 이미지 표시 루프 ---
    while True:
        # 창이 열려 있는 동안 가장 최근 프레임을 가져와 표시
        try:
            if plt.fignum_exists(fig.number):
                imgwindow.cla()  # 이전 이미지 지우기
                (ret, index) = sdk.GetTotalNumberImagesAcquired()  # 지금까지 획득된 총 프레임 수
                (ret, arr) = sdk.GetMostRecentImage16(imageSize)    # 가장 최근 16비트 이미지 가져오기
                print("Function GetMostRecentImage16 returned {} first pixel = {} size = {}".format(
                    ret, arr[0], imageSize), end="\r")
                arr = np.reshape(arr, (xpixels, ypixels))  # 1D 배열을 2D 이미지로 재구성
                imgwindow.matshow(arr)
                imgwindow.set_title("Image number: {}".format(index))
                plt.pause(0.1)  # 0.1초 대기하며 GUI 이벤트 처리
            else:
                raise Exception(
                    "can't invoke " '"update"' " command: application has been destroyed")

        except Exception as e:
            # 창이 닫히거나 오류 발생 시 루프 종료
            plt.close()
            print("")
            print(e)
            break

    # --- 정리: 카메라 드라이버 종료 ---
    print("")
    ret = sdk.ShutDown()
    print("Function Shutdown returned {}".format(ret))

else:
    print("Cannot continue, could not initialise camera {}".format(ret))
