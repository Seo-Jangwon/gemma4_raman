from backend.tools.hw_tools.SDKs.andor_codes.pyAndorSDK2 import atmcd_errors
from pyAndorSDK2 import atmcd

# --- 라이브러리 및 카메라 초기화 ---
sdk = atmcd()  # atmcd DLL 로드 및 SDK 객체 생성
ret = sdk.Initialize("")  # 카메라 드라이버 초기화
print("Function Initialize returned {}".format(ret))

if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
    HSSpeeds = []   # 수평 읽기 속도(MHz) 목록
    VSSpeeds = []   # 수직 읽기 속도(µs) 목록
    amp_modes = []  # 증폭기 모드 이름 목록

    # --- AD 채널 수 조회 ---
    (ret, ADchannel) = sdk.GetNumberADChannels()
    print("Function GetNumberADChannels returned {} number of available channels {}".format(
        ret, ADchannel))

    for channel in range(0, ADchannel):
        # --- 수평 읽기 속도(HSSpeed) 목록 조회 ---
        (ret, speed) = sdk.GetNumberHSSpeeds(channel, 0)
        print("Function GetNumberHSSpeeds {} number of available speeds {}".format(
            ret, speed))
        for x in range(0, speed):
            (ret, speed) = sdk.GetHSSpeed(channel, 0, x)  # 각 인덱스의 HSSpeed(MHz) 조회
            HSSpeeds.append(speed)

        print("Available HSSpeeds in MHz {} ".format(HSSpeeds))

        # --- 수직 읽기 속도(VSSpeed) 목록 조회 ---
        (ret, speed) = sdk.GetNumberVSSpeeds()
        print("Function GetNumberVSSpeeds {} number of available speeds {}".format(
            ret, speed))
        for x in range(0, speed):
            (ret, speed) = sdk.GetVSSpeed(x)  # 각 인덱스의 VSSpeed(µs/행) 조회
            VSSpeeds.append(speed)
        print("Available VSSpeeds in us {}".format(VSSpeeds))

        # --- 권장 최고속 수직 읽기 속도 조회 ---
        (ret, index, speed) = sdk.GetFastestRecommendedVSSpeed()
        print("Recommended VSSpeed {} index {}".format(speed, index))

        # --- 출력 증폭기(앰프) 목록 조회 ---
        (ret, amps) = sdk.GetNumberAmp()
        print("Function GetNumberAmp returned {} number of amplifiers {}".format(ret, amps))
        for x in range(0, amps):
            (ret, name) = sdk.GetAmpDesc(x, 21)  # 증폭기 설명 문자열 조회 (최대 21자)
            amp_modes.append(name)

        print("Available amplifier modes {}".format(amp_modes))

    # --- 정리: 카메라 드라이버 종료 ---
    ret = sdk.ShutDown()
    print("Function ShutDown returned {}".format(ret))

else:
    print("Cannot continue, could not initialise camera {}".format(ret))
