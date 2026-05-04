from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors

# --- 라이브러리 및 카메라 초기화 (USB iStar ICCD) ---
sdk = atmcd()  # atmcd DLL 로드 및 SDK 객체 생성
codes = atmcd_codes

ret = sdk.Initialize("")  # 카메라 드라이버 초기화
print("Function Initialize returned {}".format(ret))

if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:

    # --- 카메라 시리얼 번호 조회 ---
    (ret, iSerialNumber) = sdk.GetCameraSerialNumber()
    print("Function GetCameraSerialNumber returned {} Serial No {}".format(
        ret, iSerialNumber))

    # --- 기본 획득 파라미터 설정 ---
    ret = sdk.SetAcquisitionMode(codes.Acquisition_Mode.SINGLE_SCAN)  # 단일 스캔 모드
    print("Function SetAcquisitionMode returned {} mode = Single Scan".format(ret))

    ret = sdk.SetReadMode(codes.Read_Mode.IMAGE)  # 전체 2D 이미지 읽기 모드
    print("Function SetReadMode returned {} mode = FVB".format(ret))

    ret = sdk.SetTriggerMode(codes.Trigger_Mode.INTERNAL)  # 내부 트리거 사용
    print("Function SetTriggerMode returned {} mode = Internal".format(ret))

    (ret, xpixels, ypixels) = sdk.GetDetector()  # 검출기 픽셀 수 조회
    print("Function GetDetector returned {} xpixels = {} ypixels = {}".format(
        ret, xpixels, ypixels))

    ret = sdk.SetExposureTime(0.01)  # 노출 시간 0.01초 설정
    print("Function SetExposureTime returned {} time = 0.01s".format(ret))

    # --- DDG(디지털 지연 발생기) 게이팅 설정 (iStar ICCD 전용) ---
    ret = sdk.SetGateMode(codes.Gate_Mode.GATE_USING_DDG)  # DDG 제어 게이팅 모드
    print("Function SetGateMode returned {} mode = Gate using DDG".format(ret))

    # 게이트 지연 0ns, 게이트 폭 1ms(1,000,000,000 ps) 설정
    ret = sdk.SetDDGGateTime(0, 1000000000)
    print("Function SetDDGGateTime returned {} gate width = 1 ms".format(ret))

    ret = sdk.SetDDGExternalOutputEnabled(0, 1)  # DDG 외부 출력 A 활성화
    print("Function SetDDGExternalOutputEnabled returned {} Output A enabled".format(ret))

    # 외부 출력 A: 지연 1ms, 폭 2ms 설정
    ret = sdk.SetDDGExternalOutputTime(0, 1000000000, 2000000000)
    print("Function SetDDGExternalOutputTime returned {} output delay = 1 ms, width = 2ms".format(ret))

    ret = sdk.SetDDGInsertionDelay(1)   # 삽입 지연 모드: 1 = Fast
    print("Function SetDDGInsertionDelay returned {} mode = Fast".format(ret))

    ret = sdk.SetDDGIntelligate(1)      # MCP 인텔리게이트(MCP 게이팅) 활성화
    print("Function SetDDGIntelligate returned {} mode = MCP gating ON".format(ret))

    ret = sdk.SetMCPGain(10)            # MCP(마이크로채널판) 이득 10으로 설정
    print("Function SetMCPGain returned {} gain = 10".format(ret))

    (ret, fminExposure, fAccumulate, fKinetic) = sdk.GetAcquisitionTimings()
    print("Function GetAcquisitionTimings returned {} exposure = {} accumulate = {} kinetic = {}".format(
        ret, fminExposure, fAccumulate, fKinetic))

    ret = sdk.PrepareAcquisition()  # 획득 전 카메라 사전 준비
    print("Function PrepareAcquisition returned {}".format(ret))

    # --- 이미지 획득 ---
    ret = sdk.StartAcquisition()  # 획득 시작
    print("Function StartAcquisition returned {}".format(ret))

    ret = sdk.WaitForAcquisition()  # 획득 완료까지 블로킹 대기
    print("Function WaitForAcquisition returned {}".format(ret))

    # 16비트 이미지 데이터 읽기 (FVB 모드이므로 xpixels만큼)
    imageSize = xpixels
    (ret, arr, validfirst, validlast) = sdk.GetImages16(1, 1, imageSize)
    print("Function GetImages16 returned {} first pixel = {} size = {}".format(
        ret, arr[0], imageSize))

    # --- 형광체 상태 조회 (iStar 전용) ---
    (ret, status) = sdk.GetPhosphorStatus()
    print("Function GetPhosphorStatus returned {} status = {}".format(ret, status))

    # --- 정리: 카메라 드라이버 종료 ---
    ret = sdk.ShutDown()
    print("Function Shutdown returned {}".format(ret))

else:
    print("Cannot continue, could not initialize camera")
