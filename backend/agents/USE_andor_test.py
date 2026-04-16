import ctypes
import time
import sys
import os

# Andor SDK 상수 정의 (C# 코드의 상수 매핑)
DRV_SUCCESS = 20002
DRV_ACQUIRING = 20007
DRV_IDLE = 20073
DRV_TEMP_OFF = 20034
DRV_TEMP_STABILIZED = 20036
DRV_TEMP_NOT_REACHED = 20037
DRV_TEMP_DRIFT = 20040
DRV_TEMP_NOT_STABILIZED = 20035

# Read Modes
READ_MODE_FVB = 0
READ_MODE_MULTI_TRACK = 1
READ_MODE_RANDOM_TRACK = 2
READ_MODE_SINGLE_TRACK = 3
READ_MODE_IMAGE = 4

# Acquisition Modes
ACQ_MODE_SINGLE = 1
ACQ_MODE_ACCUMULATE = 2

# Trigger Modes
TRIGGER_MODE_INTERNAL = 0
TRIGGER_MODE_EXTERNAL = 1
TRIGGER_MODE_EXTERNAL_EXPOSURE = 7

class AndorCamera:
    def __init__(self, dll_path):
        # 1. DLL 로드
        try:
            self.dll = ctypes.cdll.LoadLibrary(dll_path)
            print(f"[INFO] DLL Loaded: {dll_path}")
        except OSError as e:
            print(f"[ERROR] DLL Load Failed. 경로를 확인하세요: {e}")
            sys.exit(1)
            
        self.width = 0
        self.height = 0

    def check_error(self, error_code, func_name):
        if error_code != DRV_SUCCESS:
            print(f"[ERROR] {func_name} failed with code: {error_code}")
            return False
        return True

    def initialize(self, config_dir):
        # 초기화 (Detector.ini가 있는 폴더 경로)
        # C#의 AndorSdk.Initialize(p.Dir) 대응
        c_dir = ctypes.create_string_buffer(config_dir.encode('utf-8'))
        ret = self.dll.Initialize(c_dir)
        
        if not self.check_error(ret, "Initialize"):
            return False
            
        print("[INFO] Camera Initialized.")
        
        # 센서 크기 가져오기
        w = ctypes.c_int()
        h = ctypes.c_int()
        self.dll.GetDetector(ctypes.byref(w), ctypes.byref(h))
        self.width = w.value
        self.height = h.value
        print(f"[INFO] Detector Size: {self.width} x {self.height}")
        return True

    def setup_acquisition(self, read_mode, exposure_time, trigger_mode, gain=0):
        # 1. Read Mode 설정 (FVB vs Image)
        self.check_error(self.dll.SetReadMode(read_mode), "SetReadMode")
        
        # 2. Acquisition Mode 설정 (Single Scan)
        self.check_error(self.dll.SetAcquisitionMode(ACQ_MODE_SINGLE), "SetAcquisitionMode")
        
        # 3. Trigger Mode 설정 (External vs Internal)
        self.check_error(self.dll.SetTriggerMode(trigger_mode), "SetTriggerMode")
        
        # 4. 노출 시간 설정
        self.check_error(self.dll.SetExposureTime(ctypes.c_float(exposure_time)), "SetExposureTime")
        
        # 5. MCP Gain (Intensifier) 설정 - 필요한 경우
        if gain > 0:
            # MCP Gating On
            self.check_error(self.dll.SetMCPGating(1), "SetMCPGating") 
            self.check_error(self.dll.SetMCPGain(gain), "SetMCPGain")
            print(f"[SETTING] MCP Gain set to {gain}")
            
        print(f"[SETTING] Setup Complete: ReadMode={read_mode}, Trigger={trigger_mode}, Exp={exposure_time}s")

    def start_acquisition_cycle(self):
        # 촬영 시작
        print("[ACTION] Waiting for Trigger...")
        ret = self.dll.StartAcquisition()
        if not self.check_error(ret, "StartAcquisition"):
            return None

        # 촬영 완료 대기 (Blocking)
        # C#의 AndorSdk.WaitForAcquisition() 대응
        ret = self.dll.WaitForAcquisition()
        if not self.check_error(ret, "WaitForAcquisition"):
            return None
            
        print("[ACTION] Acquisition Finished. Reading Data...")
        
        # 데이터 크기 계산
        # FVB 모드면 Width만큼, Image 모드면 Width * Height 만큼
        # 현재 설정된 ReadMode를 확인해야 정확하지만, 편의상 size를 넉넉히 잡거나 문맥에 맞게 처리
        # 여기서는 가장 큰 버퍼인 전체 이미지 크기로 할당
        size = self.width * self.height
        data_buffer = (ctypes.c_long * size)() # 32-bit integer array
        
        # 데이터 가져오기
        # C#의 AndorSdk.GetAcquiredData(data, npx) 대응
        ret = self.dll.GetAcquiredData(data_buffer, size)
        
        if ret == DRV_SUCCESS:
            # ctypes array를 파이썬 리스트로 변환
            return list(data_buffer)
        else:
            print(f"[ERROR] GetAcquiredData failed: {ret}")
            return None

    def set_temperature(self, target_celsius: int) -> bool:
        """목표 온도 설정 (정수 °C)."""
        ret = self.dll.SetTemperature(ctypes.c_int(target_celsius))
        return self.check_error(ret, f"SetTemperature({target_celsius})")

    def cooler_on(self) -> bool:
        """냉각기 ON."""
        ret = self.dll.CoolerON()
        return self.check_error(ret, "CoolerON")

    def cooler_off(self) -> bool:
        """냉각기 OFF."""
        ret = self.dll.CoolerOFF()
        return self.check_error(ret, "CoolerOFF")

    def get_temperature(self) -> tuple[int, int]:
        """
        현재 온도 조회.
        반환: (status_code, temperature_celsius)
        status_code: DRV_TEMP_STABILIZED(20036), DRV_TEMP_NOT_REACHED(20037),
                     DRV_TEMP_DRIFT(20040), DRV_TEMP_NOT_STABILIZED(20035), DRV_TEMP_OFF(20034)
        """
        temp = ctypes.c_int()
        status = self.dll.GetTemperature(ctypes.byref(temp))
        return status, temp.value

    def shutdown(self):
        self.dll.ShutDown()
        print("[INFO] Camera Shutdown.")

# ==========================================
# 메인 실행부 (CMD Interface)
# ==========================================
if __name__ == "__main__":
    # TODO: 실제 atmcd64d.dll 경로로 수정하세요
    dll_path = r"C:\Users\user\Desktop\gemma_raman\backend\agents" 
    
    # TODO: Detector.ini 파일이 있는 폴더 경로 (보통 Andor 설치 폴더)
    config_path = r"C:\Users\user\Desktop\gemma_raman\backend" 

    cam = AndorCamera(dll_path)

    if cam.initialize(config_path):
        try:
            # ---------------------------------------------------------
            # 시나리오: 외부 트리거를 받아 스펙트럼(FVB) 측정
            # ---------------------------------------------------------
            
            # 1. 설정: FVB 모드, 노출 0.01초, 외부 트리거, MCP 게인 100
            cam.setup_acquisition(
                read_mode=READ_MODE_FVB, 
                exposure_time=0.01, 
                trigger_mode=TRIGGER_MODE_EXTERNAL, 
                gain=100
            )

            # 2. 촬영 및 데이터 획득
            # 레이저가 쏴지고 트리거가 들어올 때까지 여기서 대기합니다.
            spectrum_data = cam.start_acquisition_cycle()

            if spectrum_data:
                # 3. 결과 출력 (상위 10개 픽셀만 예시로 출력)
                # FVB 모드이므로 데이터 길이는 Width(예: 1024 or 2048)와 같아야 함
                valid_data = spectrum_data[:cam.width] 
                print("\n[RESULT] Spectrum Data Snippet (First 10 pixels):")
                print(valid_data[:10])
                
                # CSV로 저장 예시
                with open("spectrum_output.csv", "w") as f:
                    f.write("Pixel,Count\n")
                    for i, val in enumerate(valid_data):
                        f.write(f"{i},{val}\n")
                print(f"[INFO] Data saved to spectrum_output.csv")

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            cam.shutdown()