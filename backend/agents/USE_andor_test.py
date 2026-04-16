import ctypes
import csv
import time
import sys
import os
from datetime import datetime
from pathlib import Path

import numpy as np

# ── 캘리브레이션 모듈 (같은 폴더 또는 PYTHONPATH) ───────────────────────────
# raman_calibration.py 이 같은 디렉토리에 있어야 함.
try:
    from raman_calibration import RamanCalibrator
except ImportError:
    RamanCalibrator = None  # 파일 없으면 raw 로만 동작

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
    def __init__(self, dll_path, calibrator=None):
        """
        Parameters
        ----------
        dll_path : str
            atmcd64d.dll / ATMCD64CS.dll 이 있는 폴더 경로.
        calibrator : RamanCalibrator | None
            pixel → Raman shift 변환기. None 이면 raw pixel 만 반환.
        """
        try:
            self.dll = ctypes.cdll.LoadLibrary(dll_path)
            print(f"[INFO] DLL Loaded: {dll_path}")
        except OSError as e:
            print(f"[ERROR] DLL Load Failed. 경로를 확인하세요: {e}")
            sys.exit(1)

        self.width = 0
        self.height = 0
        self.calibrator = calibrator

    # ------------------------------------------------------------------
    # ─── 캘리브레이션 주입/교체 ─────────────────────────────────────────
    # Grating motor 를 움직인 직후에는 calibrator 를 새 reference 로
    # 재생성해서 갈아끼우기만 하면 됨. 카메라 재초기화 불필요.
    # ------------------------------------------------------------------
    def set_calibrator(self, calibrator):
        self.calibrator = calibrator

    def _attach_axes(self, intensity_list):
        """raw intensity → dict (pixel, intensity, + optional axes)."""
        n = len(intensity_list)
        out = {
            "pixel": list(range(n)),
            "intensity": list(intensity_list),
            "calibrated": False,
        }
        if self.calibrator is not None and n == 1024:
            pixels = np.arange(n)
            out["raman_shift_cm-1"] = self.calibrator.pixel_to_raman_shift(pixels).tolist()
            out["wavelength_nm"] = self.calibrator.pixel_to_wavelength(pixels).tolist()
            out["laser_nm"] = self.calibrator.laser_nm
            out["calibrated"] = True
        elif self.calibrator is not None and n != 1024:
            print(f"[WARN] calibrator 는 1024 pixel 전제, 실제 {n} pixel → raw 만 반환")
        return out


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
            
            print(f"[SETTING] Setup Complete: ReadMode={read_mode}, "
              f"Trigger={trigger_mode}, Exp={exposure_time}s")

    def start_acquisition_cycle(self):
        """촬영 → 리턴: None(실패) 또는 dict (pixel/intensity + 보정 축)."""
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
        size = self.width
        data_buffer = (ctypes.c_long * size)()
        ret = self.dll.GetAcquiredData(data_buffer, size)
        if ret != DRV_SUCCESS:
            print(f"[ERROR] GetAcquiredData failed: {ret}")
            return None
        
        raw = list(data_buffer)
        return self._attach_axes(raw) 

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

# ======================================================================
# 공통 유틸: CSV 저장 (calibrated 여부 자동 분기)
# ======================================================================
def save_spectrum_csv(result: dict, path: str | Path):
    """calibrator 적용 여부와 무관하게 동작."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if result.get("calibrated"):
            # 메타 주석 (상용 파일 스타일을 약간 참고)
            f.write(f"# laser_nm,{result['laser_nm']}\n")
            f.write(f"# calibration,reference_csv_lut\n")
            w.writerow(["pixel", "raman_shift_cm-1", "wavelength_nm", "intensity"])
            for i, px in enumerate(result["pixel"]):
                w.writerow([
                    px,
                    f"{result['raman_shift_cm-1'][i]:.3f}",
                    f"{result['wavelength_nm'][i]:.4f}",
                    result["intensity"][i],
                ])
        else:
            w.writerow(["pixel", "intensity"])
            for px, inten in zip(result["pixel"], result["intensity"]):
                w.writerow([px, inten])
    print(f"[CSV] saved: {path}")


def load_calibrator(reference_csv: str | Path,
                    user_is_reversed: bool = True):
    """캘리브레이터 로드 + 실패시 명확히 None 반환."""
    if RamanCalibrator is None:
        print("[WARN] raman_calibration.py 없음 → raw 로 저장")
        return None
    ref = Path(reference_csv)
    if not ref.exists():
        print(f"[WARN] reference CSV 없음: {ref} → raw 로 저장")
        return None
    try:
        cal = RamanCalibrator.from_reference_csv(ref, user_is_reversed=user_is_reversed)
        print(f"[INFO] Calibration loaded from {ref.name}")
        print(f"       laser={cal.laser_nm}nm, "
              f"range={cal._lut.min():.0f}~{cal._lut.max():.0f} cm⁻¹, "
              f"reversed={user_is_reversed}")
        return cal
    except Exception as e:
        print(f"[WARN] Calibration load failed: {e} → raw 로 저장")
        return None


# ==========================================
# 메인 실행부
# ==========================================
if __name__ == "__main__":
    # TODO: 실제 경로로 수정
    dll_path = r"C:\Users\user\Desktop\gemma_raman\backend\agents"
    config_path = r"C:\Users\user\Desktop\gemma_raman\backend"

    # 상용 프로그램에서 현재 grating 위치로 한 번 찍어서 저장해둔 파일
    # Grating motor 를 움직이면 이 파일도 새로 만들어야 함
    reference_csv = r"C:\Users\user\Desktop\gemma_raman\backend\calibration\reference.csv"

    # ── 1. 캘리브레이터 준비 ─────────────────────────────────────────────
    calibrator = load_calibrator(reference_csv, user_is_reversed=True)

    # ── 2. 카메라 생성 + 주입 ────────────────────────────────────────────
    cam = AndorCamera(dll_path, calibrator=calibrator)

    if cam.initialize(config_path):
        try:
            cam.setup_acquisition(
                read_mode=READ_MODE_FVB,
                exposure_time=0.01,
                trigger_mode=TRIGGER_MODE_EXTERNAL,
                gain=100,
            )

            result = cam.start_acquisition_cycle()

            if result:
                # 결과 요약
                print("\n[RESULT] Spectrum Data (first 10 pixels):")
                print("  pixel intensity  " + ("raman_shift" if result["calibrated"] else ""))
                for i in range(10):
                    if result["calibrated"]:
                        print(f"  {result['pixel'][i]:5d}  {result['intensity'][i]:6d}  "
                              f"{result['raman_shift_cm-1'][i]:8.2f} cm⁻¹")
                    else:
                        print(f"  {result['pixel'][i]:5d}  {result['intensity'][i]:6d}")

                # 선택적: 캘리브레이션 sanity check
                if calibrator is not None:
                    v = calibrator.verify_with_peak(
                        np.array(result["intensity"]),
                        expected_shift=1312.3,   # 기대 피크 (샘플 맞게 변경)
                        tolerance_cm=10.0,
                    )
                    tag = "OK" if v.get("ok") else "CHECK"
                    print(f"\n[VERIFY-{tag}] {v}")

                # 저장 (파일명에 timestamp)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = Path(f"spectrum_{ts}.csv")
                save_spectrum_csv(result, out_path)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            cam.shutdown()
