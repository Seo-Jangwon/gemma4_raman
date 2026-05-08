"""
중앙 설정 모듈 — Config.ini 값을 읽어 상수로 노출.
각 파일에서 from config import ... 또는 from backend.config import ... 형태로 사용.
"""

import configparser
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "Config.ini"

_cfg = configparser.ConfigParser(strict=False)
_cfg.read(_CONFIG_PATH, encoding="cp949")

# ── Stage 한계 / 중점 ─ [STAGE_INFO] ─────────────────────────────────────────
STAGE_MAX_X    = _cfg.getfloat("STAGE_INFO", "MaxX")
STAGE_MAX_Y    = _cfg.getfloat("STAGE_INFO", "MaxY")
STAGE_CENTER_X = _cfg.getfloat("STAGE_INFO", "CenterX")
STAGE_CENTER_Y = _cfg.getfloat("STAGE_INFO", "CenterY")

# ── 카메라 해상도 ─ [TUCSEN] ──────────────────────────────────────────────────
CAMERA_WIDTH   = _cfg.getint("TUCSEN", "Width")
CAMERA_HEIGHT  = _cfg.getint("TUCSEN", "Height")

# ── 렌즈 시야각 (µm) ─ [LENS_1] ──────────────────────────────────────────────
LENS_WIDTH_UM  = _cfg.getfloat("LENS_1", "Width")
LENS_HEIGHT_UM = _cfg.getfloat("LENS_1", "Height")

# ── 라만 교정 ─ [AUTO_CALIBRATION] & [TYPE-{N}] ───────────────────────────────
_sel_rayleigh   = _cfg.getint("AUTO_CALIBRATION", "SelectedRayleigh", fallback=0)
LASER_NM        = _cfg.getfloat("AUTO_CALIBRATION", f"RayleighWaveLength{_sel_rayleigh + 1}")
SI_PEAK_OFFSET = _cfg.getfloat("AUTO_CALIBRATION", f"SiPeakOffset{_sel_rayleigh + 1}")

_sel_type       = _cfg.get("AUTO_CALIBRATION", "SelectedType", fallback="0")
FOCAL_LENGTH_MM = _cfg.getfloat(f"TYPE-{_sel_type}", "FocalLength")
PIXEL_WIDTH_UM   = _cfg.getfloat("AUTO_CALIBRATION", "PixelWidth")
GROOVE_PER_MM    = _cfg.getfloat(f"TYPE-{_sel_type}", "Groove")
TILT_ANGLE_DEG   = _cfg.getfloat(f"TYPE-{_sel_type}", "TiltAngle")
PIXEL_COUNT      = _cfg.getint("AUTO_CALIBRATION", "PixelCount")  # 1024

# ── Config.ini에 없는 기본값 (하드코딩) ───────────────────────────────────────
RAMAN_CENTER_CM1 = _cfg.getfloat(f"TYPE-{_sel_type}", "Center1")
STAGE_MIN_Z = -1.0          # Z축 최솟값 — Config.ini 항목 없음
STAGE_MAX_Z =  1.0          # Z축 최댓값 — Config.ini 항목 없음


if __name__ == "__main__":
    print(f"Config.ini 경로: {_CONFIG_PATH.resolve()}")
    print()
    print("── Stage [STAGE_INFO] ──────────────────────────────")
    print(f"  STAGE_MAX_X    = {STAGE_MAX_X}")
    print(f"  STAGE_MAX_Y    = {STAGE_MAX_Y}")
    print(f"  STAGE_CENTER_X = {STAGE_CENTER_X}")
    print(f"  STAGE_CENTER_Y = {STAGE_CENTER_Y}")
    print(f"  STAGE_MIN_Z    = {STAGE_MIN_Z}  (하드코딩)")
    print(f"  STAGE_MAX_Z    = {STAGE_MAX_Z}  (하드코딩)")
    print()
    print("── 카메라 [TUCSEN] ─────────────────────────────────")
    print(f"  CAMERA_WIDTH   = {CAMERA_WIDTH}")
    print(f"  CAMERA_HEIGHT  = {CAMERA_HEIGHT}")
    print()
    print("── 렌즈 [LENS_1] ───────────────────────────────────")
    print(f"  LENS_WIDTH_UM  = {LENS_WIDTH_UM}")
    print(f"  LENS_HEIGHT_UM = {LENS_HEIGHT_UM}")
    print()
    print(f"── 라만 교정 (SelectedRayleigh={_sel_rayleigh}, SelectedType={_sel_type}) ──")
    print(f"  LASER_NM        = {LASER_NM}")
    print(f"  SI_PEAK_OFFSET  = {SI_PEAK_OFFSET}")
    print(f"  FOCAL_LENGTH_MM = {FOCAL_LENGTH_MM}")
    print(f"  PIXEL_WIDTH_UM  = {PIXEL_WIDTH_UM}")
    print(f"  GROOVE_PER_MM   = {GROOVE_PER_MM}")
    print(f"  TILT_ANGLE_DEG  = {TILT_ANGLE_DEG}")
    print(f"  PIXEL_COUNT     = {PIXEL_COUNT}")
    print(f"  RAMAN_CENTER_CM1= {RAMAN_CENTER_CM1}")
