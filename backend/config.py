"""
중앙 설정 모듈 — Config.ini 값을 읽어 상수로 노출.
각 파일에서 from config import ... 또는 from backend.config import ... 형태로 사용.
"""

import configparser
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "Config.ini"

_cfg = configparser.ConfigParser(strict=False)
_cfg.read(_CONFIG_PATH, encoding="utf-8")

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

# ── Config.ini에 없는 기본값 (하드코딩) ───────────────────────────────────────
RAMAN_CENTER_CM1 = _cfg.getfloat("TYPE-1", "Center1")
STAGE_MIN_Z = -1.0          # Z축 최솟값 — Config.ini 항목 없음
STAGE_MAX_Z =  1.0          # Z축 최댓값 — Config.ini 항목 없음
