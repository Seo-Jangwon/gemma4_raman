"""
[역할]
  딴거싹다빼고 CSV저장만 함 (test_spectrum.py 전용 진입점)

  2026-07-30: 실제 쓰기는 spectrum_store.write_spectrum_csv 에 위임한다. 예전에는
  여기서 자체 헤더('pixel', ...)를 써서, 이 스크립트가 남긴 파일만 다른 포맷이었다
  (프로젝트에 스펙트럼 CSV writer 가 네 벌 있었던 것 중 하나 — spectrum_store 머리말 참고).
"""
from __future__ import annotations

from pathlib import Path

from backend.spectrum_store import write_spectrum_csv


def save_spectrum_csv(result: dict, path: str | Path):
    """
    스펙트럼 결과를 표준 포맷 CSV 로 저장.

    포맷: pixel_index, [raman_shift_cm-1, wavelength_nm,] intensity
    (calibrated=False 면 축 열이 빠진다.)
    """
    path = Path(path)
    calibrated = bool(result.get("calibrated"))
    meta = {k: result.get(k) for k in ("laser_nm", "exposure_time", "laser_power_pct")}
    if calibrated:
        meta["calibration"] = "factory_polynomial"
    write_spectrum_csv(
        path,
        intensity=result["intensity"],
        raman_shift=result["raman_shift_cm-1"] if calibrated else None,
        wavelength_nm=result["wavelength_nm"] if calibrated else None,
        meta=meta,
        encoding="utf-8",
    )
    print(f"[CSV] saved: {path}")
