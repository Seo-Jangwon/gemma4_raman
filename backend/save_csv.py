"""
[역할]
  딴거싹다빼고 CSV저장만 함
"""
from __future__ import annotations

import csv
from pathlib import Path


def save_spectrum_csv(result: dict, path: str | Path):
    """
    스펙트럼 결과를 CSV 로 저장.

    calibrated=True 면 4열(pixel, raman_shift, wavelength, intensity),
    calibrated=False 면 2열(pixel, intensity).
    """
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if result.get("calibrated"):
            # 메타 정보를 주석으로
            f.write(f"# laser_nm,{result.get('laser_nm', '')}\n")
            f.write(f"# calibration,factory_polynomial\n")
            if "exposure_time" in result:
                f.write(f"# exposure_time,{result['exposure_time']}\n")
            if "laser_power_pct" in result:
                f.write(f"# laser_power_pct,{result['laser_power_pct']}\n")
            w.writerow(["pixel", "raman_shift_cm-1", "wavelength_nm", "intensity"])
            for i in range(len(result["pixel"])):
                w.writerow([
                    result["pixel"][i],
                    f"{result['raman_shift_cm-1'][i]:.3f}",
                    f"{result['wavelength_nm'][i]:.4f}",
                    result["intensity"][i],
                ])
        else:
            w.writerow(["pixel", "intensity"])
            for px, val in zip(result["pixel"], result["intensity"]):
                w.writerow([px, val])
    print(f"[CSV] saved: {path}")
