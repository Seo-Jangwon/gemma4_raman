"""
raman_calibration.py  (v2, 수정본)
=====================================
까만 실리콘 측정 + 상용 MANTARAY 데이터와 대조해서 발견한 것:

1) 사용자 raw CCD pixel 순서는 상용 저장 파일과 **반대**이다.
   (user_raw_pixel k) ↔ (commercial_file_index 1023 - k)

2) 현재 setup 에서는 노치필터가 Rayleigh 를 완전히 차단하므로
   Rayleigh 피크로 anchor 를 잡는 것은 **불가능**하다.
   이전 버전의 grating-equation 기반 캘리브레이션은 틀린 가정(485 픽셀이 Rayleigh)
   위에 돌아가서 폐기한다.

3) 가장 정확한 방법은 상용 소프트웨어로 동일 grating motor 위치에서 한 번만
   reference CSV 를 저장해두고, 그 X축을 lookup table 로 사용하는 것.

Usage
-----
    cal = RamanCalibrator.from_reference_csv("SPECTRUM__1_1.csv")
    raman_shift = cal.pixel_to_raman_shift(user_raw_pixels)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


class RamanCalibrator:
    """상용 MANTARAY CSV 를 ground truth 로 하는 pixel → Raman shift 변환기."""

    def __init__(self, raman_shift_per_user_pixel: np.ndarray,
                 laser_wavelength_nm: float = 532.021):
        """
        Parameters
        ----------
        raman_shift_per_user_pixel : (1024,) array
            사용자의 raw CCD pixel 인덱스 (0..1023) 에 대응되는 Raman shift [cm⁻¹].
        laser_wavelength_nm : float
            레이저 파장. 상용 CSV 의 Rayleigh 값 사용 권장.
        """
        self._lut = np.asarray(raman_shift_per_user_pixel, dtype=float)
        assert self._lut.size == 1024, "1024 pixel 필요"
        self.laser_nm = laser_wavelength_nm

    # ------------------------------------------------------------------
    @classmethod
    def from_reference_csv(cls, csv_path: str | Path,
                            user_is_reversed: bool = True) -> "RamanCalibrator":
        """상용 MANTARAY CSV 의 X축을 캘리브레이션 기준으로 불러옴.

        user_is_reversed=True : 사용자 raw CCD pixel 순서가 상용 파일과 반대
                                 (기본값, 본 setup 에서 확인됨)
        """
        header, data = _parse_mantaray_csv(csv_path)
        shift = data["raman_shift"].values
        if shift.size != 1024:
            raise ValueError(f"예상 1024 point, 실제 {shift.size}")

        lut = shift[::-1] if user_is_reversed else shift
        laser_nm = _parse_laser_from_header(header)
        return cls(lut, laser_wavelength_nm=laser_nm)

    # ------------------------------------------------------------------
    def pixel_to_raman_shift(self, pixel) -> np.ndarray:
        p = np.asarray(pixel, dtype=float)
        return np.interp(p, np.arange(1024), self._lut)

    def pixel_to_wavelength(self, pixel) -> np.ndarray:
        rs = self.pixel_to_raman_shift(pixel)
        return 1.0 / (1.0/self.laser_nm - rs/1.0e7)

    # ------------------------------------------------------------------
    def verify_with_peak(self, user_intensity: np.ndarray,
                         expected_shift: float = 1312.3,
                         tolerance_cm: float = 5.0,
                         search_window_cm: float = 100.0) -> dict:
        """사용자 스펙트럼에서 기대 피크를 찾아 캘리브레이션 정확도 검증."""
        from scipy.ndimage import uniform_filter1d

        y = uniform_filter1d(user_intensity.astype(float), size=5)
        shifts = self._lut
        mask = np.abs(shifts - expected_shift) < search_window_cm
        if not np.any(mask):
            return {"ok": False, "error": "search window 에 데이터 없음"}

        local_y = y[mask]
        local_shifts = shifts[mask]
        idx = int(np.argmax(local_y))
        lo = max(0, idx-3)
        hi = min(len(local_y)-1, idx+3)
        baseline = np.median(y)
        weights = np.clip(local_y[lo:hi+1] - baseline, 0, None)
        if weights.sum() == 0:
            return {"ok": False, "error": "피크가 baseline 위로 올라오지 않음"}
        found_shift = float(np.sum(local_shifts[lo:hi+1] * weights) / weights.sum())
        err = found_shift - expected_shift
        return {
            "ok": abs(err) < tolerance_cm,
            "found_shift_cm-1": found_shift,
            "expected_cm-1": expected_shift,
            "error_cm-1": err,
            "prominence": float(local_y[idx] - baseline),
        }


# ---------- helpers ------------------------------------------------------
def _parse_mantaray_csv(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header, start = {}, 0
    for i, line in enumerate(lines):
        if "X Axis Data" in line:
            start = i + 1
            break
        parts = line.strip().split(",")
        if len(parts) == 2 and parts[0] and parts[1]:
            header[parts[0].strip()] = parts[1].strip()

    rows = []
    for line in lines[start:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return header, pd.DataFrame(rows, columns=["raman_shift", "intensity"])


def _parse_laser_from_header(header: dict) -> float:
    v = header.get("Rayleigh", "532.021nm")
    return float(str(v).replace("nm", "").strip())


# ---------- 실사용 예시 ----------------------------------------------------
if __name__ == "__main__":
    cal = RamanCalibrator.from_reference_csv(
        "/mnt/user-data/uploads/SPECTRUM__1_1.csv",
        user_is_reversed=True,
    )
    print(f"Laser λ (상용 CSV): {cal.laser_nm} nm")
    print(f"Raman shift range: {cal._lut.min():.1f} ~ {cal._lut.max():.1f} cm⁻¹")
    print(f"User pixel 0   → Δν = {cal.pixel_to_raman_shift(0):.2f} cm⁻¹")
    print(f"User pixel 485 → Δν = {cal.pixel_to_raman_shift(485):.2f} cm⁻¹")
    print(f"User pixel 1023→ Δν = {cal.pixel_to_raman_shift(1023):.2f} cm⁻¹")

    du = pd.read_csv("/mnt/user-data/uploads/spectrum_20260416_202347_exp0_1s_pwr20pct.csv")
    print(f"\n검증: {cal.verify_with_peak(du['intensity'].values, expected_shift=1312.3)}")
