from __future__ import annotations

import configparser
import math
from pathlib import Path

import numpy as np

__all__ = ["RamanCalibrator", "wl_p_calib"]

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "Config.ini"


def wl_p_calib(px, n0, offset_adjust, wl_center, m_order, d_grating,
               x_pixel, f, delta, gamma, curvature=0.0):
    """픽셀 인덱스 → 파장 변환 (Czerny-Turner 분광기 회절 격자 방정식).

    모든 길이 인자(wl_center, d_grating, x_pixel, f)는 동일 단위여야 함.
    반환값 단위 = wl_center/d_grating 단위.
    """
    n = px - (n0 + offset_adjust * wl_center)
    psi = np.arcsin(m_order * wl_center / (2 * d_grating * np.cos(gamma / 2)))
    eta = np.arctan(n * x_pixel * np.cos(delta) / (f + n * x_pixel * np.sin(delta)))
    return (
        (d_grating / m_order)
        * (np.sin(psi - 0.5 * gamma) + np.sin(psi + 0.5 * gamma + eta))
        + curvature * n ** 2
    )


class RamanCalibrator:
    def __init__(self, laser_nm: float, lut_cm1: np.ndarray):
        self.laser_nm = float(laser_nm)
        self._lut = lut_cm1   # cm⁻¹ shape (pixel_count,)
        self._wl_nm: np.ndarray | None = None  # from_factory_calibration에서 채움

    def pixel_to_wavelength(self, px) -> np.ndarray:
        return self._wl_nm[np.asarray(px, dtype=int)]

    def pixel_to_raman_shift(self, px) -> np.ndarray:
        return self._lut[np.asarray(px, dtype=int)]

    @classmethod
    def from_factory_calibration(cls, config_path=None,
                                  raman_center_cm1: float = 1200.0,
                                  laser_nm: float | None = None,
                                  f_mm: float | None = None) -> "RamanCalibrator":
        path = Path(config_path) if config_path else _DEFAULT_CONFIG

        cp = configparser.ConfigParser(strict=False)
        with open(path, encoding="utf-8", errors="ignore") as fh:
            cp.read_file(fh)

        auto = cp["AUTO_CALIBRATION"]
        selected_type = int(auto.get("SelectedType", "0"))
        selected_rayleigh = int(auto.get("SelectedRayleigh", "0"))
        pixel_count = int(auto.get("PixelCount", "1024"))
        pixel_width_um = float(auto.get("PixelWidth", "23.8902"))
        x_pixel_mm = pixel_width_um / 1000.0

        if laser_nm is None:
            rayleigh_key = f"RayleighWaveLength{selected_rayleigh + 1}"
            laser_nm = float(auto.get(rayleigh_key, "532.021"))

        type_section = f"TYPE-{selected_type}"
        t = cp[type_section]

        groove = float(t["Groove"])
        d_grating_mm = 1.0 / groove
        n0 = pixel_count // 2  # 센서 실제 중앙 픽셀
        gamma = math.radians(float(t.get("dV", "14.755")))
        delta = math.radians(float(t.get("TiltAngle", "0")))
        if f_mm is None:
            f_mm = float(t["FocalLength"])

        # wl_center = 격자가 현재 조준하는 중심 파장 (레이저 파장 ≠ 격자 중심 파장)
        grating_center_nm = 1.0 / (1.0 / laser_nm - raman_center_cm1 / 1e7)
        wl_center_mm = grating_center_nm * 1e-6  # nm → mm

        pixels = np.arange(pixel_count, dtype=float)
        wl_mm = wl_p_calib(
            pixels,
            n0=n0,
            offset_adjust=0.0,
            wl_center=wl_center_mm,
            m_order=1,
            d_grating=d_grating_mm,
            x_pixel=x_pixel_mm,
            f=f_mm,
            delta=delta,
            gamma=gamma,
            curvature=0.0,
        )
        wl_nm = wl_mm * 1e6
        lut_cm1 = (1.0 / laser_nm - 1.0 / wl_nm) * 1e7

        cal = cls(laser_nm=laser_nm, lut_cm1=lut_cm1)
        cal._wl_nm = wl_nm
        return cal


if __name__ == "__main__":
    cal = RamanCalibrator.from_factory_calibration(laser_nm=532.021, f_mm=580.0)
    print(f"laser     : {cal.laser_nm} nm")
    print(f"wl range  : {cal._wl_nm.min():.1f} ~ {cal._wl_nm.max():.1f} nm")
    print(f"raman range: {cal._lut.min():.0f} ~ {cal._lut.max():.0f} cm-1")
