"""
raman_calibration.py — Raman 스펙트럼 캘리브레이션 모듈
=========================================================

[핵심 역할]
  사용자의 raw CCD pixel index (0~1023) 를
  Raman shift (cm⁻¹) 또는 wavelength (nm) 로 변환.

[캘리브레이션 방법 3가지]

  1) from_factory_calibration()  ★ 권장 — 외부 파일 불필요
     상용 MANTARAY 프로그램에서 측정한 데이터 3장을 기반으로
     4차 다항식 계수를 코드에 영구 내장.
     max 오차 0.007 cm⁻¹ (dispersion 2.5 cm⁻¹/pixel 대비 무시 가능).

  2) from_reference_csv(csv_path)
     상용 프로그램 CSV 의 X축을 lookup table 로 직접 사용.
     grating motor 이동 등으로 재캘리브레이션 시 활용.

  3) from_polynomial(coefs, laser_nm)
     사용자가 직접 다항식 계수를 지정.

[pixel 축 방향 주의]
  - 사용자 raw CCD (GetAcquiredData) 의 pixel 0 은 **장파장** 쪽.
  - 상용 소프트웨어 CSV 의 index 0 은 **단파장** 쪽.
  - 즉 user_pixel k ↔ commercial_index (1023-k).
  - 이 뒤집기는 from_factory_calibration 에 이미 반영됨.

[의존성]
  numpy, (scipy — verify_with_peak 사용 시만)

[사용 예시]
  cal = RamanCalibrator.from_factory_calibration()
  shift = cal.pixel_to_raman_shift(np.arange(1024))
  wl    = cal.pixel_to_wavelength(np.arange(1024))
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# 영구 보정 계수
# ══════════════════════════════════════════════════════════════════════════════
#
# 아래 값은 상용 MANTARAY 프로그램으로 동일 위치에서 촬영한
# SPECTRUM__1_1.csv, SPECTRUM__2_2.csv, SPECTRUM__3_3.csv 3장의 X축 평균을
# 4차 다항식으로 적합한 결과.
#
# 적용 조건:
#   - 레이저: 532.021 nm
#   - 그레이팅: 1200 gr/mm (Config.txt TYPE-2)
#   - 그레이팅 모터 위치 변경 시 재적합 필요
#
# 다항식: shift(user_pixel) = c[0]*p^4 + c[1]*p^3 + c[2]*p^2 + c[3]*p + c[4]
#   user pixel 0    → 2435.17 cm⁻¹  (Stokes 장파장 끝)
#   user pixel 1023 → -136.58 cm⁻¹  (anti-Stokes 쪽)
#   max 잔차: 0.007 cm⁻¹
#   rms 잔차: 0.003 cm⁻¹

_FACTORY_POLY_COEFS = [
    -4.588977798139755e-12,   # p^4 계수
    -3.382527398823247e-08,   # p^3 계수
    -0.0003292723774643111,   # p^2 계수
    -2.1367768084995604,      # p^1 계수
     2435.1717685704175,      # p^0 (상수항)
]
_FACTORY_LASER_NM = 532.021  # 상용 CSV 헤더의 Rayleigh 파장


class RamanCalibrator:
    """
    pixel → Raman shift / wavelength 변환기.

    내부적으로 1024 pixel 에 대한 lookup table (LUT) 을 보유.
    입력이 정수가 아닌 sub-pixel 값이면 선형 보간으로 처리.

    Attributes
    ----------
    _lut : np.ndarray, shape (1024,)
        user raw pixel index 순서의 Raman shift 값 [cm⁻¹].
    laser_nm : float
        레이저 파장 [nm]. wavelength 역변환에 사용.
    """

    def __init__(self, raman_shift_per_user_pixel: np.ndarray,
                 laser_wavelength_nm: float = 532.021):
        """
        Parameters
        ----------
        raman_shift_per_user_pixel : (1024,) array
            user raw CCD pixel 0~1023 에 대응하는 Raman shift [cm⁻¹].
        laser_wavelength_nm : float
            레이저 파장 [nm].
        """
        self._lut = np.asarray(raman_shift_per_user_pixel, dtype=float)
        assert self._lut.size == 1024, f"1024 pixel 필요, {self._lut.size} 받음"
        self.laser_nm = laser_wavelength_nm

    # ══════════════════════════════════════════════════════════════════════════
    # 생성 메서드 3가지
    # ══════════════════════════════════════════════════════════════════════════

    @classmethod
    def from_factory_calibration(cls) -> "RamanCalibrator":
        """
        코드에 내장된 영구 보정 계수로 캘리브레이터 생성.

        외부 파일이 전혀 필요 없음.
        상용 MANTARAY 데이터 3장 평균에서 4차 다항식으로 적합한 결과 사용.

        Returns
        -------
        RamanCalibrator
            즉시 사용 가능한 캘리브레이터.

        Notes
        -----
        - 적용 조건: 532.021nm 레이저, 1200 gr/mm 그레이팅, 현재 모터 위치
        - 그레이팅 모터를 움직였다면 from_reference_csv() 로 재캘리브레이션 필요
        """
        pixels = np.arange(1024)
        lut = np.polyval(_FACTORY_POLY_COEFS, pixels)
        return cls(lut, laser_wavelength_nm=_FACTORY_LASER_NM)

    @classmethod
    def from_reference_csv(cls, csv_path: str | Path,
                           user_is_reversed: bool = True) -> "RamanCalibrator":
        """
        상용 MANTARAY CSV 파일의 X축을 lookup table 로 사용.

        Parameters
        ----------
        csv_path : str | Path
            상용 프로그램에서 저장한 CSV 파일 경로.
            헤더에 'Rayleigh,532.021nm' 형식의 레이저 파장 정보 포함.
        user_is_reversed : bool
            True: 사용자 raw pixel 순서가 상용 파일과 반대 (본 setup 기본값).
            Config.txt [ANDOR_IDUS] Reverse=True 이면 True.

        Returns
        -------
        RamanCalibrator
        """
        header, data = _parse_mantaray_csv(csv_path)
        shift = data["raman_shift"].values
        if shift.size != 1024:
            raise ValueError(f"1024 point 예상, {shift.size} 받음")

        # user_pixel k → commercial_idx (1023-k) → shift[1023-k]
        lut = shift[::-1] if user_is_reversed else shift
        laser_nm = _parse_laser_from_header(header)
        return cls(lut, laser_wavelength_nm=laser_nm)

    @classmethod
    def from_polynomial(cls, coefs: list[float],
                        laser_nm: float = 532.021) -> "RamanCalibrator":
        """
        사용자 지정 다항식 계수로 캘리브레이터 생성.

        Parameters
        ----------
        coefs : list[float]
            np.polyval 형식 계수 (최고차항 먼저).
            예: [c4, c3, c2, c1, c0] → shift = c4*p^4 + ... + c0
        laser_nm : float
            레이저 파장 [nm].
        """
        pixels = np.arange(1024)
        lut = np.polyval(coefs, pixels)
        return cls(lut, laser_wavelength_nm=laser_nm)

    # ══════════════════════════════════════════════════════════════════════════
    # 변환 API
    # ══════════════════════════════════════════════════════════════════════════

    def pixel_to_raman_shift(self, pixel) -> np.ndarray:
        """
        user raw CCD pixel → Raman shift [cm⁻¹].

        Parameters
        ----------
        pixel : int, float, or array-like
            CCD pixel index (0~1023). sub-pixel 값은 선형 보간.

        Returns
        -------
        np.ndarray
            Raman shift [cm⁻¹]. Stokes 가 양수.
        """
        p = np.asarray(pixel, dtype=float)
        return np.interp(p, np.arange(1024), self._lut)

    def pixel_to_wavelength(self, pixel) -> np.ndarray:
        """
        user raw CCD pixel → wavelength [nm].

        Raman shift → wavelength 역변환:
            λ = 1 / (1/λ_laser - Δν/1e7)

        Parameters
        ----------
        pixel : int, float, or array-like

        Returns
        -------
        np.ndarray
            wavelength [nm].
        """
        rs = self.pixel_to_raman_shift(pixel)
        return 1.0 / (1.0 / self.laser_nm - rs / 1.0e7)

    # ══════════════════════════════════════════════════════════════════════════
    # 검증 유틸
    # ══════════════════════════════════════════════════════════════════════════

    def verify_with_peak(self, user_intensity: np.ndarray,
                         expected_shift: float = 520.7,
                         tolerance_cm: float = 5.0,
                         search_window_cm: float = 100.0) -> dict:
        """
        측정된 스펙트럼에서 기대 피크를 찾아 캘리브레이션 정확도 검증.

        사용 예: Si wafer (520.7 cm⁻¹) 로 검증

        Parameters
        ----------
        user_intensity : np.ndarray
            raw CCD intensity 배열 (1024 길이).
        expected_shift : float
            기대하는 피크 위치 [cm⁻¹]. Si 기준 520.7.
        tolerance_cm : float
            허용 오차 [cm⁻¹]. 이 안에 들어오면 ok=True.
        search_window_cm : float
            피크 검색 범위 ± [cm⁻¹].

        Returns
        -------
        dict
            ok: bool, found_shift_cm-1, expected_cm-1, error_cm-1, prominence
        """
        from scipy.ndimage import uniform_filter1d

        y = uniform_filter1d(user_intensity.astype(float), size=5)
        shifts = self._lut

        # 기대 피크 주변만 탐색
        mask = np.abs(shifts - expected_shift) < search_window_cm
        if not np.any(mask):
            return {"ok": False, "error": "search window 에 데이터 없음"}

        local_y = y[mask]
        local_shifts = shifts[mask]

        # 가장 큰 값의 centroid (±3 pixel)
        idx = int(np.argmax(local_y))
        lo = max(0, idx - 3)
        hi = min(len(local_y) - 1, idx + 3)
        baseline = np.median(y)
        weights = np.clip(local_y[lo:hi + 1] - baseline, 0, None)
        if weights.sum() == 0:
            return {"ok": False, "error": "피크가 baseline 위로 올라오지 않음"}

        found_shift = float(np.sum(local_shifts[lo:hi + 1] * weights) / weights.sum())
        err = found_shift - expected_shift

        return {
            "ok": abs(err) < tolerance_cm,
            "found_shift_cm-1": found_shift,
            "expected_cm-1": expected_shift,
            "error_cm-1": err,
            "prominence": float(local_y[idx] - baseline),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 내부 헬퍼 — 상용 CSV 파싱
# ══════════════════════════════════════════════════════════════════════════════

def _parse_mantaray_csv(path):
    """
    MANTARAY 상용 소프트웨어 CSV 파일 파싱.

    파일 구조:
        System
        Type,MANTARAY
        Unit,WAVENUMBER
        Rayleigh,532.021nm
        Grating,1200gr
        (빈 줄)
        Measuring Condition
        Acquisition Mode,SINGLE
        ...
        X Axis Data,Y Axis Data
        -136.59,98
        -133.65,99
        ...

    Returns
    -------
    header : dict
        헤더 키-값 쌍 (예: {'Rayleigh': '532.021nm', ...})
    data : pd.DataFrame
        columns=['raman_shift', 'intensity'], 1024 행
    """
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
    """헤더에서 레이저 파장 추출 (예: '532.021nm' → 532.021)."""
    v = header.get("Rayleigh", "532.021nm")
    return float(str(v).replace("nm", "").strip())


# ══════════════════════════════════════════════════════════════════════════════
# 단독 실행 — 자체 검증
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 외부 파일 없이 factory calibration 테스트
    cal = RamanCalibrator.from_factory_calibration()
    print(f"레이저 파장: {cal.laser_nm} nm")
    print(f"Raman shift 범위: {cal._lut.min():.1f} ~ {cal._lut.max():.1f} cm⁻¹")
    print(f"User pixel 0    → Δν = {cal.pixel_to_raman_shift(0):.2f} cm⁻¹")
    print(f"User pixel 485  → Δν = {cal.pixel_to_raman_shift(485):.2f} cm⁻¹")
    print(f"User pixel 1023 → Δν = {cal.pixel_to_raman_shift(1023):.2f} cm⁻¹")
