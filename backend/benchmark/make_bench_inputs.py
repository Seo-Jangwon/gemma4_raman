# -*- coding: utf-8 -*-
"""
[역할] 라만 벤치마크의 '입력 신호 파일'을 합성해 에이전트가 바로 볼 수 있는
       data/uploads/<날짜>/ 폴더에 떨궈두는 스탠드얼론 생성기.

  왜 이 폴더인가:
    upload_store.list_uploads()/inspect_upload()/load_upload() 는 그냥 이 폴더를
    스캔한다. 즉 프론트 첨부 버튼이나 HTTP 업로드를 거치지 않고 파일만 놔둬도
    에이전트의 list_uploaded_files / inspect_file / run_analysis(file_ids=...) 가
    그대로 인식한다. → 에이전트 코드는 한 줄도 손대지 않는다.

  채점은 수동이므로 정답(ground-truth) JSON 은 만들지 않는다. 오직 '입력'만 편하게.

  실행:
    python -m backend.benchmark.make_bench_inputs
    python -m backend.benchmark.make_bench_inputs --date 2026-07-24
    python -m backend.benchmark.make_bench_inputs --clean     # 이 스크립트가 만든 것만 지우고 다시

  생성물은 전부 CSV(헤더 있음). upload_store 의 파서가 헤더/구분자를 자동 인식한다.
  파일명은 타임스탬프 없이 고정 → file_id = "<날짜>/<파일명>", 맨 파일명으로도 조회된다.

  [주의] 피크표는 '대략적인 문헌값'이다. 벤치마크 목적엔 충분하지만, 정밀한 값이
  필요하면 아래 MATERIALS 를 직접 손봐라(수동 채점이라 어차피 사람이 눈으로 본다).

  [RamanSPy] 지금은 순수 numpy 로 baseline/스파이크/노이즈/혼합을 만든다(설치 불요).
  더 '논문에 실린' artifact 모델을 원하면 아래 add_* 함수 몸통만 ramanspy.synth.mix(...)
  호출로 갈아끼우면 된다 — 파라미터 이름을 일부러 RamanSPy 와 비슷하게 맞춰뒀다.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# data/uploads — upload_store.UPLOADS_ROOT 와 동일 규칙(하드코딩 의존 없이 경로만 계산)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOADS_ROOT = _PROJECT_ROOT / "data" / "uploads"

# 공통 파수축: 100–2000 cm⁻¹, 1024 포인트(CCD 픽셀 수와 맞춤 — 필수는 아님).
AXIS_MIN, AXIS_MAX, AXIS_N = 100.0, 2000.0, 1024
AXIS = np.linspace(AXIS_MIN, AXIS_MAX, AXIS_N)

# 세기 스케일(카운트 느낌): 가장 큰 피크가 대략 이 값이 되도록 정규화.
PEAK_MAX = 40000.0
DARK_OFFSET = 300.0        # 상시 배경(검출기 dark 느낌)
SAT_LIMIT = 65535.0        # 포화 한계(T102 등)


# ── 물질별 피크표 ────────────────────────────────────────────────────────────
# (center_cm⁻¹, 상대세기, FWHM_cm⁻¹). 대략적 문헌값 — 필요시 수정.
MATERIALS: dict[str, list[tuple[float, float, float]]] = {
    "polystyrene": [
        (620.9, 0.30, 8), (795.8, 0.15, 10), (1001.4, 1.00, 6), (1031.8, 0.60, 7),
        (1155.3, 0.25, 9), (1450.5, 0.35, 12), (1583.1, 0.30, 10), (1602.3, 0.50, 8),
    ],
    "silicon": [
        (520.45, 1.00, 4),
    ],
    "PET": [
        (632, 0.20, 9), (858, 0.25, 9), (1096, 0.40, 8), (1117, 0.35, 8),
        (1288, 0.60, 9), (1614, 0.70, 8), (1726, 1.00, 10),
    ],
    "PMMA": [
        (601, 0.20, 9), (812, 0.60, 8), (970, 0.30, 8), (1122, 0.30, 8),
        (1188, 0.50, 9), (1240, 0.40, 9), (1450, 0.70, 12), (1730, 1.00, 10),
    ],
    "calcite": [
        (155, 0.20, 6), (282, 0.40, 7), (712, 0.30, 6), (1086, 1.00, 5), (1435, 0.15, 12),
    ],
    "aragonite": [
        (152, 0.25, 6), (206, 0.30, 6), (704, 0.35, 6), (1085, 1.00, 5), (1462, 0.12, 12),
    ],
    # 라이브러리 distractor(매칭 태스크에서 '정답이 아닌' 채움용) — 임의 위치.
    "distractorA": [
        (430, 0.5, 10), (900, 1.0, 9), (1350, 0.6, 11), (1700, 0.4, 10),
    ],
    "distractorB": [
        (350, 0.6, 10), (760, 0.5, 9), (1200, 1.0, 10), (1550, 0.5, 11),
    ],
}

# 라이브러리 코드(짧은 식별자) → 물질명
CODE2MAT = {
    "PS": "polystyrene", "PET": "PET", "PMMA": "PMMA",
    "CAL": "calcite", "ARA": "aragonite", "SI": "silicon",
    "XA": "distractorA", "XB": "distractorB",
}


# ── 합성 코어 ────────────────────────────────────────────────────────────────

def _lorentzian(axis, center, amp, fwhm):
    g = fwhm / 2.0
    return amp * (g * g) / ((axis - center) ** 2 + g * g)


def clean_spectrum(material: str, axis=AXIS, peak_max=PEAK_MAX) -> np.ndarray:
    """물질의 깨끗한 스펙트럼(로렌치안 합) + 상시 dark 오프셋."""
    y = np.zeros_like(axis)
    for c, a, w in MATERIALS[material]:
        y += _lorentzian(axis, c, a, w)
    if y.max() > 0:
        y = y / y.max() * peak_max
    return y + DARK_OFFSET


# ── artifact 주입(전부 순수 numpy; 파라미터명은 RamanSPy 와 유사하게) ────────────

def add_fluorescence(y, axis=AXIS, baseline_amplitude=0.6, rng=None):
    """파수 증가할수록 커지는 완만한 형광 배경을 더한다(T42/101/124)."""
    t = (axis - axis.min()) / (axis.max() - axis.min())
    bg = baseline_amplitude * PEAK_MAX * (0.3 * t + 0.7 * t ** 2)
    return y + bg


def add_noise(y, noise_amplitude=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    sigma = noise_amplitude * PEAK_MAX
    return y + rng.normal(0.0, sigma, size=y.shape)


def add_cosmic_spikes(y, n_spikes=5, spike_amplitude=1.5, rng=None):
    """폭 1–2 픽셀의 비정상 스파이크(T43/104/60)."""
    rng = rng or np.random.default_rng(0)
    y = y.copy()
    idx = rng.choice(len(y), size=n_spikes, replace=False)
    for i in idx:
        y[i] += spike_amplitude * PEAK_MAX * rng.uniform(0.8, 1.5)
        if i + 1 < len(y) and rng.random() < 0.4:
            y[i + 1] += spike_amplitude * PEAK_MAX * 0.5
    return y


def saturate(y, limit=SAT_LIMIT):
    return np.clip(y, None, limit)


def shift_axis(axis, delta_cm=5.0):
    """모든 피크가 같은 방향으로 delta 만큼 이동한 것처럼 축을 민다(T139/107/119)."""
    return axis + delta_cm


# ── CSV 라이터 ───────────────────────────────────────────────────────────────

def _round(v, nd=4):
    return round(float(v), nd)


def write_spectrum(path: Path, axis, intensity):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raman_shift_cm-1", "intensity"])
        for x, y in zip(axis, intensity):
            w.writerow([_round(x), _round(y)])


def write_two_column_library(path: Path, axis, columns: dict[str, np.ndarray]):
    """와이드 참조 라이브러리: 1열 파수축 + 스펙트럼당 1열(열 이름 = 식별자)."""
    codes = list(columns.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raman_shift_cm-1", *codes])
        for i, x in enumerate(axis):
            w.writerow([_round(x), *[_round(columns[c][i]) for c in codes]])


def write_peak_library(path: Path, entries: list[tuple[str, str, float, float]]):
    """롱 포맷 피크 라이브러리: code, material, peak_position_cm-1, rel_intensity."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "material", "peak_position_cm-1", "rel_intensity"])
        for code, mat, pos, inten in entries:
            w.writerow([code, mat, _round(pos, 2), _round(inten, 3)])


def write_map(path: Path, axis, grid_spectra: dict[tuple[float, float], np.ndarray]):
    """롱 포맷 라만 맵: x, y, raman_shift_cm-1, intensity."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "raman_shift_cm-1", "intensity"])
        for (x, y), spec in grid_spectra.items():
            for xx, yy in zip(axis, spec):
                w.writerow([_round(x, 3), _round(y, 3), _round(xx), _round(yy)])


# ── 빌드 플랜 ────────────────────────────────────────────────────────────────
# 각 항목: (파일명, 생성함수, "서비스하는 과제(참고)")
# 파일은 전부 out_dir 에 쓰인다. 이 목록의 파일명 집합이 --clean 대상이기도 하다.

def build_all(out_dir: Path, seed: int) -> list[tuple[str, str]]:
    made: list[tuple[str, str]] = []
    r = np.random.default_rng(seed)

    def emit(name, axis, intensity, tasks):
        write_spectrum(out_dir / name, axis, intensity)
        made.append((name, tasks))

    # ── 물질 clean 스펙트럼 ──────────────────────────────────────────────────
    ps = clean_spectrum("polystyrene")
    emit("ps_clean.csv", AXIS, add_noise(ps, 0.005, r),
         "T41,45,46,47,48,52,53,56,57,58,59")
    emit("si_clean.csv", AXIS, add_noise(clean_spectrum("silicon"), 0.005, r),
         "T133")
    emit("pet_clean.csv", AXIS, add_noise(clean_spectrum("PET"), 0.005, r), "T126,137 참고")
    emit("pmma_clean.csv", AXIS, add_noise(clean_spectrum("PMMA"), 0.005, r), "T126,137 참고")
    emit("calcite_clean.csv", AXIS, add_noise(clean_spectrum("calcite"), 0.005, r), "T134 참고")
    emit("aragonite_clean.csv", AXIS, add_noise(clean_spectrum("aragonite"), 0.005, r), "T134 참고")

    # ── PS 가공 변형(전처리·트러블슈팅 입력) ────────────────────────────────
    emit("ps_fluorescence.csv", AXIS, add_noise(add_fluorescence(ps, baseline_amplitude=0.6), 0.005, r),
         "T42(베이스라인),101,124")
    emit("ps_strong_bg.csv", AXIS, add_noise(add_fluorescence(ps, baseline_amplitude=1.4), 0.005, r),
         "T120(강한 배경)")
    emit("ps_spikes.csv", AXIS, add_cosmic_spikes(add_noise(ps, 0.005, r), n_spikes=6, rng=r),
         "T43,60,104(스파이크/코스믹레이)")
    emit("ps_noisy.csv", AXIS, add_noise(ps, 0.06, r),
         "T44(평활화 입력),106,128(저SNR PS)")
    emit("ps_saturated.csv", AXIS, saturate(add_noise(clean_spectrum("polystyrene", peak_max=90000), 0.01, r)),
         "T102(포화)")
    emit("ps_shifted.csv", shift_axis(AXIS, 5.0), add_noise(ps, 0.005, r),
         "T139(+5cm⁻¹ 시프트),107,119")

    # ── 경계·예외 입력 ──────────────────────────────────────────────────────
    emit("spectrum_allzero.csv", AXIS, np.zeros_like(AXIS),
         "T98(무신호 all-zero)")
    ax_narrow = np.linspace(400.0, 1800.0, AXIS_N)
    emit("spectrum_400_1800.csv", ax_narrow,
         add_noise(clean_spectrum("polystyrene", axis=ax_narrow), 0.005, r),
         "T97(5000–6000 구간 없음)")

    # ── 두 스펙트럼 비교(겹쳐그리기/피크차이) ────────────────────────────────
    ps_a = add_noise(ps, 0.005, r)
    # 일부 피크를 살짝 이동시킨 변형(5cm⁻¹ 이상 차이 나는 쌍이 생기도록)
    mat_b = [(c + (7 if c > 1500 else 0), a, w) for c, a, w in MATERIALS["polystyrene"]]
    yb = np.zeros_like(AXIS)
    for c, a, w in mat_b:
        yb += _lorentzian(AXIS, c, a, w)
    yb = yb / yb.max() * PEAK_MAX + DARK_OFFSET
    emit("ps_two_a.csv", AXIS, ps_a, "T51,55(두 스펙트럼 A)")
    emit("ps_two_b.csv", AXIS, add_noise(yb, 0.005, r), "T51,55(두 스펙트럼 B)")

    # ── Cat 7 미지(unknown) 스펙트럼 ─────────────────────────────────────────
    emit("unknown_polystyrene.csv", AXIS, add_noise(ps, 0.02, r),
         "T121,122,123,128,132,138,139")
    emit("unknown_PET.csv", AXIS, add_noise(clean_spectrum("PET"), 0.02, r),
         "T126(PET 판별)")
    emit("unknown_PMMA.csv", AXIS, add_noise(clean_spectrum("PMMA"), 0.02, r),
         "T137(라벨 불일치 검증)")
    emit("unknown_calcite.csv", AXIS, add_noise(clean_spectrum("calcite"), 0.02, r),
         "T134(calcite/aragonite 구분)")
    # PET 70% + PMMA 30% 혼합(T127)
    mix = 0.7 * clean_spectrum("PET") + 0.3 * clean_spectrum("PMMA")
    emit("unknown_mixture_PET70_PMMA30.csv", AXIS, add_noise(mix, 0.01, r),
         "T127(혼합 우세성분)")
    # 라이브러리에 없는 물질(OOD, T125)
    ood = np.zeros_like(AXIS)
    for c, a, w in [(480, 1.0, 8), (1050, 0.6, 9), (1500, 0.8, 10)]:
        ood += _lorentzian(AXIS, c, a, w)
    ood = ood / ood.max() * PEAK_MAX + DARK_OFFSET
    emit("unknown_OOD.csv", AXIS, add_noise(ood, 0.02, r), "T125(신뢰 일치 없음)")

    # ── 참조 라이브러리(와이드) ──────────────────────────────────────────────
    def ref_col(code, extra_noise=0.01):
        return add_noise(clean_spectrum(CODE2MAT[code]), extra_noise, r)

    lib_full = {
        "PS_01": ref_col("PS"), "PS_02": ref_col("PS"), "PS_03": ref_col("PS"),
        "PET_01": ref_col("PET"), "PET_02": ref_col("PET"),
        "PMMA_01": ref_col("PMMA"), "PMMA_02": ref_col("PMMA"),
        "CAL_01": ref_col("CAL"), "CAL_02": ref_col("CAL"),
        "ARA_01": ref_col("ARA"), "ARA_02": ref_col("ARA"),
        "SI_01": ref_col("SI"),
        "XA_01": ref_col("XA"), "XB_01": ref_col("XB"),
    }
    write_two_column_library(out_dir / "reference_library.csv", AXIS, lib_full)
    made.append(("reference_library.csv",
                 "T121,122,123,124,126,127,132,134,135,136,138,140 (와이드 참조; 열이름=식별자)"))

    # 정확히 8개 항목(T131 랭킹 검증)
    lib8 = {k: lib_full[k] for k in
            ["PS_01", "PET_01", "PMMA_01", "CAL_01", "ARA_01", "SI_01", "XA_01", "XB_01"]}
    write_two_column_library(out_dir / "reference_library_8.csv", AXIS, lib8)
    made.append(("reference_library_8.csv", "T131(정확히 8개 항목 랭킹)"))

    # 피크 라이브러리(롱; 역검색·피크매칭)
    entries = []
    for code, mat in CODE2MAT.items():
        for pos, inten, _w in MATERIALS[mat]:
            entries.append((code, mat, pos, inten))
    write_peak_library(out_dir / "peak_library.csv", entries)
    made.append(("peak_library.csv", "T129(피크매칭),130(역검색 620/1001/1031/1602)"))

    # ── 3×3 라만 맵(T54) ─────────────────────────────────────────────────────
    # 왼쪽 열은 PS, 오른쪽 열로 갈수록 PET 비율↑ → 1000cm⁻¹ 히트맵에 공간 패턴.
    grid = {}
    xs = [0.0, 0.1, 0.2]
    ys = [0.0, 0.1, 0.2]
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            frac_pet = ix / (len(xs) - 1)
            spec = (1 - frac_pet) * clean_spectrum("polystyrene") + frac_pet * clean_spectrum("PET")
            grid[(x, y)] = add_noise(spec, 0.01, r)
    write_map(out_dir / "raman_map_3x3.csv", AXIS, grid)
    made.append(("raman_map_3x3.csv", "T54(3×3 맵 히트맵)"))

    return made


# ── 실행 ─────────────────────────────────────────────────────────────────────

def main():
    from datetime import datetime

    ap = argparse.ArgumentParser(description="라만 벤치마크 입력 신호 생성기")
    ap.add_argument("--date", default=None, help="대상 날짜 YYYY-MM-DD (기본: 오늘)")
    ap.add_argument("--seed", type=int, default=20260725, help="난수 시드(재현용)")
    ap.add_argument("--clean", action="store_true",
                    help="이 스크립트가 만드는 파일명들을 먼저 지우고 다시 생성")
    args = ap.parse_args()

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    out_dir = UPLOADS_ROOT / day
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        # 먼저 한 번 '드라이'로 파일명을 얻어 그 이름들만 삭제(무관 파일 보호)
        names = {n for n, _ in build_all(out_dir, args.seed)}
        for n in names:
            p = out_dir / n
            if p.exists():
                p.unlink()

    made = build_all(out_dir, args.seed)

    print(f"[OK] {len(made)}개 입력 파일 생성 → {out_dir}")
    print(f"     (file_id = '{day}/<파일명>' 또는 맨 파일명으로도 조회됨)\n")
    print(f"{'파일명':38s} 서비스 과제(참고)")
    print("-" * 92)
    for name, tasks in made:
        print(f"{name:38s} {tasks}")
    print("\n사용법: 채팅에 예) '첨부된 ps_fluorescence.csv 를 불러와 5차 다항식 "
          "베이스라인 보정 후 피크를 찾아줘' 처럼 파일명을 넣어 지시하면,")
    print("      에이전트가 list_uploaded_files → inspect_file → run_analysis 로 처리한다.")
    print("\n코드 매핑: " + ", ".join(f"{c}={m}" for c, m in CODE2MAT.items()))


if __name__ == "__main__":
    main()
