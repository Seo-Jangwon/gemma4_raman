# -*- coding: utf-8 -*-
"""
문제ID별 라만 스펙트럼 생성기 (결정론적).

[왜 있나]
  파일 기반 분석/매칭 문항(T037~T056, T092/T093/T096/T104/T110, T111~T128)이
  "어떤 입력 CSV 를 써야 하는지" 프롬프트에 명시돼 있지 않아 에이전트가 헤맸다.
  이 스크립트는 문항마다 결정론적 합성 스펙트럼을 만들어 <문제ID>.csv 로 저장하고,
  프롬프트에 그 파일명을 주입한다. 실사용/재현 모두 이 파일 하나로 고정된다.

[산출물]
  1) data/uploads/<날짜>/<문제ID>.csv        — 각 문항의 입력 스펙트럼 (헤더: raman_shift_cm-1,intensity)
     · 매칭용 라이브러리:  reference_library.csv / reference_library_8.csv / peak_library.csv
     · 다중입력 문항: <ID>_a.csv, <ID>_b.csv, <ID>_ref.csv, <ID>_1..5.csv 등
  2) backend/benchmark/task_refs/<문제ID>_reference.csv  — 지정 방법을 적용한 정답 스펙트럼(자동채점 확장용)
  3) backend/benchmark/task_files.json       — 문항ID → 파일/재료/피크/라벨 매니페스트(단일 진실원)

[재료 피크표] 문헌 라만 이동값(cm-1). 폴리스티렌은 400~1800 구간에 major 7개.

실행:
  python -m backend.benchmark.make_task_spectra                 # 오늘 날짜 폴더에 생성 + 프롬프트 주입
  python -m backend.benchmark.make_task_spectra --date 2026-07-27
  python -m backend.benchmark.make_task_spectra --no-patch      # tasks_raw 프롬프트는 건드리지 않음
"""
from __future__ import annotations

import argparse
import csv
import json
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_UPLOADS = _PROJECT_ROOT / "data" / "uploads"
_REFS = _HERE / "task_refs"
_MANIFEST = _HERE / "task_files.json"
_TASKS_RAW = _HERE / "tasks_raw.json"

HEADER = ["raman_shift_cm-1", "intensity"]

# ── 공통 파형 ────────────────────────────────────────────────────────────────
AXIS = np.arange(200.0, 2000.0 + 1e-9, 1.0)          # 대부분 문항 공통 축
AXIS_400_1800 = np.arange(400.0, 1800.0 + 1e-9, 1.0)  # T092(5000-6000과 겹침 없음)
MAP_AXIS = np.arange(400.0, 1800.0 + 1e-9, 4.0)       # 맵/세션은 4 cm-1 간격(파일 크기 절약)
_LW = 6.0                                             # 로렌치안 반치폭 규모
_SCALE = 1000.0                                       # 피크 카운트 스케일
_BASE = 40.0                                          # 상시 바닥값

# ── 재료 피크표: (중심 cm-1, 상대세기) ────────────────────────────────────────
MATERIALS = {
    # 폴리스티렌 — 400~1800 major 7개 (T042의 "7개 피크"와 정확히 일치)
    "polystyrene": [(620, 0.42), (1001, 1.0), (1031, 0.55), (1155, 0.22),
                    (1450, 0.40), (1583, 0.33), (1602, 0.62)],
    "PET":         [(632, 0.30), (858, 0.35), (1096, 0.42), (1119, 0.30),
                    (1290, 0.50), (1414, 0.28), (1615, 0.48), (1727, 0.70)],
    "PMMA":        [(601, 0.30), (812, 0.50), (966, 0.28), (1122, 0.40),
                    (1450, 0.60), (1730, 0.72)],
    "calcite":     [(712, 0.30), (1086, 1.0), (1435, 0.22)],
    "aragonite":   [(701, 0.22), (705, 0.30), (1085, 1.0), (1462, 0.25)],
    "silicon":     [(520.45, 1.0)],
}


def _lorentz(axis, c, w):
    return (w * w) / ((axis - c) ** 2 + w * w)


def clean_spectrum(material, axis=AXIS, scale=_SCALE, base=_BASE):
    """재료 clean 스펙트럼(피크 합). 바닥값 base + scale*sum(로렌치안)."""
    inten = np.full_like(axis, float(base))
    for c, a in MATERIALS[material]:
        inten = inten + scale * a * _lorentz(axis, c, _LW)
    return inten


def amorphous_spectrum(axis=AXIS):
    """비정질(broad hump). 넓은 가우시안 언덕 하나 — 예리한 피크 없음."""
    hump = 800.0 * np.exp(-((axis - 480.0) ** 2) / (2 * 180.0 ** 2))
    return _BASE + hump


# ── 변형 ─────────────────────────────────────────────────────────────────────
def add_fluorescence(inten, axis, amp=2500.0, power=2.0):
    t = (axis - axis.min()) / (axis.max() - axis.min())
    return inten + amp * (t ** power)


def add_strong_bg(inten, axis, amp=6000.0):
    t = (axis - axis.min()) / (axis.max() - axis.min())
    return inten + amp * (0.15 + t) ** 2


def add_noise(inten, rng, sigma=12.0):
    return inten + rng.normal(0.0, sigma, size=inten.shape)


def add_spikes(inten, axis, rng, n=5, height=5000.0):
    out = inten.copy()
    # 실제 피크(±10 cm-1)를 피해 랜덤 위치에 1-포인트 스파이크
    peak_centers = [c for c, _ in MATERIALS["polystyrene"]]
    idxs = []
    while len(idxs) < n:
        i = int(rng.integers(3, len(axis) - 3))
        if all(abs(axis[i] - pc) > 12 for pc in peak_centers):
            idxs.append(i)
    for i in sorted(set(idxs)):
        out[i] += height
    return out, sorted(axis[i] for i in idxs)


def saturate(inten, limit=65535.0, boost=120.0):
    out = inten * boost
    out = np.clip(out, 0, limit)
    return out


def shift_material(material, d, axis=AXIS):
    inten = np.full_like(axis, float(_BASE))
    for c, a in MATERIALS[material]:
        inten = inten + _SCALE * a * _lorentz(axis, c + d, _LW)
    return inten


# ── 저장 유틸 ────────────────────────────────────────────────────────────────
def _write_spectrum(path: Path, axis, inten):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for x, y in zip(axis, inten):
            w.writerow([f"{float(x):.4f}", f"{float(y):.6f}"])


def _peaks_in_range(material, lo=None, hi=None):
    out = []
    for c, _ in MATERIALS[material]:
        if (lo is None or c >= lo) and (hi is None or c <= hi):
            out.append(round(c, 2))
    return sorted(out)


# ── 정답(reference) 계산: 지정 방법을 결정론적 입력에 적용 ─────────────────────
def _poly_baseline_removed(axis, inten, order=5):
    coef = np.polyfit(axis, inten, order)
    base = np.polyval(coef, axis)
    return inten - base


def _despike(inten, thresh=6.0, win=5):
    from scipy.signal import medfilt
    med = medfilt(inten, kernel_size=win)
    resid = inten - med
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    mask = np.abs(resid) > thresh * 1.4826 * mad
    out = inten.copy()
    out[mask] = med[mask]
    return out


def _savgol(inten, window=11, poly=3):
    from scipy.signal import savgol_filter
    return savgol_filter(inten, window_length=window, polyorder=poly, mode="interp")


def _minmax(inten):
    lo, hi = float(np.min(inten)), float(np.max(inten))
    return (inten - lo) / (hi - lo) if hi > lo else np.zeros_like(inten)


def _l2(inten):
    n = float(np.linalg.norm(inten))
    return inten / n if n > 0 else inten


# ══════════════════════════════════════════════════════════════════════════════
# 문항별 파일 스펙 — 파일을 소비하는 문항만. (하드웨어 실측/목업 재측정 문항은 제외)
# 각 항목: build(rng) -> {files:{fname:(axis,inten)}, refs:{fname:(axis,inten)},
#                          ground_truth:{...}, material, desc}
# ══════════════════════════════════════════════════════════════════════════════
def build_all(rng_master):
    def rng(tag):  # 문항별 고정 시드(crc32 기반 — 파이썬 hash 랜덤화 영향 없음) → 완전 재현
        return np.random.default_rng(zlib.crc32(tag.encode("utf-8")) & 0xFFFFFFFF)

    tasks = {}

    def put(tid, files, material, desc, refs=None, gt=None):
        tasks[tid] = {"files": files, "refs": refs or {}, "material": material,
                      "desc": desc, "ground_truth": gt or {}}

    ps = clean_spectrum("polystyrene")

    # ── Category 3: 전처리 & 피크 ──
    put("T037", {"T037.csv": (AXIS, ps)}, "polystyrene",
        "Clean polystyrene spectrum for line plotting.",
        gt={"peaks_major": _peaks_in_range("polystyrene")})

    ps_fl = add_fluorescence(ps, AXIS)
    put("T038", {"T038.csv": (AXIS, ps_fl)}, "polystyrene",
        "Polystyrene with 2nd-power fluorescence background (5th-order baseline test).",
        refs={"T038_reference.csv": (AXIS, _poly_baseline_removed(AXIS, ps_fl))})

    ps_sp, sp_pos = add_spikes(ps, AXIS, rng("T039"))
    put("T039", {"T039.csv": (AXIS, ps_sp)}, "polystyrene",
        "Polystyrene with sharp 1-point spikes to remove.",
        refs={"T039_reference.csv": (AXIS, ps)}, gt={"spike_positions_cm-1": sp_pos})

    ps_no = add_noise(ps, rng("T040"))
    put("T040", {"T040.csv": (AXIS, ps_no)}, "polystyrene",
        "Noisy polystyrene for Savitzky-Golay (11,3) smoothing.",
        refs={"T040_reference.csv": (AXIS, _savgol(ps_no))})

    put("T041", {"T041.csv": (AXIS, ps)}, "polystyrene",
        "Polystyrene for min-max (0-1) normalization.",
        refs={"T041_reference.csv": (AXIS, _minmax(ps))})

    put("T042", {"T042.csv": (AXIS, ps)}, "polystyrene",
        "Polystyrene with exactly 7 major peaks in 400-1800.",
        gt={"peaks_major": _peaks_in_range("polystyrene", 400, 1800)})

    put("T043", {"T043.csv": (AXIS, ps)}, "polystyrene",
        "Polystyrene; top-3 peaks by intensity.",
        gt={"top3_by_intensity_cm-1": [1001, 1602, 1031]})

    put("T044", {"T044.csv": (AXIS, ps)}, "polystyrene",
        "Polystyrene; FWHM of the ~1001 peak (Lorentzian, FWHM=2*w).",
        gt={"fwhm_cm-1_approx": round(2 * _LW, 2)})

    ps_meas = add_noise(ps, rng("T045"))
    put("T045", {"T045.csv": (AXIS, ps_meas), "T045_ref.csv": (AXIS, ps)}, "polystyrene",
        "Measured (noisy) PS + clean PS reference for peak match-ratio.",
        gt={"reference_peaks_cm-1": _peaks_in_range("polystyrene")})

    ps_raw = add_noise(add_fluorescence(ps_sp, AXIS), rng("T046"))
    ref46 = _minmax(_savgol(_poly_baseline_removed(AXIS, _despike(ps_raw))))
    put("T046", {"T046.csv": (AXIS, ps_raw)}, "polystyrene",
        "Raw PS (spikes+fluorescence+noise) for the full pipeline.",
        refs={"T046_reference.csv": (AXIS, ref46)})

    put("T047", {"T047_a.csv": (AXIS, ps), "T047_b.csv": (AXIS, clean_spectrum("PET"))},
        "polystyrene+PET", "Two spectra to overlay with a legend.")

    put("T048", {"T048.csv": (AXIS, ps)}, "polystyrene",
        "Polystyrene; extract 800-1200 cm-1 slice (inclusive).",
        gt={"range": [800, 1200]})

    ps_snr = add_noise(ps, rng("T049"), sigma=15.0)
    put("T049", {"T049.csv": (AXIS, ps_snr)}, "polystyrene",
        "Noisy PS; SNR = max(990-1012) / std(1800-1900).")

    # T050 3x3 맵 (long: x_mm,y_mm,raman_shift_cm-1,intensity)
    put("T050", {}, "polystyrene",
        "3x3 Raman map; value nearest 1000 cm-1 per position -> heatmap.",
        gt={"grid": "3x3", "channel_cm-1": 1000})

    ps_b = shift_material("polystyrene", 0.0)
    ps_b2 = ps_b.copy()
    # b: 1602->1610, 1155->1162 이동(>=5), 나머지 동일 -> 페어 검출 대상
    ps_shift_pair = np.full_like(AXIS, float(_BASE))
    for c, a in MATERIALS["polystyrene"]:
        cc = c + (8 if c in (1602, 1155) else 0)
        ps_shift_pair += _SCALE * a * _lorentz(AXIS, cc, _LW)
    put("T051", {"T051_a.csv": (AXIS, ps), "T051_b.csv": (AXIS, ps_shift_pair)},
        "polystyrene", "Two PS spectra; report peak pairs with >=5 cm-1 shift.",
        gt={"shifted_pairs_cm-1": [[1155, 1163], [1602, 1610]]})

    ps_bc = _poly_baseline_removed(AXIS, ps)
    put("T052", {"T052.csv": (AXIS, ps_bc)}, "polystyrene",
        "Baseline-corrected PS; trapezoidal area of 990-1012.",
        gt={"integration_range": [990, 1012]})

    put("T053", {"T053.csv": (AXIS, ps)}, "polystyrene",
        "Polystyrene; intensity ratio of peaks nearest 1001 and 1602.")

    put("T054", {"T054.csv": (AXIS, ps)}, "polystyrene",
        "Polystyrene; first numerical derivative vs wavenumber.",
        refs={"T054_reference.csv": (AXIS, np.gradient(ps, AXIS))})

    put("T055", {"T055.csv": (AXIS, ps)}, "polystyrene",
        "Polystyrene; L2 (vector-length) normalization.",
        refs={"T055_reference.csv": (AXIS, _l2(ps))})

    ps_sp2, sp_pos2 = add_spikes(ps, AXIS, rng("T056"))
    put("T056", {"T056.csv": (AXIS, ps_sp2)}, "polystyrene",
        "PS with spikes; despike then report major peaks.",
        refs={"T056_reference.csv": (AXIS, ps)},
        gt={"peaks_major": _peaks_in_range("polystyrene")})

    # ── Category 5/6: 파일 기반 예외/트러블슈팅 ──
    put("T092", {"T092.csv": (AXIS_400_1800, clean_spectrum("polystyrene", AXIS_400_1800))},
        "polystyrene", "PS limited to 400-1800; request 5000-6000 has no overlap.",
        gt={"input_range": [400, 1800], "requested": [5000, 6000]})

    put("T093", {"T093.csv": (AXIS, np.zeros_like(AXIS))}, "none",
        "All-zero intensity (no-signal input).", gt={"all_zero": True})

    put("T104", {"T104.csv": (AXIS, amorphous_spectrum())}, "amorphous",
        "Amorphous material: broad hump, no sharp peaks -> classify amorphous.",
        gt={"label": "amorphous"})

    ps_strong = add_strong_bg(ps, AXIS)
    put("T110", {"T110.csv": (AXIS, ps_strong)}, "polystyrene",
        "PS with strong background; baseline-correct then re-detect peaks.",
        refs={"T110_reference.csv": (AXIS, _poly_baseline_removed(AXIS, ps_strong))},
        gt={"peaks_major": _peaks_in_range("polystyrene")})

    put("T096", {"T096.csv": (AXIS, ps_fl)}, "polystyrene",
        "PS buried under fluorescence; diagnose + baseline-correct.",
        refs={"T096_reference.csv": (AXIS, _poly_baseline_removed(AXIS, ps_fl))},
        gt={"cause": "fluorescence", "peaks_major": _peaks_in_range("polystyrene")})

    ps_cosmic, cos_pos = add_spikes(ps, AXIS, rng("T099"), n=7)
    put("T099", {"T099.csv": (AXIS, ps_cosmic)}, "polystyrene",
        "PS with several cosmic-ray spikes at inconsistent positions; despike, preserve real peaks.",
        refs={"T099_reference.csv": (AXIS, ps)},
        gt={"cause": "cosmic_ray", "spike_positions_cm-1": cos_pos,
            "protected_peaks_cm-1": _peaks_in_range("polystyrene")})

    # ── Category 7: 유사신호 매칭 (라이브러리 + 미지 스펙트럼) ──
    # 미지 스펙트럼(문항별)
    put("T111", {"T111.csv": (AXIS, add_noise(ps, rng("T111"), 8))}, "polystyrene",
        "Unknown = polystyrene; report top-3 similar refs.", gt={"truth": "polystyrene"})
    put("T112", {"T112.csv": (AXIS, add_noise(ps, rng("T112"), 8))}, "polystyrene",
        "Unknown = polystyrene; highest cosine, is it >=0.85?", gt={"truth": "polystyrene"})
    put("T113", {"T113.csv": (AXIS, add_fluorescence(ps, AXIS, 1500))}, "polystyrene",
        "Unknown = PS + background; baseline+L2 then match.", gt={"truth": "polystyrene"})
    put("T114", {"T114.csv": (AXIS, clean_spectrum("aragonite") * 0 + amorphous_spectrum())},
        "OOD", "Unknown = out-of-distribution (amorphous, not in library) -> no reliable match.",
        gt={"truth": "no_match"})
    put("T115", {"T115.csv": (AXIS, add_noise(clean_spectrum("PET"), rng("T115"), 8))}, "PET",
        "Unknown = PET; identify among PET/PMMA.", gt={"truth": "PET"})
    put("T116", {"T116.csv": (AXIS, 0.7 * clean_spectrum("PET") + 0.3 * clean_spectrum("PMMA"))},
        "PET70/PMMA30", "Unknown = PET70/PMMA30 mixture; dominant component.", gt={"truth": "PET"})
    put("T117", {"T117.csv": (AXIS, add_noise(ps, rng("T117"), 60))}, "polystyrene",
        "Unknown = low-SNR PS; SG-smooth then identify.", gt={"truth": "polystyrene"})
    put("T118", {"T118.csv": (AXIS, add_noise(ps, rng("T118"), 8))}, "polystyrene",
        "Unknown = PS; peak-set matching (uses peak_library.csv).", gt={"truth": "polystyrene"})
    put("T119", {"T119.csv": (AXIS, add_noise(ps, rng("T119"), 8))}, "polystyrene",
        "Unknown = PS; rank all 8 library items (uses reference_library_8.csv).",
        gt={"truth": "polystyrene"})
    put("T120", {"T120.csv": (AXIS, add_noise(ps, rng("T120"), 8))}, "polystyrene",
        "Unknown = PS; return best-match id (PS_01).", gt={"truth_id": "PS_01"})
    put("T121", {"T121.csv": (AXIS, add_noise(clean_spectrum("silicon"), rng("T121"), 5))},
        "silicon", "Unknown = silicon single 520 peak.", gt={"truth": "silicon", "peak": 520.45})
    put("T122", {"T122.csv": (AXIS, add_noise(clean_spectrum("calcite"), rng("T122"), 8))},
        "calcite", "Unknown = calcite (vs aragonite).", gt={"truth": "calcite"})
    put("T123", {"T123.csv": (AXIS, add_noise(ps, rng("T123"), 8))}, "polystyrene",
        "Unknown = PS; explainable match with 2+ distinguishing peaks.", gt={"truth": "polystyrene"})
    put("T125", {"T125.csv": (AXIS, add_noise(clean_spectrum("PMMA"), rng("T125"), 8))}, "PMMA",
        "Unknown = PMMA but claimed PET; verify mismatch.", gt={"truth": "PMMA", "claimed": "PET"})
    put("T126", {"T126.csv": (AXIS, add_noise(ps, rng("T126"), 8))}, "polystyrene",
        "Unknown = PS; interpolate+baseline+L2, NN with tie rule.", gt={"truth": "polystyrene"})
    put("T127", {"T127.csv": (AXIS, shift_material("polystyrene", 5.0))}, "polystyrene",
        "Unknown = PS shifted +5 cm-1; estimate shift, correct, identify.",
        gt={"truth": "polystyrene", "shift": 5})

    # T124 / T128: 5개 미지 스펙트럼 배치
    batch = [("polystyrene", "T124_1.csv"), ("PET", "T124_2.csv"), ("PMMA", "T124_3.csv"),
             ("calcite", "T124_4.csv"), ("silicon", "T124_5.csv")]
    files124 = {fn: (AXIS, add_noise(clean_spectrum(m), rng(fn), 8)) for m, fn in batch}
    put("T124", files124, "batch", "5 unknowns in order.",
        gt={"truth_order": [m for m, _ in batch]})
    batch2 = [("polystyrene", "T128_1.csv"), ("PET", "T128_2.csv"), ("PMMA", "T128_3.csv"),
              ("calcite", "T128_4.csv"), ("silicon", "T128_5.csv")]
    files128 = {fn: (AXIS, add_noise(clean_spectrum(m), rng(fn), 8)) for m, fn in batch2}
    put("T128", files128, "batch", "5 unknowns; precision@3 macro average.",
        gt={"truth_order": [m for m, _ in batch2]})

    return tasks


# ── 매칭용 라이브러리 파일 ────────────────────────────────────────────────────
def _write_library(day_dir: Path):
    """reference_library.csv(6재료), reference_library_8.csv(8아이템),
    peak_library.csv(재료별 피크) 생성. long 포맷."""
    lib_mats = ["polystyrene", "PET", "PMMA", "calcite", "aragonite", "silicon"]
    id_prefix = {"polystyrene": "PS", "PET": "PET", "PMMA": "PMMA",
                 "calcite": "CAL", "aragonite": "ARA", "silicon": "SI"}

    def _rows(mats, reps):
        rows = []
        for m in mats:
            for r in range(1, reps + 1):
                sid = f"{id_prefix[m]}_{r:02d}"
                inten = clean_spectrum(m)
                for x, y in zip(AXIS, inten):
                    rows.append([sid, m, f"{x:.4f}", f"{y:.6f}"])
        return rows

    def _dump(path, rows):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["spectrum_id", "material", "raman_shift_cm-1", "intensity"])
            w.writerows(rows)

    _dump(day_dir / "reference_library.csv", _rows(lib_mats, 2))          # 6재료 x2 = 12
    # 8아이템: PS x3, PET x2, PMMA x1, calcite x1, silicon x1
    rows8 = (_rows(["polystyrene"], 3) + _rows(["PET"], 2)
             + _rows(["PMMA"], 1) + _rows(["calcite"], 1) + _rows(["silicon"], 1))
    _dump(day_dir / "reference_library_8.csv", rows8)

    with (day_dir / "peak_library.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["material", "peak_cm-1"])
        for m in lib_mats:
            for c, _ in MATERIALS[m]:
                w.writerow([m, f"{c:.2f}"])


def _write_map_t050(day_dir: Path):
    """T050 3x3 맵: 각 위치별 스펙트럼 long 포맷 (x_mm,y_mm,raman_shift_cm-1,intensity)."""
    coords = [(x, y) for y in (25.2, 25.3, 25.4) for x in (37.8, 37.9, 38.0)]
    axis = np.arange(900.0, 1100.0 + 1e-9, 2.0)  # 1000 근처만
    with (day_dir / "T050.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x_mm", "y_mm", "raman_shift_cm-1", "intensity"])
        for k, (x, y) in enumerate(coords):
            amp = 0.5 + 0.06 * k  # 위치별로 살짝 다른 1000 근처 세기
            inten = _BASE + _SCALE * amp * _lorentz(axis, 1000.0, _LW)
            for wv, iv in zip(axis, inten):
                w.writerow([f"{x:.3f}", f"{y:.3f}", f"{wv:.4f}", f"{iv:.6f}"])


# ── 맵/세션 파일 (T071 PCA · T072 클러스터링 · T074 세션비교) ───────────────────
def _grid_coords(n=5, x0=37.8, y0=25.2, step=0.1):
    return [(f"P{i:02d}", x0 + (i % n) * step, y0 + (i // n) * step) for i in range(n * n)]


def _write_map_long(path: Path, positions):
    """positions: [(point_id, x, y, inten[MAP_AXIS]), ...] → long CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point_id", "x_mm", "y_mm", "raman_shift_cm-1", "intensity"])
        for pid, x, y, inten in positions:
            for wv, iv in zip(MAP_AXIS, inten):
                w.writerow([pid, f"{x:.3f}", f"{y:.3f}", f"{wv:.4f}", f"{iv:.6f}"])


def _write_maps_and_sessions(day_dir: Path):
    """analysis_file 이면서 실측 불가한 제공-파일 문항용 입력 생성."""
    rng = np.random.default_rng(zlib.crc32(b"maps") & 0xFFFFFFFF)
    coords = _grid_coords()

    # T071: 균일 PS 맵(위치별 baseline/scale 변화) → 5차 baseline+L2+PCA(3성분)
    p71 = []
    for k, (pid, x, y) in enumerate(coords):
        s = clean_spectrum("polystyrene", MAP_AXIS, scale=_SCALE * (0.9 + 0.02 * (k % 5)))
        s = add_noise(add_fluorescence(s, MAP_AXIS, amp=200 + 15 * k), rng, 6)
        p71.append((pid, x, y, s))
    _write_map_long(day_dir / "T071.csv", p71)

    # T072: 두 물질 클러스터(PS/PET 교대) → k=2 클러스터링 ARI
    labels = []
    p72 = []
    for k, (pid, x, y) in enumerate(coords):
        mat = "polystyrene" if (k % 2 == 0) else "PET"
        labels.append(mat)
        p72.append((pid, x, y, add_noise(clean_spectrum(mat, MAP_AXIS), rng, 6)))
    _write_map_long(day_dir / "T072.csv", p72)

    # T074: 두 측정 세션(각 PS 5회 반복) → 1001 피크위치차·RSD·평균 코사인유사도 비교
    def _session(path: Path, scale):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rep_id", "raman_shift_cm-1", "intensity"])
            for r in range(1, 6):
                s = add_noise(clean_spectrum("polystyrene", AXIS, scale=scale), rng, 10)
                for wv, iv in zip(AXIS, s):
                    w.writerow([f"S{r:02d}", f"{wv:.4f}", f"{iv:.6f}"])
    _session(day_dir / "T074_a.csv", _SCALE)
    _session(day_dir / "T074_b.csv", _SCALE * 0.97)

    return {"T072_cluster_labels": labels}


# ── 프롬프트 주입 ─────────────────────────────────────────────────────────────
def _patch_prompts(manifest: dict):
    """tasks_raw.json 각 문항 task 끝에 'Input file(s): ...' 를 idempotent 추가."""
    raws = json.loads(_TASKS_RAW.read_text(encoding="utf-8"))
    changed = 0
    for r in raws:
        info = manifest.get(r["id"])
        if not info:
            continue
        names = info["files"]
        if not names:
            continue
        tag = " Input file(s): " + ", ".join(names) + "."
        if "Input file(s):" in r["task"]:
            # 기존 주입 갱신
            r["task"] = r["task"].split(" Input file(s):")[0].rstrip() + tag
        else:
            r["task"] = r["task"].rstrip() + tag
        changed += 1
    _TASKS_RAW.write_text(json.dumps(raws, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def main():
    ap = argparse.ArgumentParser(description="문항ID별 라만 스펙트럼 생성 + 프롬프트 주입")
    ap.add_argument("--date", default=None, help="업로드 날짜 폴더(기본: 오늘). 예: 2026-07-27")
    ap.add_argument("--no-patch", action="store_true", help="tasks_raw 프롬프트를 건드리지 않음")
    args = ap.parse_args()

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    day_dir = _UPLOADS / day
    day_dir.mkdir(parents=True, exist_ok=True)
    _REFS.mkdir(parents=True, exist_ok=True)

    tasks = build_all(np.random.default_rng(20260727))

    manifest = {}
    n_files = 0
    for tid, spec in tasks.items():
        fnames = []
        for fname, (axis, inten) in spec["files"].items():
            _write_spectrum(day_dir / fname, axis, inten)
            fnames.append(fname)
            n_files += 1
        for rname, (axis, inten) in spec["refs"].items():
            _write_spectrum(_REFS / rname, axis, inten)
        manifest[tid] = {
            "files": fnames,
            "reference_files": list(spec["refs"].keys()),
            "material": spec["material"],
            "description": spec["desc"],
            "ground_truth": spec["ground_truth"],
        }

    # 특수 파일
    _write_map_t050(day_dir)
    manifest["T050"]["files"] = ["T050.csv"]
    n_files += 1
    _write_library(day_dir)

    # 맵/세션(analysis_file 이면서 실측 불가) — T071/T072/T074
    maps_gt = _write_maps_and_sessions(day_dir)
    n_files += 4  # T071, T072, T074_a, T074_b
    manifest["T071"] = {"files": ["T071.csv"], "reference_files": [], "material": "polystyrene",
                        "description": "5x5 PS map for 5th-order baseline + L2 + 3-component PCA.",
                        "ground_truth": {}}
    manifest["T072"] = {"files": ["T072.csv"], "reference_files": [], "material": "PS/PET",
                        "description": "5x5 map with two material clusters (PS/PET) for k=2 clustering.",
                        "ground_truth": {"cluster_labels": maps_gt["T072_cluster_labels"]}}
    manifest["T074"] = {"files": ["T074_a.csv", "T074_b.csv"], "reference_files": [], "material": "polystyrene",
                        "description": "Two measurement sessions (5 PS repeats each) to compare.",
                        "ground_truth": {}}

    # 매칭 문항은 라이브러리도 입력에 포함해 프롬프트에 명시
    lib_tasks = {
        "T111": "reference_library.csv", "T112": "reference_library.csv",
        "T113": "reference_library.csv", "T114": "reference_library.csv",
        "T115": "reference_library.csv", "T116": "reference_library.csv",
        "T117": "reference_library.csv", "T118": "peak_library.csv",
        "T119": "reference_library_8.csv", "T120": "reference_library.csv",
        "T121": "peak_library.csv", "T122": "reference_library.csv",
        "T123": "reference_library.csv", "T124": "reference_library.csv",
        "T125": "reference_library.csv", "T126": "reference_library.csv",
        "T127": "reference_library.csv", "T128": "reference_library.csv",
    }
    for tid, lib in lib_tasks.items():
        if tid in manifest and lib not in manifest[tid]["files"]:
            manifest[tid]["files"].append(lib)

    _MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    patched = 0 if args.no_patch else _patch_prompts(manifest)

    print(f"[OK] {n_files}개 입력 CSV → {day_dir}")
    print(f"     라이브러리: reference_library.csv / reference_library_8.csv / peak_library.csv")
    print(f"     정답(reference) → {_REFS}")
    print(f"     매니페스트 → {_MANIFEST}  ({len(manifest)}개 문항)")
    print(f"     프롬프트 주입: {patched}개 문항 (tasks_raw.json)")
    print("\n다음: python -m backend.benchmark.build_tasks  로 tasks.json 재생성")


if __name__ == "__main__":
    main()
