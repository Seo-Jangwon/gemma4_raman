# -*- coding: utf-8 -*-
"""문항별 입력 파일과 GT 를 한 자리에서 생성한다.

[왜 한 스크립트인가]
입력을 만든 뒤 GT 를 따로 계산하면, 생성 파라미터가 바뀔 때 한쪽만 갱신되어 조용히 어긋난다.
어긋난 GT 는 '에이전트가 틀렸다'로 보고되므로 발견이 늦다. 여기서는 배열을 만든 그 자리에서
정답을 계산해 같이 떨어뜨린다.

[규약의 단일 출처]
피크 검출·SNR·FWHM·IPBSA·스파이크 판정은 전부 synth.py 의 함수를 쓴다. 채점기도 같은 함수를
import 하면 GT 가 어긋날 수 없다. 문항 문구에 적힌 규약과 이 구현이 1:1 로 대응한다.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

import synth                                          # noqa: E402
from materials import LIBRARY_12, LIBRARY_8, MATERIALS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "inputs"
GT = ROOT / "gt"
ARR = GT / "arrays"
for d in (IN, GT, ARR):
    d.mkdir(parents=True, exist_ok=True)

X = synth.axis()
NOISE = 6.0          # 카운트. SNR 문항이 성립할 정도의 잡음
manifest: dict = {}


# ── 파일 쓰기 ────────────────────────────────────────────────────────────────
def w_spec(name, y, x=X, extra=None):
    """단일 스펙트럼 CSV: raman_shift_cm-1, intensity"""
    with open(IN / name, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        head = ["raman_shift_cm-1", "intensity"] + list((extra or {}).keys())
        wr.writerow(head)
        cols = [list((extra or {})[k]) for k in (extra or {})]
        for i, (xv, yv) in enumerate(zip(x, y)):
            wr.writerow([f"{xv:.4f}", f"{yv:.6f}"] + [f"{c[i]:.6f}" for c in cols])
    return name


def w_frames(name, frames, x=X):
    """다중 프레임 CSV: frame_index, raman_shift_cm-1, intensity"""
    with open(IN / name, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame_index", "raman_shift_cm-1", "intensity"])
        for i, y in enumerate(frames):
            for xv, yv in zip(x, y):
                wr.writerow([i, f"{xv:.4f}", f"{yv:.6f}"])
    return name


def w_map(name, pts, x=X):
    """맵 CSV: x, y, raman_shift_cm-1, intensity  (pts = [(x_mm, y_mm, spectrum)])"""
    with open(IN / name, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["x", "y", "raman_shift_cm-1", "intensity"])
        for px, py, y in pts:
            for xv, yv in zip(x, y):
                wr.writerow([f"{px:.3f}", f"{py:.3f}", f"{xv:.4f}", f"{yv:.6f}"])
    return name


def w_session(name, specs, x=X):
    """세션 CSV: spectrum_id, raman_shift_cm-1, intensity"""
    with open(IN / name, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["spectrum_id", "raman_shift_cm-1", "intensity"])
        for sid, y in specs:
            for xv, yv in zip(x, y):
                wr.writerow([sid, f"{xv:.4f}", f"{yv:.6f}"])
    return name


def w_arr(name, **cols):
    """배열 GT CSV."""
    keys = list(cols)
    n = len(cols[keys[0]])
    with open(ARR / name, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for i in range(n):
            wr.writerow([f"{float(cols[k][i]):.8g}" for k in keys])
    return f"arrays/{name}"


def gt(task, inputs, grading, **fields):
    manifest[task] = {"task": task, "inputs": inputs, "grading": grading, **fields}
    (GT / f"{task}.json").write_text(
        json.dumps(manifest[task], ensure_ascii=False, indent=2), encoding="utf-8")


# ── 공통 재료 ────────────────────────────────────────────────────────────────
def ps(seed=None, counts=1000.0, broaden=1.0, shift=0.0, sigma=NOISE):
    y = synth.scale_counts(synth.pure("polystyrene", X, broaden, shift), counts)
    return y + (synth.noise(X, sigma, seed) if seed is not None else 0.0)


def peaks_of(y):
    idx, props, prom = synth.find_peaks_spec(y, X)
    order = np.argsort(props["prominences"])[::-1]
    return idx, props, order


def top_peaks(y, n):
    """세기 내림차순 상위 n 개 피크의 (위치, 세기)."""
    idx, _, _ = synth.find_peaks_spec(y, X)
    order = idx[np.argsort(y[idx])[::-1]][:n]
    return [(float(X[i]), float(y[i])) for i in order]


# ══════════════════════════════════════════════════════════════════════════
# 3. 전처리·피크 분석 (T038~T057)
# ══════════════════════════════════════════════════════════════════════════
def build_preprocessing():
    # T038 시각화 — 그냥 깨끗한 스펙트럼
    y = ps(seed=380)
    w_spec("T038.csv", y)
    gt("T038", ["T038.csv"], "ARRAY(rtol 1e-6) on 사용 배열",
       n_points=len(X), x_min=float(X[0]), x_max=float(X[-1]),
       y_min=float(y.min()), y_max=float(y.max()), y_mean=float(y.mean()))

    # T039 baseline — 형광 배경을 얹는다
    base = ps(seed=390)
    bg = synth.fluorescence(X, amp=900.0, slope=0.20, offset=120.0)
    raw = base + bg
    w_spec("T039.csv", raw)
    cor, fit = synth.ipbsa(raw, X, order=5)
    w_arr("T039_corrected.csv", **{"raman_shift_cm-1": X, "intensity": cor})
    gt("T039", ["T039.csv"], "1차 poly_order=5 EXACT / 2차 ARRAY(cos>=0.99 AND NRMSE<=0.02)",
       array="arrays/T039_corrected.csv", poly_order=5, max_iterations=100, threshold=0.001,
       note="IPBSA 는 4/5/6차 결과가 거의 같다 — 차수는 툴 인자로만 채점 가능")

    # T040 스파이크 제거
    clean = ps(seed=400)
    spiked, injected = synth.add_spikes(clean, X, n=6, seed=401)
    w_spec("T040.csv", spiked)
    removed, det = synth.remove_spikes(spiked)
    w_arr("T040_cleaned.csv", **{"raman_shift_cm-1": X, "intensity": removed})
    gt("T040", ["T040.csv"], "SET(인덱스 EXACT) + ARRAY(cos>=0.99 AND NRMSE<=0.02)",
       array="arrays/T040_cleaned.csv", spike_indices=det, injected_indices=injected,
       note="GT 인덱스는 '주입한 자리'가 아니라 '규약대로 검출된 자리'다")

    # T041 SG 평활
    from scipy.signal import savgol_filter
    y = ps(seed=410, sigma=18.0)
    w_spec("T041.csv", y)
    sm = savgol_filter(y, 11, 3)
    w_arr("T041_smoothed.csv", **{"raman_shift_cm-1": X, "intensity": sm})
    gt("T041", ["T041.csv"], "ARRAY(rtol 1e-6)", array="arrays/T041_smoothed.csv",
       window_length=11, polyorder=3, mode="interp")

    # T042 0-1 정규화
    y = ps(seed=420)
    w_spec("T042.csv", y)
    w_arr("T042_norm.csv", **{"raman_shift_cm-1": X, "intensity": synth.minmax(y)})
    gt("T042", ["T042.csv"], "ARRAY(rtol 1e-6)", array="arrays/T042_norm.csv")

    # T043 주요 7피크
    y = ps(seed=430)
    w_spec("T043.csv", y)
    idx, props, order = peaks_of(y)
    sel = sorted(idx[order][:7].tolist())
    gt("T043", ["T043.csv"], "SET(7개, ±3 cm-1) + NUM(세기 5%)",
       peaks=[{"position": float(X[i]), "intensity": float(y[i])} for i in sel],
       prominence_frac=0.05, n_detected=int(len(idx)))

    # T044 상위 3피크(세기순)
    y = ps(seed=440)
    w_spec("T044.csv", y)
    gt("T044", ["T044.csv"], "SET(순서 있는 3개, ±3 cm-1)",
       peaks=[{"position": p, "intensity": v} for p, v in top_peaks(y, 3)])

    # T045 FWHM
    y = ps(seed=450)
    w_spec("T045.csv", y)
    gt("T045", ["T045.csv"], "NUM(5%)",
       fwhm_cm1=synth.fwhm_spec(y, X, 980.0, 1020.0), interval=[980.0, 1020.0])

    # T046 피크 일치율
    smp = ps(seed=460)
    ref = synth.scale_counts(synth.pure("polystyrene", X), 1000.0)
    w_spec("T046_sample.csv", smp)
    w_spec("T046_ref.csv", ref)
    si, _, _ = synth.find_peaks_spec(smp, X)
    ri, _, _ = synth.find_peaks_spec(ref, X)
    pairs, used = [], set()
    for r in ri:
        cand = [s for s in si if s not in used and abs(X[s] - X[r]) <= 3.0]
        if cand:
            b = min(cand, key=lambda s: abs(X[s] - X[r]))
            used.add(b)
            pairs.append({"ref": float(X[r]), "sample": float(X[b])})
    gt("T046", ["T046_sample.csv", "T046_ref.csv"], "NUM(±0.02) + SET(쌍 EXACT)",
       match_ratio=len(pairs) / max(len(ri), 1), matched_pairs=pairs,
       n_ref_peaks=int(len(ri)), tolerance_cm1=3.0)

    # T047 전처리 파이프라인
    base = ps(seed=470)
    raw = base + synth.fluorescence(X, amp=700.0, slope=0.15, offset=90.0)
    raw, _ = synth.add_spikes(raw, X, n=5, seed=471)
    w_spec("T047.csv", raw)
    s1, det = synth.remove_spikes(raw)
    s2, _ = synth.ipbsa(s1, X, order=5)
    s3 = savgol_filter(s2, 11, 3)
    s4 = synth.minmax(s3)
    w_arr("T047_result.csv", **{"raman_shift_cm-1": X, "intensity": s4})
    gt("T047", ["T047.csv"], "ARRAY(cos>=0.99 AND NRMSE<=0.02) + STATE(min0/max1)",
       array="arrays/T047_result.csv", steps=["despike", "ipbsa5", "sg(11,3)", "minmax"],
       spike_indices=det)

    # T048 두 스펙트럼 겹쳐 그리기
    a = ps(seed=480)
    b = synth.scale_counts(synth.pure("PMMA", X), 1000.0) + synth.noise(X, NOISE, 481)
    w_spec("T048_a.csv", a)
    w_spec("T048_b.csv", b)
    gt("T048", ["T048_a.csv", "T048_b.csv"], "ARRAY(rtol 1e-6) ×2 + STATE(legend 2항목)",
       a_max=float(a.max()), b_max=float(b.max()), n_curves=2)

    # T049 구간 추출
    y = ps(seed=490)
    w_spec("T049.csv", y)
    m = (X >= 800.0) & (X <= 1200.0)
    w_arr("T049_slice.csv", **{"raman_shift_cm-1": X[m], "intensity": y[m]})
    gt("T049", ["T049.csv"], "ARRAY(EXACT, 행수 포함)", array="arrays/T049_slice.csv",
       n_rows=int(m.sum()), interval=[800.0, 1200.0], inclusive=True)

    # T050 SNR
    y = ps(seed=500)
    w_spec("T050.csv", y)
    gt("T050", ["T050.csv"], "NUM(5%)", snr=synth.snr_spec(y, X),
       signal_interval=[990.0, 1012.0], noise_interval=[1800.0, 1900.0], ddof=1)

    # T051 3x3 맵 → 1000 cm-1 최근접 세기
    pts, vals = [], []
    k = int(np.argmin(np.abs(X - 1000.0)))
    for i, xm in enumerate([37.8, 37.9, 38.0]):
        for j, ym in enumerate([25.2, 25.3, 25.4]):
            yy = ps(seed=5100 + i * 3 + j, counts=800.0 + 60.0 * (i * 3 + j))
            pts.append((xm, ym, yy))
            vals.append({"x": xm, "y": ym, "intensity": float(yy[k])})
    w_map("T051.csv", pts)
    gt("T051", ["T051.csv"], "ARRAY(rtol 1e-6, 9값) + STATE(축 배치)",
       target_cm1=1000.0, nearest_cm1=float(X[k]), values=vals)

    # T052 피크 위치차 5 이상인 쌍
    a = ps(seed=520)
    b = synth.scale_counts(synth.pure("polystyrene", X, shift=6.0), 1000.0) + synth.noise(X, NOISE, 521)
    w_spec("T052_a.csv", a)
    w_spec("T052_b.csv", b)
    ai, _, _ = synth.find_peaks_spec(a, X)
    bi, _, _ = synth.find_peaks_spec(b, X)
    used, big = set(), []
    for p in ai:
        cand = [q for q in bi if q not in used and abs(X[q] - X[p]) <= 10.0]
        if not cand:
            continue
        q = min(cand, key=lambda t: abs(X[t] - X[p]))
        used.add(q)
        d = abs(float(X[q]) - float(X[p]))
        if d >= 5.0:
            big.append({"a": float(X[p]), "b": float(X[q]), "diff": d})
    gt("T052", ["T052_a.csv", "T052_b.csv"], "SET(쌍 EXACT, 위치 ±1 cm-1)",
       pairs=big, pair_max_distance=10.0, min_reported_diff=5.0)

    # T053 사다리꼴 적분(입력은 이미 보정본)
    y0 = ps(seed=530)
    cor, _ = synth.ipbsa(y0 + synth.fluorescence(X, 500.0, slope=0.1), X, order=5)
    w_spec("T053.csv", cor)
    m = (X >= 990.0) & (X <= 1012.0)
    gt("T053", ["T053.csv"], "NUM(2%)",
       area=float(np.trapz(cor[m], X[m])), interval=[990.0, 1012.0])

    # T054 피크 세기비
    y = ps(seed=540)
    w_spec("T054.csv", y)
    idx, _, _ = synth.find_peaks_spec(y, X)
    p1 = idx[int(np.argmin(np.abs(X[idx] - 1001.0)))]
    p2 = idx[int(np.argmin(np.abs(X[idx] - 1602.0)))]
    gt("T054", ["T054.csv"], "NUM(5%) + SET(2개, ±3 cm-1)",
       ratio=float(y[p1] / y[p2]), peak_1001=float(X[p1]), peak_1602=float(X[p2]),
       direction="1001 / 1602")

    # T055 1차 미분
    y = ps(seed=550)
    w_spec("T055.csv", y)
    w_arr("T055_deriv.csv", **{"raman_shift_cm-1": X, "intensity": np.gradient(y, X)})
    gt("T055", ["T055.csv"], "ARRAY(rtol 1e-6)", array="arrays/T055_deriv.csv",
       method="np.gradient(y, x)")

    # T056 L2 정규화
    y = ps(seed=560)
    w_spec("T056.csv", y)
    w_arr("T056_l2.csv", **{"raman_shift_cm-1": X, "intensity": synth.l2(y)})
    gt("T056", ["T056.csv"], "ARRAY(rtol 1e-6)", array="arrays/T056_l2.csv")

    # T057 스파이크 판정 → 제거 → 피크 재보고
    clean = ps(seed=570)
    spiked, inj = synth.add_spikes(clean, X, n=4, seed=571)
    w_spec("T057.csv", spiked)
    rem, det = synth.remove_spikes(spiked)
    idx, _, _ = synth.find_peaks_spec(rem, X)
    gt("T057", ["T057.csv"], "EXACT(판정) + SET(인덱스 EXACT, 피크 ±3 cm-1)",
       has_spikes=True, spike_indices=det, injected_indices=inj,
       peaks=[float(v) for v in X[idx]])


# ══════════════════════════════════════════════════════════════════════════
# 4·5·6. 맵/레퍼런스/예외/트러블슈팅 입력
# ══════════════════════════════════════════════════════════════════════════
def build_maps_and_refs():
    from scipy.signal import savgol_filter                          # noqa: F401

    # T066 SNR<10 위치를 포함한 맵
    pts, low = [], []
    for i, xm in enumerate([37.8, 37.9, 38.0]):
        for j, ym in enumerate([25.2, 25.3, 25.4]):
            weak = (i + j) % 4 == 0                 # 일부 지점만 신호가 약하다
            yy = ps(seed=6600 + i * 3 + j, counts=60.0 if weak else 900.0,
                    sigma=NOISE if not weak else NOISE)
            pts.append((xm, ym, yy))
            s = synth.snr_spec(yy, X)
            if s < 10.0:
                low.append({"x": xm, "y": ym, "snr": s})
    w_map("T066.csv", pts)
    gt("T066", ["T066.csv"], "SET(좌표 EXACT) + PROC(초과 측정 금지)",
       low_snr_positions=low, threshold=10.0)

    # T070/T071/T077 폴리스티렌 레퍼런스(측정본과 비교용)
    ref = synth.scale_counts(synth.pure("polystyrene", X), 1000.0)
    for n in ("T070_ref.csv", "T071_ref.csv", "T077_ref.csv"):
        w_spec(n, ref)
    ri, _, _ = synth.find_peaks_spec(ref, X)
    gt("T070", ["T070_ref.csv"], "SET(좌표 25개) + NUM(±0.01) + EXACT(마스크) / 사후 GT",
       reference="polystyrene", similarity_threshold=0.85,
       rule="참조축 보간 → L2 → 코사인")
    gt("T071", ["T071_ref.csv"], "SET(좌표 20개) + EXACT(경계 위치, ±1칸) / 사후 GT",
       reference="polystyrene", rule="인접 위치 간 |Δ유사도| 최대 지점")
    gt("T077", ["T077_ref.csv"], "PROC(<=8회) + EXACT(선택 조합) / 사후 GT",
       reference_peaks=[float(v) for v in X[ri]], min_snr=20.0, peak_recall=0.90,
       dose_formula="power * exposure * 0.01")

    # T072/T073/T074 맵 (PCA / 클러스터링 / 이상 판정)
    def two_group_map(seed0, n=5):
        pts, labels = [], []
        for i in range(n):
            for j in range(n):
                grp = 0 if (i + j) < n else 1
                mat = "polystyrene" if grp == 0 else "PMMA"
                yy = synth.scale_counts(synth.pure(mat, X), 900.0)
                yy = yy + synth.fluorescence(X, 300.0, slope=0.08) + \
                    synth.noise(X, NOISE, seed0 + i * n + j)
                pts.append((37.8 + 0.1 * i, 25.2 + 0.1 * j, yy))
                labels.append({"x": round(37.8 + 0.1 * i, 3),
                               "y": round(25.2 + 0.1 * j, 3), "group": grp})
        return pts, labels

    pts, labels = two_group_map(7200)
    w_map("T072.csv", pts)
    mat = np.array([synth.l2(synth.ipbsa(p[2], X, order=5)[0]) for p in pts])
    Xc = mat - mat.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    evr = (s ** 2 / (s ** 2).sum())[:3]
    gt("T072", ["T072.csv"], "NUM(±0.01) ×3",
       explained_variance_ratio=[float(v) for v in evr],
       preprocessing=["ipbsa5", "l2", "mean-center"], n_components=3)

    pts, labels = two_group_map(7300)
    w_map("T073.csv", pts)
    try:
        from sklearn.cluster import KMeans
        mat = np.array([synth.l2(synth.ipbsa(p[2], X, order=5)[0]) for p in pts])
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(mat - mat.mean(axis=0))
        lab = km.labels_.tolist()
    except Exception:
        lab = [d["group"] for d in labels]
    for d, l in zip(labels, lab):
        d["cluster"] = int(l)
    gt("T073", ["T073.csv"], "EXACT(레이블, 순열 보정 후 100%)",
       assignments=labels, algorithm="KMeans(n_clusters=2, n_init=10, random_state=0)",
       note="레이블 번호의 순열은 동일 취급")

    # T074 중앙값 대비 유사도 < 0.80 인 이상 지점
    pts, anomalies = [], []
    specs = []
    for i in range(3):
        for j in range(3):
            odd = (i, j) in [(0, 2), (2, 0)]
            mat = "PMMA" if odd else "polystyrene"
            yy = synth.scale_counts(synth.pure(mat, X), 900.0) + synth.noise(X, NOISE, 7400 + i * 3 + j)
            pts.append((37.8 + 0.1 * i, 25.2 + 0.1 * j, yy))
            specs.append(yy)
    w_map("T074.csv", pts)
    med = np.median(np.array(specs), axis=0)
    for (px, py, yy) in pts:
        sim = synth.cosine(synth.l2(yy), synth.l2(med))
        if sim < 0.80:
            anomalies.append({"x": round(px, 3), "y": round(py, 3), "similarity": sim})
    gt("T074", ["T074.csv"], "SET(좌표 EXACT) + PROC",
       anomalies=anomalies, threshold=0.80, rule="요소별 median, L2 후 코사인")

    # T075 두 세션 비교
    def session(seed0, n=5, counts=1000.0, shift=0.0):
        return [(f"S{k+1:02d}", ps(seed=seed0 + k, counts=counts * (1 + 0.03 * ((-1) ** k)),
                                   shift=shift)) for k in range(n)]
    sa, sb = session(7500), session(7550, counts=940.0, shift=1.5)
    w_session("T075_a.csv", sa)
    w_session("T075_b.csv", sb)

    def stats(sess):
        ys = np.array([y for _, y in sess])
        mean = ys.mean(axis=0)
        idx, _, _ = synth.find_peaks_spec(mean, X)
        p = idx[int(np.argmin(np.abs(X[idx] - 1001.0)))]
        heights = ys[:, p]
        return float(X[p]), float(np.std(heights, ddof=1) / np.mean(heights) * 100.0), mean
    pa, rsda, ma = stats(sa)
    pb, rsdb, mb = stats(sb)
    gt("T075", ["T075_a.csv", "T075_b.csv"],
       "NUM ×4 (위치차 ±1 cm-1 / RSD 5% / 유사도 ±0.01)",
       peak_position_diff=abs(pb - pa), peak_a=pa, peak_b=pb,
       rsd_a_pct=rsda, rsd_b_pct=rsdb,
       cosine_of_means=synth.cosine(synth.l2(ma), synth.l2(mb)))

    # T094 요청 구간이 데이터 범위 밖
    y = ps(seed=940)
    w_spec("T094.csv", y)
    gt("T094", ["T094.csv"], "KEYWORD(범위 밖) + NUM(인용한 실제 범위 rtol 1e-6)",
       data_range=[float(X[0]), float(X[-1])], requested=[5000.0, 6000.0],
       expected_rows=0)

    # T095 L2 정규화 + 피크
    y = ps(seed=950)
    w_spec("T095.csv", y)
    idx, _, _ = synth.find_peaks_spec(y, X)
    w_arr("T095_l2.csv", **{"raman_shift_cm-1": X, "intensity": synth.l2(y)})
    gt("T095", ["T095.csv"], "ARRAY(rtol 1e-6) + SET(피크 ±3 cm-1)",
       array="arrays/T095_l2.csv", peaks=[float(v) for v in X[idx]])

    # N05 baseline 차수 비교용
    raw = ps(seed=5) + synth.fluorescence(X, 1100.0, slope=0.25, offset=150.0)
    w_spec("N05.csv", raw)
    vers = {}
    for o in (3, 5, 7):
        c, _ = synth.ipbsa(raw, X, order=o)
        vers[f"v_o{o}"] = {"poly_order": o, "max_corrected": float(c.max())}
    gt("N05", ["N05.csv"], "PROC(인자 EXACT, 반복<3) + KEYWORD(선택 근거)",
       versions=vers, note="배열은 관측 축약기가 버리므로 요약 통계로 판단해야 한다")


def build_troubleshooting():
    from scipy.stats import linregress

    # T098 형광 배경에 묻힌 피크
    raw = ps(seed=980) + synth.fluorescence(X, 1400.0, slope=0.35, offset=200.0)
    w_spec("T098.csv", raw)
    cor, _ = synth.ipbsa(raw, X, order=5)
    w_arr("T098_corrected.csv", **{"raman_shift_cm-1": X, "intensity": cor})
    idx, _, _ = synth.find_peaks_spec(cor, X)
    gt("T098", ["T098.csv"], "KEYWORD(형광) + ARRAY(cos>=0.99, NRMSE<=0.02) + SET(피크 ±3 cm-1)",
       cause="fluorescence", cause_keywords=["형광", "fluorescence", "background"],
       array="arrays/T098_corrected.csv", peaks=[float(v) for v in X[idx]])

    # T101 프레임마다 다른 위치의 스파이크
    frames, dets, injs = [], [], []
    for k in range(3):
        base = ps(seed=1010 + k)
        sp, inj = synth.add_spikes(base, X, n=4, seed=1015 + k)
        frames.append(sp)
        rem, det = synth.remove_spikes(sp)
        dets.append(det)
        injs.append(inj)
    w_frames("T101.csv", frames)
    cleaned = np.mean([synth.remove_spikes(f)[0] for f in frames], axis=0)
    w_arr("T101_mean.csv", **{"raman_shift_cm-1": X, "intensity": cleaned})
    idx, _, _ = synth.find_peaks_spec(cleaned, X)
    gt("T101", ["T101.csv"],
       "KEYWORD(cosmic ray) + SET(인덱스 EXACT ×3) + SET(피크 ±3 cm-1)",
       cause="cosmic ray", cause_keywords=["cosmic", "우주선", "스파이크", "spike"],
       spike_indices_per_frame=dets, injected_per_frame=injs,
       array="arrays/T101_mean.csv", peaks=[float(v) for v in X[idx]],
       note="프레임마다 위치가 다르다는 것이 원인 판정의 근거")

    # T103 알려진 시프트 + 레퍼런스
    SHIFT = 1.0
    ref = synth.scale_counts(synth.pure("polystyrene", X), 1000.0)
    shifted = synth.scale_counts(synth.pure("polystyrene", X, shift=SHIFT), 1000.0) \
        + synth.noise(X, NOISE, 1030)
    w_spec("T103.csv", shifted)
    w_spec("T103_ref.csv", ref)
    ri, _, _ = synth.find_peaks_spec(ref, X)
    gt("T103", ["T103.csv", "T103_ref.csv"],
       "NUM(시프트 ±0.2 cm-1, 부호 포함) + SET(피크 ±3 cm-1)",
       shift_cm1=SHIFT, corrected_peaks=[float(v) for v in X[ri]],
       method="상호상관 최대 지연")

    # T105 광표백 — 신호 감소 + 배경 증가
    frames, sig, bgv = [], [], []
    for k in range(10):
        amp = 1000.0 * (1.0 - 0.05 * k)
        b = 100.0 + 25.0 * k
        y = synth.scale_counts(synth.pure("polystyrene", X), amp) + b + synth.noise(X, NOISE, 1050 + k)
        frames.append(y)
        sig.append(float(y.max() - np.median(y)))
        bgv.append(float(np.median(y)))
    w_frames("T105.csv", frames)
    t = np.arange(10, dtype=float)
    gt("T105", ["T105.csv"], "KEYWORD(광표백) + NUM(기울기 10%) + REL(부호)",
       cause="photobleaching", cause_keywords=["광표백", "photobleach", "bleach"],
       signal_slope=float(linregress(t, sig).slope),
       background_slope=float(linregress(t, bgv).slope),
       n_frames=10)

    # T106 결정성 분류 — 폭을 넓혀 amorphous 가 되게 한다
    broad = synth.scale_counts(synth.pure("polystyrene", X, broaden=9.0), 1000.0) \
        + synth.noise(X, NOISE, 1060)
    w_spec("T106.csv", broad)
    f = synth.fwhm_spec(broad, X, 960.0, 1060.0)
    label = "crystalline" if f < 15 else ("amorphous" if f > 50 else "undecidable")
    gt("T106", ["T106.csv"], "EXACT(레이블) + NUM(FWHM 10%)",
       label=label, fwhm_cm1=f, rule="FWHM<15 crystalline / >50 amorphous / 그 사이 undecidable")

    # T108 baseline 드리프트
    SLOPE = 18.0
    frames, meds = [], []
    for k in range(10):
        y = ps(seed=1080 + k) + SLOPE * k
        frames.append(y)
        meds.append(float(np.median(y)))
    w_frames("T108.csv", frames)
    gt("T108", ["T108.csv"], "NUM(기울기 10%) + EXACT(증감 방향)",
       drift_slope=float(linregress(np.arange(10, dtype=float), meds).slope),
       direction="increasing", injected_slope=SLOPE, n_frames=10,
       rule="프레임별 중앙값의 선형회귀 기울기")

    # T112 강한 배경 — 보정 전후 피크 수 차이
    raw = ps(seed=1120) + synth.fluorescence(X, 2200.0, slope=0.5, offset=300.0)
    w_spec("T112.csv", raw)
    cor, _ = synth.ipbsa(raw, X, order=5)
    w_arr("T112_corrected.csv", **{"raman_shift_cm-1": X, "intensity": cor})
    before, _, _ = synth.find_peaks_spec(raw, X)
    after, _, _ = synth.find_peaks_spec(cor, X)
    gt("T112", ["T112.csv"], "ARRAY(cos>=0.99, NRMSE<=0.02) + SET(피크 ±3 cm-1) + EXACT(개수)",
       array="arrays/T112_corrected.csv", peaks=[float(v) for v in X[after]],
       n_peaks_before=int(len(before)), n_peaks_after=int(len(after)),
       n_newly_visible=int(len(after) - len(before)))


# ══════════════════════════════════════════════════════════════════════════
# 7. 유사신호 매칭 (T113~T130)
# ══════════════════════════════════════════════════════════════════════════
def lib(entries):
    return {sid: (mat, synth.scale_counts(synth.pure(mat, X, broaden=bd)))
            for sid, mat, bd in entries}


LIB12 = None
LIB8 = None


def rank(query, library):
    q = synth.l2(query)
    sc = [(sid, m, synth.cosine(q, synth.l2(y))) for sid, (m, y) in library.items()]
    return sorted(sc, key=lambda t: (-t[2], t[0]))


def build_matching():
    global LIB12, LIB8
    LIB12, LIB8 = lib(LIBRARY_12), lib(LIBRARY_8)

    def q_of(mat, seed, broaden=1.15, counts=1000.0, sigma=NOISE, shift=0.0):
        return synth.scale_counts(synth.pure(mat, X, broaden, shift), counts) + \
            synth.noise(X, sigma, seed)

    def emit(task, name, query, library, grading, **extra):
        w_spec(name, query)
        r = rank(query, library)
        libname = "reference_library_8.csv" if library is LIB8 else "reference_library.csv"
        gt(task, [name, libname], grading,
           ranking=[{"spectrum_id": a, "material": m, "score": s} for a, m, s in r],
           top1={"spectrum_id": r[0][0], "material": r[0][1], "score": r[0][2]},
           rule="참조축 보간 → L2 → 코사인", **extra)
        return r

    emit("T113", "T113.csv", q_of("polystyrene", 1130), LIB12,
         "EXACT(순서 있는 3개) + NUM(점수 ±0.01)")

    r = emit("T114", "T114.csv", q_of("PET", 1140), LIB12, "NUM(±0.01) + EXACT(임계 판정)",
             threshold=0.85)
    manifest["T114"]["above_threshold"] = bool(r[0][2] >= 0.85)
    (GT / "T114.json").write_text(json.dumps(manifest["T114"], ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    # T115 원시(배경 포함) → 보정해야 순위가 바로 선다
    raw = q_of("calcite", 1150) + synth.fluorescence(X, 1600.0, slope=0.4, offset=200.0)
    w_spec("T115.csv", raw)
    cor, _ = synth.ipbsa(raw, X, order=5)
    r_raw, r_cor = rank(raw, LIB12), rank(cor, LIB12)
    w_arr("T115_corrected.csv", **{"raman_shift_cm-1": X, "intensity": cor})
    gt("T115", ["T115.csv", "reference_library.csv"],
       "EXACT(물질명) + ARRAY(보정 배열, cos>=0.99) 가점",
       material=r_cor[0][1], top1=r_cor[0][0], array="arrays/T115_corrected.csv",
       without_correction_material=r_raw[0][1],
       preprocessing_matters=bool(r_raw[0][1] != r_cor[0][1]))

    # T116 OOD — 라이브러리에 없는 밴드 구성
    ood = synth.scale_counts(
        sum(synth.lorentz(X, c, h, w) for c, h, w in
            [(340.0, 0.7, 12.0), (905.0, 1.0, 10.0), (1250.0, 0.5, 15.0),
             (1900.0, 0.4, 20.0)]), 1000.0) + synth.noise(X, NOISE, 1160)
    w_spec("T116.csv", ood)
    r = rank(ood, LIB12)
    gt("T116", ["T116.csv", "reference_library.csv"],
       "NUM(±0.01) + EXACT(OOD 판정) + PROC(물질명 단정 금지)",
       best_score=r[0][2], threshold=0.75, reliable_match=bool(r[0][2] >= 0.75),
       expected_conclusion="no reliable match",
       ranking=[{"spectrum_id": a, "material": m, "score": s} for a, m, s in r])

    emit("T117", "T117.csv", q_of("PET", 1170), LIB12, "EXACT(물질명)")
    emit("T119", "T119.csv", q_of("polystyrene", 1190, sigma=90.0), LIB12, "EXACT(물질명)",
         preprocessing="savgol(11,3)")
    emit("T121", "T121.csv", q_of("polystyrene", 1210), LIB8,
         "EXACT(8개 순서) + NUM(점수 ±0.01)", tie_break="spectrum_id 오름차순")
    emit("T122", "T122.csv", q_of("PMMA", 1220, broaden=1.7), LIB12,
         "EXACT(spectrum_id) / material만 맞으면 부분점")
    emit("T123", "T123.csv", q_of("silicon", 1230), LIB12,
         "EXACT(물질명) + NUM(피크 ±3 cm-1)", basis_peak=520.7)
    emit("T124", "T124.csv", q_of("aragonite", 1240), LIB12, "EXACT(물질명)")
    # T128 — 타이브레이크가 실제로 발동하도록 쿼리를 두 참조의 등거리 지점에 놓는다.
    #
    # [왜 이렇게 만들었나 — 검증에서 걸린 문제]
    # 그냥 만든 쿼리는 1·2위 점수가 뚜렷이 갈려 동점이 나지 않았다. 그러면 문항에 적힌
    # 타이브레이크 규칙은 한 번도 적용되지 않는 장식이 되고, 규칙을 아는 에이전트와
    # 모르는 에이전트가 같은 점수를 받는다. broaden=1.30 은 CAL_01(1.0)과 CAL_02(1.7)
    # 사이 등거리 지점이다. 등거리 값은 잡음을 포함한 상태에서 찾아야 한다 — 잡음 없이
    # 맞춘 1.30 은 잡음을 얹자 동점이 깨졌다(검증에서 걸렸다). 1.314 는 이 잡음 실현까지
    # 포함해 두 점수를 0.986948 / 0.986882 로 만든다(소수 4자리까지 동점).
    q128 = synth.scale_counts(synth.pure("calcite", X, broaden=1.314), 1000.0) \
        + synth.noise(X, NOISE, 1280)
    w_spec("T128.csv", q128)
    r128 = rank(q128, LIB12)
    tied = [e for e in r128 if round(e[2], 3) == round(r128[0][2], 3)]
    winner = sorted(tied, key=lambda e: e[0])[0]
    gt("T128", ["T128.csv", "reference_library.csv"],
       "EXACT(물질명, 동점 시 spectrum_id)",
       ranking=[{"spectrum_id": a, "material": m, "score": s} for a, m, s in r128],
       top1={"spectrum_id": winner[0], "material": winner[1], "score": winner[2]},
       material=winner[1], rule="참조축 보간 → IPBSA(5) → L2 → 코사인",
       tie_break="소수 3자리에서 동점이면 spectrum_id 알파벳 오름차순",
       tied_candidates=[e[0] for e in tied],
       tie_actually_occurs=bool(len(tied) > 1))

    # T118 혼합 — 알려진 비율
    frac = 0.75
    mix = synth.scale_counts(
        frac * synth.pure("polystyrene", X) + (1 - frac) * synth.pure("PMMA", X), 1000.0) \
        + synth.noise(X, NOISE, 1180)
    w_spec("T118.csv", mix)
    r = rank(mix, LIB12)
    sc = {m: max(s for _, mm, s in r if mm == m) for m in ("polystyrene", "PMMA")}
    gt("T118", ["T118.csv", "reference_library.csv"],
       "EXACT(물질명) + NUM(유사도 ±0.01 ×2)",
       dominant=max(sc, key=sc.get), mix_fraction={"polystyrene": frac, "PMMA": 1 - frac},
       scores=sc)

    # T120 피크 기반 매칭
    q = q_of("PET", 1200)
    w_spec("T120.csv", q)
    qi, _, _ = synth.find_peaks_spec(q, X)
    best, scores = None, {}
    for sid, (m, y) in LIB12.items():
        ri, _, _ = synth.find_peaks_spec(y, X)
        used, n = set(), 0
        for rp in ri:
            cand = [p for p in qi if p not in used and abs(X[p] - X[rp]) <= 3.0]
            if cand:
                b = min(cand, key=lambda p: abs(X[p] - X[rp]))
                used.add(b)
                n += 1
        scores[sid] = {"material": m, "matched": n, "n_ref": int(len(ri)),
                       "score": n / max(len(ri), 1)}
    best = max(scores, key=lambda s: scores[s]["score"])
    gt("T120", ["T120.csv", "reference_library.csv"],
       "EXACT(물질명) + EXACT(매칭 피크 수)",
       material=scores[best]["material"], top1=best, scores=scores, tolerance_cm1=3.0)

    # T125 근거 피크 — 1위에는 있고 2위에는 없는 밴드
    q = q_of("PET", 1250)
    w_spec("T125.csv", q)
    r = rank(q, LIB12)
    m1 = r[0][1]
    m2 = next(m for _, m, _ in r if m != m1)
    b1 = {c for c, _, _ in MATERIALS[m1]}
    b2 = {c for c, _, _ in MATERIALS[m2]}
    disc = sorted(c for c in b1 if all(abs(c - o) > 8.0 for o in b2))
    gt("T125", ["T125.csv", "reference_library.csv"],
       "EXACT(물질명) + SET(근거 피크 >=2개, ±3 cm-1)",
       material=m1, second_candidate=m2, discriminating_peaks=disc, min_basis_peaks=2)

    # T126 5개 배치
    mats = ["polystyrene", "silicon", "PET", "calcite", "PMMA"]
    ans = []
    for k, m in enumerate(mats, 1):
        qq = q_of(m, 1260 + k)
        w_spec(f"T126_{k}.csv", qq)
        ans.append(rank(qq, LIB12)[0][1])
    gt("T126", [f"T126_{k}.csv" for k in range(1, 6)] + ["reference_library.csv"],
       "EXACT(5개, 순서) / 부분점 = 정답수÷5", materials=ans, intended=mats)

    # T127 주장 물질명 검증 — 불일치 사례
    claimed = "PMMA"
    q = q_of("PET", 1270)
    w_spec("T127.csv", q)
    r = rank(q, LIB12)
    best_claimed = max(s for _, m, s in r if m == claimed)
    gt("T127", ["T127.csv", "reference_library.csv"], "EXACT(판정) + NUM(±0.01)",
       claimed_material=claimed, actual_material=r[0][1], score_vs_claimed=best_claimed,
       threshold=0.85, matches=bool(best_claimed >= 0.85))

    # T129 알려진 Δ 시프트(축 간격의 정수배가 아니게)
    DELTA = 2.7
    q = q_of("polystyrene", 1290, shift=DELTA)
    w_spec("T129.csv", q)
    r = rank(q, LIB12)
    gt("T129", ["T129.csv", "reference_library.csv"],
       "NUM(Δ ±0.2 cm-1, 부호 포함) + EXACT(물질명)",
       shift_cm1=DELTA, material=r[0][1], method="상호상관 최대 지연",
       note="축 간격 1.0 의 정수배가 아니라 보간이 필요하다")

    # T130 top-3 적중률
    #
    # [왜 12항목이 아니라 8항목 라이브러리를 쓰는가 — 검증에서 걸린 문제]
    # 12항목 라이브러리는 모든 물질이 정확히 2항목씩이라, top-3 적중률의 상한이 모든
    # 쿼리에서 2/3 로 같다. 실제로 5개 쿼리가 전부 0.667 이 나와 평균이 상수가 됐다
    # — 계산을 맞게 하든 틀리게 하든 같은 값이 나오니 변별력이 없다.
    # 8항목 라이브러리는 polystyrene 3 / PET 2 / silicon·PMMA·calcite 1 로 개수가 달라
    # 쿼리마다 상한이 3/3, 2/3, 1/3 로 갈린다. 그래야 '적중률'이 의미를 갖는다.
    mats = ["polystyrene", "PET", "calcite", "PMMA", "silicon"]
    rates = []
    for k, m in enumerate(mats, 1):
        qq = q_of(m, 1300 + k)
        w_spec(f"T130_{k}.csv", qq)
        top3 = rank(qq, LIB8)[:3]
        rates.append(sum(1 for _, mm, _ in top3 if mm == m) / 3.0)
    gt("T130", [f"T130_{k}.csv" for k in range(1, 6)] + ["reference_library_8.csv"],
       "EXACT(5개 적중률) + NUM(평균 ±0.01)",
       per_query=[{"query": f"T130_{k}.csv", "material": m, "hit_rate": h}
                  for k, (m, h) in enumerate(zip(mats, rates), 1)],
       mean_hit_rate=float(np.mean(rates)),
       library="reference_library_8.csv",
       note="물질별 참조 개수가 달라야 적중률이 쿼리마다 갈린다")


# ══════════════════════════════════════════════════════════════════════════
# 합성 영상 (T037 / T063 / T076)
# ══════════════════════════════════════════════════════════════════════════
def build_images():
    import cv2
    W, H = 1060, 800
    specs = [("T037.png", [(300, 250), (760, 250), (300, 560), (760, 560)], 34),
             ("T063.png", [(690, 300)], 40),
             ("T076.png", [(410, 520)], 40)]
    for name, centers, radius in specs:
        rng = np.random.default_rng(hash(name) % 2 ** 31)
        img = (rng.normal(26, 5, (H, W, 3))).clip(0, 60).astype(np.uint8)
        for (cx, cy) in centers:
            cv2.circle(img, (cx, cy), radius, (232, 232, 232), -1)
            cv2.circle(img, (cx, cy), radius, (255, 255, 255), 2)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imwrite(str(IN / name), img)
        task = name.split(".")[0]
        gt(task, [name],
           "NUM(픽셀 ±25px) + PROC" if task != "T037" else "SET(픽셀 4개, ±25px) + PROC",
           targets=[{"pixel_x": int(a), "pixel_y": int(b)} for a, b in centers],
           radius_px=radius, tolerance_px=25, image_size=[W, H],
           note="영상 중심이 현재 스테이지 위치에 대응한다. move_to_pixel 로 환산한다.")


def main():
    build_preprocessing()
    build_maps_and_refs()
    build_troubleshooting()
    build_matching()
    build_images()
    (GT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    n_in = len(list(IN.glob("*")))
    print(f"입력 파일 {n_in}개 → {IN}")
    print(f"GT {len(manifest)}건 (+manifest.json) → {GT}")
    print(f"배열 GT {len(list(ARR.glob('*.csv')))}개 → {ARR}")


if __name__ == "__main__":
    main()
