# -*- coding: utf-8 -*-
"""부류 B 문항의 '모양새 일치' 판정 엔진.

[무엇을 재는가]
"5차 다항 baseline 보정" 같은 과제는 이름은 하나지만 구현 선택지가 여럿이고, 그에 따라
결과값이 크게 달라진다. 그래서 값 자체를 레퍼런스와 비교하면 correctness 가 아니라
구현 동일성을 재게 된다(task_class.py 의 서두 참조). 이 모듈은 대신
**어떤 정당한 구현이든 만족해야 하는 성질**만 검사한다.

    ① 피크 위치를 전부 살렸는가          recall  (±3 cm⁻¹)
    ② 없던 피크를 만들지 않았는가        precision (±3 cm⁻¹)
    ③ 피크 사이 상대 세기를 보존했는가   Δ상대세기
    ④ 전체 파형이 같은가                 pearson
    ⑤ 국소적으로 크게 어긋난 데가 없는가 max|Δ| (0~1 재정규화 후)

[임계값의 근거]
추측한 관용값이 아니라 T038 입력으로 실제 측정해서 정했다. 정당한 구현 4개는 전부
통과하고 틀린 답 5개는 전부 탈락하는 값이다 — selftest() 가 매번 이걸 재현한다.

    후보              recall  prec   Δ상대세기  pearson  max|Δ|(0-1)   판정
    plain polyfit      1.000  1.000   0.0000   1.00000    0.0000      통과
    표준화 축 polyfit   1.000  1.000   0.0000   1.00000    0.0000      통과
    iterative LMJ      1.000  1.000   0.0526   0.97917    0.0601      통과
    ALS                1.000  1.000   0.0612   0.97349    0.0797      통과
    무보정(원본)        0.857  1.000      -     0.13842    0.9102      탈락
    1차 baseline       1.000  1.000   0.1906   0.53929    0.5647      탈락
    9차 과적합          1.000  0.875   0.0541   0.96382    0.0716      탈락
    과평활(SG101)      0.429  0.600      -     0.83318    0.7807      탈락
    피크반전            0.000  0.000      -    -1.00000    1.0000      탈락

[한계 — 반드시 같이 읽을 것]
'2차 baseline'은 이 다섯 지표를 전부 통과한다(pearson 0.997, Δ상대세기 0.0099).
즉 **"5차"라는 명시된 지시를 어긴 것은 모양새로 잡히지 않는다.** 그건 절차(process)
점수로 코드·trace 에서 따로 봐야 한다. 모양새 판정은 결과(outcome) 점수일 뿐이다.

[앙상블 편차 S]
ensemble_band() 는 정당한 구현들을 실제로 돌려 그 최대 상호편차 S 를 낸다. 이 값이
"tolerance 1e-5 는 무엇을 재고 있었나"에 대한 데이터 답변이다 — 리포트에 함께 싣는다.
외부 패키지는 쓰지 않는다(numpy/scipy 만). 재현성과 버전 고정 부담 때문이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .task_class import DEFAULT_SHAPE, shape_thresholds


# ── 정당한 구현들 ────────────────────────────────────────────────────────────

def bl_plain(x, y, order=5):
    """통상 최소제곱 다항 적합. make_task_spectra 의 레퍼런스가 쓰는 방법."""
    return y - np.polyval(np.polyfit(x, y, order), x)


def bl_std_axis(x, y, order=5):
    """축을 표준화하고 적합. 수치적으로 더 안정하며 결과는 사실상 동일하다."""
    z = (x - x.mean()) / x.std()
    return y - np.polyval(np.polyfit(z, y, order), z)


def bl_lmj(x, y, order=5, n_iter=200):
    """Lieber-Mahadevan-Jansen iterative modified polyfit.

    적합 곡선보다 위에 있는 점(=피크)을 매 반복마다 곡선 값으로 눌러 내려, 피크가
    베이스라인을 끌어올리는 것을 막는다. 라만 전처리의 사실상 표준 방법 중 하나다.
    """
    yy = y.copy()
    for _ in range(n_iter):
        b = np.polyval(np.polyfit(x, yy, order), x)
        new = np.minimum(yy, b)
        if np.allclose(new, yy, atol=1e-12):
            break
        yy = new
    return y - np.polyval(np.polyfit(x, yy, order), x)


def bl_als(y, lam=1e5, p=0.01, n_iter=20):
    """Asymmetric Least Squares (Eilers & Boelens). 다항식이 아니라 평활 스플라인이지만
    '베이스라인을 빼라'는 지시의 정당한 해석에 포함된다."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sp.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    W = sp.spdiags(w, 0, L, L)
    z = y.copy()
    for _ in range(n_iter):
        W.setdiag(w)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z


ENSEMBLES = {
    "baseline_poly5": {
        "plain polyfit": lambda x, y: bl_plain(x, y, 5),
        "표준화 축 polyfit": lambda x, y: bl_std_axis(x, y, 5),
        "iterative LMJ": lambda x, y: bl_lmj(x, y, 5),
        "ALS": lambda x, y: bl_als(y),
    },
    "sgolay_11_3": {
        "mode=interp": lambda x, y: savgol_filter(y, 11, 3, mode="interp"),
        "mode=nearest": lambda x, y: savgol_filter(y, 11, 3, mode="nearest"),
        "mode=mirror": lambda x, y: savgol_filter(y, 11, 3, mode="mirror"),
        "mode=constant": lambda x, y: savgol_filter(y, 11, 3, mode="constant"),
    },
}


def ensemble_band(x, y_in, kind: str) -> tuple[dict[str, np.ndarray], float]:
    """정당한 구현들을 돌려 {이름: 결과} 와 최대 상호편차 S 를 낸다.

    S 는 "정당한 방법들 사이에서도 이만큼 벌어진다"는 뜻이므로, 이보다 작은
    tolerance 로 pass/fail 을 가르면 그건 correctness 가 아니라 구현 동일성이다.
    """
    fns = ENSEMBLES.get(kind)
    if not fns:
        return {}, 0.0
    out: dict[str, np.ndarray] = {}
    for name, fn in fns.items():
        try:
            out[name] = np.asarray(fn(np.asarray(x, float), np.asarray(y_in, float)), float)
        except Exception:                                        # noqa: BLE001
            continue
    vals = list(out.values())
    S = max((float(np.abs(a - b).max()) for a in vals for b in vals), default=0.0)
    return out, S


# ── 모양새 지표 ──────────────────────────────────────────────────────────────

def _unit01(v: np.ndarray) -> np.ndarray:
    """0~1 재정규화. 스케일이 다른 두 결과의 '모양'만 비교하기 위한 것이다."""
    v = np.asarray(v, float)
    lo = float(v.min())
    v = v - lo
    hi = float(v.max())
    return v / hi if hi > 0 else v


def peaks_of(x, y, prom_frac: float = 0.05) -> np.ndarray:
    """prominence 를 진폭 비율로 준 피크 검출. 절대 임계는 스케일에 따라 무의미하다."""
    y = np.asarray(y, float)
    ptp = float(np.ptp(y))
    if ptp <= 0:
        return np.array([])
    idx, _ = find_peaks(y, prominence=prom_frac * ptp)
    return np.asarray(x, float)[idx]


@dataclass
class ShapeResult:
    passed: bool
    metrics: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)      # 실패한 지표와 그 값
    ok_notes: list[str] = field(default_factory=list)     # 통과한 지표의 근거

    def summary(self) -> str:
        return " · ".join(self.reasons) if self.reasons else " · ".join(self.ok_notes)


def compare_shape(x, y_out, y_ref, thresholds: dict | None = None) -> ShapeResult:
    """산출 스펙트럼이 레퍼런스와 '모양새가 같은가'를 다섯 지표로 판정한다.

    값의 절대 일치는 요구하지 않는다 — ③④⑤ 는 모두 스케일 불변 지표거나 0~1
    재정규화 후에 잰다. 그래서 baseline 추정값이 달라 전체가 상하로 밀린 결과도
    피크 구조만 같으면 통과한다.
    """
    t = {**DEFAULT_SHAPE, **(thresholds or {})}
    x = np.asarray(x, float)
    yo = np.asarray(y_out, float)
    yr = np.asarray(y_ref, float)
    m: dict = {}
    bad: list[str] = []
    ok: list[str] = []

    if yo.shape != yr.shape:
        return ShapeResult(False, {"n_out": int(yo.size), "n_ref": int(yr.size)},
                           [f"점 수가 다르다 (산출 {yo.size} vs 레퍼런스 {yr.size})"])

    tol = t["peak_tol_cm"]
    pk_o = peaks_of(x, yo, t["peak_prom_frac"])
    pk_r = peaks_of(x, yr, t["peak_prom_frac"])
    m["n_peaks_out"], m["n_peaks_ref"] = int(pk_o.size), int(pk_r.size)
    m["peaks_out"] = [round(float(v), 1) for v in pk_o]
    m["peaks_ref"] = [round(float(v), 1) for v in pk_r]

    if pk_r.size == 0:
        recall = precision = float("nan")
    else:
        hit_r = sum(1 for p in pk_r if pk_o.size and np.min(np.abs(pk_o - p)) <= tol)
        hit_o = sum(1 for p in pk_o if np.min(np.abs(pk_r - p)) <= tol) if pk_o.size else 0
        recall = hit_r / pk_r.size
        precision = hit_o / pk_o.size if pk_o.size else 0.0
    m["peak_recall"], m["peak_precision"] = recall, precision

    if recall == recall:                                    # NaN 이 아니면
        if recall < t["min_recall"]:
            miss = [round(float(p), 1) for p in pk_r
                    if not (pk_o.size and np.min(np.abs(pk_o - p)) <= tol)]
            bad.append(f"피크 recall {recall:.3f} < {t['min_recall']:.2f} (놓친 피크 {miss})")
        else:
            ok.append(f"피크 {int(recall * pk_r.size)}/{pk_r.size}개 복원 (±{tol:g} cm⁻¹)")
        if precision < t["min_precision"]:
            spur = [round(float(p), 1) for p in pk_o if np.min(np.abs(pk_r - p)) > tol]
            bad.append(f"피크 precision {precision:.3f} < {t['min_precision']:.2f} "
                       f"(없는 피크를 만듦 {spur})")

    # ③ 상대 세기 — 최대 피크를 1 로 두고 나머지 피크의 높이 비율을 비교한다.
    #    피크가 전부 매칭됐을 때만 의미가 있다.
    d_rel = float("nan")
    if pk_r.size and recall == 1.0 and pk_o.size:
        io = [int(np.argmin(np.abs(pk_o - p))) for p in pk_r]
        ao = np.interp(pk_o[io], x, yo)
        ar = np.interp(pk_r, x, yr)
        if ao.max() > 0 and ar.max() > 0:
            d_rel = float(np.abs(ao / ao.max() - ar / ar.max()).max())
    m["d_rel_intensity"] = d_rel
    if d_rel == d_rel:
        if d_rel > t["max_d_rel_intensity"]:
            bad.append(f"피크 상대세기 어긋남 Δ={d_rel:.4f} > {t['max_d_rel_intensity']:.2f}")
        else:
            ok.append(f"상대세기 보존 Δ={d_rel:.4f}")

    # ④ 전체 파형
    pear = float(np.corrcoef(yo, yr)[0, 1]) if yo.std() > 0 and yr.std() > 0 else float("nan")
    m["pearson"] = pear
    if pear == pear:
        if pear < t["min_pearson"]:
            bad.append(f"파형 상관 pearson={pear:.5f} < {t['min_pearson']:.2f}")
        else:
            ok.append(f"파형 상관 {pear:.5f}")

    # ⑤ 국소 어긋남 — 스케일 차이를 없앤 뒤 본다
    d01 = float(np.abs(_unit01(yo) - _unit01(yr)).max())
    m["max_abs_01"] = d01
    if d01 > t["max_abs_01"]:
        bad.append(f"0~1 재정규화 후 max|Δ|={d01:.4f} > {t['max_abs_01']:.2f}")
    else:
        ok.append(f"0~1 max|Δ|={d01:.4f}")

    # 참고값 — 판정에는 쓰지 않는다. 기존 reference_match 가 무엇을 재고 있었는지 보여준다.
    m["max_abs_raw"] = float(np.abs(yo - yr).max())
    return ShapeResult(not bad, m, bad, ok)


# ── despike 계열 불변량 (answer_specs 의 선례를 함수로) ──────────────────────

def despike_invariant(y_in, y_out, y_ref, spike_delta: float = 1000.0,
                      non_spike_tol: float = 1e-5, min_removal: float = 0.99) -> ShapeResult:
    """스파이크 문항(T039/T056/T099) 전용.

    레퍼런스는 '스파이크를 넣기 전의 원본'이라 파괴된 값은 어떤 알고리즘으로도 복원할 수
    없다. 그래서 전 구간 일치는 정답 조건이 될 수 없고, 물어야 할 것은 두 가지뿐이다.
      · 스파이크가 아닌 점을 건드리지 않았는가
      · 스파이크 위치에서 초과분을 얼마나 제거했는가
    스파이크 위치는 '입력과 레퍼런스의 차가 spike_delta 를 넘는 점'으로 역산한다.
    """
    yi = np.asarray(y_in, float)
    yo = np.asarray(y_out, float)
    yr = np.asarray(y_ref, float)
    if not (yi.shape == yo.shape == yr.shape):
        return ShapeResult(False, {"n_in": yi.size, "n_out": yo.size, "n_ref": yr.size},
                           [f"점 수가 다르다 (입력 {yi.size} / 산출 {yo.size} / 레퍼런스 {yr.size})"])
    spike = np.abs(yi - yr) > spike_delta
    n_sp = int(spike.sum())
    m = {"n_spikes": n_sp}
    bad, ok = [], []

    if n_sp == 0:
        return ShapeResult(False, m, ["스파이크를 역산하지 못했다 (입력-레퍼런스 차가 임계 미만)"])

    nonspike_max = float(np.abs(yo - yr)[~spike].max())
    excess = np.abs(yi - yr)[spike]
    residual = np.abs(yo - yr)[spike]
    removal = float(np.mean(1.0 - residual / excess))
    m["nonspike_max_abs"] = nonspike_max
    m["removal_rate"] = removal
    m["max_abs_raw"] = float(np.abs(yo - yr).max())

    if nonspike_max > non_spike_tol:
        bad.append(f"스파이크가 아닌 점을 건드렸다 max|Δ|={nonspike_max:.4g} > {non_spike_tol:g}")
    else:
        ok.append(f"비-스파이크 {int((~spike).sum())}점 완전일치 (max|Δ|={nonspike_max:.2g})")
    if removal < min_removal:
        bad.append(f"스파이크 제거율 {removal * 100:.1f}% < {min_removal * 100:.0f}%")
    else:
        ok.append(f"스파이크 {n_sp}개 제거율 {removal * 100:.1f}%")
    return ShapeResult(not bad, m, bad, ok)


# ── 자기검증 ─────────────────────────────────────────────────────────────────

def selftest(verbose: bool = True) -> bool:
    """T038 실측으로 임계값을 재현한다 — 정당한 4개 통과, 틀린 5개 탈락.

    임계값을 누가 나중에 손대면 여기서 바로 깨진다. 그게 이 함수의 목적이다.
    """
    import csv
    from pathlib import Path
    from . import BENCH_DIR, PROJECT_ROOT

    up = sorted((PROJECT_ROOT / "data" / "uploads").glob("*/T038.csv"),
                key=lambda p: p.stat().st_mtime, reverse=True)
    ref_p = BENCH_DIR / "task_refs" / "T038_reference.csv"
    if not up or not ref_p.exists():
        print("selftest 생략 — T038 입력/레퍼런스를 찾지 못했다")
        return True

    def rd(p):
        lines = [l for l in Path(p).read_text(encoding="utf-8-sig").splitlines()
                 if not l.lstrip().startswith("#")]
        r = list(csv.DictReader(lines))
        return (np.array([float(a["raman_shift_cm-1"]) for a in r]),
                np.array([float(a["intensity"]) for a in r]))

    x, y = rd(up[0])
    _, ref = rd(ref_p)

    good = {k: fn(x, y) for k, fn in ENSEMBLES["baseline_poly5"].items()}
    bad = {
        "무보정(원본)": y.copy(),
        "1차 baseline": bl_plain(x, y, 1),
        "9차 과적합": bl_plain(x, y, 9),
        "과평활(SG101)": savgol_filter(bl_plain(x, y, 5), 101, 3),
        "피크반전": -bl_plain(x, y, 5),
    }
    th = shape_thresholds("T038")
    okall = True
    if verbose:
        print(f"{'후보':16} {'recall':>6} {'prec':>6} {'Δ상대':>7} {'pearson':>8} "
              f"{'max|Δ|01':>9} {'raw max|Δ|':>11}  판정")
    for label, cands, want in (("✅", good, True), ("❌", bad, False)):
        for k, v in cands.items():
            r = compare_shape(x, v, ref, th)
            hit = (r.passed == want)
            okall &= hit
            if verbose:
                m = r.metrics
                f = lambda z: ("  —  " if z != z else f"{z:.4f}")     # noqa: E731
                print(f"{label}{k:15} {f(m.get('peak_recall')):>6} {f(m.get('peak_precision')):>6} "
                      f"{f(m.get('d_rel_intensity')):>7} {f(m.get('pearson')):>8} "
                      f"{f(m.get('max_abs_01')):>9} {m.get('max_abs_raw', 0):>11.4g}  "
                      f"{'통과' if r.passed else '탈락'}{'' if hit else '  ← 기대와 다름!'}")
                if not r.passed:
                    print(f"{'':17}└ {r.summary()}")

    _, S = ensemble_band(x, y, "baseline_poly5")
    # SG 앙상블은 그 문항의 입력(T040)에서 재야 의미가 있다. T038 은 형광 배경이 커서
    # edge 효과가 과대평가된다.
    up40 = sorted((PROJECT_ROOT / "data" / "uploads").glob("*/T040.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    S2 = ensemble_band(*rd(up40[0]), "sgolay_11_3")[1] if up40 else float("nan")
    if verbose:
        print(f"\n앙상블 편차 S(baseline_poly5, T038) = {S:.4g} intensity units "
              f"= reference_match tolerance(1e-5) 의 {S / 1e-5:.2g}배")
        print(f"앙상블 편차 S(sgolay_11_3,   T040) = {S2:.4g} "
              f"= tolerance(1e-5) 의 {S2 / 1e-5:.2g}배")
        print(f"\nselftest: {'통과' if okall else '실패'}")
    return okall


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(0 if selftest() else 1)
