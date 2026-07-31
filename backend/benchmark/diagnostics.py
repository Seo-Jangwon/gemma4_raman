# -*- coding: utf-8 -*-
"""문항별 '자동 진단' — 정답 기준을 문장이 아니라 수치로 찍어 준다.

[왜 필요한가]
채점 콘솔에 산출 스펙트럼을 그려 놨어도, 정작 판정이 안 되는 문항이 많았다. 예를 들어
T049(SNR)는 기계검증기가 `tool_called` 하나뿐이라 "run_analysis 를 불렀다"만 확인되고
"보고한 74.24 가 맞는 값인가"는 사람이 손으로 계산해야 했다. T048(구간 추출)도 "원본과
행 순서·값이 완전히 같은가"를 눈으로는 확인할 방법이 없었다.

그래서 이 모듈은 문항마다 채점기준에 적힌 계산을 <b>입력 파일로 직접 다시 수행</b>하고,
  기준(정확한 통과 조건) · 실측(다시 계산한 값) · 에이전트 보고값 · 판정
네 칸을 표로 낸다. 사람은 그 표만 보고 정답/오답을 찍으면 된다.

[설계 원칙]
· 기준은 문항의 grading_criteria 문장을 그대로 수식으로 옮긴다. 추측한 관용 기준을
  쓰지 않는다 — 예: T049 는 "sample standard deviation" 이라 적혀 있으므로 ddof=1 이
  기준이고, 에이전트가 쓴 ddof=0 과 둘 다 계산해 차이를 보여 준다.
· 답변 텍스트에서 숫자를 찾을 때는 '기대값 근처의 값이 있는가'로 본다. 답변에는
  중간계산·파일명 숫자가 섞여 있어 위치로 특정할 수 없다.
· 계산이 불가능한 항목(그림의 시각적 판단 등)은 verdict='info' 로 두고 사람에게 넘긴다.
  자동 판정을 못 하는 걸 pass 로 눙치면 채점이 오히려 나빠진다.
· 한 문항의 진단이 실패해도 나머지는 나와야 한다 — 항목마다 예외를 잡는다.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

# sklearn(T071 PCA · T072 KMeans)이 끌고 오는 joblib/loky 는 Windows 에서 물리 코어 수를
# 세려고 외부 명령(wmic)을 실행하는데, 그게 없으면 매 실행마다 스택트레이스를 낀
# UserWarning 을 뿜는다(WinError 2). 채점 결과와 무관하지만 진짜 오류로 오해하기 쉽다.
#
# 두 가지를 같이 해야 조용해진다.
#  ① LOKY_MAX_CPU_COUNT 를 실제 코어 수보다 '작게' 준다. joblib 은 이 값이 os 코어 수보다
#     작을 때만 물리 코어 조회를 건너뛴다(같은 값이면 그대로 조회하고 실패한다).
#     여기 쓰는 PCA/KMeans 는 25×351 크기라 코어 하나 덜 쓰는 비용이 사실상 없다.
#  ② 그래도 다른 경로로 조회가 일어날 때를 대비해 그 경고 하나만 좁게 끈다.
#     (트레이스백은 joblib 이 traceback.print_tb 로 따로 찍어서 경고 필터로는 못 막는다 —
#      그래서 ① 이 본체이고 ② 는 보조다.)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 4) - 1)))

import warnings as _warnings
_warnings.filterwarnings(
    "ignore", message=r"(?s).*Could not find the number of physical cores.*",
    category=UserWarning)

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_UPLOADS = _PROJECT_ROOT / "data" / "uploads"

PASS, FAIL, INFO = "pass", "fail", "info"


# ── 입출력 로딩 ──────────────────────────────────────────────────────────────

def _rows(path: Path) -> list[dict]:
    """'#' 메타 주석행을 건너뛰고 CSV 를 dict 목록으로. 측정 자동저장 CSV 가 그 형식이다."""
    lines = [l for l in path.read_text(encoding="utf-8-sig").splitlines()
             if not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def find_input(name: str) -> Path | None:
    hits = sorted(_UPLOADS.glob(f"*/{name}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def load_xy(name: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    p = find_input(name)
    if p is None:
        return None, None
    rs = _rows(p)
    if not rs:
        return None, None
    xk = next((k for k in rs[0] if "raman_shift" in k or "wavenumber" in k), None)
    yk = next((k for k in rs[0] if k.strip().lower() == "intensity"), None)
    if xk is None or yk is None:
        return None, None
    return (np.array([float(r[xk]) for r in rs]),
            np.array([float(r[yk]) for r in rs]))


def load_groups(name: str, gkey: str | None = None) -> dict:
    """point_id / rep_id / spectrum_id 로 묶인 다중 스펙트럼 파일 → {키: (x, y)}."""
    p = find_input(name)
    if p is None:
        return {}
    rs = _rows(p)
    if not rs:
        return {}
    cols = list(rs[0].keys())
    gk = gkey or next((k for k in ("point_id", "rep_id", "spectrum_id") if k in cols), None)
    xk = next((k for k in cols if "raman_shift" in k), None)
    yk = next((k for k in cols if k.strip().lower() == "intensity"), None)
    if not (gk and xk and yk):
        return {}
    out: dict[str, list] = {}
    meta: dict[str, dict] = {}
    for r in rs:
        out.setdefault(r[gk], []).append((float(r[xk]), float(r[yk])))
        if r[gk] not in meta and "x_mm" in cols:
            meta[r[gk]] = {"x_mm": float(r["x_mm"]), "y_mm": float(r["y_mm"])}
    res = {}
    for k, v in out.items():
        res[k] = (np.array([a for a, _ in v]), np.array([b for _, b in v]))
    return {"data": res, "meta": meta}


# ── 답변에서 숫자 찾기 ───────────────────────────────────────────────────────

def answer_numbers(rec: dict) -> list[float]:
    from verifiers import _extract_numbers
    return _extract_numbers(rec.get("answer") or rec.get("final_report") or "")


def find_reported(rec: dict, expected: float, rel: float = 0.01,
                  absolute: float | None = None) -> float | None:
    """답변에 기대값 근처의 수가 있으면 그걸 돌려준다. 없으면 None.

    답변에는 중간계산·파일명 숫자가 섞여 있어 '몇 번째 숫자'로 특정할 수 없다. 그래서
    허용범위 안에 드는 값을 찾는 방식을 쓴다 — 있으면 '보고했다', 없으면 '못 찾았다'.
    """
    tol = absolute if absolute is not None else abs(expected) * rel
    best, bd = None, None
    for v in answer_numbers(rec):
        d = abs(v - expected)
        if d <= tol and (bd is None or d < bd):
            best, bd = v, d
    return best


def _decimals(v: float) -> int:
    """숫자를 몇 자리로 표시했는지. 0.6667 → 4."""
    t = repr(float(v))
    return len(t.split(".")[1]) if "." in t else 0


def compare_reported(rec: dict, expected: float, rel: float = 0.01,
                     absolute: float | None = None) -> dict:
    """답변의 값을 기준값과 비교한다. → {value, err, verdict, note}

    허용오차만으로 보면 <b>표시 반올림 때문에 정답이 오답으로 뒤집힌다</b> — 예: T128 은
    허용오차가 1e-6 인데 참값 0.666666… 을 답변이 0.6667 로 적으면 오차가 3.3e-5 라
    실패로 잡힌다. 그래서 '적은 자리수로 참값을 반올림한 값과 같은가'를 따로 보고,
    그 경우 통과로 두면서 사유를 note 에 남긴다(채점자가 뒤집을 수 있게).
    """
    tol = absolute if absolute is not None else abs(expected) * rel
    nums = answer_numbers(rec)
    if not nums:
        return {"value": None, "err": None, "verdict": FAIL, "note": "답변에 숫자가 없다"}
    hit = min(nums, key=lambda v: abs(v - expected))
    err = abs(hit - expected)
    if err <= tol:
        return {"value": hit, "err": err, "verdict": PASS, "note": ""}
    for v in nums:                        # 표시 반올림으로 설명되는가
        d = _decimals(v)
        if d and abs(round(expected, d) - v) < 10 ** (-d - 3):
            return {"value": v, "err": abs(v - expected), "verdict": PASS,
                    "note": (f"허용오차({tol:g})보다는 크지만 소수 {d}자리로 반올림하면 참값과 "
                             f"정확히 같다 — 표시 반올림이지 계산 오류가 아니다")}
    return {"value": hit, "err": err, "verdict": FAIL,
            "note": f"허용범위 밖 — 가장 가까운 값 {hit:g}, 오차 {err:.3g} > 허용 {tol:g}"}


def code_of(rec: dict, tool: str = "run_analysis") -> str:
    """이 실행에서 그 툴에 넘긴 코드 전부를 이어 붙인다(여러 번 호출할 수 있다)."""
    out = []
    for c in rec.get("tool_calls") or []:
        if c.get("name") == tool:
            out.append(str((c.get("args") or {}).get("code") or ""))
    return "\n".join(out)


def stdout_of(rec: dict, tool: str = "run_analysis") -> str:
    out = []
    for c in rec.get("tool_calls") or []:
        if c.get("name") != tool:
            continue
        r = c.get("result")
        if isinstance(r, dict) and r.get("stdout"):
            out.append(str(r["stdout"]))
    return "\n".join(out)


def _fmt(v, nd=4) -> str:
    if v is None:
        return "—"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_fmt(x, nd) for x in v) + "]"
    if isinstance(v, (bool, np.bool_)):
        return "예" if v else "아니오"
    if isinstance(v, str):
        return v
    a = abs(float(v))
    if a != 0 and (a < 1e-4 or a >= 1e6):
        return f"{v:.3e}"
    return f"{float(v):.{nd}f}".rstrip("0").rstrip(".")


def row(name: str, criterion: str, measured, reported=None, verdict: str = INFO,
        note: str = "") -> dict:
    return {"name": name, "criterion": criterion, "measured": _fmt(measured),
            "reported": _fmt(reported), "verdict": verdict, "note": note}


# ── 피크 검출 (여러 문항이 공유) ─────────────────────────────────────────────

def detect_peaks(x, y, prom_frac=0.05) -> np.ndarray:
    from scipy.signal import find_peaks
    idx, _ = find_peaks(y, prominence=prom_frac * (y.max() - y.min()))
    return x[idx]


# ── 문항별 진단 ──────────────────────────────────────────────────────────────

def _plot_integrity(rec: dict, files: list[str], expect_lines: int) -> list[dict]:
    """플로팅 문항 공통 — '값을 변형하지 않고 그렸는가'는 코드로 확인할 수 있다.

    그림 자체가 맞게 그려졌는지(선 개수·범례)는 사람이 봐야 하지만, 정규화·스무딩·로그
    같은 값 변형이 코드에 있는지는 기계로 잡힌다. 채점기준의 '원본과 좌표값이 동일'
    조항이 바로 그것이다.
    """
    code = code_of(rec)
    rows = []
    # 값을 바꾸는 연산들. plot/xlabel 등 표현 관련 호출은 제외한다.
    bad = [p for p in ("savgol_filter", "medfilt", "normalize", "polyfit", "np.log",
                       "log10", "/ *max\\(", "minmax", "gaussian_filter", "detrend")
           if re.search(p, code)]
    rows.append(row("값 변형 연산", "코드에 정규화·스무딩·로그·베이스라인 등 값 변형이 없어야 한다",
                    "없음" if not bad else f"발견: {', '.join(bad)}",
                    verdict=PASS if not bad else FAIL))
    plotted = [f for f in files if f in code]
    rows.append(row("플롯 대상 파일", f"입력 {len(files)}개가 모두 그려져야 한다",
                    f"{len(plotted)}/{len(files)} ({', '.join(plotted) or '없음'})",
                    verdict=PASS if len(plotted) == len(files) else FAIL))
    lab = {k: re.search(rf"{k}\(\s*['\"]([^'\"]+)", code) for k in ("xlabel", "ylabel")}
    rows.append(row("축 라벨", "가로축=라만 시프트(cm⁻¹), 세로축=세기",
                    " / ".join(m.group(1) if m else "없음" for m in lab.values()),
                    verdict=PASS if all(lab.values()) else FAIL))
    has_leg = "legend(" in code
    if expect_lines == 1:
        rows.append(row("범례", "선이 하나뿐이라 범례는 필수가 아니다",
                        "legend() 있음" if has_leg else "없음(선 1개라 무관)", verdict=PASS))
    else:
        rows.append(row("범례", f"선 {expect_lines}개를 구분하는 범례가 있어야 한다",
                        "legend() 호출 있음" if has_leg else "legend() 없음",
                        verdict=PASS if has_leg else FAIL,
                        note="라벨이 어느 파일인지 맞는지는 아래 그림으로 확인"))
    rows.append(row("그림 산출", "그림 파일이 실제로 생성되어야 한다(아래 갤러리)",
                    "코드에 show()/savefig 있음" if re.search(r"show\(|savefig", code) else "없음",
                    verdict=PASS if re.search(r"show\(|savefig", code) else FAIL,
                    note="선 개수·범례 표기의 시각적 정확성은 아래 그림으로 사람이 확인"))
    return rows


def d_T037(rec, task, tf):
    x, y = load_xy("T037.csv")
    rows = _plot_integrity(rec, ["T037.csv"], 1)
    if x is not None:
        rows.append(row("입력 데이터", "참고: 입력 스펙트럼의 실제 범위",
                        f"{len(x)}점, x {x.min():.0f}~{x.max():.0f} cm⁻¹, y {y.min():.2f}~{y.max():.2f}"))
    return rows


def d_T042(rec, task, tf):
    x, y = load_xy("T042.csv")
    exp = (tf.get("ground_truth") or {}).get("peaks_major") or []
    rows = []
    if x is not None:
        det = detect_peaks(x, y, 0.03)
        rows.append(row("입력에서 재검출한 피크", "정답 키와 일치해야 한다(키 검증)",
                        [round(float(v), 1) for v in det], exp,
                        PASS if len(det) == len(exp) and all(
                            min(abs(det - e)) <= 3.0 for e in exp) else INFO))
    got = [find_reported(rec, e, absolute=3.0) for e in exp]
    ok = all(g is not None for g in got)
    rows.append(row("답변의 피크 위치", f"{len(exp)}개 전부 ±3 cm⁻¹ 안에 보고",
                    exp, got, PASS if ok else FAIL,
                    note="" if ok else f"누락 {[e for e, g in zip(exp, got) if g is None]}"))
    return rows


def d_T043(rec, task, tf):
    x, y = load_xy("T043.csv")
    rows = []
    exp = (tf.get("ground_truth") or {}).get("top3_by_intensity_cm-1") or []
    if x is not None:
        det = detect_peaks(x, y, 0.03)
        idx = [int(np.argmin(np.abs(x - p))) for p in det]
        top = sorted(idx, key=lambda i: -y[i])[:3]
        rows.append(row("입력에서 재계산한 top-3", "세기 내림차순 상위 3개(키 검증)",
                        [f"{x[i]:.0f}(I={y[i]:.1f})" for i in top], exp,
                        PASS if [round(float(x[i])) for i in top] == list(exp) else INFO))
    nums = answer_numbers(rec)
    seq, pos = [], -1
    for e in exp:                                  # 등장 '순서'까지 봐야 한다
        hit = next((i for i, v in enumerate(nums) if i > pos and abs(v - e) <= 3.0), None)
        seq.append(None if hit is None else nums[hit])
        pos = hit if hit is not None else pos
    ok = all(s is not None for s in seq)
    rows.append(row("답변의 top-3 (순서 포함)", "세기 내림차순으로 ±3 cm⁻¹ 안에 보고",
                    exp, seq, PASS if ok else FAIL))
    return rows


def _fwhm(x, y, baseline, interp=True):
    ys = y - baseline
    i = int(np.argmax(ys))
    hm = ys[i] / 2

    def cross(a, b, ya, yb):
        return a + (hm - ya) * (b - a) / (yb - ya)
    l = int(np.where(ys >= hm)[0][0])
    xl = cross(x[l - 1], x[l], ys[l - 1], ys[l]) if (interp and l > 0) else x[l]
    r = int(np.where(ys[i:] < hm)[0][0]) + i
    xr = cross(x[r - 1], x[r], ys[r - 1], ys[r]) if interp else x[r - 1]
    return float(xr - xl)


def d_T044(rec, task, tf):
    x, y = load_xy("T044.csv")
    exp = float((tf.get("ground_truth") or {}).get("fwhm_cm-1_approx") or 12.0)
    lo, hi = exp * 0.95, exp * 1.05
    rows = [row("정답(참값)", "합성에 쓴 로렌치안의 해석적 FWHM = 2w = 2×6.0", exp,
                note="채점기준: 레퍼런스의 5% 이내")]
    if x is None:
        return rows
    m = (x >= 980) & (x <= 1020)
    xc, yc = x[m], y[m]
    edge = (yc[0] + yc[-1]) / 2
    variants = [("양끝 평균을 베이스라인 + 보간 (에이전트 방법)", edge, True),
                ("구간 최소값을 베이스라인 + 보간", float(yc.min()), True),
                ("스펙트럼 실제 바닥을 차감 + 보간", float(y.min()), True),
                ("베이스라인 미차감 + 보간", 0.0, True),
                ("베이스라인 미차감 + 보간 없음", 0.0, False)]
    for name, b, it in variants:
        v = _fwhm(xc, yc, b, it)
        # 이 행들은 '방법을 바꾸면 값이 이렇게 달라진다'는 참고자료다. 에이전트 책임이
        # 아니므로 판정을 붙이지 않는다 — 붙이면 종합판정이 참고행 때문에 실패로 뒤집힌다.
        rows.append(row(f"　{name}", f"허용 구간 {lo:.2f} ~ {hi:.2f} 안에 드는가",
                        f"{v:.4f} → {'안' if lo <= v <= hi else '밖'}",
                        note=f"베이스라인={b:.2f}, 참값 대비 {100*(v-exp)/exp:+.2f}%"))
    rep = find_reported(rec, exp, rel=0.05)
    allnums = [v for v in answer_numbers(rec) if 5 < v < 30]
    rows.append(row("답변의 FWHM", f"{lo:.2f} ~ {hi:.2f} 안의 값을 보고", f"{lo:.2f}~{hi:.2f}",
                    rep if rep is not None else f"허용구간 밖 (후보 {allnums})",
                    PASS if rep is not None else FAIL))
    rows.append(row("왜 좁게 나오나", "참고 — 구간 양끝은 순수 바닥이 아니다",
                    f"y(980)={yc[0]:.2f}, y(1020)={yc[-1]:.2f} vs 실제 바닥 {y.min():.2f}",
                    note="양끝값에는 1001 피크의 자기 꼬리와 30cm⁻¹ 옆 1031 피크 꼬리가 "
                         "섞여 있다. 그걸 베이스라인으로 빼면 반치 높이가 올라가 폭이 좁아진다."))
    return rows


def d_T045(rec, task, tf):
    xm, ym = load_xy("T045.csv")
    xr, yr = load_xy("T045_ref.csv")
    rows = []
    if xm is None or xr is None:
        return [row("입력", "T045.csv / T045_ref.csv 필요", "파일을 못 찾음", verdict=INFO)]
    ref_pk = detect_peaks(xr, yr, 0.05)
    meas_pk = detect_peaks(xm, ym, 0.05)
    # 중복 없는 1:1 매칭 — 가까운 쌍부터 그리디로 짝지어야 '중복 없이' 조건을 만족한다
    pairs = sorted(((abs(r - m), float(r), float(m)) for r in ref_pk for m in meas_pk
                    if abs(r - m) <= 3.0))
    used_r, used_m, matched = set(), set(), []
    for d, r, m in pairs:
        if r in used_r or m in used_m:
            continue
        used_r.add(r); used_m.add(m); matched.append((r, m, d))
    ratio = len(matched) / len(ref_pk) if len(ref_pk) else 0.0
    rows.append(row("레퍼런스 피크 수", "레퍼런스 스펙트럼에서 검출된 피크",
                    f"{len(ref_pk)}개 {[round(float(v),1) for v in ref_pk]}"))
    rows.append(row("측정 피크 수", "측정 스펙트럼에서 검출된 피크(잡음 때문에 많을 수 있다)",
                    f"{len(meas_pk)}개"))
    rows.append(row("1:1 매칭 결과", "±3 cm⁻¹, 중복 없이 짝지음",
                    f"{len(matched)}개 매칭 → {[f'{r:.0f}↔{m:.0f}' for r, m, _ in matched]}"))
    rep = find_reported(rec, ratio * 100, rel=0.02) or find_reported(rec, ratio, rel=0.02)
    rows.append(row("일치율", f"매칭수/레퍼런스수 = {len(matched)}/{len(ref_pk)}",
                    f"{ratio:.4f} (= {ratio*100:.2f}%)", rep,
                    PASS if rep is not None else FAIL,
                    note="피크 검출 파라미터에 따라 측정 피크 수가 달라진다 — "
                         "에이전트가 쓴 검출 기준이 타당한지는 아래 코드로 확인"))
    return rows


def d_T047(rec, task, tf):
    rows = _plot_integrity(rec, ["T047_a.csv", "T047_b.csv"], 2)
    for n in ("T047_a.csv", "T047_b.csv"):
        x, y = load_xy(n)
        if x is None:
            continue
        rows.append(row(f"{n} 실측", "참고: 그림의 선과 대조할 원본 특징",
                        f"{len(x)}점, y {y.min():.2f}~{y.max():.2f}, "
                        f"피크 {[round(float(v)) for v in detect_peaks(x, y, 0.05)]}"))
    return rows


def d_T048(rec, task, tf):
    import spectra_panel as sp
    lo, hi = (tf.get("ground_truth") or {}).get("range") or [800, 1200]
    src = find_input("T048.csv")
    csvs, _ = sp.find_outputs(rec)
    rows = [row("추출 구간", f"{lo} ≤ x ≤ {hi} cm⁻¹ (경계 포함)", f"{lo} ~ {hi}")]
    if src is None or not csvs:
        rows.append(row("산출 파일", "구간만 담은 CSV 가 저장되어야 한다",
                        "저장 파일 없음" if not csvs else "입력을 못 찾음", verdict=FAIL))
        return rows
    orig = _rows(src)
    xk = next(k for k in orig[0] if "raman_shift" in k)
    exp = [(float(r[xk]), float(r["intensity"])) for r in orig
           if lo <= float(r[xk]) <= hi]
    out = _rows(csvs[0])
    xk2 = next((k for k in out[0] if "raman_shift" in k), None)
    got = [(float(r[xk2]), float(r["intensity"])) for r in out] if xk2 else []
    xs = [a for a, _ in got]
    rows.append(row("행 수", f"구간 내 원본 행 수와 같아야 한다 ({len(exp)}행)",
                    f"{len(got)}행", verdict=PASS if len(got) == len(exp) else FAIL))
    rows.append(row("경계 포함", f"{lo} 과 {hi} 이 모두 들어 있어야 한다",
                    f"{lo}: {'있음' if float(lo) in xs else '없음'}, "
                    f"{hi}: {'있음' if float(hi) in xs else '없음'}",
                    verdict=PASS if (float(lo) in xs and float(hi) in xs) else FAIL))
    n_out = sum(1 for v in xs if v < lo or v > hi)
    rows.append(row("구간 밖 행", "0개여야 한다", f"{n_out}개",
                    verdict=PASS if n_out == 0 else FAIL))
    rows.append(row("행 순서", "원본 순서(오름차순) 유지", "유지" if xs == sorted(xs) else "뒤바뀜",
                    verdict=PASS if xs == sorted(xs) else FAIL))
    same = exp == got
    note = ""
    if not same and len(exp) == len(got):
        note = f"세기 max|Δ| = {max(abs(a[1]-b[1]) for a, b in zip(exp, got)):.3e}"
    rows.append(row("값 일치", "구간 내 모든 (x, 세기) 가 원본과 완전히 같아야 한다",
                    "완전 일치" if same else "불일치", verdict=PASS if same else FAIL, note=note))
    extra = [k for k in (out[0].keys() if out else []) if k not in ("raman_shift_cm-1", "intensity")]
    if extra:
        rows.append(row("추가 열", "참고 — save_result 가 붙이는 열은 감점 대상이 아니다",
                        ", ".join(extra)))
    return rows


def d_T049(rec, task, tf):
    x, y = load_xy("T049.csv")
    if x is None:
        return [row("입력", "T049.csv 필요", "파일을 못 찾음", verdict=INFO)]
    sig = y[(x >= 990) & (x <= 1012)]
    noi = y[(x >= 1800) & (x <= 1900)]
    S = float(sig.max())
    sd1, sd0 = float(noi.std(ddof=1)), float(noi.std(ddof=0))
    snr1, snr0 = S / sd1, S / sd0
    rows = [
        row("신호", "990~1012 cm⁻¹ 구간의 최대 세기", S,
            find_reported(rec, S, rel=0.001),
            PASS if find_reported(rec, S, rel=0.001) is not None else FAIL),
        row("잡음 (기준: 표본 표준편차)", "1800~1900 cm⁻¹ 세기의 <b>표본</b> 표준편차 ddof=1",
            sd1, note=f"{len(noi)}점"),
        row("잡음 (모표준편차 ddof=0)", "참고 — numpy 기본값. 채점기준이 요구한 값이 아니다", sd0),
        row("SNR (기준값)", "신호 ÷ 표본표준편차, 독립계산과 1% 이내 일치", snr1),
        row("SNR (모표준편차 기준)", "참고", snr0),
    ]
    cmp = compare_reported(rec, snr1, rel=0.01)
    rows.append(row("답변의 SNR", f"기준값 {snr1:.4f} 의 ±1% = {snr1*0.99:.4f} ~ {snr1*1.01:.4f}",
                    f"{snr1*0.99:.4f}~{snr1*1.01:.4f}", cmp["value"], cmp["verdict"],
                    note=("보고값이 1% 안이지만, 코드가 ddof=0 을 썼는지 아래에서 확인할 것 — "
                          "n 이 커서 두 정의 차이가 0.5% 밖에 안 나 허용오차에 흡수된다.")))
    code = code_of(rec)
    used = ("ddof=1 (표본)" if re.search(r"ddof\s*=\s*1|\bstdev\(|\.std\(.*ddof=1", code)
            else ("ddof=0 (모표준편차, np.std 기본값)" if re.search(r"np\.std\(|\.std\(", code)
                  else "std 호출을 못 찾음"))
    rows.append(row("코드가 쓴 표준편차 정의", "채점기준은 <b>표본</b> 표준편차(ddof=1)를 요구", used,
                    verdict=PASS if "ddof=1" in used else FAIL,
                    note="이 항목이 FAIL 이면 '수치는 허용오차 안이지만 정의는 틀림' 이다. "
                         "정의를 요구할지 수치만 볼지는 채점자가 정한다."))
    return rows


def d_T050(rec, task, tf):
    gt = tf.get("ground_truth") or {}
    ch = float(gt.get("channel_cm-1") or 1000)
    g = load_groups("T050.csv")
    p = find_input("T050.csv")
    rows = []
    if p is None:
        return [row("입력", "T050.csv 필요", "파일을 못 찾음", verdict=INFO)]
    rs = _rows(p)
    pts: dict[tuple, list] = {}
    for r in rs:
        pts.setdefault((float(r["x_mm"]), float(r["y_mm"])), []).append(
            (float(r["raman_shift_cm-1"]), float(r["intensity"])))
    axis = [s for s, _ in next(iter(pts.values()))]
    near = min(axis, key=lambda s: abs(s - ch))
    rows.append(row("최근접 채널", f"{ch:g} cm⁻¹ 에 가장 가까운 채널을 써야 한다",
                    f"{near:g} cm⁻¹ (축 {min(axis):g}~{max(axis):g}, 간격 {axis[1]-axis[0]:g})"))
    ux = sorted({k[0] for k in pts})
    uy = sorted({k[1] for k in pts})
    mat = [[dict(pts[(x, y)])[near] for x in ux] for y in uy]        # 행=y 오름차순
    grid_txt = "<br>".join(
        f"y={uy[j]:g}: " + "  ".join(f"{mat[j][i]:g}" for i in range(len(ux)))
        for j in range(len(uy) - 1, -1, -1))
    rows.append(row("정답 3×3 값", f"각 위치의 {near:g} cm⁻¹ 세기 (위가 큰 y)", grid_txt,
                    note=f"x = {', '.join(f'{v:g}' for v in ux)}"))
    flat = [v for r_ in mat for v in r_]
    got = [find_reported(rec, v, rel=1e-6) for v in flat]
    ok = all(v is not None for v in got)
    rows.append(row("답변의 9개 값", "9개 전부 상대오차 1e-6 이내",
                    f"{sum(1 for v in got if v is not None)}/9 일치",
                    verdict=PASS if ok else FAIL,
                    note="" if ok else f"누락 {[v for v, g_ in zip(flat, got) if g_ is None]}"))
    code = code_of(rec)
    orig = re.search(r"origin\s*=\s*['\"](\w+)", code)
    rows.append(row("좌표 방위", "작은 y 가 아래로 오게 그려야 한다(imshow 는 origin='lower')",
                    f"origin='{orig.group(1)}'" if orig else "imshow origin 지정 없음(기본 upper)",
                    verdict=PASS if (orig and orig.group(1) == "lower") else FAIL,
                    note="pcolormesh/contourf 를 썼다면 기본이 아래가 작은 y 라 정상 — "
                         "아래 그림으로 확인"))
    return rows


def d_T051(rec, task, tf):
    pairs = (tf.get("ground_truth") or {}).get("shifted_pairs_cm-1") or []
    flat = [v for pr in pairs for v in pr]
    got = [find_reported(rec, v, absolute=1.0) for v in flat]
    return [row("이동한 피크쌍", "각 쌍의 두 위치를 ±1 cm⁻¹ 안에 보고", pairs, got,
                PASS if all(g is not None for g in got) else FAIL)]


def d_T052(rec, task, tf):
    lo, hi = (tf.get("ground_truth") or {}).get("integration_range") or [990, 1012]
    x, y = load_xy("T052.csv")
    if x is None:
        return [row("입력", "T052.csv 필요", "파일을 못 찾음", verdict=INFO)]
    m = (x >= lo) & (x <= hi)
    area = float(np.trapezoid(y[m], x[m])) if hasattr(np, "trapezoid") else float(np.trapz(y[m], x[m]))
    cmp = compare_reported(rec, area, rel=0.01)
    return [
        row("적분 구간", f"{lo} ~ {hi} cm⁻¹, 사다리꼴 적분", f"{m.sum()}점 사용"),
        row("면적 (기준값)", f"np.trapz(y, x) over [{lo}, {hi}]", area),
        row("답변의 면적", f"기준값 ±1% = {area*0.99:.4f} ~ {area*1.01:.4f}",
            f"{area*0.99:.4f}~{area*1.01:.4f}", cmp["value"], cmp["verdict"],
            note="입력이 이미 베이스라인 보정된 스펙트럼이라 추가 차감은 필요 없다"),
    ]


def d_T053(rec, task, tf):
    x, y = load_xy("T053.csv")
    if x is None:
        return [row("입력", "T053.csv 필요", "파일을 못 찾음", verdict=INFO)]
    rows = []
    vals = {}
    for target in (1001.0, 1602.0):
        pk = detect_peaks(x, y, 0.03)
        cand = [p for p in pk if abs(p - target) <= 3.0]
        if cand:
            pos = min(cand, key=lambda p: abs(p - target))
            i = int(np.argmin(np.abs(x - pos)))
        else:                                       # 피크 검출이 실패하면 구간 최대로
            m = (x >= target - 3) & (x <= target + 3)
            i = int(np.argmax(np.where(m, y, -np.inf)))
        vals[target] = (float(x[i]), float(y[i]))
        rows.append(row(f"{target:.0f} cm⁻¹ 근처 피크", "±3 cm⁻¹ 안의 피크를 써야 한다",
                        f"x={x[i]:.1f}, 세기={y[i]:.4f}"))
    ratio = vals[1001.0][1] / vals[1602.0][1]
    cmp = compare_reported(rec, ratio, rel=0.01)
    rows.append(row("세기 비 (기준값)", "I(1001) ÷ I(1602), 레퍼런스 ±1%", ratio))
    rows.append(row("답변의 비", f"{ratio*0.99:.4f} ~ {ratio*1.01:.4f}",
                    f"{ratio*0.99:.4f}~{ratio*1.01:.4f}", cmp["value"], cmp["verdict"],
                    note=cmp["note"]))
    return rows


def _pre_pipeline(x, y):
    """T071/T072 가 지정한 전처리 — 5차 다항 베이스라인 → 벡터길이 정규화."""
    yb = y - np.polyval(np.polyfit(x, y, 5), x)
    n = np.linalg.norm(yb)
    return yb / n if n else yb


def d_T071(rec, task, tf):
    g = load_groups("T071.csv")
    if not g or not g.get("data"):
        return [row("입력", "T071.csv 필요", "파일을 못 찾음", verdict=INFO)]
    data = g["data"]
    keys = sorted(data)
    X = np.vstack([_pre_pipeline(*data[k]) for k in keys])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3).fit(X)
    evr = [float(v) for v in pca.explained_variance_ratio_]
    got = [find_reported(rec, v, absolute=0.02) for v in evr]
    got += [find_reported(rec, v * 100, absolute=2.0) for v in evr]   # % 로 보고할 수도
    hit = [g_ is not None for g_ in got[:3]] or []
    hit = [(got[i] is not None) or (got[i + 3] is not None) for i in range(3)]
    code = code_of(rec)
    return [
        row("스펙트럼 수", "맵의 모든 위치에 같은 전처리를 적용해야 한다", f"{len(keys)}개 위치"),
        row("전처리 순서", "5차 다항 베이스라인 → 벡터길이 정규화 → PCA(3성분)",
            "polyfit 있음" if "polyfit" in code else "polyfit 없음",
            note="normalize/norm 호출: " + ("있음" if re.search(r"norm|normalize", code) else "없음")),
        row("설명분산비 (기준값)", "위 순서대로 계산한 3개 값. 각 절대오차 0.02 이내",
            [round(v, 4) for v in evr]),
        row("답변의 설명분산비", "3개 각각 ±0.02 (또는 % 표기로 ±2%p)",
            [round(v, 4) for v in evr],
            [got[i] if got[i] is not None else got[i + 3] for i in range(3)],
            PASS if all(hit) else FAIL,
            note="" if all(hit) else f"불일치 성분 {[i+1 for i, h in enumerate(hit) if not h]}"),
    ]


def d_T072(rec, task, tf):
    truth = (tf.get("ground_truth") or {}).get("cluster_labels") or []
    g = load_groups("T072.csv")
    rows = [row("정답 라벨", "위치별 참 물질(ARI 계산의 기준)",
                f"{len(truth)}개: " + ", ".join(truth[:6]) + (" …" if len(truth) > 6 else ""))]
    if g and g.get("data"):
        data = g["data"]
        keys = sorted(data)
        X = np.vstack([_pre_pipeline(*data[k]) for k in keys])
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
        if len(truth) == len(keys):
            ari = adjusted_rand_score([truth[i] for i in range(len(keys))], lab)
            rows.append(row("참조 구현의 ARI", "지정 전처리 + KMeans(2) 로 얻은 ARI. 기준 0.90 이상",
                            f"{ari:.4f}", verdict=PASS if ari >= 0.9 else INFO,
                            note="이 값은 '이 문항이 풀리는가'를 보여 주는 상한 참고치다 — "
                                 "에이전트 점수가 아니다."))
    rows.append(row("에이전트 라벨", "답변·stdout 의 라벨 배열로 ARI ≥ 0.90 인지 확인",
                    "자동 추출 불가 — 아래 stdout/코드에서 라벨 배열을 찾아 대조할 것",
                    verdict=INFO))
    rows.append(row("좌표 맵", "두 군집이 좌표 위에 올바르게 배치됐는지", "아래 그림으로 확인",
                    verdict=INFO))
    return rows


def d_T074(rec, task, tf):
    ga, gb = load_groups("T074_a.csv"), load_groups("T074_b.csv")
    if not (ga.get("data") and gb.get("data")):
        return [row("입력", "T074_a.csv / T074_b.csv 필요", "파일을 못 찾음", verdict=INFO)]

    def stats(g):
        keys = sorted(g["data"])
        X = np.vstack([g["data"][k][1] for k in keys])
        x = g["data"][keys[0]][0]
        m = (x >= 995) & (x <= 1007)
        pos = [float(x[m][np.argmax(row_[m])]) for row_ in X]
        amp = [float(row_[m].max()) for row_ in X]
        return x, X.mean(axis=0), float(np.mean(pos)), np.array(amp), len(keys)

    xa, ma, pa, aa, na = stats(ga)
    xb, mb, pb, ab, nb = stats(gb)
    dpos = pb - pa
    rsd_a = float(aa.std(ddof=1) / aa.mean() * 100)
    rsd_b = float(ab.std(ddof=1) / ab.mean() * 100)
    cos = float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb)))
    _c = compare_reported(rec, cos, absolute=0.001)
    return [
        row("세션 규모", "두 세션만 비교해야 한다", f"a={na}회, b={nb}회 반복"),
        row("1001 피크 위치차 (기준값)", "각 세션 평균 피크위치의 차", f"{dpos:+.4f} cm⁻¹",
            find_reported(rec, abs(dpos), absolute=0.5),
            note=f"a={pa:.3f}, b={pb:.3f}"),
        row("세기 상대표준편차 (기준값)", "각 세션 피크세기의 RSD = std(ddof=1)/mean",
            f"a={rsd_a:.3f}%, b={rsd_b:.3f}%",
            f"a측: {find_reported(rec, rsd_a, rel=0.1)}, b측: {find_reported(rec, rsd_b, rel=0.1)}"),
        row("평균 스펙트럼 코사인 유사도 (기준값)", "두 세션 평균 스펙트럼의 코사인 유사도",
            f"{cos:.6f}", _c["value"], _c["verdict"], note=_c["note"]),
        row("보고 항목 3개", "세 값이 모두 보고서에 있어야 한다",
            "아래 답변에서 확인", verdict=INFO,
            note="피크 위치·RSD 는 구간·검출법에 따라 조금 달라진다 — 값이 근처면 방법을 확인할 것"),
    ]


def d_T128(rec, task, tf):
    order = (tf.get("truth_order") or (tf.get("ground_truth") or {}).get("truth_order") or [])
    lib = find_input("reference_library.csv")
    if lib is None or not order:
        return [row("입력", "reference_library.csv 필요", "파일을 못 찾음", verdict=INFO)]
    rs = _rows(lib)
    specs: dict[str, dict] = {}
    for r in rs:
        d = specs.setdefault(r["spectrum_id"], {"material": r["material"], "y": []})
        d["y"].append(float(r["intensity"]))
    ids = sorted(specs)
    M = np.vstack([np.array(specs[i]["y"]) for i in ids])
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    precisions = []
    detail = []
    for k, truth in enumerate(order, start=1):
        x, y = load_xy(f"T128_{k}.csv")
        if y is None:
            continue
        q = y / np.linalg.norm(y)
        sim = Mn @ q
        top = np.argsort(-sim)[:3]
        same = sum(1 for i in top if specs[ids[i]]["material"] == truth)
        precisions.append(same / 3)
        detail.append(f"Q{k}({truth}): top3={[specs[ids[i]]['material'] for i in top]} → {same}/3")
    mean = float(np.mean(precisions)) if precisions else 0.0
    cmp = compare_reported(rec, mean, absolute=1e-6)
    mats = {i: specs[i]["material"] for i in ids}
    import collections
    per_mat = collections.Counter(mats.values())
    # 채점기준은 '질의별 top-3 → 개별 정밀도 → 그 산술평균'을 요구한다. 평균값만 보면
    # 질의별 계산이 틀렸는데 우연히 평균이 맞는 답도 통과한다. 그래서 답변 표에서
    # 질의별 top-3 물질을 뽑아 기준값과 대조한다.
    ans = rec.get("answer") or rec.get("final_report") or ""
    _AL = {"polystyrene": "polystyrene", "ps": "polystyrene", "pet": "PET",
           "pmma": "PMMA", "calcite": "calcite", "aragonite": "aragonite",
           "silicon": "silicon", "si": "silicon"}
    rep_rows = []
    for line in ans.splitlines():
        if line.count("|") < 3:
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        mats = []
        for c in cells:
            for w in re.findall(r"[A-Za-z]+", c):
                v = _AL.get(w.lower())
                if v:
                    mats.append(v)
        if len(mats) >= 3:
            rep_rows.append(mats[-3:] if len(mats) > 3 else mats)
    gt_top3 = [[m for m in d.split("top3=[")[1].split("]")[0].replace("'", "").split(", ")]
               for d in detail] if detail else []
    per_ok = None
    if len(rep_rows) >= len(gt_top3) > 0:
        per_ok = all(sorted(a) == sorted(b) for a, b in zip(rep_rows, gt_top3))

    # '1e-6 이내'는 채점기준 원문이다. 0.6667 은 그 기준을 넘는다(오차 3.3e-5).
    # 표시 반올림으로 볼 수 있지만, 그 판단을 주석에 묻지 않고 별도 행으로 드러낸다.
    reported = cmp["value"]
    strict = (reported is not None and abs(float(reported) - mean) <= 1e-6)

    out = [
        row("라이브러리", "참조 스펙트럼 수와 물질별 개수", f"{len(ids)}개 · " +
            ", ".join(f"{k} {v}개" for k, v in sorted(per_mat.items())), "—",
            note=("물질당 참조가 2개뿐이므로 <b>top-3 의 최대 정밀도는 2/3 = 0.6667</b> 이다 — "
                  "1.0 을 보고했다면 오히려 계산이 틀린 것이다.")),
        row("질의별 top-3 (기준 조항)", "질의마다 같은 물질 개수 ÷ 3 = 개별 정밀도",
            "<br>".join(detail),
            ("<br>".join(", ".join(r) for r in rep_rows[:len(gt_top3)])
             if rep_rows else "답변 표에서 못 읽음"),
            (PASS if per_ok else FAIL) if per_ok is not None else INFO,
            note="" if per_ok is not False else "질의별 top-3 가 기준값과 다르다"),
        row("개별 정밀도", "5개 값", str([round(v, 4) for v in precisions]), "—"),
        row("평균 정밀도", "5개의 산술평균", f"{mean:.6f}", reported, cmp["verdict"],
            note=cmp["note"] or "유사도 정의(코사인/상관)에 따라 top-3 이 바뀔 수 있다"),
        row("기준의 1e-6 조항", "채점기준 원문: 레퍼런스와 <b>1e-6 이내</b>",
            f"{mean:.6f}",
            (f"{reported} → 오차 {abs(float(reported) - mean):.2g}"
             if reported is not None else "값 없음"),
            PASS if strict else INFO,
            note="" if strict else
                 "<b>기준을 문자 그대로 적용하면 미달이다.</b> 다만 답변이 2/3 를 소수 "
                 "4자리로 표기한 것이라면 계산이 아니라 표시의 문제다 — 위 '질의별 top-3' "
                 "행이 통과면 표시 반올림으로 보고 정답 처리하는 것이 타당하다. "
                 "그 판단은 채점자가 명시적으로 내려야 한다."),
    ]
    return out


def d_T092(rec, task, tf):
    gt = tf.get("ground_truth") or {}
    x, y = load_xy("T092.csv")
    rows = [row("입력 실제 범위", "참고 — 요청 구간이 입력 범위 밖임을 보고해야 한다",
                f"{x.min():g} ~ {x.max():g} cm⁻¹" if x is not None else "파일 없음"),
            row("요청 구간", "프롬프트가 요구한 구간", f"{gt.get('requested')}"),
            row("겹침", "두 구간은 겹치지 않는다", "겹치는 점 0개",
                note="따라서 정답은 '추출 불가/범위 밖'을 보고하는 것이고, "
                     "억지로 파일을 만들면 오답이다.")]
    import spectra_panel as sp
    csvs, _ = sp.find_outputs(rec)
    rows.append(row("산출 파일", "범위 밖 구간을 억지로 저장하지 않았는가",
                    f"{len(csvs)}개" + (f" ({csvs[0].name})" if csvs else ""),
                    verdict=PASS if not csvs else FAIL))
    return rows


def d_T093(rec, task, tf):
    x, y = load_xy("T093.csv")
    if y is None:
        return [row("입력", "T093.csv 필요", "파일을 못 찾음", verdict=INFO)]
    allz = bool(np.all(y == 0))
    return [row("입력 세기", "전 구간 0 = 무신호임을 보고해야 한다",
                f"최소 {y.min():g}, 최대 {y.max():g} → {'전부 0' if allz else '0 아님'}",
                verdict=PASS if allz else INFO,
                note="정답은 '신호 없음'을 보고하는 것. 피크를 찾아냈다고 하면 오답이다.")]


def d_T104(rec, task, tf):
    x, y = load_xy("T104.csv")
    rows = [row("정답 라벨", "이 스펙트럼의 참값", (tf.get("ground_truth") or {}).get("label"))]
    if y is not None:
        pk = detect_peaks(x, y, 0.15)
        rows.append(row("피크 특징", "참고 — 뾰족한 결정 피크가 없고 넓은 혹만 있어야 한다",
                        f"뚜렷한 피크 {len(pk)}개 {[round(float(v)) for v in pk]}, "
                        f"y {y.min():.1f}~{y.max():.1f}"))
    return rows


CHECKS = {
    "T037": d_T037, "T042": d_T042, "T043": d_T043, "T044": d_T044, "T045": d_T045,
    "T047": d_T047, "T048": d_T048, "T049": d_T049, "T050": d_T050, "T051": d_T051,
    "T052": d_T052, "T053": d_T053, "T071": d_T071, "T072": d_T072, "T074": d_T074,
    "T092": d_T092, "T093": d_T093, "T104": d_T104, "T128": d_T128,
}

# 파일처리 문항 중 여기 없던 29개(T038/039/040/041/046/054/055/056/083/096/099/110 +
# 매칭 블록 T111~T127)의 진단은 filegrade 패키지가 들고 있다. 그쪽은 정답 기준을
# 부류 A(GT 확정) / 부류 B(모양새)로 나눠 판정한다 — filegrade/task_class.py 참조.
# 임포트가 실패해도 기존 19문항 진단은 그대로 나와야 하므로 조용히 넘어간다.
try:
    from filegrade.checks import EXTRA_CHECKS as _EXTRA
    CHECKS.update(_EXTRA)
except Exception as _e:                                    # noqa: BLE001
    import warnings
    warnings.warn(f"filegrade 진단을 불러오지 못했다: {type(_e).__name__}: {_e}")


# ── 산출물 ↔ 입력 일반 대조 (문항별 진단이 없어도 항상 낸다) ──────────────────

def generic_output_rows(rec: dict, task: dict, tf: dict) -> list[dict]:
    """저장된 스펙트럼이 입력에서 '어떻게' 바뀌었는지 수치로. 전처리 문항 공통 근거."""
    import spectra_panel as sp
    files = tf.get("files") or []
    if not files:
        return []
    src = find_input(files[0])
    if src is None:
        return []
    head = _rows(src)[:1]
    if head and any(k in head[0] for k in ("point_id", "x_mm", "spectrum_id", "rep_id")):
        return []              # 맵/라이브러리 파일은 단일 곡선 대조가 성립하지 않는다
    xi, yi = load_xy(files[0])
    csvs, _ = sp.find_outputs(rec)
    if xi is None or not csvs:
        return []
    rows = []
    for p in csvs[:2]:
        c = sp.read_curves(p, max_groups=1)
        if not c or not c[0]["y"]:
            continue
        xo = np.array(c[0]["x"], float)
        yo = np.array(c[0]["y"], float)
        bits = [f"{len(yo)}점"]
        bits.append("입력과 점 수 같음" if len(yo) == len(yi) else f"입력 {len(yi)}점과 다름")
        if len(xo) == len(xi) and np.allclose(xo, xi):
            bits.append("x축 동일")
        else:
            bits.append(f"x {xo.min():g}~{xo.max():g}")
        bits.append(f"y {yo.min():.4g}~{yo.max():.4g}")
        if abs(yo.min()) < 1e-9 and abs(yo.max() - 1.0) < 1e-9:
            bits.append("<b>0~1 정규화됨</b>")
        if len(yo) == len(yi):
            d = np.abs(yo - yi)
            bits.append(f"입력 대비 max|Δ|={d.max():.4g}")
            # 입력이 상수(T093 은 전 구간 0)면 1차 적합이 성립하지 않는다 — polyfit 이
            # 0 으로 나누며 RuntimeWarning 을 뿜는다. 그 경우는 이 항목만 건너뛴다.
            if float(yi.std()) > 0:
                k, b = np.polyfit(yi, yo, 1)
                bits.append(f"선형관계 y_out ≈ {k:.6f}·y_in + {b:.4g}")
        rows.append(row(f"산출 {p.name}", "참고 — 입력이 어떻게 변형됐는지", "<br>".join(bits)))
    return rows


def run(rec: dict, task: dict, tf_entry: dict) -> list[dict]:
    """이 (문항×실행)의 진단 행 목록. 실패해도 빈 목록/오류행만 돌려주고 죽지 않는다."""
    tid = str(task.get("id") or rec.get("id") or "")
    rows: list[dict] = []
    from verifiers import _no_answer
    if _no_answer(rec.get("answer") or rec.get("final_report") or ""):
        # 이 경우 아래의 '답변의 …' 행들은 전부 '못 찾음'이 된다. 이유를 먼저 밝혀 둔다.
        rows.append(row("무응답", "모델이 최종 답변을 못 만들었다 → 자동 오답",
                        "Failed to generate a response.", verdict=FAIL,
                        note="아래 기준값들은 '정답이 무엇이었는지'를 남겨 두는 참고자료다."))
    fn = CHECKS.get(tid)
    if fn is not None:
        try:
            rows += fn(rec, task, tf_entry or {})
        except Exception as e:                     # noqa: BLE001
            rows.append(row("진단 오류", "이 문항의 자동 진단이 실패했다",
                            f"{type(e).__name__}: {e}", verdict=INFO))
    try:
        rows += generic_output_rows(rec, task, tf_entry or {})
    except Exception:                              # noqa: BLE001
        pass
    return rows


def overall(rows: list[dict]) -> str | None:
    """진단 행들의 종합 — 하나라도 FAIL 이면 fail, 전부 pass 면 pass, 그 외 None."""
    vs = [r["verdict"] for r in rows]
    if FAIL in vs:
        return FAIL
    if PASS in vs:
        return PASS
    return None
