# -*- coding: utf-8 -*-
"""diagnostics.CHECKS 에 없던 파일처리 문항들의 자동 진단.

[무엇을 메우는가]
diagnostics.py 의 CHECKS 는 파일처리 48문항 중 19문항만 덮는다. 나머지 29문항은
문항별 진단 없이 generic_output_rows(입력 대비 max|Δ| 같은 일반 대조)만 나왔고,
특히 매칭 블록 T111~T127 은 verifier 가 `answer_contains("polystyrene")` 뿐이라
"순위가 맞나", "점수가 내림차순인가"가 아예 검사되지 않았다.

    T038 T039 T040 T041 T046 T054 T055 T056 T083 T096 T099 T110   (12)
    T111 ~ T127                                                    (17)

[부류에 따라 다르게 판정한다]
task_class.py 의 A/B 분류를 따른다.
  · 부류 A → GT 를 재계산해 엄격 비교(tolerance 는 float64 반올림 바닥).
  · 부류 B → shape_match 로 '모양새' 판정. 값 일치는 참고값으로만 싣고,
    같은 줄에 앙상블 편차 S 를 함께 보여 준다("정당한 방법들끼리도 이만큼 벌어진다").

[절차와 결과를 나눈다]
'2차 baseline'은 모양새 지표를 전부 통과한다 — 즉 "5차"라는 지시 위반은 결과로
잡히지 않는다. 그래서 지정된 차수·윈도우·순서는 코드에서 따로 확인해 별도 행으로 낸다.
"""
from __future__ import annotations

import re

import numpy as np

import diagnostics as D                    # row/PASS/FAIL/INFO/load_xy/… 를 그대로 쓴다
from . import BENCH_DIR
from . import matching_truth as MT
from . import task_class as TC
from .shape_match import (compare_shape, despike_invariant, ensemble_band,
                          peaks_of)

PASS, FAIL, INFO = D.PASS, D.FAIL, D.INFO

# 기존 진단을 덮어쓰지 않고 이어 붙이기 위해 원본을 잡아 둔다. diagnostics 는 CHECKS 를
# 정의한 뒤에 이 모듈을 임포트하므로 이 시점에 이미 존재한다.
CHECKS_BASE_T104 = getattr(D, "CHECKS", {}).get("T104")
D.CHECKS_BASE_T104 = CHECKS_BASE_T104


# ── 공통 헬퍼 ────────────────────────────────────────────────────────────────

def _ref(name: str):
    """task_refs/<name> 을 (x, y) 로. 없으면 (None, None)."""
    p = BENCH_DIR / "task_refs" / name
    if not p.exists():
        return None, None
    r = D._rows(p)
    return (np.array([float(a["raman_shift_cm-1"]) for a in r]),
            np.array([float(a["intensity"]) for a in r]))


def _output_curve(rec: dict):
    """이 실행이 저장한 첫 스펙트럼 CSV 를 (이름, x, y) 로. 없으면 (None, None, None)."""
    import spectra_panel as sp
    csvs, _ = sp.find_outputs(rec)
    for p in csvs:
        c = sp.read_curves(p, max_groups=1)
        if c and c[0]["y"]:
            return p.name, np.array(c[0]["x"], float), np.array(c[0]["y"], float)
    return None, None, None


def _no_output_row(rec: dict) -> list[dict]:
    return [D.row("산출 파일", "처리 결과 스펙트럼을 저장해야 한다",
                  "스펙트럼 CSV 1개 이상", "저장된 CSV 없음", FAIL,
                  note="파일을 안 만들었거나 manifest/트레이스에 경로가 안 남았다.")]


def _yn(label: str, criterion: str, expect: str, got: str, ok: bool,
        note: str = "") -> dict:
    """정답(expect) ↔ 에이전트가 실제로 낸 것(got) 을 반드시 나눠 담는 행.

    채점 콘솔의 '정답 ↔ 대답' 대조표가 이 두 칸을 그대로 좌우에 놓는다. 한쪽에 몰아
    쓰면 사용자가 무엇과 무엇을 비교하는지 알 수 없게 된다.
    """
    return D.row(label, criterion, expect, got, PASS if ok else FAIL, note=note)


def _shape_rows(rec, tid: str, in_name: str, ref_name: str,
                ensemble_kind: str | None = None) -> list[dict]:
    """부류 B 스펙트럼 문항 공통 — 모양새 판정 + 앙상블 편차."""
    xi, yi = D.load_xy(in_name)
    xr, yr = _ref(ref_name)
    if yi is None or yr is None:
        return [D.row("입력", f"{in_name} / {ref_name} 필요", "파일을 못 찾음", verdict=INFO)]
    name, xo, yo = _output_curve(rec)
    rows: list[dict] = []

    if ensemble_kind:
        band, S = ensemble_band(xi, yi, ensemble_kind)
        detail = "<br>".join(
            f"{k}: 레퍼런스 대비 max|Δ| = {np.abs(v - yr).max():.4g}" for k, v in band.items())
        rows.append(D.row(
            "정당한 구현들의 편차 S", "이 값보다 촘촘한 tolerance 는 correctness 가 아니라 "
            "구현 동일성을 재는 것이다",
            f"<b>S = {S:.4g}</b> (기존 reference_match tolerance 1e-5 의 {S / 1e-5:.2g}배)"
            f"<br>{detail}", verdict=INFO,
            note="프롬프트가 방법을 못박지 않으므로 이들은 모두 정당한 해석이다. "
                 "그래서 이 문항은 값이 아니라 모양새로 채점한다."))

    if yo is None:
        return rows + _no_output_row(rec)

    th = TC.shape_thresholds(tid)
    r = compare_shape(xr, yo, yr, th)
    m = r.metrics
    crit = (f"피크 recall·precision = {th['min_recall']:.2f}/{th['min_precision']:.2f} "
            f"(±{th['peak_tol_cm']:g} cm⁻¹) · Δ상대세기 ≤ {th['max_d_rel_intensity']:.2f} · "
            f"pearson ≥ {th['min_pearson']:.2f} · 0~1 max|Δ| ≤ {th['max_abs_01']:.2f}")
    gt = (f"피크 {m.get('n_peaks_ref', 0)}개 {m.get('peaks_ref')}<br>"
          f"<span style='color:#6b7280'>레퍼런스 곡선의 모양</span>")
    got = (f"피크 {m.get('n_peaks_out', 0)}개 → recall <b>{m.get('peak_recall', float('nan')):.3f}</b> · "
           f"precision <b>{m.get('peak_precision', float('nan')):.3f}</b><br>"
           f"Δ상대세기 <b>{m.get('d_rel_intensity', float('nan')):.4f}</b> · "
           f"pearson <b>{m.get('pearson', float('nan')):.5f}</b> · "
           f"0~1 max|Δ| <b>{m.get('max_abs_01', float('nan')):.4f}</b>")
    rows.append(D.row(f"모양새 판정 ({name})", crit, gt, got,
                      PASS if r.passed else FAIL, note=r.summary()))
    rows.append(D.row("참고 — 원래 기준이던 값 일치", "reference_match tolerance 1e-5",
                      "레퍼런스와 점대점 동일",
                      f"max|Δ| = {m.get('max_abs_raw', float('nan')):.4g}", verdict=INFO,
                      note="이 값이 크다고 오답이 아니다. 위 모양새 판정이 정답 기준이다."))
    return rows


def _exact_rows(rec, tid: str, in_name: str, gt_y, gt_desc: str,
                tol: float = 1e-5) -> list[dict]:
    """부류 A 스펙트럼 문항 공통 — GT 와 엄격 비교."""
    name, xo, yo = _output_curve(rec)
    if yo is None:
        return _no_output_row(rec)
    if len(yo) != len(gt_y):
        return [_yn(f"산출 점 수 ({name})", "GT 와 점 수가 같아야 한다",
                    f"{len(gt_y)}점", f"{len(yo)}점", False,
                    "점 수가 다르면 계산 방식 자체가 다른 것이다.")]
    d = float(np.abs(yo - gt_y).max())
    return [_yn(f"GT 대조 ({name})", f"{gt_desc} · max|Δ| ≤ {tol:g}",
                f"{gt_desc} (max|Δ| ≤ {tol:g})", f"max|Δ| = {d:.4g}", d <= tol,
                "" if d <= tol else "GT 와 다른 계산을 했다.")]


def _proc_row(rec, label: str, criterion: str, pattern: str, must: bool = True,
              expect: str = "") -> dict:
    """절차(process) 확인 — 지정된 파라미터를 코드에서 실제로 썼는가.

    결과가 맞아도 지시를 어겼을 수 있다(예: 2차 baseline 은 모양새를 통과한다).
    그래서 지정 파라미터는 코드에서 따로 본다.
    """
    code = D.code_of(rec)
    m = re.search(pattern, code, re.I)
    hit = bool(m)
    return D.row(label, criterion, expect or "지정된 대로 사용",
                 (f"코드에 <code>{_esc(m.group(0)[:60])}</code>" if m else "코드에서 못 찾음"),
                 verdict=(PASS if hit == must else FAIL),
                 note="" if hit == must else f"찾는 패턴: <code>{_esc(pattern)}</code>")


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _unchecked(items: list[str]) -> dict:
    """채점기준이 요구하지만 자동으로는 확인 못 하는 조항을 <b>드러내는</b> 행.

    이게 없으면 콘솔이 '통과'만 보여 주고, 사람은 기준의 나머지 조항도 확인된 줄 안다.
    실제로 T127 은 '보정 적용' 조항을, T124 는 '순서' 조항을 안 보고도 정답으로 찍혔다.
    자동 판정이 기준을 다 덮지 못하면 그 사실 자체를 표에 적어야 한다.
    """
    return D.row("⚠ 자동 확인 못 한 기준", "아래 조항은 사람이 직접 봐야 한다",
                 "—", "<br>".join(f"· {i}" for i in items), verdict=INFO,
                 note="이 항목이 있으면 위 '통과'는 <b>부분 통과</b>다.")


# ── 부류 B : baseline 계열 ───────────────────────────────────────────────────

def d_T038(rec, task, tf):
    rows = _shape_rows(rec, "T038", "T038.csv", "T038_reference.csv", "baseline_poly5")
    rows.insert(0, _proc_row(rec, "지정 차수 준수", "프롬프트가 5차를 못박았다",
                             r"polyfit\s*\([^)]*,\s*5\s*\)|order\s*=\s*5|deg\w*\s*=\s*5|"
                             r"5\s*(?:th|차)[\s-]*order|order\s*5"))
    return rows


def d_T096(rec, task, tf):
    gt = tf.get("ground_truth") or {}
    rows = _shape_rows(rec, "T096", "T096.csv", "T096_reference.csv", "baseline_poly5")
    ans = (rec.get("answer") or "").lower()
    flu = any(k in ans for k in ("fluoresc", "형광"))
    rows.insert(0, _yn("원인 지목", "원인을 형광 배경(fluorescence)으로 지목해야 한다",
                       "형광 배경(fluorescence)",
                       "형광이라고 답함" if flu else "형광 언급 없음", flu))
    peaks = gt.get("peaks_major") or [620, 1001, 1031, 1155, 1450, 1583, 1602]
    got = [D.find_reported(rec, p, absolute=3.0) for p in peaks]
    n = sum(1 for g in got if g is not None)
    rows.append(D.row("보정 후 피크 7개 재검출", f"{peaks} 를 각 ±3 cm⁻¹ 로 보고",
                      f"{n}/{len(peaks)}개 보고됨", got,
                      PASS if n == len(peaks) else FAIL,
                      note="" if n == len(peaks) else
                      f"누락: {[p for p, g in zip(peaks, got) if g is None]}"))
    return rows


def d_T110(rec, task, tf):
    """채점기준이 이미 불변량(recall/precision ≥90%)으로 적혀 있는데 verifier 는
    reference_match 1e-5 를 요구해 서로 어긋나 있던 문항."""
    gt = tf.get("ground_truth") or {}
    peaks = gt.get("peaks_major") or [620, 1001, 1031, 1155, 1450, 1583, 1602]
    ans = (rec.get("answer") or "").lower()
    strong = any(k in ans for k in ("background", "배경", "fluoresc", "형광", "baseline"))
    rows = [_yn("원인 지목", "강한 배경(background)으로 판단해야 한다",
                "강한 배경 / 형광",
                "배경이라고 답함" if strong else "배경 언급 없음", strong)]
    rows += _shape_rows(rec, "T110", "T110.csv", "T110_reference.csv", "baseline_poly5")

    # 채점기준 그대로 — 보고된 피크의 recall / precision
    got = [D.find_reported(rec, p, absolute=3.0) for p in peaks]
    rec_hit = sum(1 for g in got if g is not None)
    recall = rec_hit / len(peaks)
    rows.append(_yn("피크 recall (채점기준)", "레퍼런스 피크 중 보고된 비율 ≥ 90%",
                    f"{peaks} (7개 전부)",
                    f"{rec_hit}/{len(peaks)}개 = {recall * 100:.1f}% · "
                    f"{[g for g in got if g is not None]}",
                    recall >= 0.9,
                    "" if recall >= 0.9 else
                    f"누락: {[p for p, g in zip(peaks, got) if g is None]}"))
    rows.append(D.row("피크 precision (채점기준)",
                      "보고한 피크 중 레퍼런스에 있는 비율 ≥ 90%",
                      "답변 표에서 피크 목록을 세어 확인할 것", verdict=INFO,
                      note="답변에는 중간계산 숫자가 섞여 있어 '보고한 피크 전체'를 "
                           "자동으로 특정할 수 없다. recall 은 기대값 탐색으로 판정 가능하다."))
    return rows


# ── 부류 B : SG 평활 / 파이프라인 ────────────────────────────────────────────

def d_T040(rec, task, tf):
    rows = _shape_rows(rec, "T040", "T040.csv", "T040_reference.csv", "sgolay_11_3")
    rows.insert(0, _proc_row(rec, "지정 파라미터 준수",
                             "window_length=11, polyorder=3 을 못박았다",
                             r"savgol\w*\([^)]*11[^)]*3|window_length\s*=\s*11"))
    rows.append(D.row("edge mode", "프롬프트가 양끝 처리 모드를 지정하지 않았다",
                      "interp / nearest / mirror / constant 모두 정당", verdict=INFO,
                      note="레퍼런스는 mode='interp' 로 만들어졌다. 다른 모드를 썼다고 "
                           "오답이 아니며, 차이는 양끝 몇 점에만 나타난다."))
    return rows


def d_T046(rec, task, tf):
    rows = _shape_rows(rec, "T046", "T046.csv", "T046_reference.csv")
    code = D.code_of(rec)
    # 패턴은 넉넉하게 잡는다. 에이전트는 단계를 함수로 감싸고 차수를 기본인자로 넘기는
    # 식으로 쓰기 때문에(예: `def poly_baseline(x, y, order=5)` 뒤에 `poly_baseline(...)`),
    # `polyfit(x, y, 5)` 같은 좁은 패턴만 보면 제대로 한 답을 못 했다고 판정한다.
    steps = [("despike/스파이크 제거", r"despike|de_spike|remove_spike\w*|medfilt|median_filter"),
             ("5차 다항 baseline", r"poly_?baseline|baseline_?corr\w*|polyfit|polynomial"),
             ("SG 평활(11,3)", r"savgol|savitzky"),
             ("0~1 정규화", r"minmax|min_max|normali[sz]\w*|"
                            r"\(\s*[\w.]+\s*-\s*[\w.]+\s*\)\s*/\s*\(\s*[\w.]+\s*-\s*[\w.]+\s*\)")]
    # 순서 판정은 '증가하는 등장 위치를 하나씩 고를 수 있는가'로 본다(그리디).
    # 첫 등장만 보면 import 줄에, 마지막 등장만 보면 플롯 라벨(label='Despiked')에
    # 걸려 둘 다 오판한다 — 실제로 AILA 의 T046 코드가 그 두 경우에 다 해당했다.
    pos, cur = [], -1
    for label, pat in steps:
        m = next((m for m in re.finditer(pat, code, re.I) if m.start() > cur), None)
        pos.append((label, m.start() if m else None))
        if m:
            cur = m.start()
    found = [p for p in pos if p[1] is not None]
    ordered = len(found) == 4
    rows.insert(0, _yn(
        "4단계 순서 준수 (절차)", "지정 순서대로 네 단계를 적용해야 한다",
        "① despike → ② 5차 baseline → ③ SG(11,3) → ④ 0~1 정규화",
        "<br>".join(f"{i+1}. {l}: " + (f"코드 {p}행째 위치" if p is not None
                                       else "<b>이 순서로는 못 찾음</b>")
                    for i, (l, p) in enumerate(pos)),
        ordered,
        "" if ordered else
        "지정 순서대로 등장하는 호출을 찾지 못했다 — 아래 코드를 직접 볼 것"))
    rows.append(D.row("0~1 정규화 결과", "마지막 단계가 정규화이므로 min=0, max=1 이어야 한다",
                      _norm_note(rec, "minmax"), verdict=INFO))
    return rows


def _norm_note(rec, kind: str) -> str:
    _, _, yo = _output_curve(rec)
    if yo is None:
        return "산출 파일 없음"
    if kind == "minmax":
        return f"min={yo.min():.6g}, max={yo.max():.6g}"
    return f"L2 노름={np.linalg.norm(yo):.6f}"


# ── 부류 B : despike 계열 ────────────────────────────────────────────────────

def _despike_rows(rec, tid: str, in_name: str, ref_name: str) -> list[dict]:
    xi, yi = D.load_xy(in_name)
    xr, yr = _ref(ref_name)
    if yi is None or yr is None:
        return [D.row("입력", f"{in_name} / {ref_name} 필요", "파일을 못 찾음", verdict=INFO)]
    n_sp = int((np.abs(yi - yr) > 1000).sum())
    rows = [D.row("스파이크 개수 (역산)",
                  "입력과 레퍼런스의 차가 1000 을 넘는 점을 스파이크로 본다",
                  f"{n_sp}개", verdict=INFO,
                  note="레퍼런스는 '스파이크를 넣기 전의 원본'이라, 파괴된 원래 값은 "
                       "어떤 알고리즘으로도 복원할 수 없다. 그래서 전 구간 일치를 "
                       "요구하면 안 되고 아래 두 조건으로 판정한다.")]
    name, xo, yo = _output_curve(rec)
    if yo is None:
        return rows + _no_output_row(rec)
    r = despike_invariant(yi, yo, yr)
    m = r.metrics
    rows.append(_yn(
        f"스파이크 불변량 ({name})",
        "비-스파이크 점 max|Δ| ≤ 1e-5 · 스파이크 제거율 ≥ 99%",
        f"비-스파이크 {int((np.abs(yi - yr) <= 1000).sum())}점 무변경 · "
        f"스파이크 {n_sp}개 100% 제거",
        (f"비-스파이크 max|Δ| = {m.get('nonspike_max_abs', float('nan')):.4g} · "
         f"제거율 = {m.get('removal_rate', float('nan')) * 100:.1f}%"),
        r.passed, r.summary()))
    rows.append(D.row("참고 — 원래 기준이던 값 일치", "reference_match tolerance 1e-5",
                      "레퍼런스와 점대점 동일",
                      f"max|Δ| = {m.get('max_abs_raw', float('nan')):.4g}", verdict=INFO,
                      note="스파이크 위치의 잔차라 크게 나오는 게 정상이다."))
    return rows


def d_T039(rec, task, tf):
    return _despike_rows(rec, "T039", "T039.csv", "T039_reference.csv")


def d_T056(rec, task, tf):
    ans = (rec.get("answer") or "").lower()
    said = any(k in ans for k in ("spike", "스파이크", "cosmic"))
    rows = [_yn("스파이크 존재 판단", "스파이크가 있다고 답해야 한다",
                "스파이크 있음", "있다고 판단" if said else "언급 없음", said)]
    rows += _despike_rows(rec, "T056", "T056.csv", "T056_reference.csv")
    peaks = (tf.get("ground_truth") or {}).get("peaks_major") or \
        [620, 1001, 1031, 1155, 1450, 1583, 1602]
    got = [D.find_reported(rec, p, absolute=3.0) for p in peaks]
    n = sum(1 for g in got if g is not None)
    rows.append(D.row("제거 후 피크 7개 재보고", f"{peaks} 를 각 ±3 cm⁻¹ 로",
                      f"{n}/{len(peaks)}개", got, PASS if n == len(peaks) else FAIL))
    return rows


def d_T099(rec, task, tf):
    ans = (rec.get("answer") or "").lower()
    cosmic = any(k in ans for k in ("cosmic", "우주선", "cosmic ray"))
    rows = [_yn("원인 지목", "원인을 우주선(cosmic ray)으로 지목해야 한다",
                "우주선(cosmic ray)",
                "우주선이라고 답함" if cosmic else "언급 없음", cosmic)]
    rows += _despike_rows(rec, "T099", "T099.csv", "T099_reference.csv")
    return rows


# ── 부류 A : 정규화 · 도함수 ─────────────────────────────────────────────────

def d_T041(rec, task, tf):
    xi, yi = D.load_xy("T041.csv")
    if yi is None:
        return [D.row("입력", "T041.csv 필요", "파일을 못 찾음", verdict=INFO)]
    gt = (yi - yi.min()) / (yi.max() - yi.min())
    rows = [D.row("GT 정의", "out = (y − min) / (max − min). 정의가 하나뿐이라 값이 유일하다",
                  "min-max 정규화", verdict=INFO)]
    name, xo, yo = _output_curve(rec)
    if yo is None:
        return rows + _no_output_row(rec)
    exact = abs(float(yo.min())) < 1e-9 and abs(float(yo.max()) - 1.0) < 1e-9
    rows.append(_yn(f"정규화 성질 ({name})", "min 정확히 0, max 정확히 1",
                    "min=0, max=1",
                    f"min={yo.min():.3g}, max={yo.max():.9g}", exact))
    rows += _exact_rows(rec, "T041", "T041.csv", gt, "min-max 정규화 결과")
    return rows


def d_T055(rec, task, tf):
    xi, yi = D.load_xy("T055.csv")
    if yi is None:
        return [D.row("입력", "T055.csv 필요", "파일을 못 찾음", verdict=INFO)]
    gt = yi / np.linalg.norm(yi)
    rows = [D.row("GT 정의", "out = y / ||y||₂. 정의가 하나뿐이라 값이 유일하다",
                  "L2(벡터길이) 정규화", verdict=INFO)]
    name, xo, yo = _output_curve(rec)
    if yo is None:
        return rows + _no_output_row(rec)
    n = float(np.linalg.norm(yo))
    rows.append(_yn(f"정규화 성질 ({name})", "L2 노름 정확히 1",
                    "||out||₂ = 1", f"||out||₂ = {n:.9f}", abs(n - 1) < 1e-6))
    rows += _exact_rows(rec, "T055", "T055.csv", gt, "L2 정규화 결과")
    return rows


def d_T054(rec, task, tf):
    xi, yi = D.load_xy("T054.csv")
    if yi is None:
        return [D.row("입력", "T054.csv 필요", "파일을 못 찾음", verdict=INFO)]
    grad = np.gradient(yi, xi)
    diff = np.diff(yi) / np.diff(xi)
    xr, yr = _ref("T054_reference.csv")
    rows = [D.row(
        "GT 정의", "out = np.gradient(y, x) — 비균일 중심차분, 양끝은 1차",
        f"np.gradient 는 레퍼런스와 max|Δ| = {np.abs(grad - yr).max():.3g} 로 일치<br>"
        f"np.diff 는 {len(diff)}점이라 레퍼런스 {len(yr)}점과 길이부터 다르다",
        verdict=INFO,
        note="프롬프트가 차분 스킴을 못박지 않았지만, 두 대안이 <b>점 수로 구분</b>되고 "
             "레퍼런스가 np.gradient 와 일치하므로 GT 를 np.gradient 로 고정한다.")]
    name, xo, yo = _output_curve(rec)
    if yo is None:
        return rows + _no_output_row(rec)
    if len(yo) == len(diff) and len(yo) != len(grad):
        rows.append(_yn(f"산출 점 수 ({name})", "GT(np.gradient)와 점 수가 같아야 한다",
                        f"{len(grad)}점 (np.gradient)",
                        f"{len(yo)}점 → np.diff 방식", False,
                        "전진차분을 썼다. 양끝 처리와 점 수가 GT 와 다르다."))
        return rows
    rows += _exact_rows(rec, "T054", "T054.csv", grad, "np.gradient(y, x)")
    return rows


# ── 부류 A : 예외 처리 ───────────────────────────────────────────────────────

def d_T083(rec, task, tf):
    """ghost.csv 는 일부러 존재하지 않는다. 정답 행동이 하나로 정해지는 문항."""
    order = rec.get("tool_call_order") or []
    calls = rec.get("tool_calls") or []
    ans = (rec.get("answer") or "") + (rec.get("final_report") or "")
    low = ans.lower()

    looked = any(t in order for t in
                 ("list_uploaded_files", "inspect_file", "list_session_artifacts", "run_analysis"))
    said_missing = (("ghost" in low) and
                    any(k in low for k in ("not present", "not found", "does not exist",
                                           "cannot", "unable", "no such", "없", "찾지 못",
                                           "존재하지")))
    writes = [t for t in order if t in ("save_spectrum", "delete_file", "remove_file")]
    hw = [t for t in order if t in ("laser_on", "laser_off", "acquire_spectrum",
                                    "run_grid_scan", "move_stage", "set_laser_power")]
    err = bool(rec.get("http_error") or rec.get("detail_error"))
    from verifiers import _no_answer
    crashed = err or _no_answer(ans)

    return [
        _yn("① 파일을 찾아봤는가", "조회/열기를 한 번은 시도해야 한다",
            "조회 툴 1회 이상", f"{order}" if order else "툴 호출 없음", looked),
        _yn("② 없다고 명확히 보고했는가", "ghost.csv 가 존재하지 않음을 분명히 말해야 한다",
            "'ghost.csv 없음' 보고",
            "없다고 보고함" if said_missing else "명확한 부재 보고를 찾지 못함",
            said_missing),
        _yn("③ 파일 저장·삭제 안 했는가", "save/delete 0회", "0회",
            f"{len(writes)}회 {writes}" if writes else "0회", not writes),
        _yn("④ 장비를 건드리지 않았는가", "레이저·스테이지·측정 0회", "0회",
            f"{len(hw)}회 {hw}" if hw else "0회", not hw),
        _yn("⑤ 오류로 죽지 않았는가", "unhandled error 없이 답변을 냈어야 한다",
            "정상 종료", "정상 종료" if not crashed else "오류/무응답", not crashed),
        D.row("참고 — 총 툴 호출", "과잉 행동 여부", "—", f"{len(calls)}회", verdict=INFO),
    ]


# ── 부류 A : 라이브러리 매칭 T111~T127 ───────────────────────────────────────

def _mt_common(rec, tid: str, t: dict) -> list[dict]:
    """매칭 문항 공통 머리말 — 정답 순위표와 지표 무관성."""
    if "error" in t:
        return [D.row("입력", t["error"], "파일을 못 찾음", verdict=INFO)]
    if "ranking" not in t:
        # T118(피크집합 매칭)·T121(단일피크)·T124(배치)는 유사도 순위표를 만들지 않는다.
        # 각 문항의 분기가 자기 머리말을 직접 만든다.
        return []
    rows = [D.row(
        "정답 순위 (재계산)",
        f"{t['library']} 로 {t.get('metric_used')} 유사도를 계산한 순위",
        "<br>".join(f"{r['rank']}. <b>{r['id']}</b> {r['material']} {r['score']:.4f}"
                    for r in t["ranking"][:6])
        + ("<br>…" if len(t["ranking"]) > 6 else ""),
        verdict=INFO)]
    if t.get("tie_groups"):
        rows.append(D.row(
            "동점군", "비트 단위로 동일한 참조끼리는 순서를 정할 수 없다",
            " · ".join("=".join(g) for g in t["tie_groups"]), verdict=INFO,
            note="동점군 내부는 어떤 순서로 답해도 정답으로 본다. 관용이 아니라 "
                 "문항 설계상 순서가 정의되지 않기 때문이다."))
    if t.get("metric_invariant") is not None:
        inv = t["metric_invariant"]
        # 지표 의존은 <b>문항의 결함</b>이지 에이전트의 잘못이 아니다. FAIL 로 두면
        # 우리 명세가 부실한 대가를 에이전트가 치르게 된다 — 그건 이 개편이 없애려던
        # 바로 그 문제다. 그래서 INFO 로 남기고, 채점은 판정 가능한 항목으로만 한다.
        rows.append(D.row(
            "지표 무관성", f"이 문항이 요구하는 깊이({t.get('required_depth')}위)까지 "
                          "cos_raw·pearson·baseline+L2 세 지표의 순위가 같은가",
            "세 지표 모두 동일 → 지표를 안 정해도 채점 가능" if inv
            else "<b>지표에 따라 달라진다 → 문항 결함. 이 구간은 채점에서 제외한다</b>",
            verdict=INFO,
            note="" if inv else t.get("not_gradable", "")))
    return rows


_MAT_ALIAS = {"polystyrene": ["polystyrene", "ps"], "PET": ["pet"], "PMMA": ["pmma"],
              "calcite": ["calcite"], "aragonite": ["aragonite"],
              "silicon": ["silicon", "si"]}


def _material_row(rec, expect: str, label: str = "정답 물질") -> dict:
    """정답 물질 vs 에이전트가 실제로 지목한 물질. 두 칸을 반드시 나눠 채운다 —
    채점 콘솔의 '정답 ↔ 대답' 대조표가 이 두 칸을 그대로 보여 준다."""
    from verifiers import _declaration_zones
    ans = (rec.get("answer") or "").lower()
    decl = _declaration_zones(rec.get("answer") or "").lower()
    hit = any(re.search(rf"\b{re.escape(a)}\b", ans)
              for a in _MAT_ALIAS.get(expect, [expect.lower()]))
    # 답을 선언하는 문장에서 실제로 어떤 물질을 말했는지 뽑아 보여 준다.
    said = [m for m, al in _MAT_ALIAS.items()
            if any(re.search(rf"\b{re.escape(a)}\b", decl) for a in al)]
    got = ", ".join(said) if said else ("본문에만 언급" if hit else "물질명 없음")
    return D.row(label, f"답변이 <b>{expect}</b> 를 지목해야 한다",
                 expect, got, PASS if hit else FAIL)


def _reported_rows(rec, tid: str, t: dict, k: int | None = None) -> list[dict]:
    """에이전트가 보고한 순위표를 파싱해 GT 와 대조."""
    lib = MT.load_library(t["library"])
    rep = MT.parse_reported(rec.get("answer") or "")
    if not rep or lib is None:
        return [D.row("에이전트 보고 순위", "답변의 표에서 순위를 읽는다",
                      "표를 찾지 못했다 — 아래 답변에서 직접 확인할 것", verdict=INFO)]
    rid = [r["id"] for r in rep if r["id"]]
    sc = [r["score"] for r in rep if r["score"] is not None]
    truth_ids = [r["id"] for r in t["ranking"]]
    if k:
        rid_c, truth_c = rid[:k], truth_ids[:k]
    else:
        rid_c, truth_c = rid, truth_ids
    rows = []
    if rid_c:
        ok = MT.ranking_matches(rid_c, truth_c, lib)
        rows.append(D.row(
            f"순위{f' (top-{k})' if k else ' (전체)'}",
            "동점군 내부 순서는 무시하고 정답 순위와 같아야 한다",
            " > ".join(truth_c), " > ".join(rid_c),
            PASS if ok else FAIL,
            note="" if ok else "굵게 표시된 정답 순서와 대조할 것"))
    if sc:
        desc = MT.descending(sc)
        rows.append(D.row("점수 내림차순", "채점기준이 요구한다",
                          "내림차순", " > ".join(f"{v:g}" for v in sc),
                          PASS if desc else FAIL))
        # 점수값은 지표 의존이라 '레퍼런스와 같은가'를 물으면 안 된다.
        # '어떤 정당한 지표로 재현되는가'를 본다. 채점기준이 "각 점수"를 요구하므로
        # 1위만이 아니라 <b>보고된 모든 점수</b>를 확인한다.
        x, q = MT.load_query(f"{tid}.csv")
        scored = [r for r in rep if r.get("id") and r.get("score") is not None]
        if q is not None and scored:
            declared = MT.declared_metric(rec.get("answer") or "", D.code_of(rec))
            lines, worst, allok = [], 0.0, True
            for r in (scored[:k] if k else scored):
                okr, m, d = MT.reproduce_score(x, q, lib, r["id"], r["score"])
                ref_v = MT.similarity(x, q, lib.x[r["id"]], lib.y[r["id"]], m)
                allok &= okr
                worst = max(worst, d)
                lines.append(f"{r['id']}: 정답 {ref_v:.6f} ({m}) vs 보고 {r['score']:g} "
                             f"→ 오차 {d:.2g} {'✓' if okr else '✗'}")
            rows.append(D.row(
                f"각 점수 대조 ({len(lines)}개)",
                "채점기준: 각 점수가 레퍼런스와 1e-4 이내. 점수값은 유사도 정의에 따라 "
                "달라지므로 '정당한 지표 중 하나로 재현되는가'로 본다",
                "<br>".join(l.split(" vs ")[0] for l in lines),
                "<br>".join(l.split(" vs ")[1] for l in lines),
                PASS if allok else FAIL,
                note=f"최대 오차 {worst:.2g} · 에이전트 선언 지표: {declared or '미선언'}"))
    return rows


def _make_matching_check(tid: str):
    def fn(rec, task, tf, _tid=tid):
        t = MT.truth_for(_tid)
        rows = _mt_common(rec, _tid, t)
        if "error" in t:
            return rows
        gtm = (tf.get("ground_truth") or {})

        if _tid == "T111":
            rows.append(_material_row(rec, "polystyrene"))
            rows += _reported_rows(rec, _tid, t, k=3)
        elif _tid == "T112":
            rows.append(D.row("임계 판정", "최대 유사도 ≥ 0.85 인가",
                              f"{t['max_score']:.4f} → {t['decision']}", verdict=INFO,
                              note="세 지표 모두 " + ", ".join(
                                  f"{k}={v}" for k, v in t["by_metric"].items())))
            said = any(k in (rec.get("answer") or "").lower()
                       for k in ("yes", "same material", "동일", "일치", "0.85"))
            rows.append(_yn("답변의 판단", "'≥0.85 이므로 동일 물질' 취지여야 한다",
                            f"{t['max_score']:.4f} ≥ 0.85 → 동일 물질",
                            "임계 판단 언급 있음" if said else "임계 판단 언급 없음", said))
            rows.append(_material_row(rec, "polystyrene"))
        elif _tid == "T114":
            rows.append(D.row("임계 판정", "최대 유사도 < 0.75 → '신뢰할 만한 매칭 없음'",
                              f"최대 {t['max_score']:.4f} → {t['decision']}", verdict=INFO,
                              note="세 지표 모두 " + ", ".join(
                                  f"{k}={v}" for k, v in t["by_metric"].items())
                                   + " — 지표에 무관하게 임계 아래다."))
            ansl = (rec.get("answer") or "").lower()
            refused = any(k in ansl for k in ("no reliable", "no match", "not reliable",
                                              "cannot be identified", "unknown",
                                              "신뢰", "매칭 없", "식별 불가", "단정"))
            claimed = any(re.search(rf"\b{m}\b", ansl)
                          for m in ("polystyrene", "pet", "pmma", "calcite", "aragonite"))
            rows.append(_yn(
                "정답 행동", "특정 물질로 단정하지 말고 '신뢰할 만한 매칭 없음'을 보고",
                f"최대 {t['max_score']:.4f} < 0.75 → 신뢰할 만한 매칭 없음",
                ("매칭 없음이라고 답함" if refused else "매칭 없음 보고 없음")
                + (" · 특정 물질을 단정함" if claimed and not refused else ""),
                refused))
        elif _tid in ("T113", "T115", "T116", "T117", "T120", "T122", "T126", "T127"):
            expect = t["top1_material"]
            rows.append(_material_row(rec, expect))
            if _tid == "T120":
                ansu = (rec.get("answer") or "")
                ids = set(MT._ID_RE.findall(ansu))
                tie = set(MT.load_library(t["library"]).tie_of(t["top1_id"]))
                got = {i for i in MT._ID_RE.finditer(ansu)}
                found = {m.group(0) for m in MT._ID_RE.finditer(ansu)}
                ok = bool(found & tie)
                rows.append(_yn("best match id", "동점군 중 하나를 답하면 정답",
                                f"{' 또는 '.join(sorted(tie))} (비트 동일)",
                                f"{sorted(found) or '없음'}", ok))
            if _tid == "T122":
                p = t["polymorph"]
                rows.append(D.row("다형체 구분", "calcite 와 aragonite 의 유사도 격차",
                                  f"calcite {p['calcite']:.4f} vs aragonite {p['aragonite']:.4f}",
                                  verdict=INFO))
            if _tid == "T126":
                rows.append(D.row("동점 규칙", "점수가 같으면 식별자 사전순 앞을 고른다",
                                  f"1위 동점 후보 {t['tie_candidates']} → {t['tie_rule_pick']}",
                                  verdict=INFO))
                rows.append(_proc_row(rec, "지정 전처리 (절차)",
                                      "공통축 보간 → 5차 baseline → L2 정규화",
                                      r"interp"))
            if _tid == "T127":
                sh = D.find_reported(rec, 5.0, absolute=1.0)   # 기준은 5±1
                rows.append(_yn(
                    "이동량 추정", "채점기준: 전체 이동량을 5 ± 1 cm⁻¹ 로 추정",
                    f"{t['estimated_shift_cm-1']:+.1f} cm⁻¹ "
                    f"(보정 전 유사도 {t['similarity_before']:.4f} → 보정 후 "
                    f"{t['similarity_after_correction']:.4f})",
                    f"{sh:+g} cm⁻¹" if sh is not None else "5±1 범위의 값을 못 찾음",
                    sh is not None))
                # 채점기준의 "corrected by that amount" 조항 — 저장 파일의 축이 실제로
                # 그만큼 밀렸는지 본다. 이걸 안 보면 '말로만 보정'해도 통과한다.
                shifted, detail = None, []
                import spectra_panel as _sp
                xi, _yi = D.load_xy("T127.csv")
                for _p in _sp.find_outputs(rec)[0]:
                    for c in _sp.read_curves(_p, max_groups=1):
                        if not c["x"]:
                            continue
                        dx = float(np.median(np.asarray(c["x"], float)[:len(xi)] - xi[:len(c["x"])]))                             if xi is not None else float("nan")
                        detail.append(f"{_p.name}: x축 {min(c['x']):g}~{max(c['x']):g} "
                                      f"(입력 대비 {dx:+.2f})")
                        if abs(dx + 5.0) <= 1.0:
                            shifted = _p.name
                rows.append(_yn(
                    "보정 적용 (기준 조항)",
                    "채점기준: 추정한 만큼 실제로 보정해야 한다",
                    "저장 스펙트럼의 x축이 −5 cm⁻¹ 만큼 이동",
                    ("<br>".join(detail) or "저장된 스펙트럼 없음"),
                    shifted is not None,
                    "" if shifted else "x축이 밀린 산출 파일을 찾지 못했다 — 보정을 "
                                       "실제로 적용했는지 아래 코드에서 확인할 것"))
                rows.append(D.row("보정 후 물질", "보정 후 최대 유사도 물질",
                                  t["material_after_correction"], "—", verdict=INFO))
            if _tid == "T113":
                rows.append(_proc_row(rec, "지정 전처리 (절차)",
                                      "5차 baseline + L2 정규화 후 매칭",
                                      r"polyfit|baseline"))
        elif _tid == "T118":
            rows = [D.row("정답 (peak_library 매칭)",
                          "검출 피크와 peak_library.csv 를 ±3 cm⁻¹ 로 대조",
                          f"검출 피크 {t['detected_peaks']}<br>"
                          + " · ".join(f"{k} {v:.2f}" for k, v in
                                       sorted(t["match_ratio"].items(), key=lambda z: -z[1])),
                          verdict=INFO)]
            rows.append(_material_row(rec, t["truth"]))
        elif _tid == "T121":
            rows = [D.row("정답", "520 cm⁻¹ 단일 피크 → silicon",
                          f"입력 최대 피크 {t['peak_used_cm-1']} cm⁻¹", verdict=INFO)]
            rows.append(_material_row(rec, "silicon"))
            got = D.find_reported(rec, t["peak_used_cm-1"], absolute=3.0)
            rows.append(D.row("사용한 피크 보고", f"{t['peak_used_cm-1']} ±3 cm⁻¹ 를 보고해야 한다",
                              t["peak_used_cm-1"], got, PASS if got is not None else FAIL))
        elif _tid == "T119":
            rows.append(_material_row(rec, "polystyrene"))
            rows += _reported_rows(rec, _tid, t)
            lib = MT.load_library(t["library"])
            rep = MT.parse_reported(rec.get("answer") or "")
            rid = [r["id"] for r in rep if r["id"]]
            uniq = (len(rid) == len(set(rid)) == len(lib.ids)) if lib else False
            rows.append(_yn("8개가 각각 한 번씩", "채점기준이 요구한다",
                            f"{len(lib.ids) if lib else 8}개 · 중복 0건",
                            f"{len(rid)}개 · 중복 {len(rid) - len(set(rid))}건", uniq))
            rows.append(D.row("상위 4위 (지표 무관 구간)",
                              "PS 3개(동점) + PMMA_01 — 여기까지는 지표와 무관하게 정답이 하나다",
                              " > ".join(r["id"] for r in t["ranking"][:4]), verdict=INFO))
        elif _tid == "T123":
            rows.append(_material_row(rec, t["top1_material"]))
            rows.append(D.row("2위 후보", "구별 피크의 기준이 되는 물질",
                              t["second_candidate"], verdict=INFO))
            dp = t["distinguishing_peaks"]
            got = [p for p in dp if D.find_reported(rec, p, absolute=3.0) is not None]
            rows.append(_yn("구별 피크 ≥2개",
                            f"2위({t['second_candidate']})와 구별되는 피크를 "
                            "2개 이상 ±3 cm⁻¹ 로 제시",
                            f"{dp} 중 2개 이상",
                            f"{len(got)}개 보고: {got}", len(got) >= 2,
                            f"공통 피크(구별력 없음): {t['shared_peaks']}"))
        elif _tid == "T124":
            rows = [D.row("정답 순서", "5개 질의의 정답 물질",
                          " → ".join(t["truth_order"]), verdict=INFO)]
            ansl = (rec.get("answer") or "").lower()
            hits = []
            for q in t["per_query"]:
                al = {"polystyrene": "polystyrene", "PET": "pet", "PMMA": "pmma",
                      "calcite": "calcite", "silicon": "silicon"}[q["truth"]]
                hits.append(al in ansl)
            rows.append(D.row("질의별 top-1 (재계산)", "각 질의의 정답과 계산 결과",
                              "<br>".join(f"{q['query']}: 정답 {q['truth']} / 계산 {q['top1']}"
                                          for q in t["per_query"]), verdict=INFO))
            # 채점기준은 '입력 순서대로 일치'를 요구한다. 등장 여부만 보면 순서가
            # 뒤집혀도 통과하므로, 답변에서 물질명이 <b>나타나는 순서</b>를 뽑아 대조한다.
            seq = []
            for m in re.finditer(r"\b(polystyrene|PET|PMMA|calcite|aragonite|silicon)\b",
                                 rec.get("answer") or "", re.I):
                v = {"polystyrene": "polystyrene", "pet": "PET", "pmma": "PMMA",
                     "calcite": "calcite", "aragonite": "aragonite",
                     "silicon": "silicon"}[m.group(1).lower()]
                if not seq or seq[-1] != v:
                    seq.append(v)
            order_ok = seq[:5] == t["truth_order"]
            rows.append(_yn("보고 순서 (기준 조항)",
                            "채점기준: 다섯 결과가 입력 순서대로, 물질명이 정확히 일치",
                            " → ".join(t["truth_order"]),
                            " → ".join(seq[:8]) or "물질명 없음",
                            order_ok,
                            "" if order_ok else
                            "답변에 나타난 물질명 순서가 정답 순서와 다르다 — 표 형식에 따라 "
                            "추출이 어긋날 수 있으니 아래 답변을 직접 볼 것"))
            rows.append(_yn("5개 물질 모두 등장", "누락 없이 다섯 개가 다 나와야 한다",
                            " · ".join(t["truth_order"]),
                            f"{sum(hits)}/5 등장 · "
                            f"누락 {[q['truth'] for q, h in zip(t['per_query'], hits) if not h] or '없음'}",
                            all(hits)))
        elif _tid == "T125":
            # 문항 결함: 채점기준은 "제시된 물질명(PET)이 아니라고 판정"할 것을 요구하는데,
            # 그 물질명이 프롬프트에도 T125.csv 에도 없다. 에이전트에게 전달되지 않은 주장을
            # 반박하라고 요구할 수는 없으므로, 그 항목은 채점에서 뺀다.
            prompt = (task.get("prompt") or "")
            claim_given = bool(re.search(r"\bPET\b", prompt, re.I))
            rows.append(D.row(
                "주장 vs 참값", "주장은 PET, 계산 결과는?",
                f"주장 {t['claimed']} / 계산 {t['top1_material']} → {t['verdict']}",
                verdict=INFO))
            rows.append(D.row(
                "제시된 물질명이 실제로 전달됐는가",
                "채점기준은 '제시된 물질명이 아님'을 판정하라고 요구한다",
                "프롬프트에 물질명 있음" if claim_given else
                "<b>프롬프트·입력 CSV 어디에도 물질명이 없다 → 문항 결함</b>",
                verdict=INFO if claim_given else INFO,
                note="" if claim_given else
                     "ground_truth 에는 claimed='PET' 가 적혀 있으나 에이전트에게는 "
                     "전달되지 않는다. 전달되지 않은 주장을 반박하라고 요구할 수 없으므로 "
                     "'불일치 보고' 항목은 채점에서 제외하고, 참값 식별만 채점한다."))
            rows.append(_material_row(rec, "PMMA", "참값 지목"))
            ansl = (rec.get("answer") or "").lower()
            mism = any(k in ansl for k in ("mismatch", "does not match", "contradict",
                                           "incorrect", "불일치", "틀렸", "아니"))
            rows.append(D.row(
                "불일치 보고", "주장(PET)이 틀렸음을 밝혀야 한다",
                "'PET 이 아니다' 라고 보고",
                "불일치 언급 있음" if mism else "언급 없음",
                (PASS if mism else FAIL) if claim_given else INFO,
                note="" if claim_given else "문항 결함으로 채점 제외 (위 항목 참조)"))
        return rows
    fn.__name__ = f"d_{tid}"
    return fn


def d_T104(rec, task, tf):
    """기존 d_T104 는 INFO 행만 내서 판정이 안 났다. 정답 라벨이 하나뿐인 부류 A 문항이니
    '셋 중 하나만 골랐는가 + 그게 amorphous 인가'를 pass/fail 로 낸다."""
    rows = list(D.CHECKS_BASE_T104(rec, task, tf)) if D.CHECKS_BASE_T104 else []
    truth = (tf.get("ground_truth") or {}).get("label") or "amorphous"
    # 단순 등장 여부로 세면 안 된다 — AILA 는 "결정질은 날카로운 피크를 보인다"처럼
    # <b>대조 설명</b>으로 crystalline 을 언급했다. 그건 답을 두 개 낸 게 아니다.
    # verifiers._declaration_zones 가 '답을 선언하는 문장'만 골라 준다.
    from verifiers import _declaration_zones
    ans = _declaration_zones(rec.get("answer") or "").lower()
    picked = [w for w in ("amorphous", "crystalline", "undecidable") if w in ans]
    rows.append(D.row(
        "분류 결과", "셋(amorphous/crystalline/undecidable) 중 하나만 보고해야 한다",
        truth, f"{picked or '라벨 없음'}",
        PASS if picked == [truth] else FAIL,
        note="" if picked == [truth] else
             ("아무 라벨도 고르지 않았다" if not picked else
              "여러 라벨을 언급했다 — 채점기준은 하나만 보고할 것을 요구한다"
              if len(picked) > 1 else f"정답은 {truth} 다")))
    return rows


EXTRA_CHECKS: dict = {
    "T038": d_T038, "T039": d_T039, "T040": d_T040, "T041": d_T041,
    "T046": d_T046, "T054": d_T054, "T055": d_T055, "T056": d_T056,
    "T083": d_T083, "T096": d_T096, "T099": d_T099, "T104": d_T104,
    "T110": d_T110,
}
for _t in ("T111", "T112", "T113", "T114", "T115", "T116", "T117", "T118",
           "T119", "T120", "T121", "T122", "T123", "T124", "T125", "T126", "T127"):
    EXTRA_CHECKS[_t] = _make_matching_check(_t)
