# -*- coding: utf-8 -*-
"""스펙트럼 규약 — 에이전트가 저장한 파일을 읽고, 문항이 말한 방식 그대로 다시 계산한다.

[왜 다시 계산하는가 — '사후 GT']
장비로 실제 측정한 값의 절대치는 시료·정렬·그날 상태에 따라 달라져서 미리 정답을 적어 둘
수 없다. 하지만 **에이전트 자신이 저장한 파일들의 함수**는 결정적이다:
  · 5회 측정의 평균은 그 5개 파일로 정해진다
  · 노출을 4배로 올리면 SNR 은 커진다 — 어떤 시료든
  · 두 스펙트럼의 코사인 유사도는 그 둘로 정해진다
그래서 채점기가 에이전트의 파일을 다시 읽어 규약대로 계산하고, 보고값과 맞춰 본다.

[규약은 여기 한 곳에만 적는다]
문항 본문(xlsx)에 적힌 정의와 이 파일이 어긋나면 채점이 조용히 틀린다. 정의를 바꿀 일이
있으면 여기와 문항 본문을 같이 고쳐야 한다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# ── T050 이 정한 SNR 정의 ────────────────────────────────────────────────────
# 신호 = 990~1012 cm-1 구간의 최대값
# 잡음 = 1800~1900 cm-1 구간의 표본표준편차(ddof=1)
# 이 구간은 장비 파수축이 덮어야 한다. 못 덮으면 SNR 은 계산 자체가 불가능하고, 그건
# 에이전트 잘못이 아니다 — Task.windows 로 미리 걸러서 '채점 불가'로 뺀다.
SNR_SIGNAL = (990.0, 1012.0)
SNR_NOISE = (1800.0, 1900.0)

# 피크 검출 규약 — prominence 는 세기 범위의 5%
PEAK_PROMINENCE_FRAC = 0.05

# 저장 CSV 에서 찾는 열 이름(프로젝트가 쓰는 표기들)
X_COLUMNS = ("raman_shift_cm-1", "raman_shift", "wavenumber", "wavenumber_cm-1", "shift")
Y_COLUMNS = ("corrected_intensity", "intensity", "counts", "y")


# ══════════════════════════════════════════════════════════════════════════════
# 파일 읽기
# ══════════════════════════════════════════════════════════════════════════════
def load_saved(artifacts, data_root: Path):
    """에이전트가 저장한 스펙트럼들 — [(경로, x, y), ...] 를 저장 순서대로.

    스펙트럼이 아닌 산출물(그림·요약 json)은 조용히 건너뛴다.
    """
    out = []
    for rel in artifacts or []:
        p = Path(rel)
        p = p if p.is_absolute() else (data_root / rel)
        if p.suffix.lower() != ".csv":
            continue
        xy = read_xy(p)
        if xy is not None:
            out.append((str(rel), xy[0], xy[1]))
    return out


def read_xy(path: Path):
    """CSV 한 개 → (x, y). 못 읽으면 None.

    프로젝트의 두 포맷을 모두 받는다: 주석 메타행(#)이 붙은 것과 순수 헤더만 있는 것.
    BOM 은 utf-8-sig 로 흡수한다 — 안 떼면 첫 열 이름이 어긋난다.

    x 축이 없으면 **None 을 돌려준다**. 예전에는 인덱스를 x 로 대체했는데, 그러면
    피크 위치와 SNR 이 조용히 '픽셀 단위'가 되어 전혀 다른 값을 정답처럼 내놓는다.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        lines = p.read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return None
    body = [ln for ln in lines if not ln.lstrip().startswith("#")]
    if not body:
        return None
    rd = csv.DictReader(body)
    rows = list(rd)
    cols = rd.fieldnames or []
    if not rows or not cols:
        return None

    xc = _pick(cols, X_COLUMNS)
    yc = _pick(cols, Y_COLUMNS)
    if yc is None:
        numeric = [c for c in cols if _numeric(rows, c)]
        if not numeric:
            return None
        yc = numeric[-1]
        xc = xc or (numeric[0] if len(numeric) > 1 and numeric[0] != yc else None)
    if xc is None:
        return None
    try:
        x = np.array([float(r[xc]) for r in rows])
        y = np.array([float(r[yc]) for r in rows])
    except (TypeError, ValueError):
        return None
    return (x, y) if x.size and x.size == y.size else None


def _pick(cols, names):
    low = {c.strip().lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def _numeric(rows, c):
    try:
        for r in rows[:5]:
            float(r[c])
        return True
    except (TypeError, ValueError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 규약 계산
# ══════════════════════════════════════════════════════════════════════════════
def snr(x, y):
    """T050 정의의 SNR. 구간이 축 밖이면 None — 0 이 아니라 '계산 불가'다."""
    s = (x >= SNR_SIGNAL[0]) & (x <= SNR_SIGNAL[1])
    n = (x >= SNR_NOISE[0]) & (x <= SNR_NOISE[1])
    if s.sum() == 0 or n.sum() < 2:
        return None
    sd = float(np.std(y[n], ddof=1))
    if sd <= 0:
        return None
    v = float(np.max(y[s]) / sd)
    return v if np.isfinite(v) else None


def peaks(x, y):
    """규약대로 검출한 피크 **위치**(cm-1). scipy 는 인덱스를 주므로 여기서 변환한다."""
    from scipy.signal import find_peaks
    prom = (float(np.max(y)) - float(np.min(y))) * PEAK_PROMINENCE_FRAC
    idx, _ = find_peaks(y, prominence=prom)
    return [float(x[i]) for i in idx]


def strongest_peak(x, y):
    """가장 센 피크의 위치(cm-1). 피크가 없으면 최대값 위치."""
    p = peaks(x, y)
    if not p:
        return float(x[int(np.argmax(y))])
    return max(p, key=lambda pos: float(y[int(np.argmin(np.abs(x - pos)))]))


def band_max(x, y, lo, hi):
    """구간 최대 세기. 구간이 비면 None."""
    m = (x >= lo) & (x <= hi)
    return float(np.max(y[m])) if m.sum() else None


def l2(v):
    n = float(np.linalg.norm(v))
    return np.asarray(v, float) / n if n > 0 else np.asarray(v, float)


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def cosine_distance(a, b):
    return 1.0 - cosine(a, b)


def on_common_axis(xa, ya, xb, yb):
    """두 스펙트럼을 겹치는 구간의 공통 축으로 맞춘다.

    np.interp 는 범위 밖을 양끝 값으로 **상수 외삽**한다. 축이 다른 두 스펙트럼을 그냥
    보간하면 겹치지 않는 구간이 평평한 선으로 채워져 유사도가 실제보다 높게 나온다.
    그래서 공통 구간으로 자른 뒤 보간한다.
    """
    lo = max(float(np.min(xa)), float(np.min(xb)))
    hi = min(float(np.max(xa)), float(np.max(xb)))
    if not (hi > lo):
        return None
    grid = np.linspace(lo, hi, min(len(xa), len(xb)))
    return grid, np.interp(grid, xa, ya), np.interp(grid, xb, yb)


def rsd_percent(values):
    """표본 상대표준편차(%) — ddof=1."""
    v = np.asarray([x for x in values if x is not None], float)
    if v.size < 2:
        return None
    m = float(v.mean())
    return float(np.std(v, ddof=1) / m * 100.0) if m else None


def saturated_count(y, ceiling=65535):
    """검출기 상한에 붙은 점의 개수."""
    y = np.asarray(y, float)
    return int((y >= ceiling * 0.999).sum())


def covers(axis, lo, hi, need=2) -> bool:
    """파수축이 그 구간을 충분히 덮는가."""
    if axis is None or len(axis) == 0:
        return False
    a = np.asarray(axis, float)
    return int(((a >= lo) & (a <= hi)).sum()) >= need
