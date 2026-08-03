# -*- coding: utf-8 -*-
"""스펙트럼 합성 유틸 — 입력 파일과 GT 가 같은 자리에서 나오도록 하는 기본 블록.

모든 난수는 호출부가 넘긴 시드로 만든 np.random.default_rng 만 쓴다. 전역 난수를 쓰면
생성 순서가 바뀔 때 파일이 조용히 달라져 GT 와 어긋난다.
"""
from __future__ import annotations

import numpy as np

from materials import AXIS_MIN, AXIS_MAX, AXIS_STEP, MATERIALS


def axis() -> np.ndarray:
    """공통 파수축 (200~2000 cm-1, 1801점)."""
    n = int(round((AXIS_MAX - AXIS_MIN) / AXIS_STEP)) + 1
    return AXIS_MIN + AXIS_STEP * np.arange(n)


def lorentz(x, center, height, fwhm):
    """라만 밴드는 가우시안보다 로렌치안에 가깝다(꼬리가 길다)."""
    g = fwhm / 2.0
    return height * g ** 2 / ((x - center) ** 2 + g ** 2)


def pure(material: str, x=None, broaden: float = 1.0, shift: float = 0.0) -> np.ndarray:
    """물질의 순수 스펙트럼.

    broaden — 모든 밴드의 반치폭에 곱하는 배율. 라이브러리에서 같은 물질의 서로 다른
              항목을 만드는 수단이다(결정성·측정조건 차이에 해당). 결정론적이라
              물질과 무관하게 일정한 차이를 낸다 — materials.py 의 주석 참조.
    shift   — 전체 밴드를 같은 방향으로 이동(cm-1). T103·T129 의 파수 어긋남 주입용.
    """
    if x is None:
        x = axis()
    y = np.zeros_like(x, dtype=float)
    for c, h, w in MATERIALS[material]:
        y += lorentz(x, c + shift, h, max(w * broaden, 1.0))
    return y


def scale_counts(y, peak_counts=1000.0):
    """상대세기를 검출기 카운트 스케일로. 최대 피크가 peak_counts 가 되게 맞춘다."""
    m = float(np.max(y))
    return y * (peak_counts / m) if m > 0 else y


def fluorescence(x, amp, center=1400.0, width=900.0, slope=0.0, offset=0.0):
    """형광 배경 — 넓은 가우시안 + 선형 성분. baseline 문항의 '제거 대상'."""
    return amp * np.exp(-((x - center) / width) ** 2) + slope * (x - x[0]) + offset


def noise(x, sigma, seed):
    return np.random.default_rng(seed).normal(0.0, sigma, size=len(x))


SPIKE_MIN_GAP = 8          # 주입 스파이크 사이의 최소 간격(점)
SPIKE_Z = 100.0            # 검출 임계 (detect_spikes 의 k). 근거는 아래 주석.


def add_spikes(y, x, n, seed, height_mult=(6.0, 14.0)):
    """폭 1점짜리 인위 스파이크를 넣고, 넣은 인덱스를 함께 돌려준다.

    [왜 최소 간격을 강제하는가 — 검증에서 걸린 문제]
    처음에는 위치를 무작위로만 뽑았는데, 두 스파이크가 인접(872, 873)한 경우가 나왔다.
    폭 1점을 전제로 한 3점 중앙값 필터는 2점 연속 스파이크를 지우지 못해 편차가 0 이 되고,
    그 스파이크는 영영 검출되지 않는다. GT 가 '검출 불가능한 정답'을 담게 되는 셈이다.
    SPIKE_MIN_GAP 으로 떨어뜨려 이 경우를 없앤다.

    GT 는 '넣은 인덱스'가 아니라 '규약대로 검출되는 인덱스'다 — 둘이 다를 수 있으므로
    여기서는 넣은 자리만 반환하고 정답은 detect_spikes 로 다시 구한다.
    """
    rng = np.random.default_rng(seed)
    y = y.copy()
    span = float(np.max(y) - np.min(y))
    lo, hi = 20, len(x) - 20
    idx: list[int] = []
    guard = 0
    while len(idx) < n and guard < 10000:
        guard += 1
        c = int(rng.integers(lo, hi))
        if all(abs(c - p) >= SPIKE_MIN_GAP for p in idx):
            idx.append(c)
    idx.sort()
    for i in idx:
        y[i] += span * rng.uniform(*height_mult) / 10.0
    return y, idx


def detect_spikes(y, k=SPIKE_Z):
    """스파이크 판정 — 3점 중앙값 필터 대비 편차가 강건 잡음의 k 배를 넘는 점.

    [왜 이 규약인가 — 두 번의 실패 끝에]
    (1) '5점 이동중앙값 대비 5·MAD' 로 시작했더니 주입 6개에 검출 52개가 나왔다.
        폭이 좁은 진짜 라만 밴드(1001 cm-1 은 FWHM 6점)가 이동중앙값에서 크게 벗어나기
        때문이다. 스파이크와 날카로운 피크를 '편차 크기'로는 가를 수 없다.
    (2) Whitaker-Hayes(1차 차분의 수정 z-점수)도 같은 이유로 과검출했다 — 가파른 실제
        피크의 미분값이 스파이크만큼 크다.
    (3) 가르는 것은 크기가 아니라 폭이다. 3점 중앙값은 1점 스파이크를 완전히 지우고
        6점 폭의 실제 밴드는 거의 그대로 두므로, 편차 자체가 스파이크에서만 크게 남는다.
        실측: 주입 스파이크 292~643 σ vs 실제 피크 정점 최대 59 σ — 6배 이상 벌어진다.
        임계 100 은 그 사이에 두어 양쪽에 마진을 남긴다(verify_dataset 이 매번 확인한다).
    """
    from scipy.signal import medfilt
    dev = np.asarray(y, dtype=float) - medfilt(np.asarray(y, dtype=float), 3)
    sigma = 1.4826 * float(np.median(np.abs(dev - np.median(dev))))
    if sigma <= 0:
        sigma = 1e-12
    return np.where(np.abs(dev) > k * sigma)[0].tolist()


def remove_spikes(y, k=SPIKE_Z):
    """검출된 스파이크를 이웃 선형보간으로 대체. (정제배열, 검출인덱스) 반환."""
    idx = detect_spikes(y, k)
    out = np.asarray(y, dtype=float).copy()
    if idx:
        good = np.setdiff1d(np.arange(len(out)), np.array(idx))
        out[idx] = np.interp(np.array(idx, dtype=float), good.astype(float), out[good])
    return out, idx


def ipbsa(y, x, order=5, max_iter=100, thr=1e-3):
    """IPBSA(반복 다항 배경 제거) — 문항 규약과 동일한 구현.

    apply_background_subtraction 툴과 같은 계열이어야 2차 채점(배열 유사도)이 성립한다.
    """
    yy = np.asarray(y, dtype=float).copy()
    for _ in range(max_iter):
        b = np.polyval(np.polyfit(x, yy, order), x)
        new = np.minimum(yy, b)
        prev = np.linalg.norm(yy)
        if prev > 0 and np.linalg.norm(new - yy) / prev < thr:
            yy = new
            break
        yy = new
    bg = np.polyval(np.polyfit(x, yy, order), x)
    return y - bg, bg


def l2(y):
    n = float(np.linalg.norm(y))
    return y / n if n > 0 else y


def minmax(y):
    lo, hi = float(np.min(y)), float(np.max(y))
    return (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def find_peaks_spec(y, x, prominence_frac=0.05):
    """문항 규약의 피크 검출 — prominence = 세기 범위의 5%.

    scipy 를 쓰되, 채점기도 같은 호출을 하도록 이 함수를 단일 출처로 둔다.
    """
    from scipy.signal import find_peaks
    prom = (float(np.max(y)) - float(np.min(y))) * prominence_frac
    idx, props = find_peaks(y, prominence=prom)
    return idx, props, prom


def snr_spec(y, x, sig=(990.0, 1012.0), noi=(1800.0, 1900.0)):
    """T050 정의의 SNR — 신호 구간 최대 / 잡음 구간 표본표준편차(ddof=1)."""
    s = (x >= sig[0]) & (x <= sig[1])
    n = (x >= noi[0]) & (x <= noi[1])
    sd = float(np.std(y[n], ddof=1))
    return float(np.max(y[s])) / sd if sd > 0 else float("inf")


def fwhm_spec(y, x, lo, hi):
    """T045 규약의 FWHM — 반높이는 (피크세기 - 구간최소)/2, 교차점은 선형보간."""
    m = (x >= lo) & (x <= hi)
    xs, ys = x[m], y[m]
    i = int(np.argmax(ys))
    half = (ys[i] - float(np.min(ys))) / 2.0 + float(np.min(ys))
    left = i
    while left > 0 and ys[left] > half:
        left -= 1
    right = i
    while right < len(ys) - 1 and ys[right] > half:
        right += 1
    xl = np.interp(half, [ys[left], ys[left + 1]], [xs[left], xs[left + 1]])
    xr = np.interp(half, [ys[right], ys[right - 1]], [xs[right], xs[right - 1]])
    return float(abs(xr - xl))
