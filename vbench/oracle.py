# -*- coding: utf-8 -*-
"""정답 — 에피소드가 시작되기 **전에** 확정된다.

에이전트가 무엇을 하든 정답은 바뀌지 않는다. 월드를 만드는 순간 여기서 계산해
`oracle.json` 으로 적어 두고, 채점기는 그것만 읽는다.

[왜 상수를 여기 다시 적지 않는가]
정답을 구하는 물리는 world.predict() 하나뿐이다. 손상 지수나 형광 세기를 튜닝해도
정답이 자동으로 따라오고, 시뮬레이터와 채점 기준이 갈라질 수 없다.

[허용폭도 손으로 정하지 않는다]
세 축 모두 "SNR 이 10% 나빠지는 지점"까지를 허용폭으로 삼는다. 이유는 설계문서 §6.2:
최적점은 점이 아니라 **능선**이라, 실질적으로 같은 답에 다른 점수를 주면 실력이 아니라
운을 재게 된다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field as dc_field
from pathlib import Path

import numpy as np
from scipy import ndimage

from vbench.world import VirtualWorld, FIELD_DS, P_REF, ALPHA

# 허용폭을 정하는 기준. SNR 이 최적의 이 비율 아래로 떨어지는 지점까지가 허용폭이다.
# 0.95 로 잡아 봤더니 노출 허용폭이 ±0.4s 로 나왔다 — 20% 여유를 둔 신중한 답이
# 0.2 점을 받는다는 뜻이라 부당하다. 0.90 이면 실질적으로 같은 성능인 범위를 덮는다.
BAND_LEVEL = 0.90

# 탐색 범위. 실장비 한계(레이저 0.004~100%, CCD 노출)에 맞춘다.
P_MIN, P_MAX = 2.0, 100.0
T_MIN, T_MAX = 0.02, 60.0

# 타겟 색(호박색)을 사람 말로 옮긴 것. 시각적 특징 보고를 대조할 때 쓴다(점수는 아님).
VISUAL_KEYWORDS = ["amber", "orange", "golden", "yellow", "warm", "tan", "ochre",
                   "주황", "호박", "노란", "황", "누런"]


def _max_safe_exposure(w: VirtualWorld, x, y, power, iters: int = 34) -> float:
    """이 파워에서 안전한(포화 여유 O, 손상 임계 이하) 최대 노출. 없으면 0.

    안전 여부는 노출에 대해 단조롭다(노출이 길수록 조사량도 카운트도 는다)는 점을
    이용한 이분탐색이다.
    """
    if not w.predict(x, y, power, T_MIN)["safe"]:
        return 0.0
    lo, hi = T_MIN, T_MAX
    if w.predict(x, y, power, hi)["safe"]:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if w.predict(x, y, power, mid)["safe"]:
            lo = mid
        else:
            hi = mid
    return lo


def best_recipe(w: VirtualWorld, x, y, n_coarse: int = 40) -> tuple[float, float, float]:
    """이 자리에서 안전하게 얻을 수 있는 최고 SNR 과 그때의 (파워, 노출).

    Returns (snr, power_pct, exposure_s).
    """
    def snr_at(P):
        t = _max_safe_exposure(w, x, y, P)
        return (w.predict(x, y, P, t)["snr"] if t > 0 else 0.0), t

    grid = np.linspace(P_MIN, P_MAX, n_coarse)
    vals = [snr_at(P) for P in grid]
    i = int(np.argmax([v[0] for v in vals]))
    # 거친 격자에서 고른 지점 주변을 한 번 더 좁혀 본다(능선이 완만해 정밀도가 필요하다).
    lo = grid[max(0, i - 1)]
    hi = grid[min(len(grid) - 1, i + 1)]
    fine = np.linspace(lo, hi, 21)
    for P in fine:
        s, t = snr_at(P)
        if s > vals[i][0]:
            vals[i] = (s, t)
            grid[i] = P
    return float(vals[i][0]), float(grid[i]), float(vals[i][1])


def _candidate_spots(w: VirtualWorld, top_pct: float = 99.5, max_spots: int = 12):
    """타겟이 가장 진한 자리들의 스테이지 좌표. 군집마다 하나씩 고른다.

    최적점이 하나뿐이라고 가정하면 안 된다 — 동급으로 좋은 자리가 여러 곳이면
    그중 아무 데나 맞혀도 맞힌 것이다(설계문서 §6.2 규칙 2).
    """
    ft = w.field[..., 2]
    thr = float(np.percentile(ft, top_pct))
    lab, n = ndimage.label(ft >= thr)
    if n == 0:
        r, c = np.unravel_index(int(np.argmax(ft)), ft.shape)
        return [w.map_to_stage(c * FIELD_DS, r * FIELD_DS)]
    sizes = ndimage.sum(np.ones_like(ft), lab, index=range(1, n + 1))
    order = np.argsort(sizes)[::-1][:max_spots]
    out = []
    for k in order:
        m = (lab == k + 1)
        idx = int(np.argmax(np.where(m, ft, -1)))
        r, c = np.unravel_index(idx, ft.shape)
        out.append(w.map_to_stage(c * FIELD_DS, r * FIELD_DS))
    return out


def _band(f, center, lo_limit, hi_limit, target, iters: int = 28):
    """f(v) >= target 을 만족하는 구간 [lo, hi]. center 는 반드시 만족한다고 가정."""
    lo = lo_limit
    if f(lo_limit) < target:                      # 아래쪽 경계 이분탐색
        a, b = lo_limit, center
        for _ in range(iters):
            m = 0.5 * (a + b)
            if f(m) >= target:
                b = m
            else:
                a = m
        lo = b
    hi = hi_limit
    if f(hi_limit) < target:                      # 위쪽 경계
        a, b = center, hi_limit
        for _ in range(iters):
            m = 0.5 * (a + b)
            if f(m) >= target:
                a = m
            else:
                b = m
        hi = a
    return float(lo), float(hi)


@dataclass
class Oracle:
    """이 에피소드의 정답. 결과 파일에 그대로 실린다."""
    level: str
    seed: int
    best_spots_mm: list                 # 동급으로 좋은 자리들 [[x,y], ...]
    power_pct: float                    # 정답 파워 (%)
    exposure_s: float                   # 정답 노출 (s)
    snr: float                          # 그때 얻는 SNR — 이 에피소드의 상한
    pos_tol_mm: float                   # 위치 허용폭 (반경, mm)
    exposure_tol_s: float               # 노출 허용폭 (양쪽 공통 — build() 주석 참고)
    power_band: list = dc_field(default_factory=list)      # 파워는 대역이 곧 허용폭
    exposure_band: list = dc_field(default_factory=list)
    f_tgt_at_best: float = 0.0
    f_tgt_band_edge: float = 0.0
    good_area_frac: float = 0.0
    band_level: float = BAND_LEVEL
    visual_signature: list = dc_field(default_factory=lambda: list(VISUAL_KEYWORDS))

    # ── 채점이 쓰는 두 함수 — 정답 판정 로직도 정답과 같은 곳에 둔다 ──────────

    @staticmethod
    def _rel(value, center, band, floor) -> float:
        """|value-center| 를 **그쪽 방향의** 허용폭으로 나눈 값.

        [왜 한쪽 폭을 양쪽에 쓰면 안 되나]
        능선은 비대칭이다. 파워 대역이 [12.6, 19.1] 인데 최적은 14.6 이라, 위쪽 여유(4.5)가
        아래쪽(2.0)의 두 배가 넘는다. 넓은 쪽 하나를 양쪽에 쓰면 P=10 처럼 대역 밖으로
        한참 나간 답이 '가장자리에 걸친 답'과 같은 점수를 받는다(실측: 대역 [11.2,25.1] 에서
        P=2 가 d=1.0 으로 나왔다). 방향별로 나누면 '대역 가장자리 = d 1' 이 양쪽에서 성립한다.
        """
        lo, hi = float(band[0]), float(band[1])
        width = (center - lo) if value < center else (hi - center)
        return abs(float(value) - center) / max(width, floor)

    def position_error_mm(self, world, x_mm, y_mm) -> float:
        """보고한 자리에서 **'충분히 좋은 구역'까지의 거리**(mm). 그 안이면 0.

        [왜 '가장 좋은 점까지의 거리'가 아닌가 — 스케일을 바꾸고 드러난 문제]
        처음에는 오라클이 뽑아 둔 최적 자리 12곳 중 가장 가까운 곳까지의 거리를 썼다.
        군집이 컸을 때는 그 12곳이 좋은 영역을 거의 덮었지만, 군집을 잘게 만든 뒤로는
        f_tgt 가 최고급인 자리가 수백 곳이 됐다. 그중 12곳만 정답으로 치면
        **똑같이 좋은 자리에 정확히 내려놓고도 '딴 데'로 채점된다** — 실측으로 세 기준선이
        전부 거리 79 를 받았고, 그 값은 위치 항 하나가 만든 것이었다.

        지금은 'f_tgt 가 대역 가장자리 이상인 모든 지점'을 정답 구역으로 보고, 거기까지의
        거리를 잰다. 좋은 자리에 내려놓았으면 어디든 0 이다 — 재려는 것이 정확히 그것이다.
        """
        from vbench.world import FIELD_DS
        dm = getattr(self, "_distmap", None)
        if dm is None:
            return 0.0
        col, row = world.stage_to_map(x_mm, y_mm)
        r = int(round(row / FIELD_DS))
        c = int(round(col / FIELD_DS))
        h, w = dm.shape
        if not (0 <= r < h and 0 <= c < w):
            # 월드 밖을 보고했다 — 가장자리까지의 거리에 벗어난 만큼을 더한다.
            r_, c_ = min(max(r, 0), h - 1), min(max(c, 0), w - 1)
            extra = float(np.hypot(r - r_, c - c_))
            return float((dm[r_, c_] + extra) * FIELD_DS * (world.mm_x + world.mm_y) / 2.0)
        return float(dm[r, c] * FIELD_DS * (world.mm_x + world.mm_y) / 2.0)

    def distance(self, world, x_mm, y_mm, power_pct, exposure_s) -> float:
        """정답까지의 정규화 유클리드 거리 (설계문서 §6.2).

            d = √( (Δ위치/위치허용폭)² + (Δ파워/파워허용폭)² + (Δ노출/노출허용폭)² )

        허용폭은 'SNR 이 최적의 90% 로 떨어지는 지점까지'다. 즉 **d = 1 이면 실질적으로
        같은 성능인 범위의 가장자리**이고, 그때 점수가 정확히 0.5 가 된다.
        """
        dr = self.position_error_mm(world, x_mm, y_mm)
        return float(np.sqrt(
            (dr / max(self.pos_tol_mm, 1e-4)) ** 2
            + self._rel(float(power_pct), self.power_pct, self.power_band, 0.5) ** 2
            + (abs(float(exposure_s) - self.exposure_s) / max(self.exposure_tol_s, 0.1)) ** 2))

    def score(self, world, x_mm, y_mm, power_pct, exposure_s, safe: bool) -> float:
        """점수 = 1/(1+d). 안전 위반이면 0 (태우는 레시피를 권고한 답)."""
        if not safe:
            return 0.0
        return float(1.0 / (1.0 + self.distance(world, x_mm, y_mm, power_pct, exposure_s)))

    def to_json(self, path: Path | None = None) -> dict:
        d = asdict(self)
        if path is not None:
            Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        return d


_CACHE: dict = {}


def build(w: VirtualWorld, use_cache: bool = True) -> Oracle:
    """월드 하나의 정답을 계산한다. 손상 상태를 건드리지 않는다(predict 만 쓴다).

    [난이도를 캐시 키에 넣지 않는 이유]
    난이도는 **버퍼 색 하나만** 바꾼다(설계문서 §7). 조성 필드·광안정성·형광 세기는 전부
    시드에서 나오므로 easy/medium/hard 의 정답은 완전히 같다. 계산이 에피소드당 8~10초라
    세 난이도를 따로 구하면 그만큼이 그냥 버려진다. `_distmap` 은 읽기 전용이라 공유해도 된다.
    """
    key = (w.seed, round(w.target_area, 6))
    if use_cache and key in _CACHE:
        import copy
        orc = copy.copy(_CACHE[key])
        orc.level = w.level
        return orc
    orc = _build_uncached(w)
    if use_cache:
        _CACHE[key] = orc
    return orc


def _build_uncached(w: VirtualWorld) -> Oracle:
    # ── 1. 어느 자리가 가장 좋은가 ───────────────────────────────────────────
    cands = _candidate_spots(w)
    scored = [(best_recipe(w, x, y), (x, y)) for x, y in cands]
    scored.sort(key=lambda s: -s[0][0])
    (snr_star, p_star, t_star), (bx, by) = scored[0]

    # 최적의 BAND_LEVEL 이상을 내는 자리는 모두 '동급'으로 본다.
    best_spots = [[float(x), float(y)] for (s, _, _), (x, y) in scored
                  if s >= BAND_LEVEL * snr_star]

    # ── 2. 파워 허용폭 — 각 파워에서 '안전한 최선'을 썼을 때의 SNR 능선 ──────
    def snr_of_power(P):
        t = _max_safe_exposure(w, bx, by, P)
        return w.predict(bx, by, P, t)["snr"] if t > 0 else 0.0

    p_lo, p_hi = _band(snr_of_power, p_star, P_MIN, P_MAX, BAND_LEVEL * snr_star)

    # ── 3. 정답 노출을 확정한다 ──────────────────────────────────────────────
    # [왜 그냥 최대 안전 노출을 쓰지 않는가 — 실측으로 걸린 버그]
    # t_star 는 '안전한 최대'라 안전 경계에 정확히 붙어 있다. 이걸 round() 로 저장하면
    # 위로 반올림될 때 경계를 넘어 **정답 그대로 답해도 안전위반 0점**이 됐다
    # (시드 2·4·5 에서 실제로 발생). 게다가 정답이 절벽 위에 있는 것 자체가 나쁜 설계다 —
    # 실제 분석자라면 한계에 딱 맞추지 않고 조금 물러선 조건을 쓴다.
    # 그래서 내림으로 자르고, 확정한 값이 정말 안전한지 확인까지 한다.
    p_star = float(np.floor(p_star * 1000) / 1000)
    t_star = _max_safe_exposure(w, bx, by, p_star)
    t_star = float(np.floor(t_star * 1000) / 1000)
    while t_star > T_MIN and not w.predict(bx, by, p_star, t_star)["safe"]:
        t_star -= 0.001                      # 부동소수점 여유 — 최대 몇 번이면 끝난다
    snr_star = w.predict(bx, by, p_star, t_star)["snr"]

    # ── 4. 노출 허용폭 — 최적 파워에서 노출만 움직인다 ───────────────────────
    def snr_of_expo(t):
        r = w.predict(bx, by, p_star, t)
        return r["snr"] if r["safe"] else 0.0

    t_lo, t_hi = _band(snr_of_expo, t_star, T_MIN, T_MAX, BAND_LEVEL * snr_star)

    # ── 5. 위치 — '충분히 좋은 구역'과 그 구역의 크기 ────────────────────────
    _, ft_edge = _position_tolerance(w, bx, by, BAND_LEVEL * snr_star)
    distmap, pos_tol = _good_region(w, ft_edge)

    # 파워 허용폭은 **대역이 곧 허용폭**이라 따로 저장하지 않는다(방향별로 다르다).
    orc = None  # noqa: F841  (아래에서 만든다)
    #
    # 노출은 다르다. 위쪽 끝이 안전 경계라 t_hi == t_star 이고, 그래서 '위쪽 허용폭'이 0 이다.
    # 그 값으로 나누면 조금만 길게 답해도 거리가 폭발한다. 어차피 t_star 위는 전부 안전위반
    # (0점)이라 위쪽 폭은 쓰일 일이 없으므로, **실제로 잰 아래쪽 폭을 양쪽에 쓴다.**
    orc = Oracle(
        level=w.level, seed=w.seed,
        best_spots_mm=best_spots,
        power_pct=p_star, exposure_s=t_star, snr=round(float(snr_star), 2),
        pos_tol_mm=round(pos_tol, 5),
        power_band=[round(p_lo, 2), round(p_hi, 2)],
        exposure_tol_s=round(max(t_star - t_lo, 0.1), 3),
        exposure_band=[round(t_lo, 2), round(t_hi, 2)],
        f_tgt_at_best=round(float(w.composition_at(bx, by)[2]), 4),
        f_tgt_band_edge=round(ft_edge, 4),
        good_area_frac=round(float((w.field[..., 2] >= ft_edge).mean()), 4),
    )
    # 거리맵은 JSON 에 싣지 않는다(수백만 칸짜리 배열이다). 채점은 같은 프로세스에서
    # 도므로 객체에 붙여 둔다 — asdict() 는 선언된 필드만 보므로 직렬화에 끼지 않는다.
    orc._distmap = distmap
    return orc


def _good_region(w: VirtualWorld, ft_edge: float):
    """'충분히 좋은 구역'까지의 거리맵(필드 셀 단위)과, 그 구역의 대표 크기(mm).

    거리 변환을 한 번 계산해 두면 채점은 배열 조회 한 번이다.
    허용폭은 **구역의 굵기**로 잡는다 — 한 구역 폭만큼 벗어났으면 명백히 다른 곳을
    가리킨 것이므로 d=1(0.5점)이 되는 것이 자연스럽다.
    """
    from vbench.world import FIELD_DS
    good = w.field[..., 2] >= ft_edge
    if not good.any():                       # 있을 수 없지만, 0 나눗셈은 막는다
        return np.zeros_like(good, dtype=np.float32), 0.01
    distmap = ndimage.distance_transform_edt(~good).astype(np.float32)
    inner = ndimage.distance_transform_edt(good)
    scale_cells = 2.0 * float(inner[good].mean())        # 평균 반경 × 2 ≈ 구역 폭
    mm = float(FIELD_DS * (w.mm_x + w.mm_y) / 2.0)
    return distmap, max(scale_cells * mm, 0.002)


def _position_tolerance(w: VirtualWorld, bx, by, target_snr,
                        n_levels: int = 12, max_mm: float = 0.6):
    """최적 자리에서 SNR 이 target 아래로 떨어지기까지의 거리(mm), 그리고 그때의 f_tgt.

    [왜 광선을 따라 직접 재지 않는가]
    처음에는 사방 12방향으로 나가며 각 점에서 best_recipe() 를 불렀다 — predict() 를
    12만 번 부르는 셈이라 에피소드마다 수십 초가 든다. 그런데 달성 가능한 SNR 은
    사실상 f_tgt 의 단조함수다. 그래서 두 단계로 쪼갠다:
      (1) f_tgt 를 12 단계로 훑어 SNR(f_tgt) 곡선을 만든다  → best_recipe 12번
      (2) 능선 가장자리에 해당하는 f_tgt 를 보간으로 찾고, 거리는 **필드에서 직접** 잰다
          (numpy 연산뿐이라 사실상 공짜)
    결과는 같고 비용은 1/1000 이다.
    """
    ft = w.field[..., 2]
    f_best = float(w.composition_at(bx, by)[2])

    # (1) f_tgt 수준별 대표점 — 최적 자리에서 가까운 것을 골라 f_buf 조건을 비슷하게 유지한다.
    bcol, brow = w.stage_to_map(bx, by)
    bc, br = bcol / FIELD_DS, brow / FIELD_DS
    rr, cc = np.mgrid[0:ft.shape[0], 0:ft.shape[1]]
    d2 = (rr - br) ** 2 + (cc - bc) ** 2

    levels = np.linspace(f_best, 0.02, n_levels)
    pts, snrs = [], []
    for lv in levels:
        near = np.abs(ft - lv) < 0.02
        if not near.any():
            continue
        idx = int(np.argmin(np.where(near, d2, np.inf)))
        r, c = np.unravel_index(idx, ft.shape)
        x, y = w.map_to_stage(c * FIELD_DS, r * FIELD_DS)
        s, _, _ = best_recipe(w, x, y, n_coarse=12)
        pts.append(float(ft[r, c]))
        snrs.append(s)

    # (2) SNR 이 target 이 되는 f_tgt (단조 감소이므로 뒤집어 보간)
    pts, snrs = np.asarray(pts), np.asarray(snrs)
    o = np.argsort(snrs)
    ft_edge = float(np.interp(target_snr, snrs[o], pts[o]))

    # 필드에서 'f_tgt < ft_edge' 인 가장 가까운 셀까지의 거리
    outside = ft < ft_edge
    if not outside.any():
        return max_mm, ft_edge
    dcell = float(np.sqrt(np.min(d2[outside])))          # 필드 셀 단위
    mm = dcell * FIELD_DS * float((w.mm_x + w.mm_y) / 2.0)
    return float(min(mm, max_mm)), ft_edge


def dose_of(power_pct, exposure_s) -> float:
    """조사량(기준초) — 답의 안전성을 볼 때 채점기가 쓴다. world 와 같은 식."""
    return float((float(power_pct) / P_REF) ** ALPHA * float(exposure_s))
