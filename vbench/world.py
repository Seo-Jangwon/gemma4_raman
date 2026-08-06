# -*- coding: utf-8 -*-
"""가상 시료 월드 — 이 벤치마크의 물리 전부.

에피소드 하나의 **유일한 진실**이다. 조성 필드·화면 렌더·스펙트럼 합성·손상 상태가
모두 여기 있고, 다른 모듈(oracle / drivers / score)은 이 객체를 읽기만 한다.
한 곳에 모은 이유: 같은 물리를 시뮬레이터와 채점기가 각자 구현하면 조용히 갈라지는데,
갈라지는 대상이 하필 '정답이 무엇인가'다.

설계 문서: 가상환경_벤치마크_설계.md
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, zoom

# 픽셀↔mm 변환은 실장비와 같은 단일 출처를 쓴다. 여기서 따로 정의하면
# move_to_pixel() 이 쓰는 변환과 어긋나 좌표가 조용히 틀어진다.
from backend.hw_tools import optics_map as _om
from backend.config import STAGE_CENTER_X, STAGE_CENTER_Y


# ══════════════════════════════════════════════════════════════════════════════
# 기하 — 월드와 화면
# ══════════════════════════════════════════════════════════════════════════════

MAP_W, MAP_H   = 8000, 6000     # 월드 맵 (map px)
VIEW_W, VIEW_H = 1060, 800      # 카메라 화면 — Config.ini 의 CAMERA_WIDTH/HEIGHT 와 같아야 한다

# 조성 필드는 맵 해상도로 들고 있지 않는다(8000×6000×3 = 144M 실수, 1GB 이상).
# 필요할 때만 보이는 부분을 이중선형 보간한다. 화면 안에서 보이는 결까지 담아야 하므로
# 1/4 로 둔다 — 1/8 로는 가장 잔 옥타브가 2~3 셀밖에 안 돼 뭉개진다.
FIELD_DS = 4

# 조성 노이즈의 공간 규모. **화면 크기를 기준으로 잡는다.**
#
# 군집 하나가 약 76 map px (≈31 µm) 이 되도록 맞췄다. 화면(1060 map px)에 군집이
# 14개쯤 들어오는 크기다 — 실제 시료 위의 입자 분포처럼 보이고, 에이전트가 한 화면 안에서
# '여기는 진하고 저기는 옅다'를 바로 비교할 수 있다.
#
# [두 번 고쳤다]
#   base=6   : 가장 큰 덩어리가 1300 map px → 화면이 통째로 단색이 됐다.
#   base=14  : 571 map px → 화면에 덩어리 2개. 구조는 보이지만 '대륙' 같아서
#              한 화면 안에서 비교할 것이 거의 없었다.
#   base=105 : 지금 값.
# 옥타브를 6→4 로 줄인 것은 필드 해상도 때문이다. base=105 에 옥타브 6 이면 가장 잔 결이
# 0.6 필드셀이라 표본화 한계 아래로 내려가 에일리어싱만 남는다.
FBM_BASE, FBM_OCTAVES = 105, 4

# 손상 격자 셀 크기(map px). 레이저 스팟(σ=4 map px)보다 작아야 가우시안 확산이
# 실제로 여러 셀에 걸친다 — 셀이 스팟보다 크면 확산이 한 셀에 갇혀 무의미해지고,
# '좌표를 조금 옮겨 손상을 회피'하는 편법을 막지 못한다.
DAMAGE_CELL   = 4
SPOT_SIGMA_PX = 4.0


# ══════════════════════════════════════════════════════════════════════════════
# 스펙트럼
# ══════════════════════════════════════════════════════════════════════════════

AXIS_MIN, AXIS_MAX, N_BINS = 150.0, 2300.0, 1024

# (중심 cm-1, 상대높이, FWHM cm-1)
BANDS = {
    "tgt": [(1001.4, 1.00, 8.0), (1031.0, 0.35, 8.0),
            (1602.0, 0.62, 12.0), (621.0, 0.20, 10.0)],   # 폴리스티렌
    "bg":  [(520.7, 0.80, 5.0)],                          # Si 기판 — §4 함정
    "buf": [(1640.0, 0.05, 90.0)],                        # 버퍼 — 사실상 배경
}
TARGET_CM1 = 1001.4                     # 채점 대상 밴드
NOISE_WIN  = (1750.0, 1900.0)           # SNR 의 잡음 구간
SIGNAL_HALFWIDTH = 10.0                 # 피크 높이를 찾을 때 볼 반폭
# 피크 높이의 기준선을 잡는 두 구간. **어떤 밴드도 들어 있지 않은 곳**이어야 한다.
# 처음에는 '피크 좌우 12~36 cm-1 의 중앙값'을 썼는데, 그 창이 PS 의 1031 밴드를 통째로
# 품어서 기준선이 부풀고 피크 높이가 실제의 1/4 로 나왔다.
BASELINE_WINS = ((955.0, 980.0), (1055.0, 1085.0))

G_RAMAN   = 90.0                        # counts /(%·s), 상대높이 1.0 기준
FLUOR     = {"tgt": 6000.0, "buf": 3000.0, "bg": 800.0}   # counts/s (형광 정점)
FLUOR_C, FLUOR_W = 1450.0, 700.0        # 형광 배경 가우시안
DARK      = 60.0                        # counts/s — 노출에만 비례
READ_SIGMA = 12.0                       # counts — 노출·파워 무관
FULL_WELL = 65535.0
# '안전한 조건'을 고를 때 남겨 두는 포화 여유.
# 정답을 포화 경계에 딱 붙여 놓으면, 그 조건으로 실제 측정했을 때 산탄잡음이 절반쯤은
# 상한을 넘겨 버린다(실측). 정답대로 했는데 포화되는 정답은 정답이 아니다.
SAT_HEADROOM = 0.93


# ══════════════════════════════════════════════════════════════════════════════
# 광손상
# ══════════════════════════════════════════════════════════════════════════════

P_REF   = 20.0      # 기준 파워(%) — 20%로 1초 = 1.0 기준초
# 손상의 파워 지수. **반드시 > 2** — 안 그러면 파워를 올릴수록 SNR 이 좋아져서
# 벤치마크가 가르치려는 것과 반대 행동에 상을 준다(설계문서 §5.1).
#
# [왜 2.6 이 아니라 3.5 인가 — 오라클을 돌려 보고 올렸다]
# 손상 한계를 지키는 능선 위에서 SNR ∝ P^(1-α/2) 이다. α=2.6 이면 지수가 -0.30 이라
# 능선이 거의 평평해서, SNR 이 90% 이상인 파워 대역이 [11.2, 25.1] 로 나왔다 —
# 파워를 사실상 안 재는 셈이라 '세 가지를 찾는다'는 과제가 두 가지로 줄어든다.
# α=3.5 면 지수가 -0.75 로 대역이 [12.6, 19.1] 이 되어 파워가 실제로 변별력을 갖는다.
# 열손상 임계는 조사강도에 가파르게 의존해서(열폭주·다광자 흡수) 실측 지수가 대략 2~4 에
# 걸치므로, 이 값은 물리적으로도 타당한 범위 안이다.
ALPHA   = 3.5
E0      = 21.0      # 기준초. f_tgt=0 일 때의 임계값 (시료마다 아래 편차가 곱해진다)
KAPPA   = 3.2       # 타겟이 순수할수록 잘 탄다

# ── 시료마다 달라지는 것 ──────────────────────────────────────────────────────
# [왜 필요한가 — 오라클을 돌려 보고 알았다]
# 이것들이 고정이면 최적 자리의 조성이 늘 (0,0,1) 이라, **정답이 시드·난이도와 무관하게
# 매번 똑같이 나온다**(실측: 9개 월드가 전부 14.56% / 8.95s / SNR 56.4). 그러면 에이전트는
# 한 번 알아낸 숫자를 외우기만 하면 되고, 우리가 재려던 '적응'이 아니라 '암기'를 재게 된다.
# CoALA 의 장기기억이 이기는 것은 당연해지고 그 결과는 아무 뜻이 없다.
#
# 두 값을 시료마다 흔든다. 둘 다 실제로 배치마다 변하는 양이고, 각각 **다른 제약**을
# 움직이므로 최적점이 파워·노출 두 방향 모두로 이동한다:
#   · 광안정성(E0)  → 손상 한계    → 쓸 수 있는 파워가 달라진다
#   · 형광 세기     → 포화 한계    → 쓸 수 있는 노출이 달라진다
# 이제 에이전트는 에피소드마다 한계를 **직접 재 봐야** 한다. 기억이 도움이 된다면 그건
# 정답 숫자가 아니라 '어떻게 알아내는가'(저파워로 탐색 → 배경 상승 관찰 → 후퇴)여야 하고,
# 그것이야말로 이 벤치마크가 재려는 것이다.
E0_SIGMA    = 0.30      # 로그정규 편차 — 광안정성 (±35% 정도)
FLUOR_SIGMA = 0.25      # 로그정규 편차 — 형광 배경 (±28% 정도)
W_DECAY = 0.30      # 임계 초과 후 붕괴 속도
C_DMG   = 9000.0    # counts/s — 완전 탄화 시 형광 혹의 정점

# 예비 경고의 시작점과 크기.
#
# [왜 0.60/0.12 에서 0.50/0.30 으로 올렸나 — 실측]
# 처음 값에서는 u=0.70 일 때 배경이 3% 밖에 안 올라갔다. 사람이 그래프로 보면 보이지만,
# 에이전트는 1024개 숫자로 받는다 — 3% 는 사실상 관측 불가다. 그러면 경고가 u≥0.9 에서야
# 읽히는데 그건 붕괴 직전이라, '넘기 전에 물러설 기회'라는 목적을 못 이룬다.
# 지금 값이면 u=0.70 에서 배경이 16%, u=0.85 에서 36% 오른다 — 두 측정을 비교하면 확실히 읽힌다.
# 완전 탄화(배경 100%↑ + 피크 소실)와는 여전히 뚜렷이 구분된다.
WARN_AT   = 0.50
WARN_GAIN = 0.30
CARBON_BANDS = [(1350.0, 180.0), (1580.0, 120.0)]   # 탄소 D/G
CARBON_FRAC  = 0.25


# ══════════════════════════════════════════════════════════════════════════════
# 색 — 난이도는 '버퍼 색이 타겟 색에 얼마나 가까운가' 하나로만 조절한다
# ══════════════════════════════════════════════════════════════════════════════

C_BG       = np.array([35.0, 40.0, 55.0])      # 어두운 청회색 — 기판
C_BUF_BASE = np.array([150.0, 205.0, 215.0])   # 연한 시안 — 버퍼
C_TGT      = np.array([225.0, 170.0, 55.0])    # 호박색 — 타겟

# 난이도 = 버퍼 색을 타겟 색 쪽으로 끌어당기는 비율. 이것 **하나만** 바뀐다.
# 키를 영어로 두는 이유: 결과 JSON·그림 라벨·파일 이름에 그대로 실리는데, 이 프로젝트는
# cp949 콘솔에서 돌고 matplotlib 기본 폰트에 한글 글리프가 없다(실제로 깨졌다).
LEVELS = {"easy": 0.00, "medium": 0.45, "hard": 0.78}

VIGNETTE  = 0.18    # 화면 가장자리 밝기 감소 — 픽셀 단위 색 역산을 막는다
CAM_SIGMA = 3.0     # 카메라 노이즈 (counts)


# 파수축은 **한 번만** 만들어 공유한다. 매번 새로 만들면 _baseline_fit 의 캐시 키(id)가
# 매번 달라져 캐시가 통째로 무효가 되고, 캐시 dict 만 무한히 자란다.
_AXIS = np.linspace(AXIS_MIN, AXIS_MAX, N_BINS)


def axis() -> np.ndarray:
    """공통 파수축 (cm-1), 1024점. 읽기 전용으로 쓸 것."""
    return _AXIS


def _lorentz(x, c, h, fwhm):
    g = fwhm / 2.0
    return h * g ** 2 / ((x - c) ** 2 + g ** 2)


def _fbm(rng, w, h, octaves=4, base=6) -> np.ndarray:
    """fBm 노이즈 [0,1]. 외부 라이브러리 없이 '거친 난수 격자 + 3차 보간'을 옥타브로 합친다."""
    out = np.zeros((h, w), dtype=np.float64)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gw = max(2, int(base * 2 ** o))
        gh = max(2, int(round(gw * h / w)))
        g = rng.random((gh, gw))
        # 낮은 옥타브(큰 무늬)만 3차 보간을 쓴다. 잔 옥타브는 진폭이 1/4 이하라
        # 선형 보간과 눈으로 구분되지 않는데 3차는 몇 배 비싸다.
        up = zoom(g, (h / gh, w / gw), order=(3 if o < 2 else 1), mode="nearest")
        # zoom 은 반올림 때문에 1픽셀 어긋날 수 있다 — 자르거나 가장자리 복제로 맞춘다.
        if up.shape[0] < h or up.shape[1] < w:
            up = np.pad(up, ((0, max(0, h - up.shape[0])), (0, max(0, w - up.shape[1]))), mode="edge")
        out += amp * up[:h, :w]
        total += amp
        amp *= 0.5
    return _rank_uniform(out / total)


def _rank_uniform(a: np.ndarray) -> np.ndarray:
    """값을 순위로 바꿔 [0,1] 균등분포로 만든다.

    [왜 min-max 정규화가 아닌가]
    fBm 은 옥타브의 합이라 종 모양으로 뭉쳐 있다. min-max 로 펴면 높은 문턱값 위의
    면적이 시드마다 들쭉날쭉하고, 실측으로 θ=0.30 에서도 목표 면적이 5.5% 밖에 안 나왔다
    (설계 목표는 10~15%). 순위변환을 쓰면 분포가 정확히 균등해져서
    **타겟 면적을 시드와 무관하게 정확히 지정**할 수 있다 — 난이도를 재현 가능하게 만드는 조건이다.
    """
    flat = a.ravel().astype(np.float32, copy=False)
    order = np.argsort(flat)   # 연속 노이즈라 동점이 사실상 없다 — 안정정렬이 필요 없다
    ranks = np.empty(flat.size, dtype=np.float32)
    ranks[order] = np.arange(flat.size, dtype=np.float32)
    return (ranks / max(flat.size - 1, 1)).reshape(a.shape)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class VirtualWorld:
    """가상 시료 하나. 시드가 같으면 완전히 같은 월드가 나온다.

    좌표계가 셋이라 헷갈리기 쉽다 — 이 클래스 안에서는 이렇게 부른다:
      · map (col, row)  월드 맵 픽셀. 정수/실수 모두 가능
      · view (px, py)   화면 픽셀 0..VIEW_W/H
      · stage (x, y)    실제 스테이지 mm — 도구 계층이 쓰는 좌표

    stage 와 map 은 **거울 관계**다(optics_map.SIGN_X = -1). 반면 view 와 map 은
    같은 방향이라 화면 렌더가 단순 crop 이 된다 — 둘 다 같은 SIGN 을 통과하기 때문이다.
    """

    def __init__(self, seed: int, level: str = "medium", target_area: float = 0.12):
        """target_area — `f_tgt > 0.5` 인 지점이 맵에서 차지하는 비율.
        순위변환 덕에 이 값이 시드와 무관하게 **정확히** 지켜진다."""
        if level not in LEVELS:
            raise ValueError(f"level 은 {list(LEVELS)} 중 하나여야 합니다: {level!r}")
        self.seed = int(seed)
        self.level = level
        rng = np.random.default_rng(seed)

        self.mm_x, self.mm_y = _om.mm_per_px(VIEW_W, VIEW_H)     # map px 1개의 mm 크기
        self.center_x, self.center_y = float(STAGE_CENTER_X), float(STAGE_CENTER_Y)

        # ── 조성 필드 (거친 해상도로 보관) ────────────────────────────────────
        fw, fh = MAP_W // FIELD_DS, MAP_H // FIELD_DS
        n1 = _fbm(rng, fw, fh, octaves=FBM_OCTAVES, base=FBM_BASE)
        n2 = _fbm(rng, fw, fh, octaves=FBM_OCTAVES, base=max(2, FBM_BASE - 2))

        # n1 이 균등분포이므로 P(f_tgt > 0.5) = (1-θ)·(1 - 0.5^(1/γ)) 가 정확히 성립한다.
        # 원하는 면적에서 θ 를 역산한다 — 손으로 맞출 값이 아니다.
        gamma = 1.8
        span = 1.0 - 0.5 ** (1.0 / gamma)
        theta = float(np.clip(1.0 - target_area / span, 0.0, 0.95))
        self.target_area, self.theta = float(target_area), theta

        f_tgt = np.clip((n1 - theta) / (1.0 - theta), 0.0, 1.0) ** gamma
        f_buf = (1.0 - f_tgt) * _smoothstep(n2)
        f_bg = 1.0 - f_tgt - f_buf
        self.field = np.stack([f_bg, f_buf, f_tgt], axis=-1).astype(np.float32)   # (fh,fw,3)

        # ── 색 (난이도) ──────────────────────────────────────────────────────
        mix = LEVELS[level]
        self.c_buf = C_BUF_BASE + mix * (C_TGT - C_BUF_BASE)
        self.colors = np.stack([C_BG, self.c_buf, C_TGT], axis=0)                # (3,3) RGB

        # ── 이 시료의 개성 (에피소드마다 다르다 — E0_SIGMA 주석 참고) ────────
        self.e0 = float(E0 * np.exp(rng.normal(0.0, E0_SIGMA)))
        self.fluor_scale = float(np.exp(rng.normal(0.0, FLUOR_SIGMA)))

        # ── 손상 상태 ────────────────────────────────────────────────────────
        dw, dh = MAP_W // DAMAGE_CELL, MAP_H // DAMAGE_CELL
        self.dose = np.zeros((dh, dw), dtype=np.float32)          # 기준초
        # 손상 격자와 조성 필드가 같은 해상도면 리샘플링할 것이 없다.
        # (지금 DAMAGE_CELL == FIELD_DS == 4 라 정확히 그 경우다. 그냥 부르면 300만 점을
        #  이중선형 보간하느라 월드 생성이 몇 초 더 걸린다 — 실측으로 확인.)
        ft_d = (self.field[..., 2] if (dh, dw) == self.field.shape[:2]
                else self._sample_field_grid(dw, dh)[..., 2])
        self.e_th = (self.e0 / (1.0 + KAPPA * ft_d)).astype(np.float32)

        # 손상 확산 커널 (셀 단위)
        s = SPOT_SIGMA_PX / DAMAGE_CELL
        r = max(1, int(np.ceil(3 * s)))
        gy, gx = np.mgrid[-r:r + 1, -r:r + 1]
        self.kernel = np.exp(-(gx ** 2 + gy ** 2) / (2 * s ** 2)).astype(np.float32)
        self.kernel_r = r

        self.axis = axis()
        self._band_cache = {k: sum(_lorentz(self.axis, c, h, w) for c, h, w in v)
                            for k, v in BANDS.items()}
        self._fluor_shape = np.exp(-((self.axis - FLUOR_C) / FLUOR_W) ** 2)
        self._carbon_shape = sum(_lorentz(self.axis, c, 1.0, w) for c, w in CARBON_BANDS)

        self.history: list[dict] = []

    # ── 좌표 변환 ────────────────────────────────────────────────────────────

    def stage_to_map(self, x_mm, y_mm):
        """스테이지 mm → 맵 픽셀. SIGN_X=-1 때문에 X 는 뒤집힌다."""
        col = MAP_W / 2.0 - (float(x_mm) - self.center_x) / self.mm_x
        row = MAP_H / 2.0 + (float(y_mm) - self.center_y) / self.mm_y
        return col, row

    def map_to_stage(self, col, row):
        x = self.center_x - (float(col) - MAP_W / 2.0) * self.mm_x
        y = self.center_y + (float(row) - MAP_H / 2.0) * self.mm_y
        return x, y

    def bounds_mm(self):
        """월드가 덮는 스테이지 범위 [xmin, xmax, ymin, ymax] (mm)."""
        x0, y0 = self.map_to_stage(0, 0)
        x1, y1 = self.map_to_stage(MAP_W, MAP_H)
        return [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]

    # ── 조성 조회 ────────────────────────────────────────────────────────────

    def _sample_field(self, cols, rows):
        """맵 픽셀 좌표 배열에서 조성 (…,3) 을 이중선형 보간해 뽑는다."""
        fc = np.asarray(cols, dtype=np.float64) / FIELD_DS
        fr = np.asarray(rows, dtype=np.float64) / FIELD_DS
        out = [map_coordinates(self.field[..., k], [fr, fc], order=1, mode="nearest")
               for k in range(3)]
        return np.stack(out, axis=-1)

    def _sample_field_grid(self, w, h):
        """필드를 w×h 격자로 리샘플 — 손상 격자·전체 조감도용."""
        cols = (np.arange(w) + 0.5) * (MAP_W / w)
        rows = (np.arange(h) + 0.5) * (MAP_H / h)
        cc, rr = np.meshgrid(cols, rows)
        return self._sample_field(cc, rr)

    def composition_at(self, x_mm, y_mm) -> np.ndarray:
        """스테이지 좌표에서의 (f_bg, f_buf, f_tgt)."""
        col, row = self.stage_to_map(x_mm, y_mm)
        return self._sample_field(np.array([col]), np.array([row]))[0]

    # ── 화면 렌더 ────────────────────────────────────────────────────────────

    def render(self, x_mm, y_mm, width=VIEW_W, height=VIEW_H) -> np.ndarray:
        """현 스테이지 위치에서 본 화면. BGR uint8 (cv2 규약)."""
        col_c, row_c = self.stage_to_map(x_mm, y_mm)
        cols = col_c - width / 2.0 + np.arange(width)
        rows = row_c - height / 2.0 + np.arange(height)
        cc, rr = np.meshgrid(cols, rows)
        f = self._sample_field(cc, rr)                       # (h,w,3) 조성
        rgb = f @ self.colors                                # (h,w,3) 색

        # 비네팅 — 화면 중심 기준이므로 같은 지점도 화면 위치에 따라 색이 조금 달라진다.
        yy, xx = np.mgrid[0:height, 0:width]
        r2 = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
        rgb *= (1.0 - VIGNETTE * (r2 / 2.0))[..., None]

        rng = np.random.default_rng((self.seed * 1000003 + int(col_c) * 31 + int(row_c)) % (2**32))
        rgb += rng.normal(0.0, CAM_SIGMA, rgb.shape)
        return np.clip(rgb, 0, 255).astype(np.uint8)[..., ::-1].copy()   # RGB→BGR

    def overview(self, width=800) -> np.ndarray:
        """월드 전체 조감도 (진단·문서용. 에이전트에게는 절대 주지 않는다)."""
        height = int(width * MAP_H / MAP_W)
        f = self._sample_field_grid(width, height)
        rgb = np.clip(f @ self.colors, 0, 255).astype(np.uint8)
        return rgb[..., ::-1].copy()

    # ── 손상 ─────────────────────────────────────────────────────────────────

    def _fluor_rate(self, f_bg, f_buf, f_tgt) -> float:
        """형광 배경의 정점 세기(counts/s). 시료마다 다른 fluor_scale 이 곱해진다."""
        return float(self.fluor_scale *
                     (FLUOR["tgt"] * f_tgt + FLUOR["buf"] * f_buf + FLUOR["bg"] * f_bg))

    def _dose_of(self, power_pct, exposure_s) -> float:
        """이 조사의 누적량(기준초). 파워에 초선형인 것이 이 벤치마크의 핵심이다."""
        return float((power_pct / P_REF) ** ALPHA * exposure_s)

    def _cells(self, col, row):
        """스팟 주변 손상 셀의 (행슬라이스, 열슬라이스, 커널)."""
        dh, dw = self.dose.shape
        c0 = int(round(col / DAMAGE_CELL))
        r0 = int(round(row / DAMAGE_CELL))
        r = self.kernel_r
        rs, re = max(0, r0 - r), min(dh, r0 + r + 1)
        cs, ce = max(0, c0 - r), min(dw, c0 + r + 1)
        if rs >= re or cs >= ce:
            return None
        k = self.kernel[rs - (r0 - r):re - (r0 - r), cs - (c0 - r):ce - (c0 - r)]
        return slice(rs, re), slice(cs, ce), k

    def _center_cell(self, col, row):
        """스팟 중심이 놓인 손상 셀 (행, 열). 범위를 벗어나면 None."""
        dh, dw = self.dose.shape
        c = int(round(col / DAMAGE_CELL))
        r = int(round(row / DAMAGE_CELL))
        return (r, c) if 0 <= r < dh and 0 <= c < dw else None

    # [왜 커널 가중평균이 아니라 중심 셀인가 — 2026-08-05 수정]
    # 처음에는 u 와 A 를 스팟 커널로 가중평균했다. 그런데 커널 가장자리는 조사량을 적게
    # 받으므로 평균이 중심값의 절반쯤으로 희석된다 — 중심이 u=1.06(이미 탄 상태)인데
    # 보고되는 값은 0.53 이었다. 손상 임계값이 사실상 2배로 물러져 §5.3 의 최적점 구조가
    # 통째로 무너졌다(실측으로 확인). 수집 광학은 조사 스팟과 같은 자리를 보는 공초점이라
    # **중심값이 관측에 해당한다.** 커널은 '이웃 셀에도 조사량이 번진다'(좌표를 조금 옮겨
    # 손상을 피하는 편법 차단)는 목적에만 쓴다.

    def survival_at(self, x_mm, y_mm) -> float:
        """그 지점 물질의 현재 생존율 A ∈ [0,1]. 1.0 이면 무손상."""
        col, row = self.stage_to_map(x_mm, y_mm)
        rc = self._center_cell(col, row)
        if rc is None:
            return 1.0
        u = float(self.dose[rc] / self.e_th[rc])
        return float(np.exp(-max(0.0, u - 1.0) / W_DECAY))

    def _apply_dose(self, col, row, dose):
        sel = self._cells(col, row)
        if sel is None:
            return
        rs, cs, k = sel
        self.dose[rs, cs] += (dose * k).astype(np.float32)

    def _u_at(self, col, row, extra=0.0) -> float:
        """스팟 중심의 손상 진행도 u = D/E. extra 는 아직 반영 전인 조사량."""
        rc = self._center_cell(col, row)
        if rc is None:
            return 0.0
        return float((self.dose[rc] + extra) / self.e_th[rc])

    # ── 측정 ─────────────────────────────────────────────────────────────────

    def measure(self, x_mm, y_mm, power_pct: float, exposure_s: float,
                accumulations: int = 1) -> dict:
        """레이저를 쏘고 1024점 스펙트럼을 얻는다. **손상 상태를 갱신한다(불가역).**

        [조사 중에 타는 경우를 어떻게 다루는가]
        손상은 노출이 진행되는 동안 일어나므로, 임계를 넘긴 바로 그 측정은 '멀쩡한 앞부분'과
        '탄 뒷부분'이 섞인다. 그래서 생존율을 **조사 전 누적량이 아니라 조사 중간값**
        (D_before + ΔD/2)으로 평가한다. 조사 전으로 잡으면 죽인 측정이 멀쩡하게 나와
        에이전트가 원인을 다음 측정에서야 보게 되고, 조사 후로 잡으면 한 번에 전멸한 것처럼
        보여 '경고 → 후퇴' 학습이 불가능해진다. 중간값이 둘 다 피한다.
        """
        power_pct = float(power_pct)
        exposure_s = float(exposure_s)
        n = max(1, int(accumulations))
        col, row = self.stage_to_map(x_mm, y_mm)
        f_bg, f_buf, f_tgt = self._sample_field(np.array([col]), np.array([row]))[0]

        d_shot = self._dose_of(power_pct, exposure_s) * n
        u_before = self._u_at(col, row)
        u_mid = self._u_at(col, row, extra=0.5 * d_shot)
        self._apply_dose(col, row, d_shot)                     # 불가역
        u_after = self._u_at(col, row)

        a = float(np.exp(-max(0.0, u_mid - 1.0) / W_DECAY))    # 생존율

        # ── 신호 ─────────────────────────────────────────────────────────────
        raman = G_RAMAN * power_pct * (
            f_tgt * a * self._band_cache["tgt"]
            + f_bg * self._band_cache["bg"]                    # 기판은 타지 않는다
            + f_buf * a * self._band_cache["buf"])

        fluor_rate = self._fluor_rate(f_bg, f_buf, f_tgt)
        bg_spec = fluor_rate * self._fluor_shape

        # 예비 경고 — 피크는 멀쩡한데 배경만 살짝 부푼다.
        #
        # [왜 u_mid 가 아니라 u_after 인가 — 실측으로 걸린 문제]
        # 붕괴(A)는 노출 전체의 평균이라 u_mid 가 맞다. 하지만 경고를 u_mid 로 판정하면
        # 새 자리에 한 방 쏘는 경우 u_mid = u_after/2 이므로, u_after 가 1.2 를 넘어야
        # 비로소 경고가 뜬다 — **이미 태운 뒤에 뜨는 경고**라 아무 쓸모가 없다.
        # 실측: (P=50, t=0.5) 는 u=1.06 으로 임계를 넘겼는데 경고도 손상도 안 보였다.
        # 열적 전조는 노출이 끝나는 시점의 상태이므로 u_after 로 판정한다. 이제
        # u_after=0.7 인 측정은 피크가 멀쩡한 채 배경만 부풀어, 넘기 **전에** 물러설 수 있다.
        if u_after > WARN_AT:
            w = _smoothstep((min(u_after, 1.0) - WARN_AT) / (1.0 - WARN_AT))
            bg_spec = bg_spec + WARN_GAIN * C_DMG * float(w) * self._fluor_shape
        # 탄화 — 넓은 형광 혹 + 탄소 D/G 밴드
        if a < 1.0:
            burn = C_DMG * (1.0 - a)
            bg_spec = bg_spec + burn * self._fluor_shape + CARBON_FRAC * burn * self._carbon_shape

        rate = raman + bg_spec + DARK
        ideal = rate * exposure_s * n

        rng = np.random.default_rng(
            (self.seed * 7919 + len(self.history) * 104729) % (2 ** 32))
        noisy = ideal + rng.normal(0.0, np.sqrt(np.maximum(ideal, 1.0))) \
                      + rng.normal(0.0, READ_SIGMA * np.sqrt(n), ideal.shape)

        sat_level = FULL_WELL * n
        saturated = bool(np.any(noisy >= sat_level))
        data = np.clip(noisy, 0.0, sat_level)

        rec = {
            "stage_x": float(x_mm), "stage_y": float(y_mm),
            "power_pct": power_pct, "exposure_s": exposure_s, "accumulations": n,
            "f_tgt": float(f_tgt), "dose_shot": d_shot,
            "u_before": float(u_before), "u": float(u_after), "survival": a,
            "warned": bool(u_after > WARN_AT),
            "saturated": saturated,
            "snr": snr_of(data, self.axis),
            "peak": peak_height(data, self.axis),
        }
        self.history.append(rec)
        return {"data": data, "axis": self.axis, "saturated": saturated,
                "survival": a, "u": u_after, "record": rec}

    def predict(self, x_mm, y_mm, power_pct: float, exposure_s: float,
                fresh: bool = True) -> dict:
        """**잡음 없이** 기대 SNR·포화·손상을 계산한다 — oracle 이 정답을 구할 때 쓴다.

        measure() 와 같은 밴드·형광·손상 상수를 쓰므로 시뮬레이터와 채점 기준이 갈라질 수 없다.
        `fresh=True` 는 '아직 조사한 적 없는 시료라면' 이라는 뜻이다(정답은 무손상 기준).

        측정하지 않으므로 **손상 상태를 바꾸지 않는다.**
        """
        power_pct, exposure_s = float(power_pct), float(exposure_s)
        col, row = self.stage_to_map(x_mm, y_mm)
        f_bg, f_buf, f_tgt = self._sample_field(np.array([col]), np.array([row]))[0]

        d_shot = self._dose_of(power_pct, exposure_s)
        u_before = 0.0 if fresh else self._u_at(col, row)
        u_after = u_before + d_shot / max(float(self.e_th[self._center_cell(col, row)]), 1e-9) \
            if self._center_cell(col, row) is not None else 0.0
        a = float(np.exp(-max(0.0, (u_before + u_after) / 2.0 - 1.0) / W_DECAY))

        raman = G_RAMAN * power_pct * (
            f_tgt * a * self._band_cache["tgt"]
            + f_bg * self._band_cache["bg"]
            + f_buf * a * self._band_cache["buf"])
        fluor_rate = self._fluor_rate(f_bg, f_buf, f_tgt)
        bg_spec = fluor_rate * self._fluor_shape
        if u_after > WARN_AT:
            w = _smoothstep((min(u_after, 1.0) - WARN_AT) / (1.0 - WARN_AT))
            bg_spec = bg_spec + WARN_GAIN * C_DMG * float(w) * self._fluor_shape
        if a < 1.0:
            burn = C_DMG * (1.0 - a)
            bg_spec = bg_spec + burn * self._fluor_shape + CARBON_FRAC * burn * self._carbon_shape

        ideal = (raman + bg_spec + DARK) * exposure_s
        saturated = bool(np.any(ideal >= FULL_WELL))
        clipped = np.clip(ideal, 0.0, FULL_WELL)

        # 잡음 없는 배열의 표준편차는 0 이므로, 기대 잡음을 해석적으로 구한다:
        # 산탄잡음(√카운트)과 판독잡음의 제곱합.
        nwin = (self.axis >= NOISE_WIN[0]) & (self.axis <= NOISE_WIN[1])
        sigma = float(np.sqrt(np.mean(clipped[nwin]) + READ_SIGMA ** 2))
        peak = peak_height(clipped, self.axis)
        headroom_ok = bool(np.max(ideal) <= FULL_WELL * SAT_HEADROOM)
        return {"snr": peak / sigma if sigma > 0 else 0.0, "peak": peak, "sigma": sigma,
                "saturated": saturated, "u": float(u_after), "survival": a,
                "dose": d_shot, "f_tgt": float(f_tgt),
                "headroom_ok": headroom_ok,
                # '안전하다' = 포화 여유가 있고, 손상 임계를 넘지 않는다.
                "safe": headroom_ok and u_after <= 1.0}

    # ── 진단용 요약 ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        ft = self.field[..., 2]
        burned = np.exp(-np.maximum(0.0, self.dose / self.e_th - 1.0) / W_DECAY) < 0.5
        ft_d = self._sample_field_grid(*self.dose.shape[::-1])[..., 2]
        rich = ft_d > 0.6
        return {
            "level": self.level, "seed": self.seed,
            "f_tgt_max": float(ft.max()),
            "target_area_frac": float((ft > 0.5).mean()),
            "measurements": len(self.history),
            "burned_cells": int(burned.sum()),
            "burned_target_frac": float((burned & rich).sum() / max(rich.sum(), 1)),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 스펙트럼 판독 — 시뮬레이터와 채점기가 **같은 함수**를 쓴다
# ══════════════════════════════════════════════════════════════════════════════

_FIT_CACHE: dict = {}


def _baseline_fit(ax, center):
    """기준선 직선을 뽑는 선형연산자를 미리 만들어 둔다.

    [왜 캐시하는가 — 실측]
    기준선 창도 파수축도 고정인데 매 호출마다 np.polyfit 을 돌고 있었다. oracle.build 는
    predict 를 3만 번 넘게 부르고 그 하나하나가 peak_height 를 부르므로, 이 한 줄이
    정답 계산을 32초로 만들었다. 설계행렬이 상수이니 유사역행렬을 한 번만 구하면 된다.
    """
    key = (id(ax), float(center))
    got = _FIT_CACHE.get(key)
    if got is None:
        bm = np.zeros_like(ax, dtype=bool)
        for lo, hi in BASELINE_WINS:
            bm |= (ax >= lo) & (ax <= hi)
        sig = np.abs(ax - center) <= SIGNAL_HALFWIDTH
        if bm.sum() >= 2:
            A = np.stack([ax[bm], np.ones(int(bm.sum()))], axis=1)
            # 기준선 값 = [center, 1] @ pinv(A) @ data[bm]  → 벡터 하나로 접는다
            row = np.array([float(center), 1.0]) @ np.linalg.pinv(A)
        else:
            row = None
        got = _FIT_CACHE[key] = (bm, sig, row)
    return got


def peak_height(data, ax=None, center=TARGET_CM1) -> float:
    """타겟 밴드의 배경 위 높이.

    기준선은 밴드가 없는 두 구간(BASELINE_WINS)을 지나는 **직선**으로 잡는다. 형광 배경이
    이 근방에서 기울어져 있어서, 한쪽 중앙값만 쓰면 기울기만큼 높이가 치우친다.
    """
    ax = axis() if ax is None else ax
    data = np.asarray(data, dtype=float)
    bm, sig, row = _baseline_fit(ax, center)
    if not sig.any():
        return 0.0
    base = float(row @ data[bm]) if row is not None else float(np.median(data))
    return float(np.max(data[sig]) - base)


def noise_sigma(data, ax=None) -> float:
    """잡음 구간의 **추세 제거 후** 표준편차.

    [왜 추세를 빼야 하는가 — 실측으로 걸린 문제]
    1750~1900 cm-1 에서 형광 배경이 4917 → 3907 counts 로 흘러내린다. 그냥 표준편차를
    재면 이 기울기(σ≈290)가 실제 산탄잡음(σ≈67)을 4배 이상 덮어써서, SNR 이 5 로 나온다
    (참값 24). 그러면 노출을 늘려도 SNR 이 안 오르는 것처럼 보여 최적화 문제 자체가 사라진다.
    1차 추세를 빼면 남는 것이 진짜 잡음이다 — 실제 분광 분석에서 하는 일과 같다.
    """
    ax = axis() if ax is None else ax
    data = np.asarray(data, dtype=float)
    n = (ax >= NOISE_WIN[0]) & (ax <= NOISE_WIN[1])
    if n.sum() < 3:
        return 0.0
    k, b = np.polyfit(ax[n], data[n], 1)
    return float(np.std(data[n] - (k * ax[n] + b), ddof=1))


def snr_of(data, ax=None) -> float:
    """SNR = 타겟 피크 높이 / 잡음. 시뮬레이터와 채점기의 단일 정의."""
    sd = noise_sigma(data, ax)
    return peak_height(data, ax) / sd if sd > 0 else 0.0
