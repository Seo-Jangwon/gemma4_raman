# -*- coding: utf-8 -*-
"""가상 스테이지 이미지 한 장과 그 좌표계. **이 파일이 아는 것은 '무엇이 어디 있나' 뿐이다.**

큰 이미지 한 장을 시료 전체로 놓고, 현재 스테이지 좌표가 그 이미지의 어느 부분을 보고
있는지 계산해 잘라 준다. 가상 카메라가 프레임을 만들 때 쓰고, 가상 CCD 가 '지금 어느
색 위에 있는가'를 물을 때 쓴다.

[책임 아님]
물질·스펙트럼·조사량·손상. "그것이 빛에 어떻게 반응하나"는 여기 없다. 이 파일에 물질
파라미터가 들어가기 시작하면 카메라가 스펙트럼을 알게 되고, 그 다음은 반드시 두 장치가
서로 다른 시료를 보는 상태다. 색 → 피크는 virtual_ccd._synthesize() 한 곳에만 둔다.

[왜 장치가 아니라 이 파일이 이미지를 갖는가]
실물에서는 "카메라가 지금 스테이지 위치의 시료를 본다"를 물리가 보장한다. 코드에는 그
결합이 없어서, 카메라와 CCD 가 각자 이미지를 읽으면 '화면엔 A 인데 스펙트럼은 B' 가
조용히 난다. 시료는 하나여야 하므로 시료를 파일 하나로 만들고 장치들이 그것을 참조한다.

[좌표 규약]
  · 이미지 중심 픽셀  ↔  config.STAGE_CENTER_X / STAGE_CENTER_Y
    (scene.json 의 center_stage_mm 로 덮어쓸 수 있다. 좌표값을 이 파일에 적지 않는다.)
  · 축 방향은 optics_map.SIGN_X / SIGN_Y 를 그대로 쓴다. 이미지 열이 커지면 스테이지 X 는
    작아진다(SIGN_X=-1). 이 부호가 optics_map 과 갈라지면 move_to_pixel 왕복이 깨진다.
  · 픽셀 크기는 이미지마다 다르므로 scene.json 의 um_per_px 가 정본이다. 카메라 렌즈의
    mm/px(optics_map.mm_per_px)와는 다른 값이다 — 하나는 '지도의 축척', 하나는 '카메라의 축척'.

    python -m backend.tools.hw_tools.hao_vertual.scene      자체 검사
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np

from backend.llm_config import VIRTUAL_SCENE, VIRTUAL_SCENE_ROOT
from backend.tools.hw_tools.config import STAGE_CENTER_X, STAGE_CENTER_Y
# 시야 크기와 축 부호는 optics_map 단일 출처. 여기서 다시 정의하면 실물 카메라와 가상
# 카메라가 서로 다른 시야를 보게 되고, 그건 에러 없이 좌표만 틀리는 종류다.
from backend.service.vision.optics_map import SIGN_X, SIGN_Y, fov_mm


def _default_root() -> Path:
    """씬 뿌리 기본값. llm_config 가 이 계산을 못 하는 이유는 그쪽 주석 참고."""
    from backend.service.store import DATA_ROOT
    return DATA_ROOT / "virtual_stage"


class VirtualScene:
    """이미지 한 장 + 축척 + 앵커. 상태를 바꾸지 않는다(읽기 전용)."""

    def __init__(self, image, um_per_px: float, center_mm: tuple[float, float],
                 psf_sigma_px: float = 0.0, name: str = "?"):
        self.name = name
        self.image = image                      # BGR uint8 (H, W, 3)
        self.um_per_px = float(um_per_px)
        self.center_mm = (float(center_mm[0]), float(center_mm[1]))
        self.psf_sigma_px = float(psf_sigma_px)
        self._h, self._w = image.shape[:2]

    # ── 좌표 ────────────────────────────────────────────────────────────────
    @property
    def mm_per_px(self) -> float:
        return self.um_per_px / 1000.0

    def stage_to_map_px(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        """스테이지 mm → 이미지 픽셀(u, v). 이미지 중심이 center_mm 에 앵커된다."""
        u = self._w / 2.0 + (x_mm - self.center_mm[0]) * SIGN_X / self.mm_per_px
        v = self._h / 2.0 + (y_mm - self.center_mm[1]) * SIGN_Y / self.mm_per_px
        return (u, v)

    def mm_bounds(self) -> tuple[float, float, float, float]:
        """이미지 전체가 덮는 스테이지 범위 (xmin, xmax, ymin, ymax) mm."""
        half_w = self._w / 2.0 * self.mm_per_px
        half_h = self._h / 2.0 * self.mm_per_px
        return (self.center_mm[0] - half_w, self.center_mm[0] + half_w,
                self.center_mm[1] - half_h, self.center_mm[1] + half_h)

    # ── 읽기 ────────────────────────────────────────────────────────────────
    def crop(self, cur_x: float, cur_y: float, out_w: int, out_h: int):
        """현재 스테이지 좌표의 시야를 잘라 out_w×out_h 로. 이미지 밖은 검게 채운다.

        잘라낼 영역의 물리적 크기는 optics_map.fov_mm() 이 정한다 — 실물 카메라가 보는
        시야와 같아야 하기 때문이다. 그 영역이 이미지에서 몇 픽셀인지는 이 씬의 축척
        (um_per_px)이 정한다. 두 축척을 섞지 말 것.

        이미지 밖을 검게 두는 이유: 반복(타일링)이나 가장자리 늘리기를 하면 스테이지
        한계 밖에서도 시료가 계속 보여서, 범위를 벗어난 이동이 화면상 정상으로 보인다.
        """
        w_mm, h_mm = fov_mm(out_w, out_h)
        # 이 시야가 지도에서 몇 픽셀인가. 최소 1픽셀은 보장한다(축척이 아주 거칠 때).
        crop_w = max(1, int(round(w_mm / self.mm_per_px)))
        crop_h = max(1, int(round(h_mm / self.mm_per_px)))

        cu, cv = self.stage_to_map_px(cur_x, cur_y)
        u0 = int(round(cu - crop_w / 2.0))
        v0 = int(round(cv - crop_h / 2.0))

        patch = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        # 지도와 겹치는 부분만 복사한다. 겹침이 없으면 검은 화면이 그대로 나간다.
        su0, sv0 = max(0, u0), max(0, v0)
        su1, sv1 = min(self._w, u0 + crop_w), min(self._h, v0 + crop_h)
        if su1 > su0 and sv1 > sv0:
            patch[sv0 - v0:sv1 - v0, su0 - u0:su1 - u0] = self.image[sv0:sv1, su0:su1]

        # 확대·축소 보간: 지도가 카메라보다 거칠면 확대되며 뭉개진다 — 실제로도 그렇다.
        interp = cv2.INTER_LINEAR if (crop_w < out_w or crop_h < out_h) else cv2.INTER_AREA
        view = cv2.resize(patch, (int(out_w), int(out_h)), interpolation=interp)

        if self.psf_sigma_px > 0:
            # 대물렌즈 흐림을 흉내내는 노브이지 광학 모델이 아니다. 기본은 꺼짐.
            view = cv2.GaussianBlur(view, (0, 0), self.psf_sigma_px)
        return view

    def color_at(self, x_mm: float, y_mm: float) -> tuple[int, int, int]:
        """그 좌표의 색 (B, G, R). 이미지 밖이면 (0, 0, 0).

        virtual_ccd._synthesize() 의 입력이다 — '지금 무엇 위에 레이저를 쏘고 있는가'를
        이 한 값으로 넘긴다.
        """
        u, v = self.stage_to_map_px(x_mm, y_mm)
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < self._w and 0 <= vi < self._h):
            return (0, 0, 0)
        b, g, r = self.image[vi, ui]
        # ponytail: 단일 픽셀. 레이저 스팟 크기만큼 평균이 필요해지면 반경 인자 추가.
        return (int(b), int(g), int(r))

    def __repr__(self) -> str:
        x0, x1, y0, y1 = self.mm_bounds()
        return (f"<VirtualScene {self.name} {self._w}x{self._h}px "
                f"@{self.um_per_px}um/px  x[{x0:.2f},{x1:.2f}] y[{y0:.2f},{y1:.2f}]mm>")


# ══════════════════════════════════════════════════════════════════════════════
# 로딩
# ══════════════════════════════════════════════════════════════════════════════

_SCENES: dict[str, VirtualScene] = {}
_LOCK = threading.Lock()


def scene_dir(name: str | None = None) -> Path:
    root = Path(VIRTUAL_SCENE_ROOT) if VIRTUAL_SCENE_ROOT else _default_root()
    return root / (name or VIRTUAL_SCENE)


def load_scene(name: str | None = None) -> VirtualScene:
    """씬 폴더를 읽어 VirtualScene 을 만든다. 캐시하지 않는다(get_scene 이 한다)."""
    d = scene_dir(name)
    images = sorted([p for p in d.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")])
    if not images:
        raise FileNotFoundError(
            f"가상 스테이지 이미지가 없습니다: {d}\n"
            f"이미지 한 장(map.png)과 scene.json 을 그 폴더에 넣으세요.")

    meta_path = d / "scene.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        # 조용히 넘어가지 않는다 — um_per_px 가 없으면 시야 배율이 임의값이 되고,
        # 그러면 '움직이긴 하는데 얼마나 움직였는지 모르는' 상태로 돈다.
        print(f"[WARN] {meta_path} 없음: um_per_px 기본값 1.0 으로 뜹니다. "
              f"시야 배율이 실제 시료와 맞지 않습니다.")

    img_path = d / meta["image"] if meta.get("image") else images[0]
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {img_path}")

    center = meta.get("center_stage_mm") or (STAGE_CENTER_X, STAGE_CENTER_Y)
    return VirtualScene(
        image=image,
        um_per_px=float(meta.get("um_per_px", 1.0)),
        center_mm=(float(center[0]), float(center[1])),
        psf_sigma_px=float(meta.get("psf_sigma_px", 0.0)),
        name=d.name,
    )


def get_scene(name: str | None = None) -> VirtualScene:
    """씬 싱글턴. hardware_manager.get_manager() 와 같은 규약이다.

    장치 둘(카메라·CCD)이 같은 객체를 받아야 같은 시료를 본다. 이름별로 캐시하므로
    씬을 바꿔 가며 여러 개를 띄우는 것도 가능하다(벤치가 그렇게 쓴다).
    """
    key = name or VIRTUAL_SCENE
    with _LOCK:
        if key not in _SCENES:
            _SCENES[key] = load_scene(key)
            print(f"[VSCENE] {_SCENES[key]}")
        return _SCENES[key]


# ──────────────────────────────────────────────────────────────────────────────
# 자체 점검:  python -m backend.tools.hw_tools.hao_vertual.scene
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from backend.service.vision.optics_map import pixel_to_stage
    from backend.tools.hw_tools.config import CAMERA_HEIGHT, CAMERA_WIDTH

    sc = get_scene()
    print(sc)

    # ① 앵커 — 중심 좌표에서 자른 화면의 한가운데는 지도의 한가운데여야 한다.
    u, v = sc.stage_to_map_px(*sc.center_mm)
    assert abs(u - sc.image.shape[1] / 2) < 1e-6 and abs(v - sc.image.shape[0] / 2) < 1e-6, (u, v)

    # ② 왕복 — 화면의 픽셀 p 를 도구 계층 변환(pixel_to_stage)으로 스테이지 좌표로 바꾼 뒤,
    #    그 좌표를 씬이 다시 지도 픽셀로 옮기면, 잘라낸 영역 안의 같은 자리여야 한다.
    #    두 변환이 부호나 축척에서 갈라지면 move_to_pixel 이 엉뚱한 곳으로 간다.
    cx, cy = sc.center_mm
    w_mm, h_mm = fov_mm(CAMERA_WIDTH, CAMERA_HEIGHT)
    cu, cv = sc.stage_to_map_px(cx, cy)
    for px, py in ((0, 0), (CAMERA_WIDTH - 1, 0), (CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2),
                   (0, CAMERA_HEIGHT - 1)):
        sx, sy = pixel_to_stage(px, py, cx, cy, CAMERA_WIDTH, CAMERA_HEIGHT)
        mu, mv = sc.stage_to_map_px(sx, sy)
        # 화면 픽셀 비율 == 지도 픽셀 비율
        want_u = cu + (px - CAMERA_WIDTH / 2) * (w_mm / sc.mm_per_px) / CAMERA_WIDTH
        want_v = cv + (py - CAMERA_HEIGHT / 2) * (h_mm / sc.mm_per_px) / CAMERA_HEIGHT
        assert abs(mu - want_u) < 1e-6 and abs(mv - want_v) < 1e-6, (px, py, mu, want_u, mv, want_v)

    # ③ 잘라내기 — 출력 크기가 요청대로 나오고, 지도 밖은 검다.
    view = sc.crop(cx, cy, CAMERA_WIDTH, CAMERA_HEIGHT)
    assert view.shape == (CAMERA_HEIGHT, CAMERA_WIDTH, 3), view.shape
    x0, x1, y0, y1 = sc.mm_bounds()
    far = sc.crop(x1 + 10.0, y1 + 10.0, CAMERA_WIDTH, CAMERA_HEIGHT)
    assert far.max() == 0, "지도 밖인데 내용이 보인다 — 타일링/가장자리 늘리기가 섞였다"

    # ④ 색 조회 — 지도 밖은 검정, 안은 실제 픽셀.
    assert sc.color_at(x1 + 10.0, y1 + 10.0) == (0, 0, 0)
    b, g, r = sc.color_at(cx, cy)
    assert (b, g, r) == tuple(int(c) for c in sc.image[sc.image.shape[0] // 2,
                                                       sc.image.shape[1] // 2])

    print(f"통과: 앵커 · 왕복(optics_map 과 부호·축척 일치) · 시야 {w_mm:.4f}x{h_mm:.4f}mm "
          f"= 지도 {w_mm / sc.mm_per_px:.1f}x{h_mm / sc.mm_per_px:.1f}px · 지도 밖 검정 · 색 조회")
