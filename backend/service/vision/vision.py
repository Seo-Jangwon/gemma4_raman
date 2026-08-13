# -*- coding: utf-8 -*-
"""
[역할] 카메라 프레임 전처리와 가이드빔 스팟 면적 측정의 **단일 출처**.

[왜 이 모듈이 생겼는가 — 2026-07-30]
"프레임을 받아 8bit BGR 로 만든다"가 네 곳에 복사돼 있었고, 미묘하게 서로 달랐다:

  camera_tools.analyze_microscope_image  >>8,  GRAY→BGR, 뷰 해상도로 resize
  acquire_tools.preview_grid_scan         >>8,  GRAY→BGR, 뷰 해상도로 resize  (위와 거의 동일)
  camera_tools.capture_scene             /256, BGR→RGB,  **resize 없음**
  camera_tools.run_autofocus._to_uint8   /256, BGR→GRAY
  USE_autofocus_local._to_uint8         /256, BGR→GRAY  (또 한 벌)

앞의 둘은 좌표계가 move_to_pixel 의 입력과 일치하도록 뷰 해상도로 정규화하는데,
capture_scene 만 센서 네이티브 해상도를 그대로 저장했다. 그래서 capture_scene 이 남긴
이미지 위에서 잰 픽셀 좌표는 move_to_pixel 에 넣을 수 없었다(다른 좌표계인데 겉보기로는
구분이 안 된다). 여기로 모아 '뷰 해상도 = 도구 좌표계'를 한 곳에서 보장한다.

스팟 면적(가이드빔 OFF/ON 차분 → Otsu → 픽셀 수)도 camera_tools.run_autofocus 와
USE_autofocus_local._capture_diff 에 각각 구현돼 있었다. 이 값은 오토포커스의 목적함수
자체라, 두 구현이 갈라지면 같은 시료에서 다른 Z 로 수렴한다.
"""
from __future__ import annotations

import cv2
import numpy as np

try:                                    # 패키지 모드
    from backend.tools.hw_tools.config import CAMERA_WIDTH, CAMERA_HEIGHT
except ImportError:                     # 단독 스크립트 모드
    from config import CAMERA_WIDTH, CAMERA_HEIGHT   # type: ignore[no-redef]


def to_uint8_gray(frame) -> "np.ndarray":
    """프레임 → 8bit 그레이스케일. 스팟 면적·선명도 계산의 공통 입력."""
    img = np.asarray(frame)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def to_view_bgr(frame, width: int | None = None, height: int | None = None) -> "np.ndarray":
    """프레임 → 8bit BGR, **뷰 해상도(CAMERA_WIDTH×CAMERA_HEIGHT)로 정규화**.

    이 해상도가 곧 도구 좌표계다: 여기서 읽은 픽셀 좌표를 move_to_pixel 에 그대로
    넣을 수 있고, optics_map 의 mm/px 도 같은 기준으로 계산된다. 화면을 다루는 모든
    도구(analyze_microscope_image / capture_scene / preview_grid_scan)가 이 함수를
    거치므로, 세 도구가 보고하는 좌표는 서로 교환 가능하다.
    """
    w = int(width or CAMERA_WIDTH)
    h = int(height or CAMERA_HEIGHT)
    img = np.asarray(frame)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    if bgr.shape[:2] != (h, w):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return bgr


def sharpness_score(gray) -> float:
    """라플라시안 분산 — 상대 비교용 선명도 지표.

    주의: 오토포커스의 목적함수가 아니다. run_autofocus 는 guide_beam_spot_area()
    (가이드빔 스팟 '면적 최소화')를 쓴다. 두 지표는 무관해서 서로 다른 Z 로 수렴한다.
    """
    return float(np.var(cv2.Laplacian(np.asarray(gray).astype(np.float32), cv2.CV_32F)))


def _flush_frames(camera, n: int = 3) -> None:
    """레이저 상태를 바꾼 뒤 카메라 버퍼에 남은 '변경 이전' 프레임을 버린다."""
    for _ in range(n):
        camera.get_latest_frame()


def _mean_frame(camera, n: int = 3):
    """연속 n 장의 평균(8bit 그레이). 한 장도 못 받으면 None."""
    frames = []
    for _ in range(n):
        f = camera.get_latest_frame()
        if f is not None:
            frames.append(to_uint8_gray(f))
    if not frames:
        return None
    return np.mean(frames, axis=0).astype(np.uint8)


def capture_laser_diff(camera, laser, n_avg: int = 3) -> dict:
    """레이저 OFF/ON 차분으로 빔 스팟을 분리하고 면적(픽셀 수)을 잰다.

    절차: OFF → 버퍼 flush → 배경 n장 평균 → ON → flush → 레이저 n장 평균
          → clip(laser - ref, 0) → GaussianBlur → Otsu → 비영 픽셀 수.

    Returns
    -------
    dict — {ok, area_px, ref, laser_frame, diff_clip, diff_absdiff}
           프레임을 못 받으면 {"ok": False, "area_px": 0, ...}.
           area_px 는 초점이 맞을수록 작아진다(스팟이 날카로워진다).
    """
    laser.laser_off()
    _flush_frames(camera, n_avg)
    ref = _mean_frame(camera, n_avg)

    laser.laser_on()
    _flush_frames(camera, n_avg)
    lit = _mean_frame(camera, n_avg)

    if ref is None or lit is None:
        return {"ok": False, "area_px": 0, "ref": ref, "laser_frame": lit,
                "diff_clip": None, "diff_absdiff": None}

    diff_clip = np.clip(lit.astype(np.int16) - ref.astype(np.int16), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(diff_clip, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"ok": True, "area_px": int(np.count_nonzero(binary)),
            "ref": ref, "laser_frame": lit,
            "diff_clip": diff_clip, "diff_absdiff": cv2.absdiff(lit, ref)}


def guide_beam_spot_area(camera, laser, n_avg: int = 3):
    """capture_laser_diff 의 면적만 필요할 때 쓰는 얇은 래퍼(오토포커스 목적함수).

    Returns
    -------
    int | None — 스팟 면적(픽셀 수). **프레임을 한 장도 못 받았으면 None** 이다.

    [왜 0 이 아니라 None 인가 — 2026-07-31]
    예전에는 실패도 0 으로 뭉갰다. 그런데 이 값은 '작을수록 초점이 맞다'는 목적함수라,
    0 은 최고의 점수다. 즉 카메라가 죽어 아무것도 못 본 상태가 "완벽한 초점"으로 읽혔고,
    run_autofocus 는 카메라가 통째로 멈춰도 ok:True 를 돌려줬다. 격자 스캔은 그 성공을
    믿고 초점이 안 맞은 채 모든 점을 측정했다(테스트로 재현). '못 쟀다'와 '재 봤더니 0'은
    반드시 구분돼야 한다.
    """
    d = capture_laser_diff(camera, laser, n_avg)
    if not d.get("ok"):
        return None
    return int(d.get("area_px", 0))
