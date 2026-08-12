# -*- coding: utf-8 -*-
"""
[역할] 카메라 픽셀 좌표 ↔ 스테이지 mm 좌표 변환의 **단일 출처**.

[왜 이 모듈이 생겼는가 — 2026-07-30]
같은 변환식이 네 곳에 각각 복사돼 있었고, 그중 하나는 상수까지 따로 박혀 있었다:

  · raman_tools.move_to_pixel        _UM_PER_PX_* × CALIB_FACTOR_*
  · raman_tools._mm_to_pixel         위의 역변환을 별도로 다시 구현
  · USE_scan.py                      UM_PER_PX_*/SIGN_* 를 자체 정의
  · server.py /api/stage/move-pixel  CALIB_X=1.4, CALIB_Y=1.285 를 **하드코딩**
                                     (config.CALIB_FACTOR_* 와 지금은 우연히 같을 뿐이다)

더 나빴던 것은 시야(FOV) 계산이 두 갈래로 갈린 점이다.
  preview_grid_scan : CAMERA_WIDTH × mm/px(보정 포함) → 0.427 × 0.296 mm
  capture_scene     : LENS_WIDTH_UM / 1000(보정 없음) → 0.305 × 0.230 mm
같은 화면의 물리적 크기를 1.4배 다르게 보고하고 있었고, capture_scene 이 돌려주는
extent 는 run_analysis 가 피크맵을 겹칠 때 쓰는 좌표계라 그림이 조용히 어긋났다.
(에러가 안 나고 결과만 틀리는 종류라 발견이 늦다.)

[핵심 사실] 보정된 시야 폭은 해상도와 무관하다:
    width_px × (LENS_WIDTH_UM / width_px) × CALIB = LENS_WIDTH_UM × CALIB
따라서 스트림 해상도가 몇이든 fov_mm() 은 같은 값을 준다. 반대로 '픽셀 1개당 mm' 는
해상도에 반비례하므로, 프론트가 임의 크기로 스트리밍하는 경우(server 의 move-pixel)
width/height 를 넘겨야 한다. 그래서 아래 함수들은 전부 width/height 를 인자로 받되
기본값을 Config.ini 의 뷰 해상도(CAMERA_WIDTH/HEIGHT)로 둔다.
"""
from __future__ import annotations

try:                                    # 패키지 모드 (서버·에이전트)
    from backend.hw_tools.config import (
        CAMERA_WIDTH, CAMERA_HEIGHT,
        LENS_WIDTH_UM, LENS_HEIGHT_UM,
        CALIB_FACTOR_X, CALIB_FACTOR_Y,
    )
except ImportError:                     # 단독 스크립트 모드 (USE_*.py 가 backend/ 를 sys.path 에 넣고 실행)
    from config import (                # type: ignore[no-redef]
        CAMERA_WIDTH, CAMERA_HEIGHT,
        LENS_WIDTH_UM, LENS_HEIGHT_UM,
        CALIB_FACTOR_X, CALIB_FACTOR_Y,
    )

# ── 축 부호 ───────────────────────────────────────────────────────────────────
# pixel +X(오른쪽) → stage -X   (스테이지가 +X 로 가면 화면 속 시료는 왼쪽으로 흐른다)
# pixel +Y(아래)   → stage +Y
SIGN_X = -1
SIGN_Y = +1


def mm_per_px(width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """이미지 1픽셀이 스테이지에서 몇 mm 인지(양수). 부호는 SIGN_* 로 따로 적용한다."""
    w = int(width or CAMERA_WIDTH)
    h = int(height or CAMERA_HEIGHT)
    return (LENS_WIDTH_UM  / w * CALIB_FACTOR_X / 1000.0,
            LENS_HEIGHT_UM / h * CALIB_FACTOR_Y / 1000.0)


def fov_mm(width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """카메라 시야의 물리적 크기(mm). 위 주석대로 해상도와 무관하지만, 인자를 받는
    형태를 유지해 호출부가 mm_per_px 와 같은 규약으로 쓰이게 한다."""
    mx, my = mm_per_px(width, height)
    return (mx * int(width or CAMERA_WIDTH), my * int(height or CAMERA_HEIGHT))


def pixel_delta_to_mm(dpx: float, dpy: float,
                      width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """화면 중심 기준 픽셀 변위 → 스테이지 mm 변위(부호 적용)."""
    mx, my = mm_per_px(width, height)
    return (dpx * mx * SIGN_X, dpy * my * SIGN_Y)


def pixel_to_stage(px: float, py: float, cur_x: float, cur_y: float,
                   width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """픽셀 좌표 → 절대 스테이지 좌표(mm). 이미지 중심이 현재 스테이지 위치에 대응한다."""
    w = int(width or CAMERA_WIDTH)
    h = int(height or CAMERA_HEIGHT)
    dx_mm, dy_mm = pixel_delta_to_mm(px - w / 2.0, py - h / 2.0, w, h)
    return (cur_x + dx_mm, cur_y + dy_mm)


def stage_to_pixel(sx: float, sy: float, cur_x: float, cur_y: float,
                   width: int | None = None, height: int | None = None) -> tuple[float, float]:
    """pixel_to_stage 의 역변환 — 스테이지 좌표를 현재 화면 위 픽셀로 투영한다."""
    w = int(width or CAMERA_WIDTH)
    h = int(height or CAMERA_HEIGHT)
    mx, my = mm_per_px(w, h)
    return (w / 2.0 + (sx - cur_x) * SIGN_X / mx,
            h / 2.0 + (sy - cur_y) * SIGN_Y / my)


def scene_extent(cur_x: float, cur_y: float,
                 width: int | None = None, height: int | None = None) -> list[float]:
    """현재 화면이 덮는 스테이지 좌표 범위 [xmin, xmax, ymin, ymax] (mm).

    matplotlib 의 imshow(extent=) 에 그대로 넣는 값이다. stage_to_pixel 과 같은
    mm/px 에서 유도하므로 **범위의 크기**는 화면과 정확히 일치한다(예전에는 보정계수가
    빠져 1.4배 작게 보고했다).

    [주의 — 축 방향은 여기서 다루지 않는다]
    SIGN_X=-1 이므로 이미지 왼쪽 끝이 실제로는 stage +X 쪽이고, imshow 의 기본
    origin='upper' 에서는 첫 행이 위에 그려진다. 즉 화면과 완전히 같은 방향으로
    겹치려면 호출부가 축 반전을 따로 처리해야 한다. 이 함수는 '어느 영역을 덮는가'
    (크기와 중심)만 책임진다 — 방향까지 뒤집어 돌려주면 extent 를 단순 범위로 쓰는
    기존 분석 코드가 조용히 깨지기 때문이다.
    """
    w_mm, h_mm = fov_mm(width, height)
    return [cur_x - w_mm / 2.0, cur_x + w_mm / 2.0,
            cur_y - h_mm / 2.0, cur_y + h_mm / 2.0]
