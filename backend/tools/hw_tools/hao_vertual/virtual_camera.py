# -*- coding: utf-8 -*-
"""가상 현미경 카메라. `StreamingTUCam` 덕타입.

씬(시료 이미지)에서 지금 스테이지 위치의 시야를 잘라 카메라 프레임 모양으로 만든다.

[책임 아님]
· 시료 이미지 소유 — scene 것이다.
· 현재 위치 판단 — stage 것이다.
· 빔 상태 판단 — laser 것이다.
이 파일은 '그 셋을 합쳐 한 장을 만든다'만 한다.

[왜 매니저를 참조하는가]
실물 카메라는 다른 장비를 몰라도 된다. 물리가 "카메라는 스테이지가 놓인 자리를 본다"를
보장하기 때문이다. 가상에는 그 보장이 없어서 스테이지와 레이저를 코드로 봐야 한다.
그런데 그 둘은 **재연결로 교체된다**(reconnect_hardware → _teardown → _init). 생성 시점의
객체를 붙들면 재연결 뒤 카메라만 옛 스테이지를 보게 되고, 그러면 화면이 실제 위치와
어긋난 채로 조용히 돈다. 그래서 객체가 아니라 '지금의 핸들을 들고 있는 곳'(매니저)을
참조해 매번 읽는다.

[프레임 파이프라인 — 순서 고정]
    scene.crop(스테이지 위치)      시야를 잘라 카메라 해상도로
      → 밝기(노출·auto)            잘라낸 뒤에 건다. 시야 결정과 노출은 별개다
      → 가이드빔 스팟 합성          레이저가 켜져 있을 때만
      → 노이즈
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from backend.tools.hw_tools.config import CAMERA_HEIGHT, CAMERA_WIDTH

#: 이 노출에서 이미지가 원본 밝기로 나온다. 노출을 두 배 주면 두 배 밝아진다.
_REF_EXPOSURE_MS = 50.0

#: 자동 노출이 맞추려는 평균 밝기(0-255).
_AUTO_TARGET_MEAN = 118.0

#: 가이드빔 스팟 반지름(픽셀)과 밝기. vision.guide_beam_spot_area 가 이 스팟을 잰다.
_SPOT_RADIUS_PX = 26
_SPOT_GAIN = 90.0


class VirtualCamera:
    """씬을 잘라 프레임을 만드는 카메라 대역."""

    def __init__(self, mgr, scene=None, exposure_ms: float = _REF_EXPOSURE_MS):
        """mgr : 지금의 스테이지·레이저 핸들을 들고 있는 객체(HardwareManager). 위 머리말 참고."""
        self._mgr = mgr
        if scene is None:
            from backend.tools.hw_tools.hao_vertual.scene import get_scene
            scene = get_scene()
        self.scene = scene
        self.exposure_ms = float(exposure_ms)
        self.auto_exposure = False
        self.is_streaming = False
        # 실물과 같은 이유로 RLock 이다 — MJPEG 스트림과 도구 호출이 같은 객체를 동시에
        # 만지고, stop_stream → get_latest_frame 같은 중첩 호출이 생길 수 있다.
        self._lock = threading.RLock()

    # ── 수명주기 ────────────────────────────────────────────────────────────
    def start_stream(self) -> None:
        with self._lock:
            self.is_streaming = True

    def stop_stream(self) -> None:
        with self._lock:
            self.is_streaming = False

    def close(self) -> None:
        self.stop_stream()

    # ── 설정 ────────────────────────────────────────────────────────────────
    def set_exposure(self, ms: float) -> None:
        """노출을 걸면 자동 노출은 풀린다 — 실물 SDK 도 수동 설정이 auto 를 끈다
        (StreamingTUCam.set_exposure 가 TUIDC_ATEXPOSURE 를 0 으로 내린다)."""
        with self._lock:
            self.exposure_ms = max(0.01, float(ms))
            self.auto_exposure = False

    def set_auto_exposure(self, enabled: bool) -> None:
        with self._lock:
            self.auto_exposure = bool(enabled)

    # ── 프레임 ──────────────────────────────────────────────────────────────
    def get_latest_frame(self):
        """(H, W, 3) uint8 BGR. 스트리밍 중이 아니면 None — 실물 계약과 같다."""
        with self._lock:
            if not self.is_streaming:
                return None
            view = self.scene.crop(*self._stage_xy(), CAMERA_WIDTH, CAMERA_HEIGHT)
            view = self._apply_brightness(view)
            view = self._apply_beam(view)
            return self._apply_noise(view)

    # ── 내부 ────────────────────────────────────────────────────────────────
    def _stage_xy(self) -> tuple[float, float]:
        """지금 스테이지 좌표. 스테이지가 없거나 죽었으면 씬 중심을 본다 —
        카메라만 연결한 상태에서도 화면이 나와야 한다(컨트롤러가 그 순서를 허용한다)."""
        stage = getattr(self._mgr, "stage", None)
        pos = stage.get_position() if stage is not None else None
        if pos is None:
            return self.scene.center_mm
        return (pos[0], pos[1])

    def _apply_brightness(self, view):
        gain = self.exposure_ms / _REF_EXPOSURE_MS
        if self.auto_exposure:
            mean = float(view.mean())
            # 완전히 검은 시야(지도 밖)에서는 보정하지 않는다 — 나눗셈이 폭주해 노이즈만
            # 증폭된 흰 화면이 나오고, '지도 밖'이라는 정보가 사라진다.
            if mean > 1.0:
                gain = _AUTO_TARGET_MEAN / mean
        if abs(gain - 1.0) < 1e-6:
            return view
        return np.clip(view.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    def _apply_beam(self, view):
        """레이저가 켜져 있으면 화면 중앙에 스팟을 얹는다.

        이것이 있어야 vision.capture_laser_diff / guide_beam_spot_area 가 동작한다 —
        그 함수들은 '레이저 ON 프레임 - OFF 프레임' 의 밝기 차로 스팟 면적을 잰다.
        """
        laser = getattr(self._mgr, "laser", None)
        if laser is None or not getattr(laser, "is_on", False):
            return view
        h, w = view.shape[:2]
        spot = np.zeros((h, w), np.float32)
        cv2.circle(spot, (w // 2, h // 2), _SPOT_RADIUS_PX, 1.0, -1)
        spot = cv2.GaussianBlur(spot, (0, 0), _SPOT_RADIUS_PX / 3.0)
        # 측정빔이 가이드빔보다 밝다. 파워를 올리면 스팟이 커져 보이는 것도 실제 거동이다.
        armed = bool(getattr(laser, "_power_set", False))
        strength = _SPOT_GAIN * (1.0 + (float(getattr(laser, "power_pct", 0) or 0) / 100.0)) if armed else _SPOT_GAIN * 0.5
        out = view.astype(np.float32) + spot[:, :, None] * strength
        return np.clip(out, 0, 255).astype(np.uint8)

    def _apply_noise(self, view):
        # 노이즈가 없으면 같은 자리에서 두 프레임이 완전히 동일해져, 프레임 평균·차분을
        # 쓰는 코드(vision._mean_frame, capture_laser_diff)가 실제로 도는지 확인되지 않는다.
        noise = np.random.normal(0.0, 2.0, view.shape).astype(np.float32)
        return np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def __repr__(self) -> str:
        mode = "auto" if self.auto_exposure else f"{self.exposure_ms:.1f}ms"
        return f"<VirtualCamera {'streaming' if self.is_streaming else 'idle'} exp={mode}>"
