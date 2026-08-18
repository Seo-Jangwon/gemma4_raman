# -*- coding: utf-8 -*-
"""카메라(TUCam) 도구 — 스트리밍, 캡처, 픽셀→좌표 이동, 오토포커스.

화면을 다루는 세 도구가 목적별로 갈린다: analyze_microscope_image 는 모델에게 보여주고,
capture_scene 은 파일로 남기고, preview_grid_scan(acquire_tools)은 계획을 겹쳐 보여준다.
셋 다 같은 픽셀 좌표계라 어느 것에서 읽은 좌표든 move_to_pixel 에 넘길 수 있다.
"""
from __future__ import annotations

import time

from backend.tools.hw_tools.config import CAMERA_HEIGHT, CAMERA_WIDTH, STAGE_MAX_Z, STAGE_MIN_Z
from backend.service.vision import vision as _vis
from backend.service.store.spectrum_store import save_preview_png as _store_save_preview, save_scene as _store_save_scene
from backend.service.vision import optics_map as _om
from backend.tools.result import fail, ok
from pydantic import Field
from typing import Annotated, Optional
from backend.tools.hw_tools.hw_tools import hw_core as _hw
from backend.tools.hw_tools.hw_tools.hw_core import _check_stage_target, _laser_off_quiet, _serialized, _sstate, _stage_unavailable
from backend.tools.hw_tools.hw_tools.stage_tools import move_stage


# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@_serialized("start_camera_stream")
def start_camera_stream() -> dict:
    """
    카메라 실시간 스트리밍을 시작합니다.
    USE_camera_stream.py의 StreamingTUCam.start_stream()을 호출합니다.

    반환의 already_streaming: 이 호출 "이전에" 이미 스트리밍 중이었는지.
    호출자가 스트림 소유권을 판단하는 근거다 — 남이(예: 프론트 MJPEG 뷰) 켜 둔
    스트림을 내가 끄면 그쪽 화면이 죽는다. already_streaming=True면 끄지 말 것.
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")

    try:
        # 이미 스트리밍 중인지 확인 (StreamingTUCam 내부 속성 활용)
        if getattr(_hw._camera, 'is_streaming', False):
            return ok(already_streaming=True, status="Camera is already streaming.")

        _hw._camera.start_stream()
        return ok(already_streaming=False, status="Camera streaming started successfully.")

    except Exception as e:
        return fail(f"Failed to start streaming: {str(e)}")

@_serialized("stop_camera_stream")
def stop_camera_stream() -> dict:
    """
    카메라 실시간 스트리밍을 중지합니다.
    USE_camera_stream.py의 StreamingTUCam.stop_stream()을 호출합니다.
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")

    try:
        if not getattr(_hw._camera, 'is_streaming', False):
            return ok(status="Camera is not currently streaming.")

        _hw._camera.stop_stream()
        return ok(status="Camera streaming stopped successfully.")

    except Exception as e:
        return fail(f"Failed to stop streaming: {str(e)}")


@_serialized("set_camera_exposure")
def set_camera_exposure(
    ms: Annotated[float, Field(description='Exposure time [ms]. e.g. 10.0, 50.0, 100.0')],
) -> dict:
    """카메라(TUCam) 노출 시간(ms)을 설정한다."""
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    if ms <= 0:
        return fail("Exposure time must be greater than 0.")
    try:
        _hw._camera.set_exposure(ms)
        return ok(exposure_ms=ms)
    except Exception as e:
        return fail(str(e))


@_serialized("set_camera_auto_exposure")
def set_camera_auto_exposure(
    enabled: Annotated[bool, Field(description='true = auto exposure ON, false = manual exposure')],
) -> dict:
    """카메라 자동 노출을 활성화(True) 또는 비활성화(False)한다.

    [SDK 를 직접 부르지 않는 이유 — 2026-08-15]
    예전에는 여기서 TUCAM_Capa_SetValue(_camera.TUCAMOPEN.hIdxTUCam, ...) 를 직접 불렀다.
    바로 위 set_camera_exposure 는 드라이버 메서드(set_exposure)를 부르는데 이것만 SDK
    핸들 내부를 파고드는 비대칭이었고, 그래서 카메라 드라이버를 갈아 끼우면(가상 카메라)
    다른 도구는 다 도는데 이 하나만 낄 자리가 없었다. 설정은 드라이버가 갖는다.
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    try:
        _hw._camera.set_auto_exposure(bool(enabled))
        return ok(auto_exposure=enabled)
    except Exception as e:
        return fail(str(e))


# ──────────────────────────────────────────
# 오토포커스 (카메라 선명도 기반 Z 스윕)
# ──────────────────────────────────────────

# 스팟 면적을 연속으로 이만큼 못 재면 오토포커스를 실패로 끝낸다. 한 번은 프레임 한 장을
# 놓친 것일 수 있지만(버퍼 타이밍), 연달아 못 받으면 스트림이 멈췄거나 가이드빔이 안 나오는
# 것이다. 그 상태로 계속 돌면 Z 만 흔들다가 '성공'을 보고하게 된다.
_AF_UNMEASURED_ABORT = 3

@_serialized("run_autofocus")
def run_autofocus(
    initial_z: Annotated[Optional[float], Field(description='Starting Z position for the search (mm). If omitted, keep the current Z')] = None,
    step_size: Annotated[Optional[float], Field(description='Initial Z step size (mm). Default 0.030 (30 um)')] = 0.03,
    min_step: Annotated[Optional[float], Field(description='Minimum step size (mm) - the search ends below this. Default 0.001 (1 um)')] = 0.001,
    max_steps: Annotated[Optional[int], Field(description='Maximum number of steps - forced stop when exceeded. Default 100')] = 100,
) -> dict:
    """
    가이드빔 레이저 스팟 면적 최소화 기반 힐클라이밍 오토포커스.
    USE_autofocus_local.py의 AutoFocusLocal 알고리즘을 헤드리스로 실행한다.

    동작 원리:
      각 Z 위치에서 레이저 OFF → 배경 프레임 취득 → 레이저 ON → 레이저 프레임 취득
      → clip 차분 → GaussianBlur → Otsu threshold → 스팟 픽셀 수(면적) 계산
      → 면적이 작을수록(레이저 스팟이 날카로울수록) 초점이 맞음
      → 적응형 힐클라이밍: 개선되면 같은 방향 전진, 나빠지면 방향 반전 + 보폭 절반
      → 역대 최솟값(global_best_z) 위치로 최종 귀환

    Parameters
    ----------
    initial_z : float, optional
        탐색 시작 Z 위치(mm). None이면 현재 Z 유지.
    step_size : float
        초기 Z 이동 보폭(mm). 기본 0.030mm (30µm).
    min_step : float
        최소 보폭(mm) — 이 이하면 탐색 종료. 기본 0.001mm (1µm).
    max_steps : int
        최대 스텝 수 — 초과 시 강제 종료. 기본 100.

    Returns
    -------
    dict
        optimal_z, best_area_px, step_count, z_scores, current_position
    """
    stage_err = _stage_unavailable()
    if stage_err:
        return stage_err
    # Z 축이 없는 스테이지에서는 이 도구가 성립하지 않는다 — 본질이 Z 를 훑으며 스팟 면적을
    # 최소화하는 힐클라이밍이기 때문이다. 가짜로 '수렴했다'를 돌려주면 에이전트는 초점이
    # 맞았다고 믿고 그 뒤 판단을 전부 그 위에 쌓는다.
    #
    # 판정을 llm_config.VIRTUAL_HW 가 아니라 **스테이지 객체의 속성**으로 하는 이유: 도구
    # 계층이 '지금 가상인가'를 알기 시작하면 같은 분기가 도구 수만큼 번진다. 여기서 필요한
    # 사실은 '가상인가'가 아니라 'Z 를 움직일 수 있는가' 하나다.
    if not getattr(_hw._stage, "has_z", True):
        return fail("This stage has no Z axis, so autofocus cannot run. The focus is fixed; "
                    "proceed with the measurement without focusing.")
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    if _hw._laser is None:
        return fail("Laser is not initialized.")
    if initial_z is not None:
        err = _check_stage_target(z=initial_z)
        if err:
            return err

    try:
        pos = _hw._stage.get_position()
        cur_x, cur_y, cur_z = pos[0], pos[1], pos[2]
        cur_a = pos[3] if len(pos) > 3 else 0
        n_clamped = 0

        def _goto_z(z: float) -> float:
            """Z 이동. **허용 범위로 클리핑한다** — 힐클라이밍은 목표 Z 를 스스로 밀어
            올리는 루프라, 예전처럼 검사 없이 move_absolute 를 부르면 스테이지 한계를
            그대로 넘어서는 명령이 나갔다(이 함수만 범위 검증이 없었다). 여기서는
            거부가 아니라 클리핑이 맞다 — 한계에 닿으면 면적이 개선되지 않으므로
            알고리즘이 스스로 방향을 반전한다."""
            nonlocal n_clamped
            zc = max(STAGE_MIN_Z, min(STAGE_MAX_Z, float(z)))
            if zc != float(z):
                n_clamped += 1
            _hw._stage.move_absolute(cur_x, cur_y, zc, cur_a)
            time.sleep(0.3)
            return zc

        if initial_z is not None:
            cur_z = _goto_z(initial_z)

        # 가이드빔 모드 + 카메라 스트리밍 보장
        _hw._laser.set_guide_beam()
        if not getattr(_hw._camera, 'is_streaming', False):
            _hw._camera.start_stream()

        # 목적함수는 vision.guide_beam_spot_area 단일 출처다 — USE_autofocus_local 의
        # 대화형 오토포커스와 같은 함수를 쓰므로 두 경로가 같은 Z 로 수렴한다.
        # None 은 '못 쟀다'(프레임 미수신)이고 0 과 다르다 — vision 쪽 주석 참고.
        def _capture_spot_area():
            return _vis.guide_beam_spot_area(_hw._camera, _hw._laser, n_avg=3)

        n_unmeasured = 0         # 스팟 면적을 못 잰 횟수(카메라가 프레임을 안 준다)

        # 힐클라이밍 상태
        best_z = cur_z
        best_area = float('inf')
        direction = 1
        step_count = 0
        global_best_area = float('inf')
        global_best_z = cur_z
        z_scores: list = []
        sweep_state = 'init'

        while sweep_state != 'done':
            pos_now = _hw._stage.get_position()
            z_now = pos_now[2] if pos_now else cur_z

            area = _capture_spot_area()
            if area is None:
                # 목적함수를 잴 수 없다 = 초점을 맞출 근거가 없다. 계속 돌면 Z 만 흔들다가
                # '성공'을 돌려주게 되므로 여기서 끝낸다(카메라 스트림·가이드빔을 먼저 봐야 한다).
                n_unmeasured += 1
                if n_unmeasured >= _AF_UNMEASURED_ABORT:
                    _goto_z(cur_z)          # 탐색 시작 Z 로 되돌린다(아래 문구가 사실이 되도록)
                    _laser_off_quiet()
                    return fail(f"Autofocus could not measure the guide-beam spot at all - the camera returned "
                                f"no frames {n_unmeasured} times in a row. Focus was NOT adjusted and the stage "
                                f"is back where it started. This is not something a retry fixes: check that the "
                                f"camera is streaming (start_camera_stream) and that the guide beam is actually "
                                f"emitting (get_laser_status / set_guide_beam_mode).",
                                z_scores=z_scores,
                                unmeasured_samples=n_unmeasured)
                # 아직 한도 전이면 한 칸 움직여 다시 시도한다.
                _goto_z(z_now + direction * step_size)
                continue
            z_scores.append({"z": round(z_now, 4), "area_px": area})

            if 0 < area < global_best_area:
                global_best_area = area
                global_best_z = z_now

            if sweep_state == 'init':
                best_area = area
                best_z = z_now
                _goto_z(best_z + direction * step_size)
                sweep_state = 'check'

            elif sweep_state == 'check':
                step_count += 1
                if area < best_area:
                    best_area = area
                    best_z = z_now
                    _goto_z(best_z + direction * step_size)
                else:
                    direction *= -1
                    step_size /= 2.0
                    if step_size < min_step or step_count >= max_steps:
                        sweep_state = 'done'
                    else:
                        _goto_z(best_z + direction * step_size)

        # 유효한 측정이 한 번도 없었으면(전부 면적 0 이거나 못 잼) 최적 Z 라는 것이 없다.
        # 이때 global_best_z 는 그냥 시작 Z 인데, 그걸 optimal_z 로 돌려주면 "초점을 맞췄다"는
        # 거짓 보고가 된다. 스팟이 전혀 안 보이는 상황(가이드빔 미출력, 시료 없음)이다.
        if global_best_area == float('inf'):
            _goto_z(cur_z)
            _laser_off_quiet()
            return fail("Autofocus finished without ever detecting a guide-beam spot - every sample came back "
                        "with zero spot area, so there was nothing to minimise and no focus was found. The Z "
                        "position is unchanged. Check that the guide beam is actually emitting "
                        "(set_guide_beam_mode, then look with analyze_microscope_image) and that a sample is "
                        "actually under the objective.",
                        z_scores=z_scores,
                        unmeasured_samples=n_unmeasured)

        # 역대 최솟값 위치로 최종 귀환
        _goto_z(global_best_z)
        time.sleep(0.2)
        _laser_off_quiet()

        out = ok(optimal_z=global_best_z,
                 best_area_px=global_best_area,
                 step_count=step_count,
                 z_scores=z_scores,
                 current_position={"x": cur_x, "y": cur_y, "z": global_best_z})
        if n_clamped:
            out["z_limit_hits"] = n_clamped
            out["note"] = (
                f"The search hit the Z travel limit ({STAGE_MIN_Z} to {STAGE_MAX_Z} mm) "
                f"{n_clamped} time(s), so the focus may lie outside the reachable range. "
                "If best_area_px is still large, the sample height is likely off - reposition "
                "the sample or the objective rather than repeating autofocus.")
        return out
    except Exception as e:
        _laser_off_quiet()
        return fail(str(e))


@_serialized("capture_scene")
def capture_scene() -> dict:
    """현재 현미경(카메라) 화면을 저장한다 — run_analysis 가 이 위에 피크맵을 오버레이한다.

    스테이지 위치와 보정된 시야(FOV)로 이미지의 스테이지 좌표 범위(extent, mm)를 계산해
    함께 저장하므로, 분석 코드에서 imshow(microscope_image, extent=image_extent) 후
    측정 (x,y)를 그 위에 정합해 찍을 수 있다. 카메라 스트리밍이 켜져 있어야 한다.

    [2026-07-30 수정 — 두 가지가 틀려 있었다]
    1) extent 를 LENS_*_UM/1000 으로 계산해 보정계수(CALIB_FACTOR)가 빠져 있었다.
       preview_grid_scan 이 쓰는 시야(0.427×0.296mm)와 달리 0.305×0.230mm 로 나와,
       같은 화면의 크기를 1.4배 다르게 보고했다. 이 extent 위에 측정 좌표를 찍으면
       그만큼 어긋난다(에러 없이 그림만 틀린다). 이제 optics_map.scene_extent 단일 출처.
    2) 프레임을 센서 네이티브 해상도 그대로 저장해서, 이 이미지에서 읽은 픽셀 좌표는
       move_to_pixel 에 넣을 수 없었다(analyze_microscope_image 와 다른 좌표계).
       이제 vision.to_view_bgr 로 뷰 해상도에 맞춘다 — 세 캡처 도구의 좌표계가 같다.
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    import cv2
    frame = _hw._camera.get_latest_frame()
    if frame is None:
        return fail("No camera frame. Start streaming first.")
    # 뷰 해상도로 정규화(도구 좌표계) 후 matplotlib 표시 기준인 RGB 로.
    img = cv2.cvtColor(_vis.to_view_bgr(frame), cv2.COLOR_BGR2RGB)

    extent = None
    try:
        if _hw._stage is not None:
            pos = _hw._stage.get_position()
            extent = _om.scene_extent(float(pos[0]), float(pos[1]))
    except Exception:
        pass

    saved = _store_save_scene(img, extent, {})
    if not saved.get("ok"):
        return saved

    # 마지막 캡처를 이 세션의 상태에 기억해 둔다 — save_measurement_point 가 '이 지점에서
    # 찍은 이미지'로 참조한다. 에이전트가 이미지 배열을 인자로 넘길 필요가 없게 하기 위한 것.
    _last_scene = {
        "image_url": saved["image_url"],
        "scene_npz": saved.get("scene_npz"),
        "extent": extent,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        if _hw._stage is not None:
            _p = _hw._stage.get_position()
            _last_scene["position"] = {"x": round(float(_p[0]), 4), "y": round(float(_p[1]), 4),
                                       "z": round(float(_p[2]), 4)}
    except Exception:
        pass
    _sstate()["last_scene"] = _last_scene

    return ok(image_url=saved["image_url"],
              extent=extent,
              shape=list(img.shape),
              saved={"title": "Microscope view capture", "image_url": saved["image_url"]},
              note="Usable later in run_analysis via microscope_image / image_extent, "
                   "and referenced automatically by save_measurement_point.")


@_serialized("analyze_microscope_image")
def analyze_microscope_image(
    question: Annotated[Optional[str], Field(description="What you want to check in the image (optional). e.g. 'Describe the sample position and brightness'")] = 'Find a specific object in the sample (e.g. a cell) and report its center-point pixel coordinates.',
) -> dict:
    """
    TuCam 현미경 카메라 화면을 PNG (Base64)로 캡처하여 반환.

    [왜 CAMERA_WIDTH×CAMERA_HEIGHT로 리사이즈하는가 — 두 가지 이유]

    1) 좌표계 일치 (기능 버그 수정)
       get_latest_frame()은 센서 네이티브 해상도를 그대로 준다(Config.ini의
       Width/Height는 뷰 기준 해상도일 뿐 프레임 크기가 아니다). 그런데 이 이미지를
       보고 vision LLM이 찍은 픽셀 좌표는 결국 move_to_pixel()로 들어가고,
       move_to_pixel은 이미지 중심을 (CAMERA_WIDTH/2, CAMERA_HEIGHT/2)로 가정해
       계산한다. 두 해상도가 다르면 스테이지가 엉뚱한 곳으로 이동한다.
       USE_scan.py도 같은 이유로 픽셀→스테이지 계산 전에 이 크기로 리사이즈한다.
       → 여기서 미리 맞춰 두면 이 함수의 출력 좌표계 = move_to_pixel의 입력 좌표계.

    2) API 이미지 제한
       Anthropic API는 이미지 1장당 base64 10MB, 긴 변 2576px가 상한이고 그보다 큰
       이미지는 어차피 서버에서 다운스케일된다. 네이티브 프레임을 무손실 PNG로 보내면
       12MB를 넘겨 요청 자체가 거부됐다(실제로 그랬다). 1060×800이면 ~2MB, 긴 변도
       상한 이하라 다운스케일이 없어 좌표가 보낸 그대로 유지된다.
    """
    if _hw._camera is None:
        return fail("Camera is not initialized.")
    try:
        import base64
        import cv2
        frame = _hw._camera.get_latest_frame()
        if frame is None:
            return fail("Failed to acquire frame (check whether streaming is active)")

        # 뷰 기준 해상도로 정규화 — 위 docstring의 (1)(2). capture_scene /
        # preview_grid_scan 과 같은 vision.to_view_bgr 를 쓰므로 좌표계가 동일하다.
        frame_bgr = _vis.to_view_bgr(frame)
        height, width = frame_bgr.shape[:2]
        ret, buf = cv2.imencode('.png', frame_bgr)
        enhanced_question = f"{question}\n\n[The attached image has an original resolution of {width}px wide by {height}px tall. When returning pixel coordinates, give exact pixel values based on this resolution.][Note: you return pixel coordinates, which are NOT stage coordinates. To move the stage to that location, you must use the move_to_pixel(pixel_x, pixel_y) function.]"

        if not ret:
            return fail("PNG encoding failed")
        img_b64 = base64.b64encode(buf).decode('utf-8')

        # 디스크에도 남긴다 — 이게 없으면 대화 히스토리의 base64 가 이 이미지의 **유일한
        # 사본**이 된다(2026-08-12). 히스토리는 턴 경계에서 잘려 나가므로, 저장이 없으면
        # 방금 본 화면을 다시 볼 방법이 영영 사라진다. file_id 를 결과에 실어 두면
        # ToolMessage 에 남아, base64 가 사라진 뒤에도 view_image(file_id) 로 되돌아온다.
        # (preview_grid_scan 이 이미 같은 헬퍼로 같은 일을 한다.)
        saved_img = _store_save_preview(buf.tobytes(), tag="microscope")

        # 밝기 통계와 선명도 — 예전 capture_camera_frame 이 주던 값들을 여기로 합쳤다.
        # 리사이즈·8bit 정규화를 거친 '위에서 실제로 보낸 그 이미지'에서 계산하므로,
        # 모델이 보는 화면과 숫자가 일치한다(옛 툴은 uint16 원본을 그대로 재서 어긋났다).
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        out = ok(image_base64=img_b64,
                 question=enhanced_question,
                 width=width,
                 height=height,
                 min_intensity=float(gray.min()),
                 max_intensity=float(gray.max()),
                 mean_intensity=float(gray.mean()),
                 sharpness_score=_vis.sharpness_score(gray))
        # 저장 실패가 촬영 자체를 실패로 만들면 안 된다 — 모델은 이미 이미지를 받는다.
        # 다만 file_id 가 없으면 다시 못 보므로, 그 사실을 결과에 적어 알린다.
        if saved_img.get("ok"):
            out["image_file"] = saved_img["file_id"]
            out["saved"] = {"title": "Microscope view", "image_url": saved_img["image_url"]}
        else:
            out["image_file_error"] = (
                f"The image was NOT saved to disk ({saved_img.get('error')}), so you cannot "
                f"view it again later. Extract everything you need from it in this turn.")
        return out
    except Exception as e:
        return fail(str(e))


@_serialized("move_to_pixel")
def move_to_pixel(
    pixel_x: Annotated[int, Field(description=f'Image X pixel coordinate (0 - {CAMERA_WIDTH})')],
    pixel_y: Annotated[int, Field(description=f'Image Y pixel coordinate (0 - {CAMERA_HEIGHT})')],
) -> dict:
    """
    카메라 이미지의 픽셀 좌표를 스테이지 mm 좌표로 변환해 이동한다.
    이미지 중심(CAMERA_WIDTH/2, CAMERA_HEIGHT/2)이 현재 스테이지 위치에 대응한다.

    입력 좌표계는 analyze_microscope_image / capture_scene / preview_grid_scan 이
    돌려주는 이미지와 동일하다(셋 다 vision.to_view_bgr 로 같은 해상도에 맞춘다).
    변환식은 optics_map 단일 출처 — 예전에는 이 함수, USE_scan.py, server.py 가
    각자 같은 식을 갖고 있었고 server.py 는 보정계수를 하드코딩까지 했다.
    """
    stage_err = _stage_unavailable()
    if stage_err:
        return stage_err
    try:
        pos = _hw._stage.get_position()
        if pos is None:
            return fail("Failed to query stage position")
        tx, ty = _om.pixel_to_stage(pixel_x, pixel_y, float(pos[0]), float(pos[1]))
        return move_stage(x=round(tx, 4), y=round(ty, 4))
    except Exception as e:
        return fail(str(e))
