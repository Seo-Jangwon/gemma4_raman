"""
장비 라우터 — 연결 / 상태 조회 / 직접 조작.

  POST /api/camera/connect     카메라 연결(노출시간 지정)
  POST /api/stage/connect      스테이지 연결
  POST /api/laser/connect      레이저 연결(COM 포트 지정)
  GET  /api/ccd/status         CCD 온도·안정화 상태 (프론트가 주기 폴링)
  POST /api/ccd/connect        CCD 연결/재시도
  GET  /api/camera/stream      MJPEG 라이브 영상
  POST /api/stage/move-pixel   카메라 클릭 픽셀 → 스테이지 이동
  GET  /api/hardware/state     현재 하드웨어 파라미터 전체 (파라미터 패널 동기화)
  POST /api/stage/speed        스테이지 이동 속도 설정
  POST /api/spectrum/acquire   LLM 없이 스펙트럼 측정 (수동 측정 버튼)
  GET  /api/health             서버·장비 연결 요약

여기 있는 것은 전부 '사람이 버튼으로 시키는' 경로다. 에이전트는 이 HTTP API 를 쓰지 않고
도구 계층(backend/tools/tools.py 의 TOOL_DISPATCH)을 직접 부른다.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

#: 가상/실물 판단의 단일 출처. 이 라우터가 매니저의 _init_camera/_init_laser 를 우회해
#: 직접 생성하는 자리가 둘 있어서, 그 두 곳도 같은 값을 봐야 한다.
from backend.llm_config import VIRTUAL_HW
from backend.tools.hw_tools.hw_tools.hw_core import (
    # 장비 조작 직렬화 가드. 락 순서는 항상 instrument_guard -> component_lock 이다
    # (hw_core.instrument_guard docstring 참고) — 뒤집으면 reconnect_hardware 와 교착한다.
    instrument_guard,
    InstrumentBusy,
    # 매니저의 현재 핸들 4개를 도구 계층 전역에 다시 주입하는 공용 헬퍼.
    sync_tool_handles as rt_sync_handles,
)
from backend.web_controller.setups.init_release import cool_ccd_in_background
from backend.web_controller.setups.request_log import log_ccd_reading
from backend.web_controller.setups.state import (
    AppState, CcdInitState, StateDep, run_in_worker,
)
from backend.web_controller.dto.requests import (
    AcquireSpectrumRequest,
    CameraConnectRequest,
    CCDConnectRequest,
    CCDStatusResponse,
    LaserConnectRequest,
    MovePixelRequest,
    StageConnectRequest,
    StageSpeedRequest,
)

router = APIRouter(prefix="/api", tags=["hardware"])


async def _run_connect(state: AppState, connect: Callable[[], str]) -> dict:
    """연결 작업을 워커에서 돌리고, 예외를 HTTP 상태코드로 옮긴다.

    409  측정/스캔이 도는 중이라 연결 작업을 하지 않았다. 서버 오류가 아니라
         '지금은 안 된다'이므로 구분해 알린다 — 프론트가 재시도를 안내할 수 있다.
    503  connect 안에서 직접 올린 HTTPException(스테이지 죽은 세션)은 그대로 통과시킨다.
         아래 500 이 삼키면 '재시작이 필요하다'는 정보가 사라진다.
    """
    try:
        return {"ok": True, "message": await run_in_worker(state, connect)}
    except InstrumentBusy as e:
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 장비 연결
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/camera/connect")
async def camera_connect(body: CameraConnectRequest, state: StateDep):
    hw = state.hw

    def _connect() -> str:
        # 이 엔드포인트는 노출 시간을 인자로 받으려고 _init_camera() 를 쓰지 않고
        # 직접 생성한다 → @_guarded 보호를 못 받으므로 락을 여기서 명시적으로 잡는다.
        # '이미 연결됨' 판정도 락 안에서 해야 한다: 밖에서 보면 동시 요청 둘이 모두
        # "미연결"을 보고 각자 카메라를 열어 한쪽 핸들이 고아가 된다.
        with instrument_guard("camera connect"), hw.component_lock("camera"):
            if hw.camera is not None:
                return "카메라 이미 연결됨"
            # 가상/실물 분기는 매니저의 _init_camera 와 같은 판단이다. 여기가 그것을 우회하는
            # 이유(노출시간을 인자로 받으려고)는 위 주석 참고 — 판단 자체는 갈라지면 안 된다.
            if VIRTUAL_HW:
                from backend.tools.hw_tools.hao_vertual.virtual_camera import VirtualCamera
                cam = VirtualCamera(hw, exposure_ms=body.exposure_ms)
            else:
                from backend.tools.hw_tools.hao.USE_camera_stream import StreamingTUCam
                cam = StreamingTUCam(exposure_ms=body.exposure_ms)
            cam.start_stream()
            hw.camera = cam
            rt_sync_handles(hw)
            return "카메라 연결 완료"

    return await _run_connect(state, _connect)


@router.post("/stage/connect")
async def stage_connect(body: StageConnectRequest, state: StateDep):
    hw = state.hw

    def _connect() -> str:
        # _init_stage() 자체는 @_guarded("stage") 로 보호되지만, '이미 연결됨' 판정과
        # 전역 핸들 재주입까지 한 구간으로 묶어야 동시 요청에도 안전하다.
        with instrument_guard("stage connect"), hw.component_lock("stage"):
            if hw.stage is not None:
                # 해제에 실패해 '죽은' 핸들은 고아 세션을 막으려고 일부러 남겨둔 것이다
                # (system_tools._teardown_component). 여기서 '이미 연결됨' 으로 200 을
                # 돌려주면 프론트는 초록불을 켜지만 실제로는 아무것도 못 한다 —
                # 재초기화도 불가능하다(새 핸들을 만들면 죽은 세션이 고아가 된다).
                # 그래서 재시작이 필요하다고 503 으로 정직하게 알린다.
                if getattr(hw.stage, "dead", False):
                    raise HTTPException(status_code=503, detail=(
                        f"스테이지 세션이 무효 상태입니다({getattr(hw.stage, 'dead_reason', 'unknown')}). "
                        f"DLL 세션을 해제하지 못해 재연결로는 복구할 수 없습니다 — "
                        f"서버 프로세스를 재시작해야 합니다."))
                return "스테이지 이미 연결됨"
            hw._init_stage()
            rt_sync_handles(hw)
            return "스테이지 연결 완료"

    return await _run_connect(state, _connect)


@router.post("/laser/connect")
async def laser_connect(body: LaserConnectRequest, state: StateDep):
    hw = state.hw

    def _connect() -> str:
        # 카메라와 같은 이유 — 포트를 인자로 받으려고 _init_laser() 를 우회하므로
        # 락을 직접 잡는다. COM 포트는 두 번 열면 'Access is denied' 로 죽는다.
        with instrument_guard("laser connect"), hw.component_lock("laser"):
            if hw.laser is not None:
                return "레이저 이미 연결됨"
            if VIRTUAL_HW:
                from backend.tools.hw_tools.hao_vertual.virtual_laser import VirtualLaser
                laser = VirtualLaser(port=body.port)
            else:
                from backend.tools.hw_tools.hao.USE_laser_with_power import LaserController
                laser = LaserController(port=body.port)
            if not (laser.ser and laser.ser.is_open):
                raise RuntimeError(f"레이저 포트 연결 실패 ({body.port})")
            hw.laser = laser
            rt_sync_handles(hw)
            return "레이저 연결 완료"

    return await _run_connect(state, _connect)


@router.get("/ccd/status", response_model=CCDStatusResponse)
async def ccd_status(state: StateDep):
    """프론트 상태바가 쉬지 않고 폴링한다. 콘솔에는 '사건'일 때만 한 줄 남는다
    (목표 온도 도달·온도 변화·쿨러 OFF 등 — core/request_log.log_ccd_reading 참고)."""
    ccd = state.hw.ccd
    if ccd is None:
        log_ccd_reading(False, None, None)
        return CCDStatusResponse(connected=False, temperature=None, temp_status=None)

    def _read():
        try:
            t = ccd.get_temperature()        # temperature_status_num 도 업데이트
            s = ccd.temp_status_dict.get(ccd.temperature_status_num)
            return int(t), s
        except IOError:
            # 촬영 중(DRV_ACQUIRING)이면 GetTemperature 접근 불가 — 쿨러는 그대로
            # 켜져 있으므로 '꺼짐'이 아니라 마지막으로 읽은 상태를 그대로 유지한다.
            cached_t = getattr(ccd, "temperature", None)
            cached_s = ccd.temp_status_dict.get(
                getattr(ccd, "temperature_status_num", None))
            return (int(cached_t) if cached_t is not None else None), cached_s

    try:
        temp, status_str = await run_in_worker(state, _read)
        log_ccd_reading(True,temp, status_str)
        return CCDStatusResponse(connected=True, temperature=temp, temp_status=status_str)
    except Exception:
        # 그 외 일시적 접근 불가 — 상태 미상(프론트는 '꺼짐'이 아닌 미상으로 처리)
        log_ccd_reading(True,None, None)
        return CCDStatusResponse(connected=True, temperature=None, temp_status=None)


@router.post("/ccd/connect")
async def ccd_connect(body: CCDConnectRequest, state: StateDep):
    """CCD 는 기동 시 lifespan 이 이미 백그라운드로 연결한다. 여기는 실패 후 재시도용."""
    if state.hw.ccd is not None:
        return {"ok": True, "message": "CCD 연결됨 (냉각 진행 중 또는 완료)"}
    if state.ccd_state == CcdInitState.FAILED:
        threading.Thread(
            target=cool_ccd_in_background, args=(state,), daemon=True, name="ccd-retry",
        ).start()
        return {"ok": True, "message": "CCD 재초기화 시작"}
    # COOLING / IDLE: lifespan 백그라운드 스레드가 처리 중
    return {"ok": True, "message": "CCD 초기화 진행 중 (/api/ccd/status 에서 온도 확인)"}


# ══════════════════════════════════════════════════════════════════════════════
# 카메라 스트리밍
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/camera/stream")
async def camera_stream(request: Request, state: StateDep):
    """MJPEG 스트리밍 — <img src="/api/camera/stream"> 으로 표시."""
    import cv2
    import numpy as np

    camera = state.hw.camera
    if camera is None:
        raise HTTPException(status_code=503, detail="카메라 미연결")

    async def generate():
        """
        [왜 매 루프마다 is_disconnected()를 확인하는가 — 좀비 제너레이터 방지]
        Starlette은 제너레이터가 yield해서 소켓에 쓰려고 할 때 비로소 클라이언트
        접속 종료를 알아챈다. 그런데 카메라가 프레임을 못 주면(get_latest_frame이
        None) 아래 루프는 yield를 한 번도 하지 않는다 → 브라우저가 떠나도 그 사실을
        영영 모른 채 while True를 계속 돌며 50ms마다 공용 스레드풀에 작업을 밀어 넣는다.
        새로고침할 때마다 이런 좀비가 하나씩 쌓이고, 워커가 마르면 채팅 요청
        (/api/experiment/stream)이 줄을 서서 수 분씩 지연됐다.
        → 프레임 유무와 무관하게 접속 종료를 직접 확인하고 빠져나온다.
        """
        while True:
            if await request.is_disconnected():
                break
            try:
                frame = await run_in_worker(state, camera.get_latest_frame)
            except Exception as e:
                print(f"[CAM STREAM] 프레임 오류: {e}")
                await asyncio.sleep(0.2)
                continue
            if frame is not None:
                if frame.dtype == np.uint16:
                    frame = (frame >> 8).astype(np.uint8)
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# ══════════════════════════════════════════════════════════════════════════════
# 장비 조작 / 상태
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/stage/move-pixel")
async def stage_move_pixel(body: MovePixelRequest, state: StateDep):
    """카메라 뷰 클릭 픽셀 좌표 → 스테이지 상대 이동.

    변환은 optics_map 단일 출처를 쓴다. 예전에는 여기서 CALIB_X=1.4, CALIB_Y=1.285 를
    **하드코딩**해서 config.CALIB_FACTOR_* 와 우연히 같을 뿐이었다 — 보정값을 다시 재면
    프론트 클릭 이동만 옛 값으로 남는다.
    """
    from backend.tools.hw_tools.hw_tools.stage_tools import move_stage_relative
    from backend.service.vision import optics_map

    dx_mm, dy_mm = optics_map.pixel_delta_to_mm(
        body.px - body.stream_width  / 2,
        body.py - body.stream_height / 2,
        width=body.stream_width, height=body.stream_height,
    )
    return await run_in_worker(state, lambda: move_stage_relative(dx=dx_mm, dy=dy_mm))


@router.get("/hardware/state")
async def hardware_state(state: StateDep):
    """현재 하드웨어 파라미터 조회 — 프론트엔드 파라미터 패널 동기화용.

    [왜 도구 계층 함수를 쓰는가]
    같은 값을 읽는 코드가 여기에 또 있었다(get_ccd_info / get_laser_status /
    get_stage_position / get_stage_speed 에 이은 다섯 번째 경로). 필드 폴백 규칙이
    미묘하게 달라 프론트와 에이전트가 다른 숫자를 보게 되므로, stage_tools 의
    hardware_snapshot 하나로 모았다.
    """
    from backend.tools.hw_tools.hw_tools.system_tools import hardware_snapshot
    return await run_in_worker(state, lambda: hardware_snapshot(state.hw))


@router.post("/stage/speed")
async def stage_set_speed(body: StageSpeedRequest, state: StateDep):
    """스테이지 이동 속도 직접 설정 — 파라미터 패널 수동 변경용."""
    from backend.tools.hw_tools.hw_tools.stage_tools import set_stage_speed
    return await run_in_worker(state, lambda: set_stage_speed(
        x_speed_mm_s=body.x, y_speed_mm_s=body.y, z_speed_mm_s=body.z))


@router.post("/spectrum/acquire")
async def spectrum_acquire(body: AcquireSpectrumRequest, state: StateDep):
    """파라미터 패널 수동 측정 버튼용 — LLM 없이 acquire_spectrum() 직접 호출."""
    from backend.tools.hw_tools.hw_tools.acquire_tools import acquire_spectrum

    result = await run_in_worker(state, lambda: acquire_spectrum(
        exposure=body.exposure,
        power=body.power,
        acq_mode=body.acq_mode,
        num_accumulations=body.num_accumulations,
        kinetic_count=body.kinetic_count,
        kinetic_cycle_time=body.kinetic_cycle_time,
        read_mode=body.read_mode,
        hbin=body.hbin,
        single_track_center=body.single_track_center,
        single_track_width=body.single_track_width,
        trigger_mode=body.trigger_mode,
    ))
    if not result.get("ok"):
        # 에이전트가 이미 측정/스캔 중이면 acquire_spectrum 이 장비 락을 얻지 못하고
        # busy_with 를 실은 에러를 돌려준다. 장비 고장이 아니라 '순서를 기다려야 한다'
        # 이므로 409 로 구분해 알린다.
        if result.get("busy_with"):
            raise HTTPException(status_code=409, detail=result.get("error", "장비 사용 중"))
        raise HTTPException(status_code=500, detail=result.get("error", "측정 실패"))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 헬스체크
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/health")
async def health(state: StateDep):
    hw = state.hw
    return {
        "status": "ok",
        "hardware": {
            "stage":  hw.stage  is not None,
            "ccd":    hw.ccd    is not None,
            "camera": hw.camera is not None,
            "laser":  hw.laser  is not None,
            "llm":    hw.ollama is not None,
        },
        "ccd_init_state": state.ccd_state,
    }
