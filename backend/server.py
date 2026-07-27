"""
Raman GPT — FastAPI 백엔드 서버

실행:
    python -m backend.server          (프로젝트 루트에서)
    또는
    cd backend && python server.py

포트: 8000
프론트엔드(Vite, 3000)는 /api/* 를 http://localhost:8000 으로 프록시.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_BACKEND = Path(__file__).resolve().parent
_PROJECT_ROOT = _BACKEND.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── FastAPI ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── 백엔드 모듈 ────────────────────────────────────────────────────────────────
from backend.hardware_manager import HardwareManager
import backend.llm_client as llm_client
from backend.config import CAMERA_WIDTH, CAMERA_HEIGHT, LENS_WIDTH_UM, LENS_HEIGHT_UM
from backend.hw_tools.raman_tools import init_hardware as rt_init_hardware


# ══════════════════════════════════════════════════════════════════════════════
# 앱 상태
# ══════════════════════════════════════════════════════════════════════════════

class CcdInitState(str, Enum):
    IDLE       = "idle"
    COOLING    = "cooling"
    STABILIZED = "stabilized"
    FAILED     = "failed"


@dataclass
class AppState:
    hw: HardwareManager = field(default_factory=HardwareManager)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=4)
    )
    ccd_state: CcdInitState = CcdInitState.IDLE
    ccd_error: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan (시작 / 종료)
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    app.state.s = state
    loop = asyncio.get_event_loop()

    # ── Phase 1: 빠른 장비 (스테이지/카메라/레이저/Ollama) ──
    fast_inits = [
        ("STAGE", state.hw._init_stage),
        ("CAM",   state.hw._init_camera),
        ("LASER", state.hw._init_laser),
        ("LLM",   state.hw._init_ollama),
    ]
    for name, fn in fast_inits:
        try:
            await loop.run_in_executor(state.executor, fn)
            print(f"[SERVER] {name} 초기화 완료")
        except Exception as e:
            print(f"[WARN]   {name} 초기화 실패 (건너뜀): {e}")

    if state.hw.ollama is not None:
        llm_client.setup(state.hw.ollama)

    # CCD 없이도 스테이지/레이저/카메라 툴은 즉시 사용 가능
    rt_init_hardware(
        stage=state.hw.stage,
        laser=state.hw.laser,
        ccd=None,
        camera=state.hw.camera,
    )

    # ── Phase 2: CCD — 백그라운드 스레드 (fire-and-forget) ──
    # _init_ccd()는 냉각기 ON 직후 self.ccd를 할당하므로
    # 폴링 중에도 /api/ccd/status 에서 온도 조회 가능
    def _ccd_background():
        try:
            state.ccd_state = CcdInitState.COOLING
            state.hw._init_ccd()          # 내부에서 self.ccd 즉시 할당 후 -40°C 대기
            state.ccd_state = CcdInitState.STABILIZED
            # CCD 안정화 후 툴 디스패치 갱신
            rt_init_hardware(
                stage=state.hw.stage,
                laser=state.hw.laser,
                ccd=state.hw.ccd,
                camera=state.hw.camera,
            )
            print("[SERVER] CCD 안정화 완료 — 툴 디스패치 갱신")
        except Exception as e:
            state.ccd_state = CcdInitState.FAILED
            state.ccd_error = str(e)
            print(f"[ERROR]  CCD 초기화 실패: {e}")

    threading.Thread(target=_ccd_background, daemon=True, name="ccd-init").start()

    print("[SERVER] 준비 완료 — http://localhost:8000  (CCD 냉각 진행 중)")
    yield   # ── 서버 실행 중 ──

    # ── 종료 ──
    state.executor.shutdown(wait=False)
    threading.Thread(
        target=state.hw.shutdown, daemon=False, name="hw-shutdown"
    ).start()
    print("[SERVER] 종료 신호 수신 — CCD 온도 복구 후 종료됩니다.")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI 앱
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Raman GPT API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 측정 결과(png/csv/json) 정적 서빙 — 채팅 인라인 표시·다운로드용.
# vite proxy 가 /api 만 통과시키므로 /api/results 아래에 둔다(spectrum_store.URL_PREFIX 와 일치).
from fastapi.staticfiles import StaticFiles          # noqa: E402
from backend.spectrum_store import RESULTS_ROOT       # noqa: E402
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/api/results", StaticFiles(directory=str(RESULTS_ROOT)), name="results")


def _state(request: Request) -> AppState:
    return request.app.state.s


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic 모델
# ══════════════════════════════════════════════════════════════════════════════

class CameraConnectRequest(BaseModel):
    exposure_ms: float = 10.0

class StageConnectRequest(BaseModel):
    dll_path: str = ""

class LaserConnectRequest(BaseModel):
    port: str = "COM4"
    baud: int = 115200

class CCDConnectRequest(BaseModel):
    target_temp: int = -40

class CCDStatusResponse(BaseModel):
    connected: bool
    temperature: Optional[int]
    temp_status: Optional[str]

class MovePixelRequest(BaseModel):
    px: float
    py: float
    stream_width: int = CAMERA_WIDTH
    stream_height: int = CAMERA_HEIGHT

class AcquireSpectrumRequest(BaseModel):
    exposure: float = 0.2
    power: int = 40
    acq_mode: str = 'single'
    num_accumulations: int = 1
    kinetic_count: int = 1
    kinetic_cycle_time: Optional[float] = None
    read_mode: str = 'fvb'
    hbin: int = 1
    single_track_center: Optional[int] = None
    single_track_width: int = 1
    trigger_mode: str = 'internal'

class StageSpeedRequest(BaseModel):
    x: float = 5.0
    y: float = 5.0
    z: float = 5.0

class ExperimentRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    # 어떤 에이전트 아키텍처로 실행할지 선택. 기본값 "AILA"(ReAct baseline)라
    # 이 필드를 보내지 않던 기존 프론트엔드/벤치마크는 그대로 동작한다.
    # "CoALA"를 주면 single_agent_CoALA(의사결정 사이클 + 장기기억)로 라우팅된다.
    agent: Optional[str] = "AILA"


def _agent_module(name: Optional[str]):
    """agent 이름 → (에이전트 모듈, 정규화된 이름).

    AILA↔CoALA를 가르는 판단은 이 함수 '하나'에만 있다 — 새 에이전트를 추가하거나
    분기 규칙을 바꿀 때 여기만 고치면 라우트는 손대지 않는다. 두 모듈 모두 동일 공개
    API(ALL_TOOLS / stream_experiment / run_experiment)를 노출하므로 호출부는 동일하다.
    알 수 없는 값은 기본 AILA로 폴백한다(회귀 방지)."""
    if (name or "").strip().upper() == "COALA":
        from backend.agents import single_agent_CoALA as mod
        return mod, "CoALA"
    from backend.agents import single_agent_AILA as mod
    return mod, "AILA"


# ══════════════════════════════════════════════════════════════════════════════
# 장비 초기화 엔드포인트
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/camera/connect")
async def camera_connect(body: CameraConnectRequest, request: Request):
    state = _state(request)
    hw = state.hw
    if hw.camera is not None:
        return {"ok": True, "message": "카메라 이미 연결됨"}
    loop = asyncio.get_event_loop()
    try:
        def _connect():
            from backend.hw_tools.USE_camera_stream import StreamingTUCam
            cam = StreamingTUCam(exposure_ms=body.exposure_ms)
            cam.start_stream()
            hw.camera = cam
            rt_init_hardware(stage=hw.stage, laser=hw.laser, ccd=hw.ccd, camera=hw.camera)
        await loop.run_in_executor(state.executor, _connect)
        return {"ok": True, "message": "카메라 연결 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stage/connect")
async def stage_connect(body: StageConnectRequest, request: Request):
    state = _state(request)
    hw = state.hw
    if hw.stage is not None:
        return {"ok": True, "message": "스테이지 이미 연결됨"}
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(state.executor, hw._init_stage)
        rt_init_hardware(stage=hw.stage, laser=hw.laser, ccd=hw.ccd, camera=hw.camera)
        return {"ok": True, "message": "스테이지 연결 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/laser/connect")
async def laser_connect(body: LaserConnectRequest, request: Request):
    state = _state(request)
    hw = state.hw
    if hw.laser is not None:
        return {"ok": True, "message": "레이저 이미 연결됨"}
    loop = asyncio.get_event_loop()
    try:
        def _connect():
            from backend.hw_tools.USE_laser_with_power import LaserController
            laser = LaserController(port=body.port)
            if not (laser.ser and laser.ser.is_open):
                raise RuntimeError(f"레이저 포트 연결 실패 ({body.port})")
            hw.laser = laser
            rt_init_hardware(stage=hw.stage, laser=hw.laser, ccd=hw.ccd, camera=hw.camera)
        await loop.run_in_executor(state.executor, _connect)
        return {"ok": True, "message": "레이저 연결 완료"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ccd/status", response_model=CCDStatusResponse)
async def ccd_status(request: Request):
    state = _state(request)
    ccd = state.hw.ccd

    if ccd is None:
        return CCDStatusResponse(connected=False, temperature=None, temp_status=None)

    loop = asyncio.get_event_loop()
    try:
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
        temp, status_str = await loop.run_in_executor(state.executor, _read)
        return CCDStatusResponse(connected=True, temperature=temp, temp_status=status_str)
    except Exception:
        # 그 외 일시적 접근 불가 — 상태 미상(프론트는 '꺼짐'이 아닌 미상으로 처리)
        return CCDStatusResponse(connected=True, temperature=None, temp_status=None)


@app.post("/api/ccd/connect")
async def ccd_connect(body: CCDConnectRequest, request: Request):
    state = _state(request)
    if state.hw.ccd is not None:
        return {"ok": True, "message": "CCD 연결됨 (냉각 진행 중 또는 완료)"}
    if state.ccd_state == CcdInitState.FAILED:
        # 실패 시 재시도
        def _retry():
            try:
                state.ccd_state = CcdInitState.COOLING
                state.hw._init_ccd()
                state.ccd_state = CcdInitState.STABILIZED
                rt_init_hardware(
                    stage=state.hw.stage, laser=state.hw.laser,
                    ccd=state.hw.ccd, camera=state.hw.camera,
                )
            except Exception as e:
                state.ccd_state = CcdInitState.FAILED
                state.ccd_error = str(e)
        threading.Thread(target=_retry, daemon=True, name="ccd-retry").start()
        return {"ok": True, "message": "CCD 재초기화 시작"}
    # COOLING / IDLE: lifespan 백그라운드 스레드가 처리 중
    return {"ok": True, "message": "CCD 초기화 진행 중 (/api/ccd/status 에서 온도 확인)"}


# ══════════════════════════════════════════════════════════════════════════════
# AI 에이전트 엔드포인트
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/experiment/run")
async def experiment_run(body: ExperimentRequest, request: Request):
    """단일 gemma4 에이전트 동기 실행 (세션 히스토리 없이 1회, 벤치마크/레거시용)."""
    mod, _ = _agent_module(body.agent)
    run_experiment = mod.run_experiment
    state = _state(request)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            state.executor,
            lambda: run_experiment(body.message, body.session_id or ""),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/experiment/stream")
async def experiment_stream(body: ExperimentRequest, request: Request):
    """단일 gemma4 에이전트 — SSE 스트리밍(도구 호출 진행상황 + 세션 대화 기억).

    프론트엔드는 fetch로 이 엔드포인트를 열고 ReadableStream을 파싱한다.
    이벤트 형식(표준 SSE):
        event: <type>\\n
        data:  <json>\\n\\n
    type ∈ {chat, node, done, error}. (single_agent.stream_experiment 참고)

    [동기 제너레이터를 async SSE로 잇는 방법]
    stream_experiment는 내부에서 Ollama 호출을 블로킹 실행하는 "동기" 제너레이터다.
    이를 이벤트 루프에서 직접 돌리면 서버가 멈추므로, ThreadPoolExecutor 워커에서
    돌리고 각 이벤트를 asyncio.Queue로 넘겨 async 쪽에서 흘려보낸다.
    """
    mod, _ = _agent_module(body.agent)
    stream_experiment = mod.stream_experiment

    state = _state(request)
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _producer():
        # 워커 스레드: 동기 제너레이터를 소비해 큐로 밀어넣는다.
        try:
            for event in stream_experiment(body.message, body.session_id or ""):
                loop.call_soon_threadsafe(q.put_nowait, event)
        except Exception as e:  # 방어적 — stream_experiment가 자체적으로 error 이벤트를 내지만
            loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "detail": str(e)})
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

    state.executor.submit(_producer)

    async def event_gen():
        import json
        while True:
            event = await q.get()
            if event is _SENTINEL:
                break
            etype = event.get("type", "message")
            payload = json.dumps(event, ensure_ascii=False)
            yield f"event: {etype}\ndata: {payload}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # nginx/프록시 버퍼링 방지 (SSE 즉시 전달)
        },
    )


@app.get("/api/agents/health")
async def agents_health(agent: str = "AILA"):
    """단일 에이전트 시스템 상태 확인. agent 쿼리로 AILA/CoALA 중 선택(기본 AILA)."""
    mod, name = _agent_module(agent)
    return {
        "status": "ok",
        "agents": [f"single_agent_{name}"],
        "model": "gemma4 (ollama)",
        "tools": len(mod.ALL_TOOLS),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 카메라 스트리밍 / 스테이지 클릭 이동 / 하드웨어 상태 / 직접 측정
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/camera/stream")
async def camera_stream(request: Request):
    """MJPEG 스트리밍 — <img src="/api/camera/stream"> 으로 표시."""
    import cv2
    import numpy as np

    state = _state(request)
    camera = state.hw.camera
    if camera is None:
        raise HTTPException(status_code=503, detail="카메라 미연결")

    async def generate():
        """
        [왜 매 루프마다 is_disconnected()를 확인하는가 — 좀비 제너레이터 방지]
        Starlette은 제너레이터가 yield해서 소켓에 쓰려고 할 때 비로소 클라이언트
        접속 종료를 알아챈다. 그런데 카메라가 프레임을 못 주면(get_latest_frame이
        None) 아래 루프는 yield를 한 번도 하지 않는다 → 브라우저가 떠나도 그 사실을
        영영 모른 채 while True를 계속 돌며 50ms마다 공용 스레드풀
        (ThreadPoolExecutor(max_workers=4))에 작업을 밀어 넣는다.
        새로고침할 때마다 이런 좀비가 하나씩 쌓이고, 워커가 마르면 채팅 요청
        (/api/experiment/stream의 _producer)이 줄을 서서 수 분씩 지연됐다.
        → 프레임 유무와 무관하게 접속 종료를 직접 확인하고 빠져나온다.
        """
        loop = asyncio.get_event_loop()
        while True:
            if await request.is_disconnected():
                break
            try:
                frame = await loop.run_in_executor(state.executor, camera.get_latest_frame)
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


@app.post("/api/stage/move-pixel")
async def stage_move_pixel(body: MovePixelRequest, request: Request):
    """카메라 뷰 클릭 픽셀 좌표 → 스테이지 상대 이동.
    USE_scan.py의 보정 상수를 그대로 사용한다."""
    from backend.hw_tools.raman_tools import move_stage_relative

    LENS_W_UM, LENS_H_UM = LENS_WIDTH_UM, LENS_HEIGHT_UM
    CALIB_X,   CALIB_Y   = 1.4, 1.285
    SIGN_X,    SIGN_Y     = -1, 1

    um_per_px_x = LENS_W_UM / body.stream_width
    um_per_px_y = LENS_H_UM / body.stream_height
    dx_mm = (body.px - body.stream_width  / 2) * um_per_px_x * CALIB_X / 1000.0 * SIGN_X
    dy_mm = (body.py - body.stream_height / 2) * um_per_px_y * CALIB_Y / 1000.0 * SIGN_Y

    state = _state(request)
    loop  = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        state.executor,
        lambda: move_stage_relative(dx=dx_mm, dy=dy_mm),
    )
    return result


@app.get("/api/hardware/state")
async def hardware_state(request: Request):
    """현재 하드웨어 파라미터 조회 — 프론트엔드 파라미터 패널 동기화용.

    [왜 executor로 오프로드하는가]
    아래 stage.get_position/get_velocity, ccd.get_temperature 는 직렬 통신·SDK 호출로
    수십~수백 ms 블로킹된다. async 핸들러에서 직접 부르면 그동안 이벤트 루프가 멈춰,
    같은 루프에서 도는 카메라 MJPEG 스트림(/api/camera/stream)이 끊긴다. 프론트가 이
    엔드포인트를 주기적으로 폴링하므로, 하드웨어 읽기를 스레드풀로 내려 루프를 막지 않는다.
    (getattr 캐시 읽기는 값싸지만 한 번에 같이 내려 코드를 단순하게 둔다.)
    """
    state = _state(request)
    hw = state.hw

    def _read() -> dict:
        out: dict = {"ccd": None, "laser": None, "stage": None}

        if hw.ccd is not None:
            ccd = hw.ccd
            ccd_info: dict = {
                "exposure_time": getattr(ccd, "exposure_time", None),
                "acq_mode":      getattr(ccd, "aq_mode",       "single"),
                "num_acc":       getattr(ccd, "num_acc",        1),
                "num_kin":       getattr(ccd, "num_kin",        1),
                "ro_mode":       getattr(ccd, "ro_mode",        "fvb"),
                "preamp_gain_i": getattr(ccd, "preamp_gain_i",  0),
                "preamp_gains":  getattr(ccd, "preamp_gains",   []),
                "shutter":       getattr(ccd, "shutter_mode",   "auto"),
                "temperature":   None,
            }
            try:
                ccd_info["temperature"] = int(ccd.get_temperature())
            except Exception:
                pass
            out["ccd"] = ccd_info

        if hw.laser is not None:
            out["laser"] = {
                "power_pct": getattr(hw.laser, "power_pct", None),
                "is_on":     getattr(hw.laser, "is_on",     None),
            }

        if hw.stage is not None:
            try:
                pos = hw.stage.get_position()
                out["stage"] = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
            except Exception:
                out["stage"] = {}
            try:
                # get_velocity() 는 dict 반환 {"ok",x_speed_mm_s,y_speed_mm_s,z_speed_mm_s,...}.
                # (이전 버그: vel[0] 로 dict 를 정수 인덱싱 → 항상 예외 → velocity 미기록)
                vel = hw.stage.get_velocity()
                if out["stage"] is None:
                    out["stage"] = {}
                if isinstance(vel, dict) and vel.get("ok"):
                    out["stage"]["velocity"] = {
                        "x": float(vel["x_speed_mm_s"]),
                        "y": float(vel["y_speed_mm_s"]),
                        "z": float(vel["z_speed_mm_s"]),
                    }
            except Exception:
                pass

        return out

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(state.executor, _read)


@app.post("/api/stage/speed")
async def stage_set_speed(body: StageSpeedRequest, request: Request):
    """스테이지 이동 속도 직접 설정 — 파라미터 패널 수동 변경용."""
    from backend.hw_tools.raman_tools import set_stage_speed

    state = _state(request)
    loop  = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        state.executor,
        lambda: set_stage_speed(x_speed_mm_s=body.x, y_speed_mm_s=body.y, z_speed_mm_s=body.z),
    )
    return result


@app.post("/api/spectrum/acquire")
async def spectrum_acquire(body: AcquireSpectrumRequest, request: Request):
    """파라미터 패널 수동 측정 버튼용 — LLM 없이 acquire_spectrum() 직접 호출."""
    from backend.hw_tools.raman_tools import acquire_spectrum

    state = _state(request)
    loop  = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        state.executor,
        lambda: acquire_spectrum(
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
        ),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail=result.get("error", "측정 실패"))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 첨부 데이터 파일 엔드포인트
# ══════════════════════════════════════════════════════════════════════════════
#
# 채팅창에 붙인 측정 데이터(csv/excel/txt)를 받는다. 아래 KB 업로드와 목적이 다르다:
#   /api/kb/upload    문서(pdf/md/…)를 지식베이스에 색인 — '프로토콜을 가르친다'
#   /api/files/upload 표 데이터를 에이전트가 분석 — '이 데이터를 봐 달라'
# 에이전트는 이 HTTP API를 쓰지 않는다. list_uploaded_files / inspect_file 도구로
# backend.upload_store를 직접 부른다(파일 위치 규칙은 upload_store 머리말 참고).

@app.post("/api/files/upload")
async def files_upload_endpoint(file: UploadFile = File(...)):
    """데이터 파일을 data/uploads/<날짜>/에 저장하고 file_id를 돌려준다.

    파싱은 여기서 하지 않는다 — 큰 파일이면 업로드 응답이 그만큼 늦어지고, 애초에
    '어떻게 읽을지'는 에이전트가 inspect_file로 판단할 몫이다. 여기는 저장만 한다.
    """
    from backend.upload_store import ALLOWED_SUFFIXES, save_upload

    name = Path(file.filename or "").name       # 경로 탈출 방지 — 파일명만 취한다
    if not name:
        raise HTTPException(status_code=400, detail="파일명이 없습니다")
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(f"지원하지 않는 형식입니다. 가능: {', '.join(sorted(ALLOWED_SUFFIXES))} "
                    f"(논문·프로토콜 문서는 /api/kb/upload 로)"),
        )
    try:
        data = await file.read()
        return save_upload(name, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장 실패: {e}")


@app.get("/api/files")
async def files_list_endpoint(date: Optional[str] = None):
    """해당 날짜(기본 오늘)에 올라온 첨부 파일 목록 — 프론트 표시/디버깅용."""
    from backend.upload_store import list_uploads
    return {"ok": True, "files": list_uploads(date)}


# ══════════════════════════════════════════════════════════════════════════════
# 지식베이스(KB) 엔드포인트
# ══════════════════════════════════════════════════════════════════════════════
#
# 에이전트는 이 HTTP API를 쓰지 않는다 — 에이전트는 search_knowledge_base 도구로
# knowledge.search_kb()를 직접 호출한다. 여기 있는 건 사람용이다:
#   운영: 문서 업로드 → 재색인
#   디버깅: 에이전트가 뭘 검색해 오는지 눈으로 확인
#   실험 전 점검: 지금 벡터 검색인지 키워드 폴백인지 확인 (가장 중요)

@app.get("/api/kb/status")
async def kb_status_endpoint():
    """KB 진단 — 어느 검색기가 살아있고 인덱스에 몇 개 들었는지.

    ⚠ 벤치마크를 돌리기 전에 반드시 확인할 것. retriever가 "keyword"인데 그걸
    모른 채 실험하면 "벡터 RAG를 붙인 결과"라고 쓴 게 전부 거짓이 된다.
    """
    from backend.agents.knowledge import kb_status
    return kb_status()


@app.get("/api/kb/search")
async def kb_search_endpoint(q: str, top_k: int = 3):
    """에이전트와 똑같은 경로로 KB를 검색해 본다(디버깅용).

    에이전트가 이상한 파라미터를 고를 때, 프롬프트 탓인지 검색 탓인지 가르는 데 쓴다.
    """
    from backend.agents.knowledge import search_kb
    if not q.strip():
        raise HTTPException(status_code=400, detail="q가 비어 있습니다")
    hits = search_kb(q, top_k=top_k)
    return {
        "query": q,
        "count": len(hits),
        # 항목별 _retriever와 별개로, 응답 수준에서도 한 번 더 노출한다.
        "retriever": hits[0].get("_retriever") if hits else None,
        "results": hits,
    }


@app.post("/api/kb/reload")
async def kb_reload_endpoint():
    """캐시를 비워 다음 조회 때 디스크를 다시 읽게 한다.

    knowledge_base.json을 고쳤을 때 서버를 껐다 켜지 않아도 되게 하는 용도.
    (Chroma 인덱스 자체를 갱신하려면 /api/kb/reindex가 필요하다 — 이건 캐시만 비운다.)
    """
    from backend.agents.knowledge import kb_status, reload_kb
    reload_kb()
    return {"ok": True, "message": "KB 캐시를 비웠습니다", "status": kb_status()}


@app.post("/api/kb/upload")
async def kb_upload_endpoint(file: UploadFile = File(...)):
    """문서를 kb_sources/에 저장한다. 색인은 하지 않는다.

    [왜 업로드와 색인을 분리하나]
    색인은 문서 수에 비례해 임베딩을 돌리므로 수 초~수 분이 걸린다. 업로드 요청을
    그동안 붙잡아 두면 프론트가 타임아웃난다. 여러 파일을 올린 뒤 /api/kb/reindex를
    한 번 부르는 게 임베딩 왕복도 줄인다.
    """
    from backend.agents.knowledge import KB_SOURCES_DIR

    allowed = {".pdf", ".txt", ".md", ".json"}
    name = Path(file.filename or "").name          # 경로 탈출 방지 — 파일명만 취한다
    if not name:
        raise HTTPException(status_code=400, detail="파일명이 없습니다")
    if Path(name).suffix.lower() not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 형식입니다. 가능: {', '.join(sorted(allowed))}",
        )

    KB_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    dest = KB_SOURCES_DIR / name
    try:
        dest.write_bytes(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장 실패: {e}")

    return {
        "ok": True,
        "filename": name,
        "bytes": dest.stat().st_size,
        "message": "저장됨. 검색에 반영하려면 POST /api/kb/reindex 를 호출하세요.",
    }


@app.post("/api/kb/reindex")
async def kb_reindex_endpoint(request: Request, caption: bool = False):
    """kb_sources/와 knowledge_base.json을 다시 읽어 Chroma를 재색인한다.

    caption=true면 PDF 페이지를 gemma4로 캡션한다(페이지당 VLM 1회 — 매우 느림).

    색인은 블로킹 작업(임베딩 HTTP 왕복 다수)이라 워커 스레드에서 돌린다.
    """
    from backend.agents.kb_ingest import ingest
    from backend.agents.knowledge import kb_status, reload_kb

    state = _state(request)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            state.executor,
            lambda: ingest(caption=caption, with_spectra=True),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"색인 실패: {e}")

    # 색인이 컬렉션을 지웠다 다시 만들었으므로, 검색 쪽이 들고 있는 낡은 핸들을 버린다.
    reload_kb()
    return {"ok": True, "indexed": result, "status": kb_status()}


# ── 헬스체크 ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health(request: Request):
    state = _state(request)
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


# ══════════════════════════════════════════════════════════════════════════════
# 단독 실행
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # reload=True 시 lifespan이 재실행되어 하드웨어 재초기화됨
        log_level="info",
    )
