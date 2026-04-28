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
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ── 백엔드 모듈 ────────────────────────────────────────────────────────────────
from backend.hardware_manager import HardwareManager
import backend.llm_client as llm_client
from backend.agents.raman_tool_schemas import RAMAN_TOOLS
from backend.agents.raman_tools import TOOL_DISPATCH, init_hardware as rt_init_hardware


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

class HardwareCommandRequest(BaseModel):
    command: str

class ChatRequest(BaseModel):
    message: str
    agent: str = "general"

class AutoFocusRequest(BaseModel):
    initial_z: float
    z_range: float
    z_step: float

class OptimizeRequest(BaseModel):
    sample_type: str
    purpose: str
    target_peaks: Optional[list[float]] = None

class TroubleshootRequest(BaseModel):
    issue: str = ""

class MovePixelRequest(BaseModel):
    px: float
    py: float
    stream_width: int = 1060
    stream_height: int = 800

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
            from backend.agents.USE_camera_stream import StreamingTUCam
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
            from backend.agents.USE_laser_with_power import LaserController
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
            t = ccd.get_temperature()            # temperature_status_num 도 업데이트
            s = ccd.temp_status_dict.get(ccd.temperature_status_num)
            return int(t), s
        temp, status_str = await loop.run_in_executor(state.executor, _read)
        return CCDStatusResponse(connected=True, temperature=temp, temp_status=status_str)
    except Exception:
        # 촬영 중 등 일시적 접근 불가
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

def _require_llm(state: AppState):
    if state.hw.ollama is None:
        raise HTTPException(status_code=503, detail="LLM이 연결되지 않았습니다.")


@app.post("/api/hardware-command")
async def hardware_command(body: HardwareCommandRequest, request: Request):
    state = _state(request)
    _require_llm(state)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            state.executor,
            lambda: llm_client.run_agent(
                user_message=body.command,
                tools=RAMAN_TOOLS,
                tool_dispatch=TOOL_DISPATCH,
                max_steps=20,
                verbose=True,
            ),
        )
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    state = _state(request)
    _require_llm(state)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            state.executor,
            lambda: llm_client.run_agent(
                user_message=body.message,
                tools=RAMAN_TOOLS,
                tool_dispatch=TOOL_DISPATCH,
                max_steps=20,
                verbose=True,
            ),
        )
        return {"message": result, "data": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/autofocus")
async def autofocus(body: AutoFocusRequest, request: Request):
    state = _state(request)
    _require_llm(state)
    loop = asyncio.get_event_loop()
    prompt = (
        f"자동 초점 조절을 수행하세요. "
        f"초기 Z 위치: {body.initial_z}mm, "
        f"탐색 범위: ±{body.z_range/2}mm, "
        f"이동 간격: {body.z_step}mm. "
        f"스테이지를 각 Z 위치로 이동하며 최적 초점을 찾아주세요."
    )
    try:
        result = await loop.run_in_executor(
            state.executor,
            lambda: llm_client.run_agent(
                user_message=prompt,
                tools=RAMAN_TOOLS,
                tool_dispatch=TOOL_DISPATCH,
                max_steps=30,
            ),
        )
        import numpy as np
        z_positions = list(np.arange(
            body.initial_z - body.z_range / 2,
            body.initial_z + body.z_range / 2 + body.z_step,
            body.z_step,
        ).tolist())
        optimal_z = body.initial_z
        return {
            "optimal_z": optimal_z,
            "best_score": 0.0,
            "z_positions": z_positions,
            "focus_scores": [0.0] * len(z_positions),
            "message": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize-parameters")
async def optimize_parameters(body: OptimizeRequest, request: Request):
    state = _state(request)
    _require_llm(state)
    loop = asyncio.get_event_loop()

    peaks_str = f", 목표 피크: {body.target_peaks}" if body.target_peaks else ""
    prompt = (
        f"라만 분광 실험 파라미터를 추천해 주세요.\n"
        f"- 샘플 종류: {body.sample_type}\n"
        f"- 측정 목적: {body.purpose}{peaks_str}\n"
        f"레이저 출력(%), CCD 노출 시간(초), 누적 횟수를 포함해 한국어로 답변하세요."
    )
    try:
        summary = await loop.run_in_executor(
            state.executor,
            lambda: llm_client.generate(prompt),
        )
        return {
            "summary": summary,
            "spectrometer_settings": {
                "laser_power_mw": 40,
                "grating": "1200",
                "nd_filter": 1.0,
            },
            "ccd_settings": {
                "exposure_time_s": 1.0,
                "num_accumulations": 3,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/troubleshoot")
async def troubleshoot(body: TroubleshootRequest, request: Request):
    state = _state(request)
    _require_llm(state)
    loop = asyncio.get_event_loop()
    prompt = (
        f"라만 분광기 문제 진단을 도와주세요.\n"
        f"증상: {body.issue}\n"
        f"가능한 원인과 해결책을 한국어로 설명하세요."
    )
    try:
        result = await loop.run_in_executor(
            state.executor,
            lambda: llm_client.generate(prompt),
        )
        return {"diagnosis": result, "recommendations": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/troubleshoot/upload")
async def troubleshoot_upload(
    file: UploadFile = File(...),
    request: Request = None,
):
    state = _state(request)
    _require_llm(state)
    contents = await file.read()
    loop = asyncio.get_event_loop()
    import base64
    b64 = base64.b64encode(contents).decode()
    prompt = (
        f"라만 분광기 이미지를 분석하고 문제를 진단해 주세요. "
        f"(이미지 크기: {len(contents)} bytes)\n"
        f"가능한 원인과 해결책을 한국어로 설명하세요."
    )
    try:
        result = await loop.run_in_executor(
            state.executor,
            lambda: llm_client.generate(prompt),
        )
        return {"diagnosis": result, "recommendations": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        loop = asyncio.get_event_loop()
        while True:
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
    from backend.agents.raman_tools import move_stage_relative

    LENS_W_UM, LENS_H_UM = 305.0, 230.0
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
    """현재 하드웨어 파라미터 조회 — 프론트엔드 파라미터 패널 동기화용."""
    hw   = _state(request).hw
    out  = {"ccd": None, "laser": None, "stage": None}

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
            vel = hw.stage.get_velocity()
            if out["stage"] is None:
                out["stage"] = {}
            out["stage"]["velocity"] = {"x": float(vel[0]), "y": float(vel[1]), "z": float(vel[2])}
        except Exception:
            pass

    return out


@app.post("/api/stage/speed")
async def stage_set_speed(body: StageSpeedRequest, request: Request):
    """스테이지 이동 속도 직접 설정 — 파라미터 패널 수동 변경용."""
    from backend.agents.raman_tools import set_stage_speed

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
    from backend.agents.raman_tools import acquire_spectrum

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
