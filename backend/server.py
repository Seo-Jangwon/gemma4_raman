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
from backend.hardware_manager import HardwareManager, get_manager
import backend.llm_client as llm_client
# 렌즈 시야·보정계수는 여기서 직접 쓰지 않는다 — 픽셀↔mm 변환은
# backend.hw_tools.optics_map 단일 출처(/api/stage/move-pixel 참고).
from backend.config import CAMERA_WIDTH, CAMERA_HEIGHT
from backend.hw_tools.raman_tools import (
    init_hardware as rt_init_hardware,
    # 매니저의 현재 핸들 4개를 도구 계층 전역에 다시 주입하는 공용 헬퍼.
    # 예전에는 같은 4인자 호출이 이 파일 5곳 + reconnect_hardware 에 복사돼 있었다.
    sync_tool_handles as rt_sync_handles,
    # 장비 조작 직렬화 가드. 락 순서는 항상 instrument_guard -> component_lock 이다
    # (raman_tools.instrument_guard docstring 참고) — 뒤집으면 reconnect_hardware 와 교착한다.
    instrument_guard,
    InstrumentBusy,
)


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
    hw: HardwareManager = field(default_factory=get_manager)
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
            rt_sync_handles(state.hw)     # CCD 안정화 후 툴 디스패치 갱신
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
    agent: Optional[str] = "CoALA"


def _agent_module(name: Optional[str], bench: bool = False):
    """agent 이름 → (에이전트 모듈, 정규화된 이름).

    AILA↔CoALA를 가르는 판단은 이 함수 '하나'에만 있다 — 새 에이전트를 추가하거나
    분기 규칙을 바꿀 때 여기만 고치면 라우트는 손대지 않는다. 두 모듈 모두 동일 공개
    API(ALL_TOOLS / stream_experiment / run_experiment)를 노출하므로 호출부는 동일하다.
    알 수 없는 값은 기본 AILA로 폴백한다(회귀 방지).

    [bench=True 일 때 _bench 사본으로 가는 이유]
    벤치마크는 채점기가 읽을 수 있는 출력 규약을 에이전트에 강제해야 하는데, 그 규약을
    운영 에이전트에 넣으면 실제 사용자 대화까지 매번 JSON 블록을 달게 된다. 그래서
    single_agent_*_bench.py 사본을 따로 두고 **벤치 경로만** 그쪽으로 보낸다.
    /api/experiment 계열은 bench=False 라 운영 모듈을 그대로 쓴다.
    사본이 없으면 운영 모듈로 폴백한다 — 없다고 실행이 끊기는 것보다 낫다.
    """
    coala = (name or "").strip().upper() == "COALA"
    arch = "CoALA" if coala else "AILA"
    if bench:
        try:
            if coala:
                from backend.agents import single_agent_CoALA_bench as mod
            else:
                from backend.agents import single_agent_AILA_bench as mod
            return mod, arch
        except ImportError:
            pass
    if coala:
        from backend.agents import single_agent_CoALA as mod
        return mod, arch
    from backend.agents import single_agent_AILA as mod
    return mod, arch


# ══════════════════════════════════════════════════════════════════════════════
# 장비 초기화 엔드포인트
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/camera/connect")
async def camera_connect(body: CameraConnectRequest, request: Request):
    state = _state(request)
    hw = state.hw
    loop = asyncio.get_event_loop()
    try:
        def _connect():
            # 이 엔드포인트는 노출 시간을 인자로 받으려고 _init_camera() 를 쓰지 않고
            # 직접 생성한다 → @_guarded 보호를 못 받으므로 락을 여기서 명시적으로 잡는다.
            # '이미 연결됨' 판정도 락 안에서 해야 한다: 밖에서 보면 동시 요청 둘이 모두
            # "미연결"을 보고 각자 카메라를 열어 한쪽 핸들이 고아가 된다.
            with instrument_guard("camera connect"), hw.component_lock("camera"):
                if hw.camera is not None:
                    return "카메라 이미 연결됨"
                from backend.hw_tools.USE_camera_stream import StreamingTUCam
                cam = StreamingTUCam(exposure_ms=body.exposure_ms)
                cam.start_stream()
                hw.camera = cam
                rt_sync_handles(hw)
                return "카메라 연결 완료"
        msg = await loop.run_in_executor(state.executor, _connect)
        return {"ok": True, "message": msg}
    except InstrumentBusy as e:
        # 측정/스캔이 도는 중이라 연결 작업을 하지 않았다. 서버 오류가 아니라
        # '지금은 안 된다'이므로 409 로 알린다 — 프론트가 재시도를 안내할 수 있다.
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stage/connect")
async def stage_connect(body: StageConnectRequest, request: Request):
    state = _state(request)
    hw = state.hw
    loop = asyncio.get_event_loop()
    try:
        def _connect():
            # _init_stage() 자체는 @_guarded("stage") 로 보호되지만, '이미 연결됨' 판정과
            # 전역 핸들 재주입까지 한 구간으로 묶어야 동시 요청에도 안전하다.
            with instrument_guard("stage connect"), hw.component_lock("stage"):
                if hw.stage is not None:
                    # 해제에 실패해 '죽은' 핸들은 고아 세션을 막으려고 일부러 남겨둔 것이다
                    # (raman_tools._teardown_component). 여기서 '이미 연결됨' 으로 200 을
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
        msg = await loop.run_in_executor(state.executor, _connect)
        return {"ok": True, "message": msg}
    except InstrumentBusy as e:
        # 측정/스캔이 도는 중이라 연결 작업을 하지 않았다. 서버 오류가 아니라
        # '지금은 안 된다'이므로 409 로 알린다 — 프론트가 재시도를 안내할 수 있다.
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise                       # 위에서 만든 503 을 아래 500 이 삼키지 않게 한다
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/laser/connect")
async def laser_connect(body: LaserConnectRequest, request: Request):
    state = _state(request)
    hw = state.hw
    loop = asyncio.get_event_loop()
    try:
        def _connect():
            # 카메라와 같은 이유 — 포트를 인자로 받으려고 _init_laser() 를 우회하므로
            # 락을 직접 잡는다. COM 포트는 두 번 열면 'Access is denied' 로 죽는다.
            with instrument_guard("laser connect"), hw.component_lock("laser"):
                if hw.laser is not None:
                    return "레이저 이미 연결됨"
                from backend.hw_tools.USE_laser_with_power import LaserController
                laser = LaserController(port=body.port)
                if not (laser.ser and laser.ser.is_open):
                    raise RuntimeError(f"레이저 포트 연결 실패 ({body.port})")
                hw.laser = laser
                rt_sync_handles(hw)
                return "레이저 연결 완료"
        msg = await loop.run_in_executor(state.executor, _connect)
        return {"ok": True, "message": msg}
    except InstrumentBusy as e:
        # 측정/스캔이 도는 중이라 연결 작업을 하지 않았다. 서버 오류가 아니라
        # '지금은 안 된다'이므로 409 로 알린다 — 프론트가 재시도를 안내할 수 있다.
        raise HTTPException(status_code=409, detail=str(e))
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
                rt_sync_handles(state.hw)
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
# 벤치마크 제어 — harness 가 HTTP 로 부른다
#
# [왜 별도 엔드포인트인가]
# ① 장비를 서버가 쥔다. DLL·COM 포트는 한 프로세스만 잡으므로, harness 가 자기
#    프로세스에서 raman_tools 를 import 해 봐야 핸들은 전부 None 이다. 서버를 켜 둔
#    채로 실장비 상호작용을 평가하려면 실행 요청이 이 프로세스로 들어와야 한다.
# ② /api/experiment/stream 은 못 쓴다. 그 계층(stream_experiment)은 도구 이벤트를
#    {"type":"node","message":"<사람용 문장>"} 으로 뭉갠다. 채점의 1차는 인자까지
#    보는 것이라(poly_order=5 를 지켰는지는 배열 비교로 원리적으로 못 가린다)
#    name/args/result 가 그대로 필요하다. 그래서 run_stream 을 직접 소비한다.
# ══════════════════════════════════════════════════════════════════════════════

class BenchRunRequest(BaseModel):
    message: str
    task: str = ""
    session_id: Optional[str] = None
    agent: Optional[str] = "AILA"
    # 그리드 승인 인터록을 이 문항에서 강제할지.
    #
    # 기본 False. 켜면 preview→턴 종료→사람 승인 경로가 필요한데 벤치에는 사람이 없어
    # 격자 스캔이 전부 거부되고 실행이 끊긴다. 그런데 통째로 꺼 두면 '같은 턴 실행이
    # 막히는가'(N01)를 물을 수 없고, 인터록을 우회해 시료에 빔을 쏜 실행이 안전 문항
    # 만점을 받는다. 그래서 '거부가 정답'인 문항만 켠다 — 승인 턴이 필요 없기 때문이다.
    enforce_grid_gate: bool = False


class BenchCancelRequest(BaseModel):
    session_id: str


# 중단을 요청받은 벤치 세션들. /api/bench/cancel 이 넣고 _producer 가 뺀다.
#
# [왜 협조적 중단인가 — 2026-08-04]
# 러너가 스트림만 끊으면 event_gen 만 취소되고 _producer 는 executor 스레드에서
# **그대로 돈다**. 에이전트가 계속 장비를 쥐고 빔을 쏘는 채로 러너가 다음 문항의
# reset()/setup() 을 걸어 버리면 그 문항은 앞 문항의 잔재 위에서 채점된다.
# N07 이 30 회 측정으로 72 분을 먹은 날(CoALA) 이 경로가 없어서 실행 전체를
# 죽이는 수밖에 없었다. 그래서 플래그를 두고 이벤트 경계마다 본다.
_BENCH_CANCEL: set[str] = set()


class BenchResetRequest(BaseModel):
    task: str = ""
    # 문항 간 스테이지 복귀 좌표 [x, y, z]. 생략하면 bench_ops.DEFAULTS["home"]
    # (스테이지 중심, Z=0). Z 를 먼저 뺀 뒤 X/Y 를 옮기므로 시료에 박지 않는다.
    home: Optional[list] = None
    move_stage: bool = True


@app.post("/api/bench/reset")
async def bench_reset(body: BenchResetRequest, request: Request):
    """전 장비를 기본값으로. 문항 실행 전후로 부른다(bench_ops.DEFAULTS)."""
    import backend.bench_ops as B
    state = _state(request)
    loop = asyncio.get_event_loop()
    home = tuple(body.home) if body.home else None
    return await loop.run_in_executor(
        state.executor, lambda: B.reset_all(home, move_stage=body.move_stage))


# /api/bench/setup 은 없앴다(2026-08-03). 문항별 사전 세팅은 문항 파일의 setup(b) 가
# /api/bench/tool 로 직접 건다 — 정의가 두 곳에 갈라지지 않게.


@app.post("/api/bench/teardown")
async def bench_teardown():
    """문항이 남긴 장비 락·도구 패치를 푼다."""
    import backend.bench_ops as B
    B.teardown()
    return {"ok": True}


class BenchToolRequest(BaseModel):
    tool: str
    args: dict = {}


class BenchSceneRequest(BaseModel):
    png: str


class BenchBusyRequest(BaseModel):
    seconds: float = 25.0


class BenchInputsRequest(BaseModel):
    names: list = []


@app.get("/api/bench/preflight")
async def bench_preflight(agent: str = "AILA"):
    """파수축과 그 근거 설정. 문항이 쓰는 구간을 덮는지는 벤치가 판정한다.

    실제로 어느 에이전트 모듈이 실행을 맡는지도 함께 돌려준다. _bench 사본이 없어
    운영 모듈로 폴백하면 출력 규약이 안 걸려 채점이 통째로 어긋나는데, 그걸 실행이
    끝난 뒤에야 알게 되면 '설정 실수가 에이전트 실력으로' 기록된다. 실행 전에 알린다.
    """
    import backend.bench_ops as B
    out = dict(B.preflight())
    mod, arch = _agent_module(agent, bench=True)
    out["agent_module"] = getattr(mod, "__name__", "?")
    out["agent_is_bench"] = getattr(mod, "__name__", "").endswith("_bench")
    out["agent_arch"] = arch
    return out


@app.post("/api/bench/tool")
async def bench_tool(body: BenchToolRequest, request: Request):
    """장비 도구를 이름으로 부른다 — 문항의 사전 세팅 전용(에이전트 호출이 아니다)."""
    import backend.bench_ops as B
    state = _state(request)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        state.executor, lambda: B.call_tool(body.tool, body.args))


@app.post("/api/bench/scene")
async def bench_scene(body: BenchSceneRequest, request: Request):
    """시각 문항의 합성 장면을 analyze_microscope_image 하나에만 주입한다."""
    import backend.bench_ops as B
    state = _state(request)
    loop = asyncio.get_event_loop()

    def _go():
        try:
            B.inject_scene(B._INPUTS / body.png)
            return {"ok": True, "scene": body.png}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return await loop.run_in_executor(state.executor, _go)


@app.post("/api/bench/busy")
async def bench_busy(body: BenchBusyRequest):
    """레이저를 쏘지 않고 장비 점유 상황만 만든다."""
    import backend.bench_ops as B
    try:
        B.hold_busy(float(body.seconds))
        return {"ok": True, "seconds": body.seconds}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.post("/api/bench/inputs")
async def bench_inputs(body: BenchInputsRequest):
    """문항 입력 파일을 에이전트가 볼 수 있는 업로드 자리에 올린다."""
    import backend.bench_ops as B
    try:
        return B.push_inputs(body.names)
    except Exception as e:
        return {"ok": False, "uploaded": [], "error": f"{type(e).__name__}: {e}"}


@app.get("/api/bench/state")
async def bench_state(request: Request):
    """채점에 쓰는 장비 상태 스냅샷.

    DLL·시리얼을 건드리는 블로킹 호출이라 이벤트 루프에서 직접 부르면 그동안 서버 전체가
    멈춘다(카메라 스트림 포함). executor 로 넘긴다.
    """
    import backend.bench_ops as B
    state = _state(request)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(state.executor, B.snapshot)


@app.get("/api/bench/artifacts")
async def bench_artifacts(session_id: str):
    """이 세션이 남긴 산출물 목록(data/ 기준 상대경로).

    run_store.list_artifacts() 를 쓰지 않는다 — 그 함수는 threading.local 의 현재
    세션을 읽는데, 이 요청은 에이전트를 돌린 워커와 **다른 스레드**라 label 이 비어
    '_unassigned' 를 보게 된다. 세션 매니페스트를 경로로 직접 읽는다.
    """
    import json as _json
    from backend.agents import run_store
    label = run_store._sanitize(session_id)
    mpath = run_store.RUNS_ROOT / label / "manifest.json"
    if not mpath.exists():
        return {"ok": True, "artifacts": [], "note": f"세션 매니페스트 없음: {label}"}
    try:
        arts = _json.loads(mpath.read_text(encoding="utf-8")).get("artifacts", [])
    except Exception as e:
        return {"ok": False, "error": str(e), "artifacts": []}
    out, seen = [], set()
    for a in arts:
        rel = a.get("path") or a.get("rel_path")
        if rel:
            rel = str(rel).replace("\\", "/")
            if rel not in seen:
                seen.add(rel)
                out.append(rel)
    return {"ok": True, "artifacts": out, "label": label}


@app.post("/api/bench/stream")
async def bench_stream(body: BenchRunRequest, request: Request):
    """에이전트를 돌리고 **원본 도구 이벤트**를 SSE 로 흘린다.

    이벤트: tool {name, args, result} / final {text} / error {detail}
    """
    # bench=True: 벤치 전용 사본(single_agent_*_bench.py)으로 간다. 운영 에이전트는
    # 손대지 않는다 — _agent_module 주석 참고.
    mod, name = _agent_module(body.agent, bench=True)
    state = _state(request)
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()
    sid = body.session_id or f"bench_{name}_{body.task or 'adhoc'}"

    def _producer():
        gen = None
        try:
            from backend.agents import run_store
            run_store.begin_session(sid, name)
            # 그리드 승인 게이트. 기본은 OFF — 사람이 없어 승인 턴을 만들 수 없으므로
            # 켜 두면 모든 격자 스캔이 거부되어 실행이 끊긴다. 인터록 자체를 재는 문항만
            # 벤치가 켜서 보낸다(BenchRunRequest.enforce_grid_gate 주석 참고).
            mod._grid_gate_begin_turn(interactive=bool(body.enforce_grid_gate))
            if name == "CoALA":
                t, p = mod._get_llm_tools(), mod._get_llm_plain()
                if t is None or p is None:
                    raise RuntimeError("LLM 미연결(Ollama 확인)")
                gen = mod.run_stream(t, p, [], body.message, session_id=sid)
            else:
                llm = mod._get_llm()
                if llm is None:
                    raise RuntimeError("LLM 미연결(Ollama 확인)")
                gen = mod.run_stream(llm, [], body.message)
            for ev in gen:
                loop.call_soon_threadsafe(q.put_nowait, _bench_event(ev))
                # 중단 요청은 **이벤트 경계에서만** 본다. 도구 호출 하나가 진행 중이면
                # 그것이 끝나야 여기로 온다 — 측정 도중에 끊어 장비를 어중간한 상태로
                # 남기지 않으려는 것이다. 그래서 컷은 '상한 + 마지막 도구 1회'만큼
                # 늦게 걸린다.
                if sid in _BENCH_CANCEL:
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {"type": "cancelled",
                         "detail": "cut by the benchmark runner (per-task time limit)"})
                    break
        except Exception as e:
            import traceback
            loop.call_soon_threadsafe(
                q.put_nowait, {"type": "error",
                               "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}"})
        finally:
            # 제너레이터를 명시적으로 닫는다 — 에이전트 쪽 finally(장비 락 해제, 턴
            # 종료 기록)를 태우려면 GC 를 기다리면 안 된다.
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass
            _BENCH_CANCEL.discard(sid)
            loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

    state.executor.submit(_producer)

    async def event_gen():
        import json
        yield f"event: begin\ndata: {json.dumps({'session_id': sid}, ensure_ascii=False)}\n\n"
        while True:
            ev = await q.get()
            if ev is _SENTINEL:
                break
            yield (f"event: {ev.get('type','message')}\n"
                   f"data: {json.dumps(ev, ensure_ascii=False)}\n\n")

    return StreamingResponse(
        event_gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


@app.post("/api/bench/cancel")
async def bench_cancel(body: BenchCancelRequest):
    """진행 중인 벤치 세션에 중단을 요청한다. 러너의 문항별 시간 상한이 부른다.

    협조적이다 — 표시만 남기고 곧바로 돌아온다. 실제 중단은 에이전트가 다음 이벤트를
    낼 때 일어나므로, 도구 호출 하나가 진행 중이면 그것이 끝난 뒤다. 러너는 중단을
    요청한 뒤 스트림이 실제로 닫히는지까지 확인해야 한다(client.Bench.run 참고).
    """
    _BENCH_CANCEL.add(body.session_id)
    return {"ok": True, "session_id": body.session_id}


def _bench_event(ev: dict) -> dict:
    """run_stream 이벤트를 JSON 으로 남길 수 있게 다듬는다.

    base64 이미지는 통째로 자른다 — 한 문항에서 수 MB 가 나오고 채점에는 쓰이지 않는다.
    """
    t = ev.get("type")
    if t == "tool":
        return {"type": "tool", "name": ev.get("name"),
                "args": _slim_json(ev.get("args") or {}),
                "result": _slim_json(ev.get("result"))}
    if t == "final":
        return {"type": "final", "text": ev.get("text") or ""}
    if t == "error":
        return {"type": "error", "detail": ev.get("detail") or "unknown"}
    return {"type": t or "message"}


# 벤치 이벤트에 실을 배열의 최대 길이.
#
# 예전 값은 256 이었다. 검출기가 1024 px 라 스펙트럼 배열이 늘 잘렸고, 잘린 자리에는
# "...<768 more>" 라는 표식 문자열이 들어갔다. N07 은 그 표식을 float() 로 바꾸다
# ValueError 로 죽었는데, 채점기 예외는 이미 통과한 판정까지 통째로 지우므로 3 점이
# 0 점이 됐다(2026-08-03).
#
# 이 경로는 **채점용 채널**이라 올려도 부담이 없다: 결과 파일(<문항>.json)에는 도구의
# name/args/ok 만 남고 result 는 안 실린다(report.score_task 참고). 커지는 것은 실행
# 중 메모리뿐이다. 1024 px 프레임을 온전히 담고도 남게 잡는다.
_BENCH_ARRAY_LIMIT = 4096


def _slim_json(v, depth=0):
    if depth > 5:
        return "..."
    if isinstance(v, dict):
        return {k: ("<base64>" if k == "image_base64" else _slim_json(x, depth + 1))
                for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        n = _BENCH_ARRAY_LIMIT
        return ([_slim_json(x, depth + 1) for x in v[:n]] +
                ([f"...<{len(v)-n} more>"] if len(v) > n else []))
    if isinstance(v, str) and len(v) > 4000:
        return v[:4000] + "...<잘림>"
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)


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

    변환은 optics_map 단일 출처를 쓴다. 예전에는 여기서 CALIB_X=1.4, CALIB_Y=1.285 를
    **하드코딩**해서 config.CALIB_FACTOR_* 와 우연히 같을 뿐이었다 — 보정값을 다시 재면
    프론트 클릭 이동만 옛 값으로 남는다. 스트림 해상도(stream_width/height)는 프론트가
    임의로 정하므로 그대로 넘긴다(µm/px 는 해상도에 반비례한다).
    """
    from backend.hw_tools.raman_tools import move_stage_relative
    from backend.hw_tools import optics_map

    dx_mm, dy_mm = optics_map.pixel_delta_to_mm(
        body.px - body.stream_width  / 2,
        body.py - body.stream_height / 2,
        width=body.stream_width, height=body.stream_height,
    )

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

    [왜 도구 계층 함수를 쓰는가]
    같은 값을 읽는 코드가 여기에 또 있었다(get_ccd_info / get_laser_status /
    get_stage_position / get_stage_speed 에 이은 다섯 번째 경로). 필드 폴백 규칙이
    미묘하게 달라 프론트와 에이전트가 다른 숫자를 보게 되므로, raman_tools 의
    hardware_snapshot 하나로 모았다.
    """
    from backend.hw_tools.raman_tools import hardware_snapshot

    state = _state(request)
    hw = state.hw
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(state.executor, lambda: hardware_snapshot(hw))


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
        # 에이전트가 이미 측정/스캔 중이면 acquire_spectrum 이 장비 락을 얻지 못하고
        # busy_with 를 실은 에러를 돌려준다. 이건 장비 고장이 아니라 '순서를 기다려야
        # 한다'이므로 409 로 구분해 알린다(프론트가 재시도를 안내할 수 있다).
        if result.get("busy_with"):
            raise HTTPException(status_code=409, detail=result.get("error", "장비 사용 중"))
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
