"""
서버 수명주기 — 기동 시 장비 초기화, 종료 시 안전 해제.

동작: 2단계
  Phase 1  스테이지 / 카메라 / 레이저 / Ollama — 빠르므로 순차로 기다린다.
  Phase 2  CCD — 냉각 안정화(-40°C)에 수 분 걸리므로 백그라운드 스레드로 넘기고
           서버는 먼저 뜬다. 진행 상황은 /api/ccd/status 로 본다.

종료는 CCD 온도 복구(≥ -5°C)를 기다려야 해서 오래 걸림. 그래서 별도 스레드에
넘기고 uvicorn 은 먼저 빠져나옴(HardwareManager.shutdown 참고).
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

# ★ Ollama 모델·호스트·num_ctx·타임아웃을 바꾸려면 backend/llm_config.py 한 곳이다.
#   (여기 두지 않은 이유: vbench 는 서버를 안 거치고 에이전트를 직접 import 하므로
#    이 파일이 로드되지 않는다 — llm_config 머리말 참고.)
#   파일을 고치지 않고 바꾸려면:  RAMAN_OLLAMA_MODEL=gemma4:12b python -m backend.web_controller.main
import backend.llm_config as llm_config
from backend.tools.hw_tools.hw_tools.hw_core import (
    init_hardware as rt_init_hardware,
    # 매니저의 현재 핸들 4개를 도구 계층 전역에 다시 주입하는 공용 헬퍼.
    sync_tool_handles as rt_sync_handles,
)
from backend.web_controller.setups.state import AppState, CcdInitState


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    app.state.s = state
    loop = asyncio.get_event_loop()

    # 어떤 LLM 으로 도는지 기동 로그 첫 줄에 박아 둔다. 모델을 바꿔 가며 돌릴 때
    # (예: gemma4:31b → gemma4:12b) '지금 뜬 서버가 어느 모델인가'를 로그만 보고
    # 확정할 수 있어야 한다 — 설정은 llm_config.py 한 곳이고, 파일을 고치는
    # 대신 RAMAN_OLLAMA_MODEL 환경변수로 띄우는 것이 권장 경로다.
    print(f"[SERVER] LLM: {llm_config.describe()}")

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

    # 여기 있던 llm_client.setup(state.hw.ollama) 는 삭제했다 — llm_client 는 받은
    # 클라이언트를 전역에 넣어 둘 뿐이고 그걸 읽는 곳이 하나도 없었다(모듈 자체 주석이
    # "현재 호출자는 없으며"라고 적고 있었다). 에이전트는 llm_config 를 보고 자기
    # ChatOllama 를 만든다. 지금은 모듈 파일도 없어 이 줄이 기동을 막고 있었다.

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
    threading.Thread(
        target=cool_ccd_in_background, args=(state,), daemon=True, name="ccd-init",
    ).start()

    print("[SERVER] 준비 완료 — http://localhost:8000  (CCD 냉각 진행 중)")
    yield   # ── 서버 실행 중 ──

    # ── 종료 ──
    state.executor.shutdown(wait=False)
    threading.Thread(
        target=state.hw.shutdown, daemon=False, name="hw-shutdown"
    ).start()
    print("[SERVER] 종료 신호 수신 — CCD 온도 복구 후 종료됩니다.")


def cool_ccd_in_background(state: AppState) -> None:
    """CCD 를 연결하고 -40°C 안정화까지 기다린다(수 분). 반드시 별도 스레드에서 부른다.

    기동 시 lifespan 이 한 번, 실패 후 /api/ccd/connect 로 재시도할 때 한 번 더 쓴다.
    """
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
