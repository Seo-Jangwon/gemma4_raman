"""
Raman GPT — FastAPI 백엔드 서버 (엔트리포인트)

실행:
    python -m backend.web_controller.main      (프로젝트 루트에서)

포트 8000. 프론트엔드(Vite, 3000)는 /api/* 를 http://localhost:8000 으로 프록시한다.

이 파일은 '조립'만 한다 — 미들웨어 · 정적 마운트 · 라우터 등록.
아래 include_router 블록이 이 서버의 API 목차다. 로직은 전부 routers/ 안에 있다.

    core/     상태(state) · 수명주기(init_release) · 공용 인프라(streaming, agent_module)
    schemas/  요청·응답 Pydantic 모델
    routers/  URL 별 컨트롤러
"""

from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path 에 올린다 — backend.* 절대 import 가 어디서 띄우든 풀리게.
# 반드시 아래 backend.* import 보다 먼저 실행돼야 한다.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI                                      # noqa: E402
from fastapi.middleware.cors import CORSMiddleware               # noqa: E402
from fastapi.staticfiles import StaticFiles                      # noqa: E402

from backend.service.store.spectrum_store import RESULTS_ROOT                  # noqa: E402
from backend.web_controller.setups.init_release import lifespan    # noqa: E402
from backend.web_controller.setups.request_log import RequestLogMiddleware  # noqa: E402
from backend.web_controller.controllers import (                     # noqa: E402
    agent, files, hardware, kb,
)

app = FastAPI(title="Raman GPT API", version="1.0.0", lifespan=lifespan)

# 들어온 요청을 한 줄씩 찍는다(경로 · 상태코드 · 소요시간 · 담당 라우터).
app.add_middleware(RequestLogMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 측정 결과(png/csv/json) 정적 서빙 — 채팅 인라인 표시·다운로드용.
# vite proxy 가 /api 만 통과시키므로 /api/results 아래에 둔다(spectrum_store.URL_PREFIX 와 일치).
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/api/results", StaticFiles(directory=str(RESULTS_ROOT)), name="results")

# ── API 목차 ──────────────────────────────────────────────────────────────────
app.include_router(hardware.router)   # /api/camera, /api/stage, /api/laser, /api/ccd,
                                      # /api/spectrum, /api/hardware/state, /api/health
app.include_router(agent.router)      # /api/experiment/*, /api/agents/health
app.include_router(files.router)      # /api/files/*
app.include_router(kb.router)         # /api/kb/*


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.web_controller.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # reload=True 시 lifespan이 재실행되어 하드웨어 재초기화됨
        log_level="info",
    )
