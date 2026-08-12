"""
앱 전역 상태(AppState)와, 라우터가 그것을 받는 방법.

라우터는 `state: StateDep` 한 줄로 상태를 주입받는다(FastAPI 의존성 주입).
블로킹 하드웨어 호출은 전부 `run_in_worker()` 를 거친다 — 이유는 그 함수 docstring 참고.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Callable, Optional, TypeVar

from fastapi import Depends, Request

from backend.hw_tools.hardware_manager import HardwareManager, get_manager


class CcdInitState(str, Enum):
    """CCD 초기화 진행 단계. 냉각이 수 분 걸려 '연결됨/아님'만으로는 표현이 안 된다."""
    IDLE       = "idle"
    COOLING    = "cooling"
    STABILIZED = "stabilized"
    FAILED     = "failed"


@dataclass
class AppState:
    """서버가 살아 있는 동안 유지되는 것 전부. lifespan 이 만들어 app.state.s 에 붙인다."""
    hw: HardwareManager = field(default_factory=get_manager)
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=4)
    )
    ccd_state: CcdInitState = CcdInitState.IDLE
    ccd_error: Optional[str] = None


def get_state(request: Request) -> AppState:
    return request.app.state.s


#: 라우트 시그니처에 `state: StateDep` 로 적으면 AppState 가 주입된다.
StateDep = Annotated[AppState, Depends(get_state)]


T = TypeVar("T")


async def run_in_worker(state: AppState, fn: Callable[[], T]) -> T:
    """블로킹 함수를 워커 스레드에서 돌리고 결과를 기다린다.

    [왜 반드시 이걸 거쳐야 하는가]
    stage.get_position / ccd.get_temperature / acquire_spectrum 같은 호출은 직렬 통신·SDK
    호출이라 수십 ms~수 초 블로킹된다. async 핸들러에서 그냥 부르면 그동안 이벤트 루프가
    통째로 멈춰, 같은 루프에서 도는 카메라 MJPEG 스트림(/api/camera/stream)이 끊긴다.
    프론트는 /api/hardware/state 를 1.5 초마다 폴링하므로 이건 이론이 아니라 상시 발생한다.

    워커는 AppState.executor(워커 4개) 하나를 공유한다 — 여기가 마르면 채팅 요청까지
    줄을 서므로, 여기에 넘기는 작업은 '언젠가 끝나는' 것이어야 한다.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(state.executor, fn)
