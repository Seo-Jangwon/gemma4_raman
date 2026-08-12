"""
동기 제너레이터 → SSE(Server-Sent Events) 스트리밍 브리지.

[왜 브리지가 필요한가]
에이전트의 run_stream / stream_experiment 는 내부에서 Ollama 호출을 블로킹 실행하는
**동기** 제너레이터다. 이벤트 루프에서 직접 돌리면 서버 전체가 멈춘다. 그래서
워커 스레드에서 소비하고, 각 이벤트를 asyncio.Queue 로 넘겨 async 쪽에서 흘려보낸다.

/api/experiment/stream 이 쓴다. 스트리밍 라우트가 늘어나도 이 한 곳만 부르면 된다.
"""

from __future__ import annotations

import asyncio
import json
from typing import Callable, Optional

from fastapi.responses import StreamingResponse

from backend.web_controller.setups.state import AppState

# SSE 응답 헤더. X-Accel-Buffering 은 nginx/프록시 버퍼링을 꺼 이벤트를 즉시 전달시킨다.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _frame(event_type: str, payload: dict) -> str:
    """표준 SSE 한 블록:  event: <type>\\ndata: <json>\\n\\n"""
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_response(
    state: AppState,
    produce: Callable[[Callable[[dict], None]], None],
    *,
    first_event: Optional[tuple[str, dict]] = None,
) -> StreamingResponse:
    """`produce` 를 워커 스레드에서 돌리며 그 이벤트를 SSE 로 흘리는 응답을 만든다.

    produce(emit)   워커 스레드에서 실행될 콜백. emit(dict) 로 이벤트를 밀어 넣는다.
                    이벤트의 "type" 키가 SSE 의 event 이름이 된다(없으면 "message").
                    ★ 예외 처리는 호출자가 produce 안에서 직접 한다 — 에러 메시지 형식이
                      경로마다 다르기 때문이다. 아래 except 는 마지막 안전망일 뿐이다.
    first_event     스트림 맨 앞에 붙일 (이벤트이름, payload). 없으면 생략.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def emit(event: dict) -> None:
        # 워커 스레드 → 이벤트 루프. Queue 는 스레드 안전하지 않으므로 반드시 이 경로로.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _worker():
        try:
            produce(emit)
        except Exception as e:   # produce 가 자체 처리에 실패한 경우의 안전망
            emit({"type": "error", "detail": f"{type(e).__name__}: {e}"})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    state.executor.submit(_worker)

    async def event_gen():
        if first_event is not None:
            yield _frame(first_event[0], first_event[1])
        while True:
            event = await queue.get()
            if event is DONE:
                break
            yield _frame(event.get("type", "message"), event)

    return StreamingResponse(
        event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS,
    )
