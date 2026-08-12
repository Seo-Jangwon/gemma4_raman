"""
로그 — 콘솔에 남길 것과 남기지 않을 것을 정하는 곳.

두 가지가 들어 있다. 둘 다 목적이 같다: **폴링 때문에 콘솔이 도배되는 것을 막는 것**.
프론트는 /api/hardware/state 를 1.5 초마다, /api/ccd/status 를 그보다 자주 두드린다.
들어오는 대로 다 찍으면 정작 봐야 할 요청(측정 실패, 409, 에이전트 호출)이 묻힌다.

  1. RequestLogMiddleware   요청 1건 = 로그 1줄 (폴링 경로는 상태코드가 바뀔 때만)
  2. log_ccd_reading()      CCD 온도는 '사건'이 있을 때만

이 모듈은 표준 라이브러리 말고는 아무것도 import 하지 않는다 — 그래서 하드웨어 SDK 가 없는
PC 에서도 아래 자체 검사가 그냥 돈다:  python backend/web_controller/core/request_log.py
"""

from __future__ import annotations

import time


# ══════════════════════════════════════════════════════════════════════════════
# 1. 요청 로그
# ══════════════════════════════════════════════════════════════════════════════
#
#     [REQ] POST /api/stage/speed      -> 200  (12ms)  [hardware]
#     [REQ] POST /api/spectrum/acquire -> 409  (3ms)   [hardware]
#
# [왜 라우터 파일마다 찍지 않고 여기 한 곳인가]
# 경로가 곧 담당 파일이다(main.py 의 include_router 목차 참고). 같은 print 를 32개 라우트에
# 복붙하면 새 라우트를 추가할 때마다 빠뜨릴 수 있고, 응답 상태코드·소요시간은 라우트 안에서
# 알 수도 없다. 미들웨어 하나가 지금 것과 앞으로 추가될 것 전부를 덮는다.
# 끝의 대괄호는 그 요청을 처리한 라우터(= routers/ 아래 파일)다.
#
# [왜 BaseHTTPMiddleware(@app.middleware("http")) 가 아니라 순수 ASGI 인가]
# BaseHTTPMiddleware 는 응답 본문을 anyio 메모리 스트림으로 감싼다. 이 서버에는 끊기지 않는
# MJPEG(/api/camera/stream)과 SSE(/api/experiment/stream)가 있어서, 그
# 경로에 버퍼링·연결종료 전파 문제를 얹고 싶지 않다(카메라 스트림의 좀비 가드는
# request.is_disconnected() 에 의존한다). 순수 ASGI 미들웨어는 send 메시지를 지나가며 보기만
# 하므로 스트리밍 동작에 손대지 않는다.

#: 프론트가 쉬지 않고 두드리는 폴링 경로. 상태코드가 바뀔 때만 찍는다
#: (첫 요청과 200→503 같은 오류 전환은 그대로 보인다).
#: /api/ccd/status 의 온도 자체는 아래 log_ccd_reading() 이 따로 관리한다.
POLLING_PATHS = {"/api/ccd/status", "/api/hardware/state"}

_last_polling_status: dict[str, int] = {}


def should_log_request(path: str, status: int) -> bool:
    """이 응답을 콘솔에 남길지. 폴링 경로는 상태코드가 직전과 다를 때만 True."""
    if path not in POLLING_PATHS:
        return True
    if _last_polling_status.get(path) == status:
        return False
    _last_polling_status[path] = status
    return True


def _router_label(scope) -> str:
    """이 요청을 처리한 라우터 이름(= routers/ 아래 파일). 못 알아내면 빈 문자열."""
    tags = getattr(scope.get("route"), "tags", None)
    return f"  [{tags[0]}]" if tags else ""


class RequestLogMiddleware:
    """요청 1건 = 로그 1줄. app.add_middleware(RequestLogMiddleware) 로 건다."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        started = time.perf_counter()
        path = scope.get("path", "")

        async def send_wrapper(message):
            # 응답 헤더가 나가는 시점에 찍는다 — 스트리밍 응답도 스트림이 끝날 때까지
            # 기다리지 않고 '들어왔다'를 바로 볼 수 있다.
            if message["type"] == "http.response.start":
                status = message["status"]
                if should_log_request(path, status):
                    ms = (time.perf_counter() - started) * 1000
                    print(f"[REQ] {scope.get('method', '?'):6} {path} "
                          f"-> {status}  ({ms:.0f}ms){_router_label(scope)}")
            await send(message)

        await self.app(scope, receive, send_wrapper)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CCD 온도 로그
# ══════════════════════════════════════════════════════════════════════════════
#
# 프론트 상태바(CCDStatusBar)가 /api/ccd/status 를 쉬지 않고 폴링한다. 읽은 온도를 매번
# 찍으면 냉각이 도는 20 여 분 동안 콘솔이 온도 줄로만 찬다. 그래서 '사건'일 때만 남긴다:
#     · 연결 ↔ 미연결 전환
#     · 안정화 상태 변화 (냉각 중 → 안정화 / 드리프트 / 냉각기 OFF)
#     · 온도가 CCD_LOG_STEP_C 이상 움직였을 때
#
# 하드웨어 계층(hardware_manager._init_ccd / _ccd_warmup_and_shutdown)의 진행 출력은
# 건드리지 않는다 — 거기는 기동·종료를 붙잡고 있는 블로킹 구간이라 진행 표시가 필요하다.

#: 이보다 작게 움직이면 조용히 넘어간다. 로그가 너무 잦으면 올리면 된다.
CCD_LOG_STEP_C = 1.0

_ccd_last: dict = {"connected": None, "temp": None, "status": None}


def log_ccd_reading(connected: bool, temp, status) -> bool:
    """직전 관측과 달라진 게 있을 때만 한 줄 찍는다. 찍었으면 True."""
    last = _ccd_last
    if connected == last["connected"] and status == last["status"]:
        if temp is None or last["temp"] is None:
            if temp == last["temp"]:            # 둘 다 온도 미상 — 변화 없음
                return False
        elif abs(temp - last["temp"]) < CCD_LOG_STEP_C:
            return False

    last.update(connected=connected, temp=temp, status=status)
    if not connected:
        print("[CCD] 미연결")
        return True
    reading = f"{temp}°C" if temp is not None else "온도 미상"
    print(f"[CCD] {reading}  [{status or '상태 미상'}]")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 자체 검사 — 조용해야 할 때 조용한지, 사건일 때 찍는지
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── CCD 온도 ──
    assert log_ccd_reading(False, None, None) is True,  "첫 관측은 찍어야 한다"
    assert log_ccd_reading(False, None, None) is False, "미연결 반복은 조용해야 한다"

    assert log_ccd_reading(True, 20, "냉각 중") is True,  "연결 전환은 사건이다"
    assert log_ccd_reading(True, 20, "냉각 중") is False, "같은 온도 반복은 조용해야 한다"
    assert log_ccd_reading(True, 19, "냉각 중") is True,  "1°C 이상 변화는 사건이다"
    assert log_ccd_reading(True, 19, "안정화")  is True,  "목표 도달(상태 변화)은 사건이다"
    assert log_ccd_reading(True, 19, "안정화")  is False, "안정화 유지는 조용해야 한다"
    assert log_ccd_reading(True, 19, "냉각기 OFF") is True, "쿨러 OFF 는 사건이다"
    assert log_ccd_reading(True, None, "냉각기 OFF") is True, "온도 미상 전환은 사건이다"
    assert log_ccd_reading(True, None, "냉각기 OFF") is False, "온도 미상 반복은 조용해야 한다"

    # ── 요청 로그 ──
    assert should_log_request("/api/spectrum/acquire", 200) is True,  "일반 경로는 항상 찍는다"
    assert should_log_request("/api/spectrum/acquire", 200) is True,  "일반 경로는 반복도 찍는다"
    assert should_log_request("/api/ccd/status", 200) is True,  "폴링 경로 첫 요청은 찍는다"
    assert should_log_request("/api/ccd/status", 200) is False, "같은 상태 폴링은 조용해야 한다"
    assert should_log_request("/api/ccd/status", 503) is True,  "상태코드 전환은 찍어야 한다"
    assert should_log_request("/api/ccd/status", 503) is False, "전환 후 반복은 다시 조용"

    print("\n자체 검사 통과")
