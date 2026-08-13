# -*- coding: utf-8 -*-
"""스테이지(XYZ 시료대) 도구.

    move_stage           절대 좌표(mm)로 이동
    move_stage_relative  현재 위치 기준 변위(mm)만큼 이동
    get_stage_position   현재 X·Y·Z 조회
    get_stage_speed      현재 축별 속도 조회
    set_stage_speed      축별 속도 설정(생략한 축은 유지)

규칙 셋:

1. 좌표는 거부, 속도는 클리핑.
   범위 밖 좌표는 이동하지 않고 에러를 돌려준다. 속도는 드라이버가 상한으로 잘라내므로
   같은 상한으로 미리 잘라 '실제 적용될 값'을 보고한다.

2. 한계 숫자는 여기 적지 않는다.
   좌표 한계는 config(STAGE_MAX_*), 속도 한계는 드라이버(USE_stage_test.MAX_SPEED_*)에서
   import 한다.

3. 쓰기는 잠그고 읽기는 잠그지 않는다.
   이동·속도설정은 @_serialized 로 장비 락을 잡아 reconnect_hardware 와 배타적이다.
   조회는 프론트가 주기 폴링하므로 락을 잡지 않고, 대신 _stage_read 로 감싸 재연결
   구간과 겹치면 _STAGE_BUSY 를 돌려준다.

장비 핸들은 `_hw._stage` 로 읽는다. hw_core.init_hardware() 가 그 전역을 갈아 끼우므로
`from hw_core import _stage` 로 받으면 연결 전의 None 을 계속 붙들게 된다.
"""
from __future__ import annotations

from typing import Annotated, Optional

from pydantic import Field

# 스테이지 이동 가능 범위(mm).
from backend.tools.hw_tools.config import STAGE_MAX_X, STAGE_MAX_Y, STAGE_MAX_Z, STAGE_MIN_Z
# 축별 속도 상한. 드라이버가 클리핑에 쓰는 상수를 그대로 빌려 쓴다.
from backend.tools.hw_tools.hao.USE_stage_test import MAX_SPEED_XY as _MAX_SPEED_XY, MAX_SPEED_Z as _MAX_SPEED_Z
# 장비 핸들은 모듈 속성으로 읽는다.
from backend.tools.hw_tools.hw_tools import hw_core as _hw
from backend.tools.hw_tools.hw_tools.hw_core import (
    _STAGE_BUSY,            # 재연결과 겹쳐 읽기를 건너뛰었을 때 _stage_read 가 주는 표식
    _check_stage_target,    # 좌표 범위 검증 — 모든 이동 경로가 공유
    _serialized,            # 장비 락 데코레이터(쓰기 경로 전용)
    _stage_read,            # 락 없이 읽되 재연결 중이면 비켜 주는 래퍼
    _stage_unavailable,     # 핸들 없음 → 모델이 읽을 실패 dict
)
# 도구 응답 형식. 성공 ok(**payload) / 실패 fail(사유).
from backend.tools.result import fail, ok


# ══════════════════════════════════════════════════════════════════════════════
# 조회 — 락을 잡지 않는 경로
# ══════════════════════════════════════════════════════════════════════════════

def get_stage_speed() -> dict:
    """축별 이동 속도를 mm/s 로 읽는다.

    Returns
    -------
    dict
        성공: ``{"ok": True, "x_speed_mm_s": …, "y_speed_mm_s": …, "z_speed_mm_s": …}``
        실패: 핸들 없음 · 재연결 중 · 드라이버 거부를 각각 다른 문장으로 알린다.

    Notes
    -----
    드라이버의 ``get_velocity()`` 는 dict 를 돌려준다(시퀀스가 아니다).
    """
    err = _stage_unavailable()                      # 스테이지 미연결이면 여기서 끝
    if err:
        return err

    try:
        # 조회는 장비 락을 잡지 않는다. _stage_read 가 재연결 구간이면 호출을 건너뛴다.
        vel = _stage_read(lambda: _hw._stage.get_velocity())

        if vel is _STAGE_BUSY:
            # 값을 못 읽었을 뿐 아무것도 바꾸지 않았다는 점을 명시한다(재시도 안전).
            return fail("The stage is being connected or released right now, so its speed was not read. "
                        "Nothing was changed - try again in a few seconds.")

        # 드라이버는 실패를 예외가 아니라 {"ok": False, "error": …} 로 알린다.
        if not (isinstance(vel, dict) and vel.get("ok")):
            return fail(vel.get("error", "Failed to read velocity") if isinstance(vel, dict)
                        else "Unexpected velocity type")

        # 드라이버 dict 를 그대로 흘리지 않고 세 축만 골라 결과 모양을 고정한다.
        return ok(x_speed_mm_s=vel["x_speed_mm_s"],
                  y_speed_mm_s=vel["y_speed_mm_s"],
                  z_speed_mm_s=vel["z_speed_mm_s"])
    except Exception as e:
        return fail(str(e))


def get_stage_position() -> dict:
    """현재 스테이지 위치를 mm 로 읽는다.

    Returns
    -------
    dict
        성공: ``{"ok": True, "x": …, "y": …, "z": …}``

    Notes
    -----
    이동 도구는 이동 후 좌표를 이미 돌려주므로, 이동 직후에 이 도구를 또 부를 필요는 없다.
    """
    err = _stage_unavailable()
    if err:
        return err

    try:
        pos = _stage_read(lambda: _hw._stage.get_position())     # 조회 경로 — 락 없음

        if pos is _STAGE_BUSY:
            return fail("The stage is being connected or released right now, so its position was not read. "
                        "Nothing was changed - try again in a few seconds.")
        if pos is None:                                          # 드라이버가 값을 못 준 경우
            return fail("Failed to query stage position")

        return ok(x=pos[0], y=pos[1], z=pos[2])                  # 드라이버는 (x, y, z) 튜플
    except Exception as e:
        return fail(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 설정·이동 — 장비 락을 잡는 경로
# ══════════════════════════════════════════════════════════════════════════════

@_serialized("set_stage_speed")
def set_stage_speed(
    x_speed_mm_s: Annotated[Optional[float], Field(description='X-axis movement speed (mm/s, max 5.0). Optional.')] = None,
    y_speed_mm_s: Annotated[Optional[float], Field(description='Y-axis movement speed (mm/s, max 5.0). Optional.')] = None,
    z_speed_mm_s: Annotated[Optional[float], Field(description='Z-axis movement speed (mm/s, max 0.1). Optional.')] = None,
) -> dict:
    """축별 이동 속도를 설정한다. 생략한 축은 현재 속도를 유지한다.

    Parameters
    ----------
    x_speed_mm_s, y_speed_mm_s, z_speed_mm_s : float, optional
        목표 속도(mm/s). ``None`` 이면 그 축은 건드리지 않는다.

    Returns
    -------
    dict
        ``{"ok": True, "x_speed_mm_s": …, …}`` — **실제로 적용될 값**이다. 상한에 걸린
        축이 있으면 ``clipped`` 에 요청값·적용값·상한이, ``note`` 에 안내 문장이 실린다.
    """
    err = _stage_unavailable()
    if err:
        return err

    try:
        # 1) 현재 속도를 읽는다. 생략된 축을 무엇으로 채울지가 여기서 정해진다.
        #    이 경로는 이미 락 안이라 _stage_read 가 필요 없다.
        current_vel = _hw._stage.get_velocity()
        if not current_vel.get("ok"):
            return fail(current_vel.get("error", "Failed to read current velocity"))

        # 2) 생략(None)한 축을 현재 값으로 메운다. 드라이버가 세 축을 항상 함께 요구하므로
        #    '안 건드림'을 표현할 방법이 이것뿐이다.
        req = {
            "x_speed_mm_s": x_speed_mm_s if x_speed_mm_s is not None else current_vel["x_speed_mm_s"],
            "y_speed_mm_s": y_speed_mm_s if y_speed_mm_s is not None else current_vel["y_speed_mm_s"],
            "z_speed_mm_s": z_speed_mm_s if z_speed_mm_s is not None else current_vel["z_speed_mm_s"],
        }

        # 3) 드라이버와 같은 상수로 클리핑한다. XY 와 Z 는 상한이 다르다.
        limits = {"x_speed_mm_s": _MAX_SPEED_XY,
                  "y_speed_mm_s": _MAX_SPEED_XY,
                  "z_speed_mm_s": _MAX_SPEED_Z}
        eff, clipped = {}, {}                      # eff = 적용값, clipped = 잘린 축 기록
        for k, v in req.items():
            hi = limits[k]
            e = max(-hi, min(hi, float(v)))        # 음수(역방향)도 같은 크기로 제한
            eff[k] = e
            if e != float(v):                      # 잘렸으면 무엇이 어떻게 잘렸는지 남긴다
                clipped[k] = {"requested": float(v), "applied": e, "limit_mm_s": hi}

        # 4) 컨트롤러에 명령. 네 번째 인자는 A축 속도이고 이 장비는 쓰지 않아 0.0 고정.
        #    set_velocity 는 실패를 예외가 아니라 False 로 알리므로 반드시 확인한다.
        applied = _hw._stage.set_velocity(eff["x_speed_mm_s"], eff["y_speed_mm_s"],
                                          eff["z_speed_mm_s"], 0.0)
        if applied is False:
            return fail("The controller rejected the velocity command, so the stage speed was NOT changed. "
                        "Check the stage connection and retry.")

        # 5) 요청값이 아니라 적용값을 돌려준다.
        out = ok(**eff)
        if clipped:
            out["clipped"] = clipped
            out["note"] = ("Some axes were clipped to the hardware speed limit "
                           f"(XY <= {_MAX_SPEED_XY} mm/s, Z <= {_MAX_SPEED_Z} mm/s). "
                           "The values reported here are what the stage will actually use.")
        return out

    except Exception as e:
        return fail(str(e))


@_serialized("move_stage")
def move_stage(
    x: Annotated[float, Field(ge=0, le=STAGE_MAX_X, description=f'X-axis position (mm, 0-{STAGE_MAX_X})')],
    y: Annotated[float, Field(ge=0, le=STAGE_MAX_Y, description=f'Y-axis position (mm, 0-{STAGE_MAX_Y})')],
    z: Annotated[Optional[float], Field(ge=STAGE_MIN_Z, le=STAGE_MAX_Z, description=f'Z-axis position (mm, {STAGE_MIN_Z}-{STAGE_MAX_Z}). Optional - omit to keep the current Z.')] = None,
) -> dict:
    """스테이지를 절대 좌표(mm)로 이동한다.

    Parameters
    ----------
    x, y : float
        목표 좌표(mm). 범위를 벗어나면 이동하지 않고 거부한다.
    z : float, optional
        목표 Z(mm). 생략하면 현재 Z 를 유지한다 — XY 만 움직이고 초점은 건드리지 않는다.

    Returns
    -------
    dict
        ``{"ok": True, "position": {"x": …, "y": …, "z": …}}`` — 이동 후 드라이버에서
        되읽은 값이라 보고된 좌표가 곧 실제 좌표다.

    Notes
    -----
    ``component_lock`` 은 잡지 않는다. ``@_serialized`` 의 장비 락이 이미
    reconnect_hardware 와 배타적이고, 여기서 하나 더 잡으면 hw_core 가 정한 락 순서
    (component_lock → 장비 락)를 뒤집어 교착한다.
    """
    err = _stage_unavailable()
    if err:
        return err

    # 범위 검증은 공용 함수 하나로 — 상대 이동도 같은 함수를 지난다.
    err = _check_stage_target(x, y, z)
    if err:
        return err

    try:
        kw = {"x": x, "y": y, "wait": True}        # wait=True: 이동이 끝난 뒤 반환
        if z is not None:
            kw["z"] = z
        else:
            # 드라이버가 세 축을 모두 요구하므로 Z 생략은 '현재 Z 재지정'으로 표현한다.
            kw["z"] = _hw._stage.get_position()[2]

        _hw._stage.move_absolute(**kw)

        pos = _hw._stage.get_position()            # 요청값이 아니라 이동 후 실제 좌표를 보고
        return ok(position={"x": pos[0], "y": pos[1], "z": pos[2]})
    except Exception as e:
        return fail(str(e))


@_serialized("move_stage_relative")
def move_stage_relative(
    dx: Annotated[Optional[float], Field(description='Displacement in X (mm)')] = 0.0,
    dy: Annotated[Optional[float], Field(description='Displacement in Y (mm)')] = 0.0,
    dz: Annotated[Optional[float], Field(description='Displacement in Z (mm)')] = 0.0,
) -> dict:
    """현재 위치를 기준으로 변위(mm)만큼 이동한다.

    Parameters
    ----------
    dx, dy, dz : float, optional
        축별 변위(mm). 생략하면 0 — 그 축은 움직이지 않는다.

    Returns
    -------
    dict
        ``{"ok": True, "position": {…}}`` 또는 실패 dict. 범위 밖이면 에러 문장에
        계산된 목표 좌표를 덧붙인다 — 변위만으로는 왜 막혔는지 알 수 없기 때문이다.
    """
    err = _stage_unavailable()
    if err:
        return err

    try:
        pos = _hw._stage.get_position()            # 락 안이라 _stage_read 없이 직접 읽는다
        if pos is None:
            return fail("Failed to query stage position")

        # 변위를 현재 위치에 더해 목표 절대 좌표를 만든 뒤 검사한다 — move_stage 와 같은 판정.
        err = _check_stage_target(float(pos[0]) + float(dx),
                                  float(pos[1]) + float(dy),
                                  float(pos[2]) + float(dz))
        if err:
            # 모델은 자기가 보낸 변위밖에 모르므로 계산된 목표를 알려 준다.
            err["error"] += (f" - this is the target after applying the relative move "
                             f"(dx={dx}, dy={dy}, dz={dz}) to the current position.")
            return err

        _hw._stage.move_relative(dx, dy, dz, 0)    # 네 번째는 A축 변위 — 이 장비는 쓰지 않는다

        pos = _hw._stage.get_position()            # 이동 후 실제 좌표를 되읽는다
        return ok(position={"x": pos[0], "y": pos[1], "z": pos[2]})
    except Exception as e:
        return fail(str(e))
