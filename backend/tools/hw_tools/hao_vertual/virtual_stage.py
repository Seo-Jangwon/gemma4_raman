# -*- coding: utf-8 -*-
"""가상 XY 스테이지. `TangoController` 덕타입.

이 객체가 아는 것은 **위치 하나**다. 시야도 시료도 모른다 — 카메라가 이 좌표를 읽어 갈 뿐이다.

[왜 상속하지 않는가]
TangoController 를 상속하면 Tango_DLL.dll 을 끌게 되어, 장비 없는 PC 에서 돌리려는 목적
자체가 무너진다. hw_core.init_hardware() 는 타입 검사를 하지 않고 도구 계층은 전부
`_hw._stage.get_position()` 형태로 부르므로, 이름만 맞으면 그대로 꽂힌다.

[Z 축이 없다]
가상 시료는 이미지 한 장(2D)이라 Z 라는 것이 존재하지 않는다. 그래도 get_position() 은
**4-튜플을 유지한다** — camera_tools 가 pos[2] 를 읽으므로 길이를 줄이면 IndexError 가 난다.
Z 를 요구하는 도구(run_autofocus)는 아래 has_z 속성을 보고 스스로 빠진다. 도구 계층이
llm_config.VIRTUAL_HW 를 import 하지 않게 하려는 것이다 — 가상 여부를 도구가 알기 시작하면
분기가 도구 수만큼 번진다.

[초기화 절차를 그대로 통과시킨다]
hardware_manager._init_stage 는 load_dll → create_session → connect → set_velocity →
get_position → move_absolute(중점) 순으로 돈다. 그 함수들을 전부 갖고 있으므로 매니저는
**생성자 한 줄만** 바뀌고 나머지 절차는 실물과 똑같이 실행된다.
"""
from __future__ import annotations

from backend.tools.hw_tools.config import (
    STAGE_CENTER_X, STAGE_CENTER_Y, STAGE_MAX_X, STAGE_MAX_Y,
)

#: 속도 상한은 실물과 같은 곳(USE_stage_test)을 본다 — stage_tools 도 거기서 읽으므로
#: 여기서 따로 정의하면 '도구는 막는데 드라이버는 통과시키는' 상태가 된다.
from backend.tools.hw_tools.hao.USE_stage_test import MAX_SPEED_XY, MAX_SPEED_Z


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


class VirtualStage:
    """위치·속도 상태만 가진 스테이지 대역."""

    #: 이 스테이지에는 Z 축이 없다. run_autofocus 가 이 속성으로 판정한다.
    has_z = False

    def __init__(self, dll_path: str | None = None,
                 x: float | None = None, y: float | None = None):
        # dll_path 는 받기만 하고 쓰지 않는다 — 매니저가 실물과 같은 인자로 부르기 때문이다.
        self.x = STAGE_CENTER_X if x is None else float(x)
        self.y = STAGE_CENTER_Y if y is None else float(y)
        self.z = 0.0
        self.a = 0.0
        self.vx = self.vy = self.vz = self.va = 0.1
        self.connected = False
        self.dead = False
        self.dead_reason = None
        self.LSID = -1

    # ── 수명주기 (매니저의 초기화 절차가 그대로 지나간다) ──────────────────────
    def load_dll(self) -> bool:
        return True

    def create_session(self) -> bool:
        self.LSID = 1
        return True

    def connect(self, interface: int = -1, port: str = "", baudrate: int = 57600) -> bool:
        self.connected = True
        return True

    def calibrate(self, axes: int = 3) -> bool:
        return True

    def mark_dead(self, reason: str) -> None:
        """실물에서는 DLL 세션 해제 실패로 복구 불가가 된 상태. 가상에도 남겨 둔 이유는
        컨트롤러(hardware.py)가 dead 를 보고 503 을 돌려주는 경로를 그대로 태우기 위함이다."""
        self.dead = True
        self.dead_reason = reason

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def free_session(self) -> bool:
        self.LSID = -1
        return True

    # ── 위치 ────────────────────────────────────────────────────────────────
    def get_position(self):
        """(x, y, z, a). 실패를 None 으로 알리는 실물 계약을 그대로 따른다."""
        if self.dead or not self.connected:
            return None
        return (self.x, self.y, self.z, self.a)

    def move_absolute(self, x: float, y: float, z: float = 0.0, a: float = 0.0,
                      wait: bool = True) -> bool:
        """z 는 받고 무시한다 — 시료가 2D 라 Z 라는 축이 없다.

        범위를 벗어난 목표는 클리핑한다. 실물 드라이버도 그렇게 동작하고, 도구 계층
        (_check_stage_target)이 그보다 먼저 거부하므로 여기까지 오는 값은 이미 유효하다.
        """
        if self.dead or not self.connected:
            return False
        self.x = _clip(x, 0.0, STAGE_MAX_X)
        self.y = _clip(y, 0.0, STAGE_MAX_Y)
        # ponytail: 이동 즉시 완료(wait 무시). 실제 이동 시간이 필요해지면 거리/속도 sleep.
        return True

    def move_relative(self, dx: float, dy: float, dz: float = 0.0, da: float = 0.0,
                      wait: bool = True) -> bool:
        return self.move_absolute(self.x + dx, self.y + dy, 0.0, 0.0, wait)

    # ── 속도 ────────────────────────────────────────────────────────────────
    def get_velocity(self) -> dict:
        if self.dead:
            return {"ok": False, "error": f"세션 무효 - {self.dead_reason}"}
        if not self.connected:
            return {"ok": False, "error": "연결되지 않았습니다"}
        return {"ok": True,
                "x_speed_mm_s": self.vx, "y_speed_mm_s": self.vy,
                "z_speed_mm_s": self.vz, "a_speed_mm_s": self.va}

    def set_velocity(self, vx: float, vy: float, vz: float, va: float) -> bool:
        if self.dead or not self.connected:
            return False
        self.vx = _clip(vx, 0.0, MAX_SPEED_XY)
        self.vy = _clip(vy, 0.0, MAX_SPEED_XY)
        self.vz = _clip(vz, 0.0, MAX_SPEED_Z)
        self.va = _clip(va, 0.0, MAX_SPEED_XY)
        return True

    def __repr__(self) -> str:
        return f"<VirtualStage x={self.x:.4f} y={self.y:.4f} (no Z)>"
