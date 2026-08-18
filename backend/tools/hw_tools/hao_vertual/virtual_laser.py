# -*- coding: utf-8 -*-
"""가상 레이저. `LaserController` 덕타입.

빔이 꺼졌는지 · 가이드빔인지 · 측정빔인지, 그리고 파워 몇 %인지. 그게 전부다.

[책임 아님]
시료에 주는 영향. 조사량 누적·광표백·손상은 이 층에 없다(다음 단계). 이 객체는
'무엇이 나가고 있는가'만 알고, 그것을 읽어 가는 쪽은 카메라(가이드빔 스팟 합성)와
CCD(측정빔이 켜져 있었는가)다.

[속성 네 개가 계약이다]
도구 계층이 메서드가 아니라 **속성**으로 빔 상태를 판정한다. 빠뜨리면 에러가 아니라
'파워를 설정한 적이 없다'는 거부로 나타나서 원인을 찾기 어렵다.

    _power_set   hw_core._beam_state()  — 지금 laser_on() 하면 측정빔인가 가이드빔인가
    power_pct    hw_core._beam_state()  — 마지막으로 설정한 파워(%)
    ND_MIN_PCT   hw_core._laser_power_range() — 허용 범위 하한
    ND_MAX_PCT   hw_core._laser_power_range() — 허용 범위 상한

[laser_off 가 _power_set 을 내리지 않는 이유]
실물의 의도적 설계를 그대로 따른다(USE_laser_with_power.laser_off 주석). 여기서 내리면
다음 laser_on() 이 ND 필터를 측정 위치로 옮기지 않아, 파워를 걸어 둔 채 껐다 켰을 때
조용히 가이드빔이 나간다. 측정빔에서 내려오는 길은 set_guide_beam() 하나다.
"""
from __future__ import annotations


class _DummySerial:
    """`laser.ser and laser.ser.is_open` 으로 연결을 판정하는 곳이 둘 있다
    (hardware_manager._init_laser, controllers/hardware.py). 그 판정을 통과시키는 껍데기."""

    is_open = True

    def close(self) -> None:
        self.is_open = False


class VirtualLaser:
    """빔 상태와 파워 값만 가진 레이저 대역."""

    ND_MIN_PCT, ND_MAX_PCT = 0.004, 100.0

    def __init__(self, port: str = "VIRTUAL", baud: int = 115200):
        self.port = port
        self.ser = _DummySerial()
        self._power_set = False      # 측정빔 무장 여부
        self.power_pct = None        # 마지막으로 설정한 파워(%)
        self.is_on = False           # 발진 중인가 — 카메라가 스팟을 그릴지 결정할 때 읽는다

    # ── 빔 ──────────────────────────────────────────────────────────────────
    def laser_on(self) -> bool:
        self.is_on = True
        return True

    def laser_off(self) -> bool:
        """실물은 '발진 정지를 확인 못 했다'를 False 로 알린다. 가상은 항상 확인된다.

        _power_set 은 건드리지 않는다 — 위 머리말 참고.
        """
        self.is_on = False
        return True

    def set_guide_beam(self) -> bool:
        """ND 필터를 차단 위치로. 측정빔 무장이 여기서 풀린다."""
        self._power_set = False
        return True

    # ── 파워 ────────────────────────────────────────────────────────────────
    def set_power(self, percent) -> bool:
        """실패를 예외가 아니라 False 로 알리는 실물 계약을 유지한다.

        범위 밖 값을 클리핑하는 것도 실물과 같다. 도구 계층(_apply_laser_power)이 그보다
        먼저 거부하므로 여기까지 오는 값은 이미 유효하다 — 클리핑은 마지막 방어선이다.
        """
        try:
            p = float(percent)
        except (TypeError, ValueError):
            return False
        self.power_pct = max(self.ND_MIN_PCT, min(self.ND_MAX_PCT, p))
        self._power_set = True
        return True

    def pulse_to_percent(self, pulse):
        """실물은 ND 모터 펄스 위치를 % 로 되돌린다. 가상에는 모터가 없어 그대로 돌려준다."""
        return float(pulse)

    def home_search(self) -> bool:
        return True

    def close(self) -> None:
        self.laser_off()
        self.set_guide_beam()
        self.ser.close()

    @property
    def beam(self) -> str:
        """'off' | 'guide' | 'measurement'. 로그·표현용이며 판정의 정본은 아니다
        (정본은 hw_core._beam_state)."""
        if not self.is_on:
            return "off"
        return "measurement" if self._power_set else "guide"

    def __repr__(self) -> str:
        return f"<VirtualLaser {self.beam} power={self.power_pct}%>"
