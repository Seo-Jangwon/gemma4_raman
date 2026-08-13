# -*- coding: utf-8 -*-
"""레이저 도구.

    laser_on             발진 시작
    laser_off            발진 정지
    set_laser_power      ND 필터 투과율 설정(발사하지 않는다)
    get_laser_status     발사 여부·파워·무장 상태 조회
    set_guide_beam_mode  가이드빔 대기 상태로 전환

[빔이 둘이다 — 이 파일의 핵심]
같은 "레이저 ON" 이라도 실제로 나가는 빔이 둘로 갈린다.

    ND 필터가 투과 위치  → 측정빔. 라만 신호가 나온다.
    ND 필터가 차단 위치  → 가이드빔. 정렬용이고 신호는 0 이다.

어느 쪽인지는 파워가 광학계에 **적용되어 있는가**(power_armed)로 정해진다. 그래서 이
파일의 도구는 전부 `beam` / `power_armed` 를 결과에 실어, 호출자가 "켰는데 왜 신호가
없지" 상태에 빠지지 않게 한다. 판정은 hw_core._beam_state 한 곳이 한다.

[측정빔을 실제로 쏘는 경로는 여기가 아니다]
acquire_tools.acquire_spectrum 이 파워 적용 → ON → 측정 → OFF 를 원자적으로 처리한다.
여기 도구를 이어 붙여 측정하면 그 사이(모델이 다음 수를 생각하는 시간) 동안 빔이 시편에
그대로 얹혀 있게 된다.

파워 적용은 hw_core._apply_laser_power 를 지난다 — 범위 검증과 ND 모터 정착 대기가
거기 한 곳에 있다.
"""
from __future__ import annotations

# 도구 응답 형식. 성공 ok(**payload) / 실패 fail(사유).
from backend.tools.result import fail, ok
from pydantic import Field
from typing import Annotated
# 장비 핸들은 모듈 속성으로 읽는다(init_hardware 가 전역을 갈아 끼운다).
from backend.tools.hw_tools.hw_tools import hw_core as _hw
# _apply_laser_power  파워 검증·적용·정착 대기 (통과=None, 실패=에러 dict)
# _beam_state         지금 켜면 어느 빔이 나가는지 판정
# _restore_guide_beam_quiet  ND 를 차단 위치로 — 예외를 삼키는 최후 차단용
# _serialized         장비 락 데코레이터
from backend.tools.hw_tools.hw_tools.hw_core import _apply_laser_power, _beam_state, _restore_guide_beam_quiet, _serialized


# ══════════════════════════════════════════════════════════════════════════════
# 발진 제어
# ══════════════════════════════════════════════════════════════════════════════

@_serialized("laser_on")
def laser_on() -> dict:
    """레이저를 켠다. **어떤 빔이 나가는지를 함께 보고한다.**

    Returns
    -------
    dict
        ``{"ok": True, "status": "Laser ON (measurement|guide beam)", "beam": …}``
        측정빔이면 ``power_percent`` 가, 가이드빔이면 무장 방법을 적은 ``note`` 가 붙는다.

    Notes
    -----
    이 도구는 측정빔을 무장시키지 못한다. 파워가 적용되지 않은 상태(가이드빔 모드나
    오토포커스 직후)에서는 드라이버가 발진 명령을 받아도 가이드빔만 낸다.
    """
    if _hw._laser is None:
        return fail("Laser is not initialized.")

    # 발진 전에 판정한다 — 켠 뒤에 물으면 이미 무엇이 나갔는지 모른다.
    st = _beam_state()
    armed, beam = st["power_armed"], st["beam"]

    try:
        _hw._laser.laser_on()
    except Exception as e:
        return fail(str(e))

    out = ok(status=f"Laser ON ({beam} beam)", beam=beam)
    if armed:
        out["power_percent"] = st["power_percent"]     # 측정빔 — 실제 적용된 파워를 싣는다
    else:
        # 가이드빔이면 신호가 0 으로 나오므로, 원인과 다음 수를 결과 안에 적어 준다.
        out["note"] = (
            "Only the GUIDE beam is emitted - the laser power has not been applied to the optics, "
            "so the measurement beam is still blocked by the ND filter. For a real measurement call "
            "set_laser_power(percent) first, or use acquire_spectrum(power=...) which does "
            "power -> on -> acquire -> off atomically.")
    return out


@_serialized("laser_off")
def laser_off() -> dict:
    """레이저 발진을 정지한다. 파워 설정과 광학계 위치는 그대로 둔다.

    Returns
    -------
    dict
        성공: ``{"ok": True, "status": "Laser OFF", "power_armed": …, "power_percent": …}``
        정지 미확인: ``ok=False`` + ``laser_off_unconfirmed`` / ``nd_blocked``.

    Notes
    -----
    드라이버는 정지 명령의 ACK 를 못 받으면 ``False`` 를 돌려준다. 그때 '껐다'고 보고하면
    이후의 모든 안전 판단이 그 위에 쌓이므로, ND 필터로 차단을 한 번 더 시도하되 결과는
    실패로 올린다.
    """
    if _hw._laser is None:
        return fail("Laser is not initialized.")

    try:
        confirmed = _hw._laser.laser_off()
        st = _beam_state()

        if confirmed is False:
            # 발진 정지를 확인하지 못했다 — 빔이 아직 살아 있을 수 있다.
            # 광로 차단(ND)을 최후 수단으로 한 번 더 시도한다.
            blocked = _restore_guide_beam_quiet()
            return fail("Laser stop (SSPW 0) was NOT confirmed - the beam may still be on. " +
                        ("The ND filter has been moved to the blocking position. " if blocked else
                         "ND blocking could not be confirmed either. ") +
                        "Check the laser controller link.",
                        laser_off_unconfirmed=True,
                        nd_blocked=bool(blocked),
                        power_armed=st["power_armed"])

        # 파워는 남겨 둔 채 발진만 멈춘 상태 — 다시 켜면 같은 빔이 나간다는 뜻이라 함께 보고.
        return ok(status="Laser OFF",
                  power_armed=st["power_armed"],
                  power_percent=st["power_percent"])
    except Exception as e:
        return fail(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 파워·상태
# ══════════════════════════════════════════════════════════════════════════════

@_serialized("set_laser_power")
def set_laser_power(
    percent: Annotated[float, Field(description='Laser power as ND filter transmission in percent (0.004-100).')],
) -> dict:
    """ND 필터 투과율을 설정해 측정빔을 무장한다. **발사하지는 않는다.**

    Parameters
    ----------
    percent : float
        투과율(%). 0.004~100 의 실수. 범위 밖은 잘라내지 않고 거부한다.

    Returns
    -------
    dict
        적용 후의 빔 상태(``power_armed`` · ``power_percent`` · ``beam``).

    Notes
    -----
    측정이 목적이라면 acquire_spectrum(power=…) 이 낫다 — 적용·발사·정지가 원자적이라
    빔이 켜진 채 남지 않는다. 이 도구는 파워만 미리 걸어 둘 때(정렬 후 무장, 여러 측정에
    같은 파워 재사용) 쓴다. 두 경로 모두 _apply_laser_power 를 지나므로 동작은 같다.
    """
    err = _apply_laser_power(percent)      # 범위 검증 → ND 이동 → 정착 대기까지 공용 경로
    if err:
        return err
    return ok(**_beam_state())             # 요청값이 아니라 적용 후 상태를 보고한다


def get_laser_status() -> dict:
    """레이저 상태를 조회한다 — 발사 여부, 파워(%), 그 파워의 실제 적용 여부.

    Returns
    -------
    dict
        ``is_on`` · ``power_armed`` · ``power_percent`` · ``beam_if_turned_on``.
        무장 상태가 아니면 ``last_requested_power_percent`` 와 안내 ``note`` 가 붙는다.

    Notes
    -----
    ``power_percent`` 만 보면 안 된다. 그 값은 '마지막으로 요청한 파워'라 가이드빔 모드로
    전환된 뒤에도 남아 있어서, 실제로는 ND 가 차단 위치인데도 준비된 것처럼 보인다.
    판단 기준은 항상 ``power_armed`` 다.
    """
    if _hw._laser is None:
        return fail("Laser is not initialized.")

    try:
        st = _beam_state()
        armed, last = st["power_armed"], st["power_percent"]

        out = ok(is_on=bool(getattr(_hw._laser, "is_on", False)),
                 power_armed=armed,
                 # 무장 상태일 때만 유효한 값이므로, 아니면 None 으로 비워 오해를 막는다.
                 power_percent=last if armed else None,
                 beam_if_turned_on=st["beam"])

        if not armed:
            # 값 자체는 버리지 않고 이름을 바꿔 남긴다 — '요청했었다'는 정보는 유용하다.
            out["last_requested_power_percent"] = last
            out["note"] = (
                "Power is NOT applied to the optics right now - the ND filter sits at the "
                "guide-beam/blocking position, so laser_on() would emit only the guide beam. "
                "Call set_laser_power(percent) to arm the measurement beam.")
        return out
    except Exception as e:
        return fail(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 가이드빔
# ══════════════════════════════════════════════════════════════════════════════

@_serialized("set_guide_beam_mode")
def set_guide_beam_mode() -> dict:
    """가이드빔 대기 상태로 전환한다.

    빔 스플리터(축04)를 대기 위치로, ND 필터(축02)를 메인빔 차단 위치로 옮긴다.
    측정빔을 쓰지 않고 시편 정렬·초점을 확인할 때의 상태다.

    Returns
    -------
    dict
        ``{"ok": True, "status": "Switched to guide-beam mode"}``

    Notes
    -----
    acquire_spectrum 은 측정이 끝나면 이 상태로 알아서 되돌린다(카메라가 다시 보이게).
    측정 뒤에 이 도구를 따로 부를 필요는 없다.
    """
    if _hw._laser is None:
        return fail("Laser is not initialized.")
    try:
        _hw._laser.set_guide_beam()
        return ok(status="Switched to guide-beam mode")
    except Exception as e:
        return fail(str(e))
