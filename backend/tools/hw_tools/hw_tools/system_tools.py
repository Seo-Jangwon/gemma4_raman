# -*- coding: utf-8 -*-
"""장비 연결 자체를 다루는 도구 — 상태 진단과 재연결.

다른 도구가 실패했을 때 '한 대가 죽었는가, 여러 대가 죽었는가'를 먼저 알려 주는 자리다.
재연결은 자원 점유 해제(_teardown_component)와 재초기화를 분리해서, 프로세스 락 때문에
재시도가 무의미한 경우와 장비 쪽 문제라 한 번은 재시도할 만한 경우를 구분해 보고한다.
"""
from __future__ import annotations

import time

from backend.tools.result import fail, ok
from pydantic import Field
from typing import Annotated, Literal, Optional
from backend.tools.hw_tools.hw_tools import hw_core as _hw
from backend.tools.hw_tools.hw_tools.hw_core import InstrumentBusy, _STAGE_BUSY, _last_stage_values, _read_stage_values, _stage_read, instrument_guard, sync_tool_handles


# 해제와 재연결 사이의 대기(초). DLL 세션/COM 포트/USB 핸들은 close() 가 돌아온 뒤에도
# OS 가 잠시 붙잡고 있어, 곧바로 재연결하면 '방금 내가 닫은 자원'에 스스로 걸린다.
_RECONNECT_SETTLE_S = 1.5
# 재초기화 시도 횟수. 자원 해제 타이밍 문제는 보통 1회 재시도로 풀리고,
# 진짜 장비 문제라면 몇 번을 더 해도 안 되므로 크게 잡을 이유가 없다.
_RECONNECT_ATTEMPTS = 2


def _teardown_component(mgr, comp: str) -> dict:
    """재접속 전 해당 컴포넌트를 해제하고, 무엇이 성공/실패했는지 보고한다.

    [이 함수가 왜 이렇게 장황해졌는가 — 2026-07-29 버그 수정]
    이전 구현은 모든 예외를 삼키고 무조건 `mgr.<comp> = None` 을 실행했다. 그 결과
    '해제가 실패했는데 핸들만 버리는' 상황이 생겼다:

        disconnect() 가 에러코드를 반환(실패) → 예외가 아니라 False 이므로 무시됨
        → mgr.stage = None 으로 유일한 참조를 버림
        → DLL 세션(LSID)은 여전히 열려 있는데 이제 아무도 LSX_FreeLSID 를 부를 수 없다
        → 이후 _init_stage() 의 LSX_ConnectSimple 은 영구히 "이미 점유 중"으로 실패
        → 에이전트가 reconnect_hardware 를 몇 번 불러도 절대 회복 불가(프로세스 재시작만이 답)

    이게 "하드웨어가 점유중이라 안 끊긴다"의 실체다. 그래서 이제:
      · 해제 각 단계의 성공/실패를 기록해서 돌려준다(조용히 성공한 척하지 않는다).
      · '해제가 확인된 경우에만' 참조를 버린다. 실패했으면 핸들을 남겨 다음 재시도가
        같은 객체로 다시 해제를 시도할 수 있게 한다 — 고아 세션을 만들지 않는다.

    Returns
    -------
    dict — {"released": bool, "steps": {단계명: "ok" | "실패이유"}}
           released=False 면 자원이 아직 잡혀 있다는 뜻이고, 호출자는 재초기화를
           시도하기 전에 이 사실을 알아야 한다.
    """
    steps: dict = {}

    def _try(name, fn) -> bool:
        """단계 하나 실행. 예외도, 'False 반환'도 실패로 취급한다 —
        이 프로젝트의 장비 래퍼들은 실패를 예외가 아니라 False/에러코드로 알린다."""
        try:
            r = fn()
        except Exception as e:
            steps[name] = f"{type(e).__name__}: {e}"
            return False
        if r is False:                      # None 은 '반환값 없음'이므로 성공으로 본다
            steps[name] = "returned False (device reported failure)"
            return False
        steps[name] = "ok"
        return True

    obj = getattr(mgr, comp, None)
    if obj is None:
        return {"released": True, "steps": {"skip": "was not connected"}}

    if comp == "stage":
        _try("disconnect", obj.disconnect)
        # free_session 이 성공하면 DLL 세션이 사라지므로, disconnect 가 실패했어도
        # 참조를 버려도 안전하다. 반대로 free 가 실패하면 반드시 핸들을 남겨야 한다.
        freed = _try("free_session", obj.free_session)
        released = freed
        if not released and hasattr(obj, "mark_dead"):
            # 세션을 풀지 못했다 = 이 핸들로는 아무도 자원을 되찾을 수 없다(프로세스 재시작만이 답).
            # 핸들은 남기되(고아 세션 방지) '죽었다'고 표시한다 — 안 그러면 get_hardware_status 가
            # connected: true 로 보고하고, 에이전트는 죽은 스테이지로 다음 작업을 시도한다.
            obj.mark_dead(f"release failed during reconnect ({steps})")
    elif comp == "camera":
        _try("stop_stream", obj.stop_stream)
        released = _try("close", obj.close)
    elif comp == "ccd":
        released = _try("close", obj.close)
    elif comp == "laser":
        _try("laser_off", obj.laser_off)

        def _close_serial():
            ser = getattr(obj, "ser", None)
            if ser is None:
                return None
            if getattr(ser, "is_open", False):
                ser.close()
            # 닫힌 것을 확인한다 — COM 포트가 남아 있으면 재연결이 'Access is denied' 로 죽는다.
            return not getattr(ser, "is_open", False)

        released = _try("close_serial", _close_serial)
    else:
        return {"released": False, "steps": {"error": f"unknown component '{comp}'"}}

    if released:
        setattr(mgr, comp, None)
    return {"released": released, "steps": steps}


def get_hardware_status() -> dict:
    """어느 장비가 연결되어 있는지, 안 되어 있으면 왜인지 한눈에 돌려준다.

    [왜 이 도구가 필요한가 — 2026-07-29]
    에이전트에게는 '무엇이 연결되어 있는가'를 물어볼 수단이 아예 없었다. 서버에는
    /api/hardware/state 가 있지만 그건 프론트엔드용 HTTP 엔드포인트이고 도구가 아니다.
    그래서 에이전트는 도구를 하나 찔러보고 "Stage is not initialized." 를 받은 뒤에야
    상황을 짐작했고, 여러 장비가 동시에 죽으면 무엇부터 손대야 할지 판단할 근거가 없어
    같은 reconnect 를 반복하거나 그냥 포기했다. 진단을 먼저 할 수 있게 해 준다.
    """
    try:
        from backend.tools.hw_tools.hao.hardware_manager import get_manager
        mgr = get_manager()
    except Exception as e:
        return fail(f"HardwareManager unavailable: {type(e).__name__}: {e}")

    out: dict = ok(connected={}, notes={})
    for name in ("stage", "laser", "ccd", "camera"):
        obj = getattr(mgr, name, None)
        # 핸들이 있어도 '죽은' 세션은 연결로 치지 않는다 — 해제에 실패한 핸들은 고아 세션을
        # 막으려고 일부러 남겨둔 것이지, 쓸 수 있다는 뜻이 아니다(_teardown_component 참고).
        dead = obj is not None and getattr(obj, "dead", False)
        out["connected"][name] = obj is not None and not dead
        if obj is None:
            out["notes"][name] = ("Not connected. Try reconnect_hardware(component='%s'). "
                                  "If that reports the resource is still held, the server process "
                                  "must be restarted - no tool can clear it." % name)
        elif dead:
            out["notes"][name] = (
                "Dead session (%s): the handle still exists but the DLL session could not be "
                "released, so no tool can revive it. reconnect_hardware will NOT help - the server "
                "process must be restarted. Continue without the %s if the task allows it."
                % (getattr(obj, "dead_reason", "unknown"), name))

    # 연결된 것은 실제로 응답하는지도 가볍게 확인한다 — 핸들이 살아 있어도 장비가
    # 먹통이면 '연결됨'만 보고 진행하다 뒤에서 터진다.
    # 읽기도 component_lock('stage') 안에서만 한다 — 재연결 중 DLL 동시 호출이 세션을
    # 죽인다(_stage_read 주석 참고).
    if out["connected"].get("stage"):
        try:
            probe = _stage_read(lambda: mgr.stage.get_position())
            if probe is _STAGE_BUSY:
                out["notes"]["stage"] = ("The stage is being connected or released by another thread "
                                         "right now, so its position was not read. Nothing is wrong - "
                                         "check again in a few seconds.")
            else:
                out["stage_position"] = (
                    {"x": probe[0], "y": probe[1], "z": probe[2]} if probe else None)
                if not probe:
                    out["notes"]["stage"] = ("Handle exists but get_position() returned nothing - "
                                             "stage is unresponsive.")
        except Exception as e:
            out["notes"]["stage"] = f"Handle exists but reading position raised {type(e).__name__}: {e}"
    if mgr.laser is not None:
        try:
            ser = getattr(mgr.laser, "ser", None)
            out["laser_serial_open"] = bool(getattr(ser, "is_open", False)) if ser else None
        except Exception as e:
            out["notes"]["laser"] = f"{type(e).__name__}: {e}"

    ready = [k for k, v in out["connected"].items() if v]
    out["summary"] = (f"connected: {', '.join(ready) if ready else 'none'}; "
                      f"missing: {', '.join(k for k, v in out['connected'].items() if not v) or 'none'}")
    return out


def hardware_snapshot(mgr=None) -> dict:
    """프론트 파라미터 패널용 현재 장비 설정 스냅샷.

    [왜 여기 있는가 — 2026-07-30]
    server.py 의 /api/hardware/state 가 CCD·레이저·스테이지 상태를 getattr 로 직접
    다시 읽고 있었다. get_ccd_info / get_laser_status / get_stage_position /
    get_stage_speed 와 같은 값을 읽는 다섯 번째 경로였고, 필드 이름과 폴백 규칙이
    미묘하게 달라 프론트와 에이전트가 서로 다른 숫자를 보는 일이 생겼다.

    mgr 를 주면 그 매니저의 핸들을, 생략하면 이 모듈의 전역 핸들을 읽는다 —
    서버는 매니저를, 도구 경로는 주입된 전역을 쓰기 때문이다.
    반환 키는 프론트가 그대로 소비하므로 바꾸지 말 것.
    """
    ccd    = getattr(mgr, "ccd", None)    if mgr is not None else _hw._ccd
    laser  = getattr(mgr, "laser", None)  if mgr is not None else _hw._laser
    stage  = getattr(mgr, "stage", None)  if mgr is not None else _hw._stage

    out: dict = {"ccd": None, "laser": None, "stage": None}

    if ccd is not None:
        info = {
            "exposure_time": getattr(ccd, "exposure_time", None),
            "acq_mode":      getattr(ccd, "aq_mode",       "single"),
            "num_acc":       getattr(ccd, "num_acc",        1),
            "num_kin":       getattr(ccd, "num_kin",        1),
            "ro_mode":       getattr(ccd, "ro_mode",        "fvb"),
            "preamp_gain_i": getattr(ccd, "preamp_gain_i",  0),
            "preamp_gains":  getattr(ccd, "preamp_gains",   []),
            "shutter":       getattr(ccd, "shutter_mode",   "auto"),
            "temperature":   None,
        }
        try:
            info["temperature"] = int(ccd.get_temperature())
        except Exception:
            pass                            # 촬영 중이면 온도 접근이 막힌다 — 미상으로 둔다
        out["ccd"] = info

    if laser is not None:
        out["laser"] = {"power_pct": getattr(laser, "power_pct", None),
                        "is_on":     getattr(laser, "is_on",     None)}

    if stage is not None:
        # 이 함수는 프론트가 1초마다 폴링하는 경로다. 재연결이 도는 동안 여기서 DLL 을
        # 부르면 LSX_Disconnect/LSX_FreeLSID 와 겹쳐 세션을 죽인다(_stage_read 주석).
        # 락을 못 잡으면 DLL 을 건드리지 않고 직전 값을 그대로 보여준다 — 재연결은 몇 초라
        # 패널이 잠깐 안 움직일 뿐이고, 숫자가 사라졌다 나타나는 것보다 덜 혼란스럽다.
        snap = _stage_read(lambda: _read_stage_values(stage))
        if snap is _STAGE_BUSY:
            snap = _last_stage_values.get("value", {})
        else:
            _last_stage_values["value"] = snap
        out["stage"] = snap

    return out


def reconnect_hardware(
    component: Annotated[Optional[Literal['stage', 'ccd', 'camera', 'laser', 'all']], Field(description="Which component to reconnect. Default 'all'.")] = 'all',
) -> dict:
    """카메라/스테이지/CCD/레이저 연결을 끊었다가 재초기화한다.

    component: 'stage' | 'ccd' | 'camera' | 'laser' | 'all' (기본 all).
    주의: CCD 재초기화는 -40C 냉각 안정화까지 수 분간 블로킹될 수 있다.
    재초기화 후 hw_core 전역 핸들을 새 객체로 다시 주입한다.
    """
    try:
        from backend.tools.hw_tools.hao.hardware_manager import get_manager
    except Exception as e:
        return fail(f"HardwareManager import failed: {e}")

    comp = str(component or "all").strip().lower()
    valid = {"stage", "ccd", "camera", "laser", "all"}
    if comp not in valid:
        return fail(f"component must be one of {sorted(valid)}")

    try:
        mgr = get_manager()
    except Exception as e:
        # 여기서 예외가 나면 도구가 에러 dict 가 아니라 날것의 예외를 던진다 —
        # 에이전트는 그것을 관측으로 읽을 수 없으므로 반드시 감싼다.
        return fail(f"HardwareManager unavailable: {type(e).__name__}: {e}")

    targets = ["stage", "ccd", "camera", "laser"] if comp == "all" else [comp]
    done, errors, detail = [], {}, {}

    # ── 장비 락을 '해제부터 핸들 재주입까지' 통째로 쥔다 ──────────────────────
    # [왜 컴포넌트마다가 아니라 전체 구간인가 — 2026-07-31]
    # 처음에는 컴포넌트 루프 안에서만 잡았는데, 루프가 락을 놓은 뒤 sync_tool_handles()
    # 전에 틈이 생겼다. 그 틈에 다른 스레드의 측정이 끼어들면 이미 close() 된 옛 CCD
    # 핸들로 촬영을 시도한다(전역 _hw._ccd 는 아직 교체 전이다). 테스트로 재현됐다:
    #     ['reconnect_start', 'measure_start', 'measure_end', 'reconnect_end']
    # 그래서 교체가 끝날 때까지 놓지 않는다.
    #
    # 락 순서는 instrument_guard → component_lock 로 고정한다(instrument_guard
    # docstring 참고). 서버의 connect 엔드포인트도 같은 순서를 쓴다.
    try:
        _guard = instrument_guard(f"reconnect_hardware({comp})")
        _guard.__enter__()
    except InstrumentBusy as e:
        return fail(str(e),
                    reconnected=[],
                    detail={"skipped": "instrument busy"},
                    hint="A measurement or scan is still running. Nothing was changed - "
                         "wait for it to finish (check get_hardware_status()) and retry.")

    try:
        return _reconnect_locked(mgr, targets, done, errors, detail)
    finally:
        _guard.__exit__(None, None, None)


def _reconnect_locked(mgr, targets, done, errors, detail) -> dict:
    """reconnect_hardware 의 본체 — **장비 락을 쥔 상태로만** 불린다."""
    for t in targets:
        # ── 장비 락을 '기다리지 않고' 잡는다 ──────────────────────────────────
        # 이 매니저는 서버와 공유하는 단일 인스턴스다. 락이 잡혀 있다는 것은 다른
        # 스레드(서버의 /api/*/connect 엔드포인트, 또는 수 분간 도는 CCD 냉각 스레드)가
        # 이미 이 장비를 만지고 있다는 뜻이다. 블로킹으로 기다리면 CCD 냉각 뒤에 붙어
        # 도구 호출이 몇 분간 멈춘 것처럼 보이므로, 기다리지 않고 '지금 진행 중'이라고
        # 알려 에이전트가 스스로 판단하게 한다.
        lock = mgr.component_lock(t)
        if not lock.acquire(blocking=False):
            detail[t] = {"skipped": "component busy (lock held by another thread)"}
            errors[t] = (
                f"The {t} is being connected or released by another thread right now (the "
                f"server's own reconnect endpoint, or the CCD cooling thread which runs for "
                f"several minutes). Reconnecting was NOT attempted: two threads initializing the "
                f"same device at once can orphan the handle and force a server restart. Do NOT "
                f"retry immediately - continue with the remaining hardware, then check "
                f"get_hardware_status() to see whether it came back on its own."
            )
            continue

        try:
            td = _teardown_component(mgr, t)
            detail[t] = {"teardown": td}

            if not td["released"]:
                # 자원이 아직 잡혀 있다 — 이 상태로 재초기화하면 "이미 점유 중"으로 실패한다.
                # 그걸 'reconnect 실패'로 뭉개지 말고, 원인과 다음 행동을 명시해 돌려준다.
                errors[t] = (
                    f"Could not release the {t} before reconnecting - the resource is still held "
                    f"by this process (teardown steps: {td['steps']}). Reconnecting was NOT attempted, "
                    f"because doing so would fail with an 'already in use' error and could orphan the "
                    f"handle. This is a process-level lock, not a cable problem: retrying this tool will "
                    f"not clear it. Continue with the remaining hardware if possible, report this in your "
                    f"final answer, and note that the server process must be restarted to free the {t}."
                )
                continue

            # 해제와 재연결 사이에 잠깐 쉰다 — DLL 세션/COM 포트/USB 핸들은 OS 가 즉시
            # 놓아주지 않는다. 없으면 '방금 내가 닫은 자원'에 스스로 걸려 실패한다.
            time.sleep(_RECONNECT_SETTLE_S)

            last = None
            for attempt in range(1, _RECONNECT_ATTEMPTS + 1):
                try:
                    # _init_<t> 는 같은 락을 다시 잡는다(_guarded). RLock 이라 재진입 OK.
                    getattr(mgr, f"_init_{t}")()
                    done.append(t)
                    detail[t]["init"] = f"ok (attempt {attempt})"
                    last = None
                    break
                except Exception as e:
                    last = f"{type(e).__name__}: {e}"
                    detail[t]["init"] = f"attempt {attempt} failed: {last}"
                    if attempt < _RECONNECT_ATTEMPTS:
                        time.sleep(_RECONNECT_SETTLE_S)
            if last is not None:
                errors[t] = (
                    f"Released the {t} successfully but re-initialization failed after "
                    f"{_RECONNECT_ATTEMPTS} attempts: {last}. The resource is now free, so this is a "
                    f"device-side problem (power, cable, driver, or the device is held by another "
                    f"program such as rays-on.exe) - not something you can fix by calling tools again. "
                    f"Proceed without the {t} if the task allows it and say so explicitly in your answer."
                )
        finally:
            lock.release()

    # 재초기화된 객체를 hw_core 전역에 재주입(server 의 connect 엔드포인트와 공용 경로).
    # **아직 장비 락 안이다** — 여기까지 와야 옛 핸들로 측정이 들어가는 틈이 사라진다.
    # init_hardware 도 같은 락을 잡지만 RLock 이라 재진입으로 통과한다.
    sync_tool_handles(mgr)
    now_connected = {k: getattr(mgr, k, None) is not None
                     for k in ("stage", "laser", "ccd", "camera")}

    # [왜 dict 리터럴이 아니라 ok()/fail() 인가 — 2026-08-12]
    # 예전에는 {"ok": len(errors) == 0, ..., "errors": errors} 였다. 응답 형식 규약은 실패에
    # error(단수)를 요구하는데 사유가 errors(복수)에만 있어서, normalize() 가 "사유 없는
    # 실패"로 보고 진단을 통째로 버렸다. 155초짜리 재연결의 기록이 한 줄로 뭉개졌다.
    # 여기서 규약대로 답하면 그 자리를 아예 지나가지 않는다.
    #
    # 부분 성공(넷 중 셋만 붙음)도 실패다 — 남은 하나로 측정이 안 되기 때문이다.
    # 다만 reconnected 를 함께 실어, 무엇이 살았는지는 모델이 알고 판단하게 한다.
    if errors:
        return fail("; ".join(f"{k}: {v}" for k, v in errors.items()),
                    reconnected=done, detail=detail, now_connected=now_connected)
    return ok(reconnected=done, detail=detail, now_connected=now_connected)
