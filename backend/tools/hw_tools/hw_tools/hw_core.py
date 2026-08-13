# -*- coding: utf-8 -*-
"""장비 핸들·직렬화 락·세션 상태·공용 검사 — 도구 모듈 전부가 여기에 기댄다.

여기 있는 것은 '어느 한 장비의 조작'이 아니라 **여러 도구가 같은 규칙을 쓰게 하는 것**들이다:
장비 핸들 4개와 그 주입 경로(init_hardware/sync_tool_handles), DLL 동시 호출을 막는
직렬화 락, 대화 세션별 도구 상태(_sstate), 그리고 파워 적용·범위 검증·CCD 설정 적용처럼
여러 도구에 복사되기 쉬운 정책들이다.

[핸들을 왜 이 모듈의 속성으로 읽어야 하는가]
_stage/_laser/_ccd/_camera 는 init_hardware() 가 global 로 **재바인딩**한다. 다른 모듈이
`from hw_core import _stage` 로 받으면 그 시점의 None 을 영원히 붙들고 있게 된다. 그래서
도구 모듈은 `from ... import hw_core as _hw` 로 모듈을 받아 `_hw._stage` 로 읽는다.
"""
from __future__ import annotations

import time

from backend.tools.hw_tools.config import STAGE_MAX_X, STAGE_MAX_Y, STAGE_MAX_Z, STAGE_MIN_Z
from backend.tools.result import fail


#  정의하고 있었다 — 2026-07-30 제거. 이제 config 에서 import 한다.)

_stage = None
_laser = None
_ccd   = None
_camera = None


# ══════════════════════════════════════════════════════════════════════════════
# 동시성 — 장비 조작 직렬화 (2026-07-31)
#
# [무엇이 문제였나]
# 이 서버는 ThreadPoolExecutor(max_workers=4) 로 돌고, 아래 경로들이 **같은 전역
# 핸들**을 동시에 만질 수 있다:
#     · 에이전트 도구 호출     /api/experiment/stream → 워커 스레드
#     · 패널 수동 측정         /api/spectrum/acquire  → 워커 스레드
#     · 패널 파라미터 변경     /api/stage/speed, /api/stage/move-pixel
#     · 장비 연결/재연결       /api/*/connect, reconnect_hardware
#
# 락이 없을 때 실제로 재현된 것(가짜 하드웨어, 느린 CCD):
#     laser 이벤트: [set_power, on, set_power, on, off, off]
#   ① A 가 촬영하는 도중 B 의 set_power 가 ND 필터를 옮긴다 → A 의 결과에는
#      'laser_power_pct: 20' 이 찍히지만 실제로는 80% 로 조사됐다. 에러 없이 값만
#      틀리고, 시료에는 의도한 4배가 들어간다(안전 문제이기도 하다).
#   ② A 의 finally: laser_off() 가 B 의 촬영 중에 발화한다 → B 는 조사 없는
#      스펙트럼을 얻는데 ok:True 로 돌아온다.
#
# [왜 hardware_manager 의 component_lock 으로는 부족한가]
# 그 락은 '연결/해제 수명주기'만 직렬화한다(_guarded 데코레이터). 측정·이동처럼
# 이미 연결된 장비를 **쓰는** 경로는 그 락을 지나지 않는다.
#
# [설계]
# 장비 하나(레이저·CCD·스테이지·카메라)는 물리적으로 동시에 두 작업을 할 수 없다.
# 그래서 자원별로 쪼개지 않고 '장비 조작' 하나로 직렬화한다. 쪼개면 acquire_spectrum
# 처럼 레이저+CCD+스테이지를 함께 쓰는 도구에서 락 순서 문제가 생긴다.
#
# RLock 인 이유: run_grid_scan → move_stage → acquire_spectrum 처럼 도구가 도구를
# 부른다. 같은 스레드의 재진입이므로 RLock 이 아니면 자기 자신에게 걸려 교착한다.
#
# 조회 전용 도구(get_*)는 이 락을 잡지 않는다 — 프론트가 /api/hardware/state 를
# 주기적으로 폴링하는데, 10분짜리 격자 스캔 동안 패널이 얼어붙으면 안 된다. 조회는
# 모두 None 검사와 try/except 를 갖고 있어 최악의 경우 에러 dict 로 떨어진다.
# ══════════════════════════════════════════════════════════════════════════════

import contextlib
import functools
import threading

_INSTRUMENT_LOCK = threading.RLock()

# 락 대기 상한(초). 무한 대기는 금지 — 워커 4개가 전부 대기에 잠기면 서버가 먹통이
# 된 것처럼 보인다. 측정 한 번(노출×누적)이 수 초~수십 초일 수 있어 넉넉히 잡되,
# 격자 스캔처럼 분 단위로 도는 작업 뒤에 붙으면 '지금 바쁘다'고 알리는 편이 낫다.
_BUSY_TIMEOUT_S = 30.0

# 지금 락을 쥔 작업 이름(진단용). 락 안에서만 쓰므로 별도 보호가 필요 없다.
_lock_holder: dict = {"what": None}


class InstrumentBusy(RuntimeError):
    """장비 락을 제한 시간 안에 얻지 못했다. 호출자가 사용자에게 알려야 한다."""


@contextlib.contextmanager
def instrument_guard(what: str = "hardware connect", timeout: float = None):
    """장비 조작을 직렬화하는 공개 컨텍스트 매니저 — 도구 밖(서버 엔드포인트)용.

    [락 순서 — 반드시 지킬 것]
    이 프로젝트의 유일한 순서는 **instrument_guard → component_lock** 이다.
    장비 연결/해제 경로는 전부 이 순서를 쓴다:

        with instrument_guard("camera connect"), hw.component_lock("camera"):
            ...

    반대로 잡으면(component_lock 을 쥔 채 이 가드에 들어가면) reconnect_hardware 와
    ABBA 교착이 난다. 순서를 바꾸는 코드를 새로 만들지 말 것.

    타임아웃되면 InstrumentBusy 를 던진다 — 도구 계층은 에러 dict 를 돌려주지만,
    HTTP 엔드포인트는 예외를 그대로 잡아 사용자에게 보여주는 편이 낫다.
    """
    t = _BUSY_TIMEOUT_S if timeout is None else timeout
    if not _INSTRUMENT_LOCK.acquire(timeout=t):
        busy = _lock_holder.get("what") or "another operation"
        raise InstrumentBusy(
            f"The instrument is busy with '{busy}' and did not become free within {t:.0f}s, "
            f"so '{what}' was NOT performed. Wait for the running operation to finish and retry.")
    prev = _lock_holder.get("what")
    _lock_holder["what"] = what if prev is None else prev
    try:
        yield
    finally:
        _lock_holder["what"] = prev
        _INSTRUMENT_LOCK.release()


def _serialized(what: str, timeout: float = _BUSY_TIMEOUT_S):
    """장비를 실제로 움직이거나 조사하는 도구를 직렬화하는 데코레이터.

    타임아웃되면 예외가 아니라 **관측 가능한 에러 dict** 를 돌려준다 — 에이전트는
    이것을 읽고 기다렸다 재시도하거나 다른 일을 할 수 있어야 한다.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _INSTRUMENT_LOCK.acquire(timeout=timeout):
                busy = _lock_holder.get("what") or "another operation"
                return fail(f"The instrument is busy with '{busy}' and did not become free within "
                            f"{timeout:.0f}s, so '{what}' was NOT performed. Only one operation may drive "
                            f"the hardware at a time (a measurement, a stage move, or a grid scan started "
                            f"from the parameter panel or another chat session). Nothing was changed - "
                            f"wait for it to finish and try again.",
                            busy_with=busy)
            prev = _lock_holder.get("what")
            _lock_holder["what"] = what if prev is None else prev   # 최상위 작업 이름을 유지
            try:
                return fn(*args, **kwargs)
            finally:
                _lock_holder["what"] = prev
                _INSTRUMENT_LOCK.release()
        return wrapper
    return deco


# ── 스테이지 '읽기' 경로의 DLL 동시 호출 차단 ──────────────────────────────────
# [왜 필요한가 — 2026-07-31 사고]
# 조회 전용 경로(위 주석대로 _INSTRUMENT_LOCK 을 일부러 잡지 않는다)가 스테이지 DLL 을
# 직접 부른다. 프론트는 /api/hardware/state 를 1초마다 폴링하고, 서버는 워커 4개짜리
# 스레드풀로 그것을 처리한다. 그 사이 reconnect_hardware 가 다른 스레드에서
# LSX_Disconnect / LSX_FreeLSID 를 부르면 Tango DLL 안에서 C++ 예외(0xE06D7363)가
# 터져 나온다 — 해제가 실패하고, 세션은 고아가 되고, 스테이지는 프로세스 재시작
# 전까지 죽는다. 실제로 그렇게 죽였다.
#
# 그렇다고 읽기에 _INSTRUMENT_LOCK 을 잡으면 안 된다(10분짜리 스캔 동안 패널이 얼어붙는다).
# 대신 '연결/해제를 직렬화하는' component_lock('stage') 만 **기다리지 않고** 잡는다:
#   · 평소에는 아무도 안 잡고 있으므로 즉시 통과 → 폴링 성능 그대로
#   · 재연결 중(수 초)에는 즉시 실패 → DLL 을 아예 건드리지 않는다
# 락 순서(instrument_guard → component_lock)를 어기지 않는다 — 여기서는 component_lock
# 하나만 잡고, 그 안에서 다른 락을 잡지 않는다.
_STAGE_BUSY = object()      # '락을 못 잡았다'(= 연결/해제 진행 중). DLL 실패와 구분한다.


def _stage_read(fn):
    """스테이지 DLL 읽기를 component_lock('stage') 안에서만 수행한다.

    락을 못 잡으면 fn 을 **부르지 않고** _STAGE_BUSY 를 돌려준다 — 호출자는 직전 값을
    쓰거나 '지금 재연결 중'이라고 보고해야 한다. 매니저를 얻을 수 없는 환경(단독 스크립트,
    테스트)에서는 보호할 대상도 없으므로 그냥 실행한다.
    """
    try:
        from backend.tools.hw_tools.hao.hardware_manager import get_manager
        lock = get_manager().component_lock("stage")
    except Exception:
        return fn()

    if not lock.acquire(blocking=False):
        return _STAGE_BUSY
    try:
        return fn()
    finally:
        lock.release()


def _stage_unavailable() -> dict:
    """스테이지를 쓸 수 없으면 그 이유를 담은 에러 dict, 쓸 수 있으면 None.

    'None 이면 미초기화' 하나만 보던 검사를 한 곳에 모았다 — 해제에 실패해 죽은 세션은
    핸들이 남아 있어서 예전 검사를 그대로 통과했고, 도구들은 죽은 DLL 로 명령을 계속 보냈다.
    """
    if _stage is None:
        return fail("Stage is not initialized.")
    if getattr(_stage, "dead", False):
        return fail(f"The stage session is dead and cannot be used: {getattr(_stage, 'dead_reason', 'unknown')}. "
                    f"The DLL session could not be released, so no tool can revive it - the server process must "
                    f"be restarted. Do NOT retry reconnect_hardware; continue without the stage if the task "
                    f"allows it and say so explicitly in your answer.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 세션별 도구 상태 (2026-07-31)
#
# [왜 전역이면 안 되나]
# _last_spectrum / _last_scene / _bg_versions / _grid_gate 는 모두 '이 대화에서
# 방금 무엇을 했는가'다. 모듈 전역으로 두면 채팅 탭을 두 개 열었을 때 서로의 상태를
# 덮어쓴다. 그중 _grid_gate 는 **사람 승인 인터록**이라 결과가 특히 나쁘다:
# 벤치마크 실행(run_experiment)이 grid_gate_begin_turn(interactive=False) 로
# enforce 를 끄면, 동시에 열려 있던 대화 세션의 승인 게이트까지 꺼진다 —
# 즉 사람 승인 없이 레이저 격자 스캔이 나갈 수 있었다(테스트로 확인).
#
# 세션 라벨(run_store)로 키를 잡는다. 스레드로컬이 아니라 세션 키인 이유: 게이트는
# 턴 경계를 넘어 유지돼야 하는데(미리보기는 N턴, 승인·실행은 N+1턴), 턴마다 다른
# 워커 스레드에 배정될 수 있기 때문이다.
# ══════════════════════════════════════════════════════════════════════════════

_SESSION_STATE: dict[str, dict] = {}
_SESSION_STATE_LOCK = threading.Lock()
_SESSION_STATE_MAX = 16          # 오래된 세션 상태는 버린다(프로세스가 오래 산다)


def _new_session_state() -> dict:
    return {
        "last_spectrum": None,   # 가장 최근 acquire_spectrum() 결과 캐시
        "last_scene":    None,   # 가장 최근 capture_scene() 결과(경로·좌표) 캐시
        "bg_versions":   {},     # version_label → 배경제거 처리 결과
        # 그리드 사람-승인 게이트. enforce 기본 False = 벤치마크(자율)에서는 꺼짐,
        # 대화 세션은 grid_gate_begin_turn(interactive=True) 로 켠다.
        # resolved: 미리보기가 실제로 그린 격자 중심(mm). center 를 생략해 찍은 미리보기를
        # 실행할 때 '그때 그 자리'로 고정하기 위한 값이다(_grid_gate_on_preview 주석).
        "grid_gate":     {"geom": None, "resolved": None, "state": "none", "enforce": False},
    }


def _sstate() -> dict:
    """현재 세션의 도구 상태. 세션이 열리지 않았으면 '_unassigned' 로 모은다."""
    try:
        from backend.service.store import run_store
        key = run_store.current().get("label") or "_unassigned"
    except Exception:
        key = "_unassigned"
    with _SESSION_STATE_LOCK:
        st = _SESSION_STATE.get(key)
        if st is None:
            if len(_SESSION_STATE) >= _SESSION_STATE_MAX:
                _SESSION_STATE.pop(next(iter(_SESSION_STATE)), None)
            st = _new_session_state()
            _SESSION_STATE[key] = st
        return st


def _cache_and_return(result: dict) -> dict:
    """acquire_spectrum() 결과를 변경하지 않고 이 세션의 캐시에 담아 그대로 반환."""
    if result.get("ok") and "data" in result:
        _sstate()["last_spectrum"] = result
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 공용 하드웨어 헬퍼
#   같은 동작(파워 적용·범위 검증·레이저 정지)이 여러 도구에 복사돼 있어서, 정책이
#   호출 경로마다 달랐다. 아래 함수들이 그 정책의 단일 출처다.
#   관례: 검사 함수는 '통과면 None, 실패면 error dict' 를 돌려준다
#         (이 파일의 _validate_grid_args / _grid_gate_check 와 같은 규약).
# ══════════════════════════════════════════════════════════════════════════════

# ND 필터 모터가 멈춘 뒤 광학이 정착할 때까지의 대기(초). 예전에는 같은 동작에
# 0.1(드라이버) / 0.15(set_laser_power) / 0.5(acquire_spectrum) 세 값이 섞여 있었다.
_ND_SETTLE_S = 0.15


def _laser_power_range() -> tuple[float, float]:
    """허용 파워(%) 범위. 드라이버 상수를 단일 출처로 삼는다."""
    return (float(getattr(_laser, "ND_MIN_PCT", 0.004)),
            float(getattr(_laser, "ND_MAX_PCT", 100.0)))


def _apply_laser_power(percent, settle_s: float = _ND_SETTLE_S) -> dict | None:
    """레이저 파워를 검증하고 실제로 적용한다. 통과면 None, 실패면 error dict.

    [정책 — 왜 클램핑이 아니라 거부인가]
    드라이버 set_power() 는 범위를 벗어난 값을 조용히 클램핑한다. 도구 계층에서까지
    클램핑하면 에이전트는 "200%로 설정했다"고 믿은 채 100%로 측정한 결과를 해석하게
    된다. 그래서 여기서는 거부하고 이유를 돌려준다 — 관측으로 읽고 스스로 고치도록.

    set_power() 는 실패를 예외가 아니라 False 로 알린다(ND 모터 무응답 등). 그걸
    무시하면 파워가 안 걸린 상태로 ok:True 를 돌려주게 되므로 반드시 확인한다.
    """
    if _laser is None:
        return fail("Laser is not initialized.")
    if isinstance(percent, bool) or not isinstance(percent, (int, float)):
        try:
            percent = float(percent)
        except (TypeError, ValueError):
            return fail("power must be a number (%).")
    lo, hi = _laser_power_range()
    if not (lo <= float(percent) <= hi):
        return fail(f"Valid power range: {lo} - {hi} (%)")
    try:
        applied = _laser.set_power(float(percent))
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")
    time.sleep(settle_s)
    if applied is False:
        return fail("The ND filter motor did not confirm the move, so the laser power was NOT applied. "
                    "The measurement beam is still un-armed. Retry, or check the laser controller link.")
    return None


def _beam_state() -> dict:
    """지금 laser_on() 하면 어떤 빔이 나가는지.

    power_armed 는 '파워가 실제로 광학계에 적용돼 있는가'(드라이버 _power_set)다.
    power_pct 는 마지막으로 '설정한' 값이라 가이드빔 모드로 전환된 뒤에도 남아 있어서,
    그것만 보면 "이미 40%니 켜면 된다"고 오판한다(실제로는 ND 가 차단 위치다).
    laser_on / laser_off / get_laser_status 가 같은 판정을 쓰도록 여기로 모았다.
    """
    armed = bool(getattr(_laser, "_power_set", False))
    return {"power_armed": armed,
            "power_percent": getattr(_laser, "power_pct", None),
            "beam": "measurement" if armed else "guide"}


def _laser_off_quiet() -> bool:
    """예외를 던지지 않는 레이저 정지 — finally 블록 전용(안전 보장 경로).

    [조용하되 포기하지는 않는다 — 2026-08-03]
    finally 에서 부르므로 예외를 밖으로 내보내면 안 되지만, '껐다고 치고 넘어가는' 것과
    '끄려고 끝까지 해 보는' 것은 다르다. SSPW 0 이 확인되지 않으면 ND 를 차단 위치로
    옮겨 광로를 물리적으로 막는다. 발진을 못 멈추면 빔이라도 막는 것이 순서다.
    돌려주는 bool 은 '빔이 확실히 차단됐는가'다 — 호출자가 결과에 실어 보낼 수 있게.
    """
    if _laser is None:
        return True
    try:
        if _laser.laser_off() is not False:
            return True
    except Exception:
        pass
    print("[laser] SSPW 0 unconfirmed - falling back to ND blocking.")
    return _restore_guide_beam_quiet()


def _restore_guide_beam_quiet() -> None:
    """광학계를 가이드빔(=카메라 관찰) 위치로 되돌린다. 실패해도 조용히 넘어간다.

    [왜 도구 계층에 있는가 — 2026-07-31 회귀 수정]
    축04(BEAM_SPLITTER)에는 두 위치가 있다: 측정 위치(-0303764, 빛이 분광기로 간다)와
    대기 위치(-0612828, 빛이 카메라로 간다). laser_on() 은 파워가 무장된 상태면 축04 를
    측정 위치로 옮긴다 — 그 순간부터 카메라 화면은 아무것도 못 본다.

    예전에는 드라이버의 laser_off() 가 내부에서 set_guide_beam() 을 불러 자동 복귀했지만,
    그건 '끄고 다시 켜면 측정빔이 아니라 가이드빔이 나가는' 함정을 만들어 2026-07-30 에
    제거됐다(USE_laser_with_power.laser_off docstring 참고). 그때 "가이드빔이 필요한 쪽이
    명시적으로 부른다"고 정했는데, 정작 acquire_spectrum 이 그 호출을 갖지 않아
    **측정 후 카메라 화면이 돌아오지 않는** 회귀가 생겼다. 그 명시적 호출이 여기다.

    드라이버가 아니라 여기에 두는 이유: acquire_spectrum 은 시작할 때 _apply_laser_power 로
    항상 파워를 다시 걸므로(power=None 이어도 마지막 값을 재적용한다), set_guide_beam 이
    내리는 _power_set=False 때문에 다음 측정이 막히지 않는다. 반면 laser_off() 도구는
    '껐다가 같은 파워로 다시 켠다'는 용도라 그대로 둬야 한다 — 함정이 되살아난다.

    돌려주는 bool 은 '가이드빔 위치로 옮기는 명령이 예외 없이 끝났는가'다. 소등이 실패한
    경우 이것이 마지막 차단 수단이라 성패를 호출자에게 알려야 한다.
    """
    if _laser is None:
        return False
    try:
        _laser.set_guide_beam()
        return True
    except Exception as e:
        print(f"[laser] guide-beam restore failed: {type(e).__name__}: {e}")
        return False


def _check_stage_target(x=None, y=None, z=None) -> dict | None:
    """스테이지 목표 좌표가 허용 범위 안인지 검사. 통과면 None, 실패면 error dict.

    [왜 공용인가 — 2026-07-30]
    예전에는 move_stage 만 범위를 검사했고, move_stage_relative 와 run_autofocus 는
    _stage.move_absolute() 를 직접 불러 검사를 통째로 건너뛰었다. 오토포커스는 Z 를
    스스로 밀어 올리는 루프라, 하필 검사가 가장 필요한 경로가 빠져 있었다.
    한계값은 config.py 단일 출처(STAGE_MAX_X/Y, STAGE_MIN_Z/MAX_Z).
    """
    if x is not None and not (0 <= float(x) <= STAGE_MAX_X):
        return fail(f"X out of range: {x} (allowed: 0-{STAGE_MAX_X})")
    if y is not None and not (0 <= float(y) <= STAGE_MAX_Y):
        return fail(f"Y out of range: {y} (allowed: 0-{STAGE_MAX_Y})")
    if z is not None and not (STAGE_MIN_Z <= float(z) <= STAGE_MAX_Z):
        return fail(f"Z out of range: {z} (allowed: {STAGE_MIN_Z}-{STAGE_MAX_Z})")
    return None


def init_hardware(stage=None, laser=None, ccd=None, camera=None):
    """하드웨어 객체를 주입. run_scan.py 등에서 초기화 후 호출.

    [왜 락을 잡는가 — 2026-07-31]
    이 함수는 전역 핸들 4개를 갈아끼운다. 측정이 진행 중일 때 다른 스레드(패널의
    /api/*/connect, reconnect_hardware)가 이걸 부르면, acquire_spectrum 이 A 장비로
    시작해 B 장비로 끝나거나 중간에 None 을 만난다. 그래서 '진행 중인 조작이 끝난
    뒤에' 교체한다.

    [락 순서 — 교착 방지]
    이 프로젝트의 락 순서는 항상 **component_lock → _INSTRUMENT_LOCK** 이다.
    서버의 connect 엔드포인트는 component_lock 을 쥔 채 여기로 들어오고,
    reconnect_hardware 도 component_lock 을 먼저 잡은 뒤 장비 락을 잡는다.
    반대로 잡는 경로를 새로 만들면 교착한다.

    타임아웃되면 교체를 건너뛰지 않고 경고를 남긴 뒤 그대로 진행한다 — 핸들을 갱신하지
    않으면 방금 연결한 장비를 도구가 영영 못 보게 되어(옛 핸들이 남는다) 더 나쁘다.
    """
    global _stage, _laser, _ccd, _camera
    got = _INSTRUMENT_LOCK.acquire(timeout=_BUSY_TIMEOUT_S)
    if not got:
        print(f"[WARN] hw_core.init_hardware(): could not take the instrument lock within "
              f"{_BUSY_TIMEOUT_S:.0f}s because '{_lock_holder.get('what')}' is running - "
              f"replacing the handles anyway.")
    try:
        _stage = stage
        _laser = laser
        _ccd = ccd
        _camera = camera
    finally:
        if got:
            _INSTRUMENT_LOCK.release()
    print(f"[DEBUG] hw_core.init_hardware() called: stage={_stage}, laser={_laser}, "
          f"ccd={_ccd}, camera={_camera}")


# 폴링이 재연결과 겹쳐 스킵됐을 때 프론트에 보여줄 직전 값. 한 스레드가 통째로 교체하고
# 다른 스레드는 통째로 읽으므로(부분 갱신 없음) 별도 락이 필요 없다.
_last_stage_values: dict = {"value": {}}


def _read_stage_values(stage) -> dict:
    """스테이지 위치/속도를 프론트 형식으로 읽는다 — **_stage_read 안에서만** 부를 것."""
    out: dict = {}
    try:
        pos = stage.get_position()
        out = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
    except Exception:
        out = {}
    try:
        # get_velocity() 는 dict 반환 {"ok",x_speed_mm_s,...}.
        vel = stage.get_velocity()
        if isinstance(vel, dict) and vel.get("ok"):
            out["velocity"] = {"x": float(vel["x_speed_mm_s"]),
                               "y": float(vel["y_speed_mm_s"]),
                               "z": float(vel["z_speed_mm_s"])}
    except Exception:
        pass
    return out


def sync_tool_handles(mgr) -> None:
    """매니저의 현재 핸들을 이 모듈 전역에 다시 주입한다.

    연결/재연결 뒤에 반드시 해야 하는 일이라 server.py 의 connect 엔드포인트 3곳과
    reconnect_hardware 가 각각 같은 4인자 호출을 복사해 갖고 있었다. 한 곳으로 모아
    '어느 핸들을 주입하는지'가 갈라질 여지를 없앤다.
    """
    init_hardware(stage=getattr(mgr, "stage", None), laser=getattr(mgr, "laser", None),
                  ccd=getattr(mgr, "ccd", None), camera=getattr(mgr, "camera", None))


# ══════════════════════════════════════════════════════════════════════════════
# CCD 설정 적용 — 설정 툴(set_ccd_*)과 acquire_spectrum 의 공용 경로
#
# [왜 공용인가 — 2026-07-30]
# CCD 조건을 거는 길이 항상 두 개였다: 미리 set_ccd_* 로 걸어 두거나, 측정 시
# acquire_spectrum 의 인자로 넘기거나. 두 경로가 각자 구현을 갖고 있어서 허용값과
# 동작이 어긋났다(예: set_ccd_trigger_mode 에만 external_fvb_em 이 빠져 있었고,
# read_mode 적용 코드는 두 곳에 거의 글자 단위로 복사돼 있었다). 이제 두 경로 모두
# 아래 _apply_* 를 통과하므로, 허용값·순서·부작용이 구조적으로 같다.


# ══════════════════════════════════════════════════════════════════════════════

_ACQ_MODES      = ('single', 'accumulate', 'kinetic', 'run_till_abort')
_ACQ_MODES_1D   = ('single', 'accumulate', 'kinetic')      # 1D 스펙트럼으로 조립 가능한 것
_READ_MODES     = ('fvb', 'single_track', 'image')
_READ_MODES_1D  = ('fvb', 'single_track')
_TRIGGER_MODES  = ('internal', 'external', 'external_start', 'external_exposure',
                   'external_fvb_em', 'software')
_SHUTTER_MODES  = ('auto', 'open', 'close')

# 드라이버 표기 → 이 파일의 인자 표기
_RO_MODE_TO_ARG = {'FULL_VERTICAL_BINNING': 'fvb', 'SINGLE_TRACK': 'single_track', 'IMG': 'image'}

_CCD_NOT_READY = ("The spectrometer (CCD) is not ready yet - it is cooling (stabilizing at "
                  "-40 degC) or not connected. It becomes available automatically once "
                  "stabilized; check get_ccd_info() or get_hardware_status().")


def _ccd_ready() -> dict | None:
    """CCD 사용 가능 여부. 통과면 None, 아니면 error dict.
    (같은 안내 문구가 13개 함수에 복사돼 있던 것을 한 곳으로 모았다.)"""
    return None if _ccd is not None else fail(_CCD_NOT_READY)


def _current_read_mode() -> str:
    """현재 CCD 읽기 모드를 이 파일의 인자 표기('fvb'/'single_track'/'image')로."""
    return _RO_MODE_TO_ARG.get(getattr(_ccd, 'ro_mode', ''), 'fvb')


def _check_ccd_positive(name: str, value, integer: bool = False) -> dict | None:
    """CCD 수치 파라미터가 양수인지 검사. 통과(또는 None=생략)면 None, 실패면 error dict.

    [왜 필요한가 — 2026-08-05]
    이 값들은 결국 SDK 로 들어가고, SDK 는 잘못된 값에 DRV_P1INVALID 를 돌려준다
    (_check 가 IOError 로 올린다). 문제는 그 호출 시점이다: acquire_spectrum 에서
    _apply_acq_mode / _apply_read_mode 는 **레이저를 켠 뒤**에 실행되므로, 검증을 SDK 에
    맡기면 잘못된 값 하나 때문에 시료에 빔이 들어간 뒤에야 실패한다. 셔터·트리거를
    발사 전에 검증하는 것과 정확히 같은 이유로 여기서 미리 막는다.

    [거부하되 조용히 고치지 않는다]
    0 을 1 로 바꿔 주면 호출자는 자기가 요청한 조건으로 측정됐다고 믿는다.
    _apply_laser_power 가 클램핑 대신 거부하는 것과 같은 정책이다.

    None 은 통과시킨다 — 이 파일에서 None 은 '지정 안 함 = 현재 설정 유지'다.
    bool 을 막는 이유: 파이썬에서 True 는 int 라 isinstance 검사를 그냥 통과한다.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fail(f"{name} must be a number, got {value!r}.")
    if integer:
        if int(value) != value:
            return fail(f"{name} must be a whole number, got {value!r}.")
        if int(value) < 1:
            return fail(f"{name} must be at least 1, got {value!r}. Omit the parameter to keep the "
                        f"instrument's current setting - it is not a way to switch the feature off.")
    elif float(value) <= 0:
        return fail(f"{name} must be greater than 0, got {value!r}. Omit the parameter to keep the "
                    f"instrument's current setting.")
    return None


def _apply_acq_mode(mode: str, exposure=None, num_accumulations=None,
                    kinetic_count=None, kinetic_cycle_time=None) -> dict | None:
    """취득 모드와 그 파라미터를 적용한다. 통과면 None, 실패면 error dict.

    None 인 파라미터는 드라이버가 현재 값을 그대로 둔다(드라이버의 None=유지 규약).
    **반드시 읽기 모드(_apply_read_mode)보다 먼저** 호출해야 한다 — create_buffer()
    가 aq_mode 에 의존하므로 순서가 뒤바뀌면 버퍼 모양이 어긋난다.
    """
    if mode not in _ACQ_MODES:
        return fail(f"acquisition mode must be one of {list(_ACQ_MODES)}.")
    # 수치 검증도 이 공용 경로에서 한다 — acquire_spectrum 은 발사 전에 이미 같은 검사를
    # 끝내지만(아래 사전검증 블록), set_ccd_acquisition_mode 는 여기로만 들어온다.
    # 한쪽에만 두면 같은 값이 한 도구에서는 깔끔한 에러, 다른 도구에서는 날것의 SDK
    # IOError 로 갈라진다 — 이 파일이 _apply_* 를 공용으로 만든 이유가 그것이다.
    for _n, _v, _i in (("exposure", exposure, False),
                       ("num_accumulations", num_accumulations, True),
                       ("kinetic_count", kinetic_count, True),
                       ("kinetic_cycle_time", kinetic_cycle_time, False)):
        err = _check_ccd_positive(_n, _v, integer=_i)
        if err:
            return err
    if mode == 'single':
        _ccd.set_aq_single_scan(exposure=exposure)
    elif mode == 'accumulate':
        _ccd.set_aq_accumulate_scan(exposure_time=exposure, num_acc=num_accumulations)
    elif mode == 'kinetic':
        _ccd.set_aq_kinetic_scan(
            exp_time=exposure,
            num_kin=kinetic_count,
            # >= 1 이다(> 1 이 아니다) — 2026-08-05.
            # None 은 '지정 안 함 = 유지', 숫자는 '이 값으로 걸어라'가 이 파일의 규약이다.
            # 예전에는 > 1 이라 **명시한 1 이 None 으로 바뀌어** 무시됐다. 직전에 누적이
            # 10 으로 걸려 있었다면 num_accumulations=1 로 요청해도 10 이 유지되어, 5 프레임
            # 짜리 kinetic 이 요청의 10 배로 조사된다(5x10x0.5s = 25s vs 2.5s). 관측은
            # 가능하지만(보고값도 10) 그때는 이미 시료에 들어간 뒤다.
            # acquire_spectrum 이 생략 시 '리셋'이 아니라 '유지'가 되도록 고친 2026-07-30
            # 수정과 같은 규약이다 — '값을 안 줬다'와 '값을 줬다'를 뭉개지 않는다.
            num_acc=num_accumulations if (num_accumulations or 0) >= 1 else None,
            kin_time=kinetic_cycle_time,
        )
    else:                                   # run_till_abort
        _ccd.set_aq_run_till_abort_scan()
        if exposure is not None:
            _ccd.set_exposure_time(exposure)
    # set_aq_*_scan 이 인자로 받지 않는 조합(예: single 모드인데 누적 횟수만 조정)을 메운다.
    #
    # [지우지 말 것 — 조건이 뒤집혀 보이지만 맞다, 2026-08-05]
    # "값이 중요한 모드에서만 건너뛴다"처럼 읽혀서 지우고 싶어지는 자리다. 실제로는
    # set_aq_single_scan / set_aq_run_till_abort_scan 이 num_acc 를 받지 않고, num_kin 을
    # 받는 set_aq_* 는 kinetic 하나뿐이라, 아래 두 줄이 나머지 조합을 담당한다. 지우면
    # set_ccd_acquisition_mode('single', num_accumulations=N) 이 조용히 무시된다.
    # accumulate/kinetic 모드는 위의 set_aq_* 가 SDK 와 캐시를 함께 갱신하므로 여기서
    # 다시 부를 필요가 없다(andor_ccd_interface 의 캐시 불변식 주석 참고).
    if num_accumulations is not None and mode in ('single', 'run_till_abort'):
        _ccd.set_num_accumulations(num_accumulations)
    if kinetic_count is not None and mode != 'kinetic':
        _ccd.set_num_kinetics(kinetic_count)
    return None


def _apply_read_mode(mode: str, hbin=None, single_track_center=None,
                     single_track_width=None) -> dict | None:
    """읽기 모드를 적용한다(create_buffer 가 여기서 불린다). 통과면 None, 실패면 error dict.

    None 인 파라미터는 현재 설정값을 그대로 재사용한다.
    """
    if mode not in _READ_MODES:
        return fail(f"read mode must be one of {list(_READ_MODES)}.")
    for _n, _v in (("hbin", hbin), ("single_track_center", single_track_center),
                   ("single_track_width", single_track_width)):
        err = _check_ccd_positive(_n, _v, integer=True)
        if err:
            return err
    eff_hbin = hbin if hbin is not None else _ccd.get_current_hbin()
    if mode == 'fvb':
        _ccd.set_ro_full_vertical_binning(hbin=eff_hbin)
    elif mode == 'single_track':
        center = (single_track_center if single_track_center is not None
                  else getattr(_ccd, 'ro_single_track_center', None))
        if center is None:
            return fail("single_track mode requires the single_track_center parameter.")
        _ccd.set_ro_single_track(
            center=center,
            width=(single_track_width if single_track_width is not None
                   else getattr(_ccd, 'ro_single_track_width', 1)),
            hbin=eff_hbin,
        )
    else:                                   # image
        _ccd.set_ro_image_mode(hbin=eff_hbin)
    return None


def _apply_trigger_mode(mode: str) -> dict | None:
    """트리거 모드 적용. 통과면 None, 실패면 error dict."""
    if mode not in _TRIGGER_MODES:
        return fail(f"trigger mode must be one of {list(_TRIGGER_MODES)}.")
    _ccd.set_trigger_mode(mode)
    return None


def _apply_shutter(mode: str, explicit: bool = False) -> dict | None:
    """셔터 모드 적용. 통과면 None, 실패면 error dict.

    explicit=True 는 '사람/에이전트가 셔터를 직접 지정했다'는 뜻이고, 그 사실을 CCD
    객체에 표시해 둔다(_effective_shutter 참고). 플래그를 모듈 전역이 아니라 CCD
    객체에 다는 이유: 재연결하면 새 AndorCCD 객체가 오고 셔터도 다시 close 로
    초기화되는데, 객체에 달아 두면 플래그가 자동으로 함께 리셋된다.
    """
    if mode not in _SHUTTER_MODES:
        return fail(f"shutter must be one of {list(_SHUTTER_MODES)}.")
    if mode == 'auto':
        _ccd.set_shutter_auto()
    elif mode == 'open':
        _ccd.set_shutter_open(True)
    else:
        _ccd.set_shutter_close()
    if explicit:
        try:
            _ccd.shutter_explicit = True
        except Exception:
            pass
    return None


def _effective_shutter(requested: str | None) -> str:
    """acquire_spectrum 이 실제로 걸 셔터 모드를 정한다.

    [왜 그냥 '생략=유지'가 아닌가 — 2026-07-31 수정]
    CCD 초기화(hardware_manager._init_ccd)가 셔터를 close 로 둔다("촬영 직전까지 광원
    차단"). 그래서 단순히 유지로 하면 **세션 첫 측정이 조용히 암전 프레임**이 된다.
    그렇다고 예전처럼 무조건 auto 로 되돌리면 set_ccd_shutter 가 유일하게 '유지되지
    않는' 설정 툴이 되어, 12개 CCD 설정 툴 중 이것만 규약이 반대가 된다.

    그래서 둘을 가른다:
      · 아무도 셔터를 지정한 적이 없다  → 'auto' (초기 close 로 인한 암전 방지)
      · 툴로 한 번이라도 지정했다        → 그 설정을 유지 (다른 설정 툴과 같은 규약)
    다크/배경 프레임은 set_ccd_shutter('close') 로 걸어 두고 여러 번 측정하거나,
    acquire_spectrum(shutter='close') 로 그 측정에만 걸 수 있다 — 이제 둘 다 된다.
    """
    if requested is not None:
        return requested
    if getattr(_ccd, 'shutter_explicit', False):
        return getattr(_ccd, 'shutter_mode', None) or 'auto'
    return 'auto'


# ──────────────────────────────────────────────────────────────────────────────
# 자체 점검:  python -m backend.hw_tools.hw_tools.hw_core
#
# 여기서 확인하는 것은 '분할이 깨뜨릴 수 있는 것' 딱 둘이다.
#   ① 핸들 재바인딩이 다른 모듈에서 보이는가
#      init_hardware() 는 이 모듈의 전역을 global 로 갈아 끼운다. 도구 모듈이
#      `from hw_core import _stage` 로 받았다면 그 시점의 None 을 영원히 붙들어,
#      장비를 연결해도 모든 도구가 "not connected" 를 돌려준다. import 는 성공하고
#      스키마도 멀쩡하므로 정적 검사로는 절대 안 잡힌다 — 실제로 불러 봐야 한다.
#   ② 세션 상태가 모듈 사이에서 공유되는가
#      acquire 가 캐시한 직전 스펙트럼을 bg_tools 가 source='last' 로 읽는다.
#      두 모듈이 각자 _SESSION_STATE 를 갖게 되면 배경 제거가 조용히 빈손이 된다.
#
# 가짜 장비를 쓰는 이유: 개발 PC 에는 장비가 없고, 위 두 가지는 장비 동작이 아니라
# 모듈 경계의 문제라 가짜로 충분하다.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    class _FakeStage:
        """이동 명령을 기억만 하는 스테이지. 실제 DLL 호출은 없다."""

        def __init__(self):
            self.pos = [1.0, 2.0, 0.5]
            self.moves = []

        def get_position(self, **kw):
            return tuple(self.pos)

        # 드라이버가 wait= 같은 인자를 더 받아도 이 점검은 '명령이 닿았는가'만 보므로
        # **kw 로 흘려보낸다. 인자 규약을 여기서 다시 못 박으면 드라이버를 고칠 때마다
        # 관계없는 이 검사가 깨진다.
        def move_absolute(self, x, y, z, a=0.0, **kw):
            self.moves.append(("abs", x, y, z))
            self.pos = [x, y, z]
            return True

        def move_relative(self, dx, dy, dz, da=0.0, **kw):
            self.moves.append(("rel", dx, dy, dz))
            self.pos = [self.pos[0] + dx, self.pos[1] + dy, self.pos[2] + dz]
            return True

        def get_velocity(self, **kw):
            return (1.0, 1.0, 0.05)

        def set_velocity(self, vx, vy, vz, va=0.0, **kw):
            return True

    # 정식 모듈 경유로 부른다. `python -m` 은 이 파일을 __main__ 으로 한 벌 더 올리므로,
    # 여기서 init_hardware() 를 그냥 부르면 stage_tools 가 보는 것과 다른 전역을 고치게 된다.
    from backend.tools.hw_tools.hw_tools import hw_core as core
    from backend.tools.hw_tools.hw_tools import stage_tools
    from backend.tools.non_hw_tools import bg_tools

    # ── ① 재바인딩이 보이는가 ────────────────────────────────────────────────
    assert stage_tools.get_stage_position().get("ok") is False,         "핸들이 없는데 위치를 돌려줬다 - _stage_unavailable 이 안 걸린다"

    fake = _FakeStage()
    core.init_hardware(stage=fake)

    got = stage_tools.get_stage_position()
    assert got.get("ok"), f"연결 후에도 스테이지를 못 본다(재바인딩 실패): {got}"
    assert (got["x"], got["y"]) == (1.0, 2.0), got

    moved = stage_tools.move_stage(x=3.0, y=4.0)
    assert moved.get("ok"), moved
    assert fake.moves and fake.moves[-1][0] == "abs", "이동 명령이 가짜 장비에 닿지 않았다"

    rel = stage_tools.move_stage_relative(dx=0.5)
    assert rel.get("ok"), rel
    assert fake.pos[0] == 3.5, fake.pos

    # 범위 밖은 클리핑이 아니라 거부 — 분할 후에도 공용 검사가 그대로 걸리는지.
    assert stage_tools.move_stage(x=-1.0, y=0.0).get("ok") is False, "범위 밖을 허용했다"

    # 핸들을 떼면 도구가 곧바로 다시 '미연결'로 돌아가야 한다(캐시된 참조가 없다는 뜻).
    core.init_hardware()
    assert stage_tools.get_stage_position().get("ok") is False,         "핸들을 뗐는데도 옛 장비를 보고 있다 - 어딘가 from-import 로 붙들고 있다"

    # ── ② 세션 상태 공유 ──────────────────────────────────────────────────────
    assert bg_tools.apply_background_subtraction(poly_order=5).get("ok") is False,         "직전 측정이 없는데 배경 제거가 성공했다"

    core._cache_and_return({"ok": True, "data": [10.0, 12.0, 40.0, 12.0, 10.0],
                       "mode": "single"})
    assert core._sstate()["last_spectrum"] is not None, "캐시가 이 모듈에 안 남았다"

    done = bg_tools.apply_background_subtraction(poly_order=3, version_label="selfcheck")
    assert done.get("ok"), f"bg_tools 가 acquire 의 세션 캐시를 못 봤다: {done}"
    assert bg_tools.list_bg_versions()["count"] >= 1, "버전 목록이 비었다"

    core._sstate()["last_spectrum"] = None
    core._sstate()["bg_versions"].clear()

    print("통과: 핸들 재바인딩 반영/해제 · 이동 명령 전달 · 범위 거부 "
          "· 세션 캐시 acquire↔bg 공유")
