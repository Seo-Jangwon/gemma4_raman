"""
Raman 하드웨어 tool 래퍼
- LLM agent가 호출할 수 있는 단순 함수들
- 각 함수는 dict를 반환 (LLM에게 결과를 텍스트로 전달하기 위해)
"""

from __future__ import annotations
import time
import json
import csv
from pathlib import Path

from backend.config import (
    STAGE_MAX_X, STAGE_MAX_Y, STAGE_MIN_Z, STAGE_MAX_Z,
    CAMERA_WIDTH, CAMERA_HEIGHT,
)
# 픽셀↔mm 변환과 시야(FOV)는 optics_map 한 곳에서만 정의한다. 예전에는 이 파일,
# USE_scan.py, server.py 가 같은 식을 각자 갖고 있었고 server.py 는 보정계수를
# 하드코딩까지 했다(optics_map 머리말 참고).
from backend.hw_tools import optics_map as _om
# 프레임 전처리·가이드빔 스팟 면적도 마찬가지로 vision 한 곳에서만 정의한다.
from backend.hw_tools import vision as _vis
# 조사량 상한/공식은 에이전트 계층과 공유해야 하므로 의존성 없는 모듈에 둔다.
from backend.safety_limits import MAX_DOSE_MJ_PER_GRID as _GRID_MAX_DOSE_MJ, estimate_dose_mj
# spectrum_store.save_spectrum 은 '측정 결과 자동 저장'이다(에이전트 툴이 아니다).
# 별칭으로 받아 두어 이 모듈의 다른 이름들과 헷갈리지 않게 한다.
from backend.spectrum_store import (
    save_spectrum as _store_save_spectrum,
    list_results as _store_list_results,
    combine_spectra as _store_combine_spectra,
    aggregate_spectra_csv as _store_aggregate_csv,
    bundle_results as _store_bundle_results,
    save_scene as _store_save_scene,
    save_preview_png as _store_save_preview,
    write_spectrum_csv as _store_write_csv,
)
from backend.analysis_sandbox import run_analysis as _run_analysis
from backend.web_search import web_search as _web_search
# 속도 상한은 드라이버(USE_stage_test.py)에만 하드코딩되어 있다. 툴 계층은 그 상수를
# 그대로 빌려 써서 '실제 적용될 값'을 보고한다 — 상한을 두 곳에 적어 두면 갈라진다.
from backend.hw_tools.USE_stage_test import (
    MAX_SPEED_XY as _MAX_SPEED_XY,
    MAX_SPEED_Z as _MAX_SPEED_Z,
)

# (STAGE_MIN_Z / STAGE_MAX_Z 는 config.py 에 이미 있었는데 여기서 같은 값을 다시
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
            f"장비가 '{busy}' 작업 중이라 {t:.0f}초 안에 사용할 수 없습니다 — '{what}'을(를) "
            f"수행하지 않았습니다. 진행 중인 측정이 끝난 뒤 다시 시도하세요.")
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
                return {"ok": False, "error": (
                    f"The instrument is busy with '{busy}' and did not become free within "
                    f"{timeout:.0f}s, so '{what}' was NOT performed. Only one operation may drive "
                    f"the hardware at a time (a measurement, a stage move, or a grid scan started "
                    f"from the parameter panel or another chat session). Nothing was changed - "
                    f"wait for it to finish and try again."),
                    "busy_with": busy}
            prev = _lock_holder.get("what")
            _lock_holder["what"] = what if prev is None else prev   # 최상위 작업 이름을 유지
            try:
                return fn(*args, **kwargs)
            finally:
                _lock_holder["what"] = prev
                _INSTRUMENT_LOCK.release()
        return wrapper
    return deco


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
        "grid_gate":     {"geom": None, "state": "none", "enforce": False},
    }


def _sstate() -> dict:
    """현재 세션의 도구 상태. 세션이 열리지 않았으면 '_unassigned' 로 모은다."""
    try:
        from backend.agents import run_store
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
        return {"ok": False, "error": "Laser is not initialized."}
    if isinstance(percent, bool) or not isinstance(percent, (int, float)):
        try:
            percent = float(percent)
        except (TypeError, ValueError):
            return {"ok": False, "error": "power must be a number (%)."}
    lo, hi = _laser_power_range()
    if not (lo <= float(percent) <= hi):
        return {"ok": False, "error": f"Valid power range: {lo} - {hi} (%)"}
    try:
        applied = _laser.set_power(float(percent))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    time.sleep(settle_s)
    if applied is False:
        return {"ok": False, "error": (
            "The ND filter motor did not confirm the move, so the laser power was NOT applied. "
            "The measurement beam is still un-armed. Retry, or check the laser controller link.")}
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


def _laser_off_quiet() -> None:
    """실패해도 조용히 넘어가는 레이저 정지 — finally 블록 전용(안전 보장 경로)."""
    try:
        if _laser is not None:
            _laser.laser_off()
    except Exception:
        pass


def _check_stage_target(x=None, y=None, z=None) -> dict | None:
    """스테이지 목표 좌표가 허용 범위 안인지 검사. 통과면 None, 실패면 error dict.

    [왜 공용인가 — 2026-07-30]
    예전에는 move_stage 만 범위를 검사했고, move_stage_relative 와 run_autofocus 는
    _stage.move_absolute() 를 직접 불러 검사를 통째로 건너뛰었다. 오토포커스는 Z 를
    스스로 밀어 올리는 루프라, 하필 검사가 가장 필요한 경로가 빠져 있었다.
    한계값은 config.py 단일 출처(STAGE_MAX_X/Y, STAGE_MIN_Z/MAX_Z).
    """
    if x is not None and not (0 <= float(x) <= STAGE_MAX_X):
        return {"ok": False, "error": f"X out of range: {x} (allowed: 0-{STAGE_MAX_X})"}
    if y is not None and not (0 <= float(y) <= STAGE_MAX_Y):
        return {"ok": False, "error": f"Y out of range: {y} (allowed: 0-{STAGE_MAX_Y})"}
    if z is not None and not (STAGE_MIN_Z <= float(z) <= STAGE_MAX_Z):
        return {"ok": False, "error": f"Z out of range: {z} (allowed: {STAGE_MIN_Z}-{STAGE_MAX_Z})"}
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
        print(f"[WARN] raman_tools.init_hardware(): 진행 중인 '{_lock_holder.get('what')}' 때문에 "
              f"{_BUSY_TIMEOUT_S:.0f}s 안에 장비 락을 얻지 못했습니다 — 핸들 교체를 강행합니다.")
    try:
        _stage = stage
        _laser = laser
        _ccd = ccd
        _camera = camera
    finally:
        if got:
            _INSTRUMENT_LOCK.release()
    print(f"[DEBUG] raman_tools.init_hardware() 호출됨: stage={_stage}, laser={_laser}, ccd={_ccd}, camera={_camera}")


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
        from backend.hardware_manager import get_manager
        mgr = get_manager()
    except Exception as e:
        return {"ok": False, "error": f"HardwareManager unavailable: {type(e).__name__}: {e}"}

    out: dict = {"ok": True, "connected": {}, "notes": {}}
    for name in ("stage", "laser", "ccd", "camera"):
        obj = getattr(mgr, name, None)
        out["connected"][name] = obj is not None
        if obj is None:
            out["notes"][name] = ("Not connected. Try reconnect_hardware(component='%s'). "
                                  "If that reports the resource is still held, the server process "
                                  "must be restarted - no tool can clear it." % name)

    # 연결된 것은 실제로 응답하는지도 가볍게 확인한다 — 핸들이 살아 있어도 장비가
    # 먹통이면 '연결됨'만 보고 진행하다 뒤에서 터진다.
    if mgr.stage is not None:
        try:
            pos = mgr.stage.get_position()
            out["stage_position"] = (
                {"x": pos[0], "y": pos[1], "z": pos[2]} if pos else None)
            if not pos:
                out["notes"]["stage"] = "Handle exists but get_position() returned nothing - stage is unresponsive."
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
    ccd    = getattr(mgr, "ccd", None)    if mgr is not None else _ccd
    laser  = getattr(mgr, "laser", None)  if mgr is not None else _laser
    stage  = getattr(mgr, "stage", None)  if mgr is not None else _stage

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
        try:
            pos = stage.get_position()
            out["stage"] = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
        except Exception:
            out["stage"] = {}
        try:
            # get_velocity() 는 dict 반환 {"ok",x_speed_mm_s,...}.
            vel = stage.get_velocity()
            if isinstance(vel, dict) and vel.get("ok"):
                out["stage"]["velocity"] = {"x": float(vel["x_speed_mm_s"]),
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


def reconnect_hardware(component: str = "all") -> dict:
    """카메라/스테이지/CCD/레이저 연결을 끊었다가 재초기화한다.

    component: 'stage' | 'ccd' | 'camera' | 'laser' | 'all' (기본 all).
    주의: CCD 재초기화는 -40C 냉각 안정화까지 수 분간 블로킹될 수 있다.
    재초기화 후 raman_tools 전역 핸들을 새 객체로 다시 주입한다.
    """
    try:
        from backend.hardware_manager import get_manager
    except Exception as e:
        return {"ok": False, "error": f"HardwareManager import failed: {e}"}

    comp = str(component or "all").strip().lower()
    valid = {"stage", "ccd", "camera", "laser", "all"}
    if comp not in valid:
        return {"ok": False, "error": f"component must be one of {sorted(valid)}"}

    try:
        mgr = get_manager()
    except Exception as e:
        # 여기서 예외가 나면 도구가 에러 dict 가 아니라 날것의 예외를 던진다 —
        # 에이전트는 그것을 관측으로 읽을 수 없으므로 반드시 감싼다.
        return {"ok": False, "error": f"HardwareManager unavailable: {type(e).__name__}: {e}"}

    targets = ["stage", "ccd", "camera", "laser"] if comp == "all" else [comp]
    done, errors, detail = [], {}, {}

    # ── 장비 락을 '해제부터 핸들 재주입까지' 통째로 쥔다 ──────────────────────
    # [왜 컴포넌트마다가 아니라 전체 구간인가 — 2026-07-31]
    # 처음에는 컴포넌트 루프 안에서만 잡았는데, 루프가 락을 놓은 뒤 sync_tool_handles()
    # 전에 틈이 생겼다. 그 틈에 다른 스레드의 측정이 끼어들면 이미 close() 된 옛 CCD
    # 핸들로 촬영을 시도한다(전역 _ccd 는 아직 교체 전이다). 테스트로 재현됐다:
    #     ['reconnect_start', 'measure_start', 'measure_end', 'reconnect_end']
    # 그래서 교체가 끝날 때까지 놓지 않는다.
    #
    # 락 순서는 instrument_guard → component_lock 로 고정한다(instrument_guard
    # docstring 참고). 서버의 connect 엔드포인트도 같은 순서를 쓴다.
    try:
        _guard = instrument_guard(f"reconnect_hardware({comp})")
        _guard.__enter__()
    except InstrumentBusy as e:
        return {"ok": False, "error": str(e), "reconnected": [],
                "detail": {"skipped": "instrument busy"},
                "hint": ("A measurement or scan is still running. Nothing was changed - "
                         "wait for it to finish (check get_hardware_status()) and retry.")}

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

    # 재초기화된 객체를 raman_tools 전역에 재주입(server 의 connect 엔드포인트와 공용 경로).
    # **아직 장비 락 안이다** — 여기까지 와야 옛 핸들로 측정이 들어가는 틈이 사라진다.
    # init_hardware 도 같은 락을 잡지만 RLock 이라 재진입으로 통과한다.
    sync_tool_handles(mgr)
    return {"ok": (len(errors) == 0), "reconnected": done,
            "errors": (errors or None), "detail": detail,
            "now_connected": {k: getattr(mgr, k, None) is not None
                              for k in ("stage", "laser", "ccd", "camera")}}


# ──────────────────────────────────────────
# 스테이지
# ──────────────────────────────────────────

def get_stage_speed() -> dict:
    """현재 스테이지 이동 속도를 반환한다. 단위는 mm/s."""
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    try:
        # get_velocity() 는 dict 반환. (이전 버그: speeds[0] 로 dict 를 정수 인덱싱 → 항상 에러)
        vel = _stage.get_velocity()
        if not (isinstance(vel, dict) and vel.get("ok")):
            return {"ok": False, "error": vel.get("error", "Failed to read velocity") if isinstance(vel, dict) else "Unexpected velocity type"}
        return {"ok": True,
                "x_speed_mm_s": vel["x_speed_mm_s"],
                "y_speed_mm_s": vel["y_speed_mm_s"],
                "z_speed_mm_s": vel["z_speed_mm_s"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@_serialized("set_stage_speed")
def set_stage_speed(x_speed_mm_s: float = None, y_speed_mm_s: float = None, z_speed_mm_s: float = None) -> dict:
    """스테이지 이동 속도를 설정한다. 전달되지 않은 축의 속도는 기존 값을 유지한다.

    축별 상한(USE_stage_test.MAX_SPEED_XY / MAX_SPEED_Z)을 넘는 값은 드라이버가 클리핑한다.
    예전에는 이 툴이 클리핑 전 '요청값'을 그대로 돌려줘서, Z=1.0 을 요청하면 실제로는
    0.1 로 걸렸는데도 호출자는 1.0 으로 설정됐다고 믿었다(이동 시간 예측이 10배 틀어진다).
    이제 같은 상수로 미리 클리핑해 '실제 적용될 값'을 보고하고, 잘린 축은 clipped 로 알린다.
    """
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}

    try:
        # 1. 현재 장비에 설정된 속도를 읽어옵니다.
        current_vel = _stage.get_velocity()

        if not current_vel.get("ok"):
            return {"ok": False, "error": current_vel.get("error", "Failed to read current velocity")}

        # 2. 값이 안 들어왔으면(None) 기존 속도를 그대로 사용합니다.
        req = {
            "x_speed_mm_s": x_speed_mm_s if x_speed_mm_s is not None else current_vel["x_speed_mm_s"],
            "y_speed_mm_s": y_speed_mm_s if y_speed_mm_s is not None else current_vel["y_speed_mm_s"],
            "z_speed_mm_s": z_speed_mm_s if z_speed_mm_s is not None else current_vel["z_speed_mm_s"],
        }

        # 3. 드라이버와 '같은 상수'로 클리핑한다 — 상한은 USE_stage_test.py 한 곳에만 있다.
        limits = {"x_speed_mm_s": _MAX_SPEED_XY, "y_speed_mm_s": _MAX_SPEED_XY,
                  "z_speed_mm_s": _MAX_SPEED_Z}
        eff, clipped = {}, {}
        for k, v in req.items():
            hi = limits[k]
            e = max(-hi, min(hi, float(v)))
            eff[k] = e
            if e != float(v):
                clipped[k] = {"requested": float(v), "applied": e, "limit_mm_s": hi}

        # 4. 조합된 속도로 컨트롤러에 명령을 내립니다. (va는 사용하지 않으므로 0.0)
        #    set_velocity 는 실패를 예외가 아니라 False 로 알린다 — 무시하면 안 된다.
        applied = _stage.set_velocity(eff["x_speed_mm_s"], eff["y_speed_mm_s"],
                                     eff["z_speed_mm_s"], 0.0)
        if applied is False:
            return {"ok": False, "error": (
                "The controller rejected the velocity command, so the stage speed was NOT changed. "
                "Check the stage connection and retry.")}

        out = {"ok": True, **eff}
        if clipped:
            out["clipped"] = clipped
            out["note"] = ("Some axes were clipped to the hardware speed limit "
                           f"(XY <= {_MAX_SPEED_XY} mm/s, Z <= {_MAX_SPEED_Z} mm/s). "
                           "The values reported here are what the stage will actually use.")
        return out

    except Exception as e:
        return {"ok": False, "error": str(e)}

@_serialized("move_stage")
def move_stage(x: float, y: float, z: float = None) -> dict:
    """스테이지를 절대 좌표(mm)로 이동."""
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}

    err = _check_stage_target(x, y, z)      # 범위 검증(공용 — 모든 이동 경로가 같은 한계를 쓴다)
    if err:
        return err

    try:
        kw = {"x": x, "y": y, "wait": True}
        if z is not None:
            kw["z"] = z
        else:
            kw["z"] = _stage.get_position()[2]  # 현재 Z 유지
        _stage.move_absolute(**kw)
        pos = _stage.get_position()
        return {"ok": True, "position": {"x": pos[0], "y": pos[1], "z": pos[2]}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_stage_position() -> dict:
    """현재 스테이지 위치를 반환."""
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    try:
        pos = _stage.get_position()
        return {"ok": True, "x": pos[0], "y": pos[1], "z": pos[2]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("move_stage_relative")
def move_stage_relative(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> dict:
    """스테이지를 현재 위치 기준 상대 이동(mm).

    move_stage 와 **같은 범위 한계**가 적용된다 — 현재 위치에 변위를 더한 목표를 먼저
    계산해 검사한다. 예전에는 이 함수만 검사가 없어서, 같은 목적지라도 절대 이동이면
    거부되고 상대 이동이면 통과하는 비대칭이 있었다.
    """
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    try:
        pos = _stage.get_position()
        if pos is None:
            return {"ok": False, "error": "Failed to query stage position"}
        err = _check_stage_target(float(pos[0]) + float(dx),
                                  float(pos[1]) + float(dy),
                                  float(pos[2]) + float(dz))
        if err:
            err["error"] += (f" - this is the target after applying the relative move "
                             f"(dx={dx}, dy={dy}, dz={dz}) to the current position.")
            return err
        _stage.move_relative(dx, dy, dz, 0)
        pos = _stage.get_position()
        return {"ok": True, "position": {"x": pos[0], "y": pos[1], "z": pos[2]}}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 레이저
# ──────────────────────────────────────────

@_serialized("laser_on")
def laser_on() -> dict:
    """레이저를 켠다. 어떤 빔이 나가는지(측정빔/가이드빔)를 함께 보고한다."""
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    # 드라이버는 파워가 적용되지 않은 상태(_power_set=False, 예: 가이드빔 모드나
    # 오토포커스 직후)에서 SSPW 1 을 받으면 측정빔이 아니라 가이드빔을 낸다.
    # 두 경우를 "Laser ON" 한 마디로 뭉개면 호출자는 신호가 왜 0인지 알 수 없다.
    st = _beam_state()                      # 판정은 _beam_state 단일 출처
    armed, beam = st["power_armed"], st["beam"]
    try:
        _laser.laser_on()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    out = {"ok": True, "status": f"Laser ON ({beam} beam)", "beam": beam}
    if armed:
        out["power_percent"] = st["power_percent"]
    else:
        out["note"] = (
            "Only the GUIDE beam is emitted - the laser power has not been applied to the optics, "
            "so the measurement beam is still blocked by the ND filter. For a real measurement call "
            "set_laser_power(percent) first, or use acquire_spectrum(power=...) which does "
            "power -> on -> acquire -> off atomically.")
    return out


@_serialized("laser_off")
def laser_off() -> dict:
    """레이저를 끈다(발진 정지). 파워 설정과 광학계 위치는 그대로 유지된다."""
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    try:
        _laser.laser_off()
        st = _beam_state()
        return {"ok": True, "status": "Laser OFF",
                # 끄더라도 ND 위치(=파워 설정)는 남는다 — 다시 켜면 같은 파워의
                # 측정빔이 나간다. 예전처럼 가이드빔으로 되돌아가지 않는다.
                "power_armed": st["power_armed"],
                "power_percent": st["power_percent"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_laser_power")
def set_laser_power(percent: float) -> dict:
    """레이저 출력 설정. percent: ND 필터 투과율 0.004~100 (실수 허용).

    측정 직전에 파워를 정할 거라면 acquire_spectrum(power=...) 을 쓰는 편이 낫다 —
    파워 적용 → ON → 측정 → OFF 를 원자적으로 수행해 레이저가 켜진 채 남지 않는다.
    이 도구는 파워만 미리 걸어 두고 싶을 때(가이드빔 정렬 후 무장, 여러 측정에서
    같은 파워 재사용) 쓴다. 두 경로 모두 아래 _apply_laser_power 를 통과한다.
    """
    err = _apply_laser_power(percent)       # 검증·적용·정착 대기(공용)
    if err:
        return err
    return {"ok": True, **_beam_state()}


def get_laser_status() -> dict:
    """현재 레이저 상태: 발사 여부, 파워(%), 그리고 그 파워가 실제로 광학계에 적용된 상태인지.

    power_armed 를 반드시 함께 본다. power_pct 는 '마지막으로 설정한 값'이라 가이드빔
    모드로 전환된 뒤에도 남아 있어서, 그것만 보면 "이미 40%로 맞춰져 있으니 켜면 된다"고
    오판하게 된다(실제로는 ND 가 차단 위치라 가이드빔만 나간다).
    """
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    try:
        st = _beam_state()                  # 판정은 _beam_state 단일 출처
        armed, last = st["power_armed"], st["power_percent"]
        out = {
            "ok": True,
            "is_on": bool(getattr(_laser, "is_on", False)),
            "power_armed": armed,
            "power_percent": last if armed else None,
            "beam_if_turned_on": st["beam"],
        }
        if not armed:
            out["last_requested_power_percent"] = last
            out["note"] = (
                "Power is NOT applied to the optics right now - the ND filter sits at the "
                "guide-beam/blocking position, so laser_on() would emit only the guide beam. "
                "Call set_laser_power(percent) to arm the measurement beam.")
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
    return None if _ccd is not None else {"ok": False, "error": _CCD_NOT_READY}


def _current_read_mode() -> str:
    """현재 CCD 읽기 모드를 이 파일의 인자 표기('fvb'/'single_track'/'image')로."""
    return _RO_MODE_TO_ARG.get(getattr(_ccd, 'ro_mode', ''), 'fvb')


def _apply_acq_mode(mode: str, exposure=None, num_accumulations=None,
                    kinetic_count=None, kinetic_cycle_time=None) -> dict | None:
    """취득 모드와 그 파라미터를 적용한다. 통과면 None, 실패면 error dict.

    None 인 파라미터는 드라이버가 현재 값을 그대로 둔다(드라이버의 None=유지 규약).
    **반드시 읽기 모드(_apply_read_mode)보다 먼저** 호출해야 한다 — create_buffer()
    가 aq_mode 에 의존하므로 순서가 뒤바뀌면 버퍼 모양이 어긋난다.
    """
    if mode not in _ACQ_MODES:
        return {"ok": False, "error": f"acquisition mode must be one of {list(_ACQ_MODES)}."}
    if mode == 'single':
        _ccd.set_aq_single_scan(exposure=exposure)
    elif mode == 'accumulate':
        _ccd.set_aq_accumulate_scan(exposure_time=exposure, num_acc=num_accumulations)
    elif mode == 'kinetic':
        _ccd.set_aq_kinetic_scan(
            exp_time=exposure,
            num_kin=kinetic_count,
            num_acc=num_accumulations if (num_accumulations or 0) > 1 else None,
            kin_time=kinetic_cycle_time,
        )
    else:                                   # run_till_abort
        _ccd.set_aq_run_till_abort_scan()
        if exposure is not None:
            _ccd.set_exposure_time(exposure)
    # set_aq_*_scan 이 다루지 않는 조합(예: single 모드인데 누적 횟수만 조정)도 반영한다.
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
        return {"ok": False, "error": f"read mode must be one of {list(_READ_MODES)}."}
    eff_hbin = hbin if hbin is not None else _ccd.get_current_hbin()
    if mode == 'fvb':
        _ccd.set_ro_full_vertical_binning(hbin=eff_hbin)
    elif mode == 'single_track':
        center = (single_track_center if single_track_center is not None
                  else getattr(_ccd, 'ro_single_track_center', None))
        if center is None:
            return {"ok": False,
                    "error": "single_track mode requires the single_track_center parameter."}
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
        return {"ok": False, "error": f"trigger mode must be one of {list(_TRIGGER_MODES)}."}
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
        return {"ok": False, "error": f"shutter must be one of {list(_SHUTTER_MODES)}."}
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


# ──────────────────────────────────────────
# 스펙트럼 수집 (Andor CCD)
# ──────────────────────────────────────────

def _persist_spectrum(result: dict, tag: str = "") -> dict:
    """측정 결과를 로컬(날짜/시간)에 저장하고 result['saved']에 경로·URL을 첨부한다.

    스테이지 좌표를 읽을 수 있으면 메타에 실어 제목/파일명이 좌표별로 붙게 한다
    (예: 10x10 스캔 → (x, y)). 저장 실패가 측정 결과 반환을 막지 않도록 방어적으로 처리.
    """
    meta: dict = {}
    if tag:
        meta["tag"] = tag
    try:
        if _stage is not None:
            pos = _stage.get_position()
            meta["x"], meta["y"] = round(float(pos[0]), 3), round(float(pos[1]), 3)
    except Exception:
        pass
    saved = _store_save_spectrum(result, meta or None)
    if saved.get("ok"):
        result["saved"] = saved
    else:
        result["saved_error"] = saved.get("error")
    return result


@_serialized("acquire_spectrum")
def acquire_spectrum(
    exposure: float = None,
    power: float = None,
    stabilize_sec: float = 0.5,
    acq_mode: str = None,
    num_accumulations: int = None,
    kinetic_count: int = None,
    kinetic_cycle_time: float = None,
    read_mode: str = None,
    hbin: int = None,
    single_track_center: int = None,
    single_track_width: int = None,
    trigger_mode: str = None,
    shutter: str = None,
) -> dict:
    """
    라만 스펙트럼 수집 (Single / Accumulate / Kinetic 모드 지원).

    [파라미터를 생략하면 '현재 장비 설정을 그대로 쓴다' — 2026-07-30 수정]
    예전에는 모든 파라미터에 기본값(0.2s / 40% / single / fvb / internal)이 박혀 있어,
    호출자가 생략하면 '유지'가 아니라 '기본값으로 리셋'이었다. 그래서 set_ccd_exposure,
    set_laser_power, set_ccd_acquisition_mode, set_ccd_read_mode, set_ccd_trigger_mode 로
    미리 맞춰 둔 설정이 측정 직전에 조용히 덮어써졌다(설정 툴 5개가 사실상 무효였다).
    "노출 1.0s 로 설정하고 측정" 같은 2단계 요청이 실제로는 0.2s 로 측정되던 원인이다.
    드라이버(andor_ccd_interface)는 이미 None=유지 규약을 갖고 있어 그대로 넘기면 된다.

    원자적 실행 흐름:
      1. 레이저 출력 설정 (ND filter motor 이동, 블로킹)
      2. 레이저 ON + 안정화 대기
      3. CCD 취득 모드 설정 (set_aq_* → set_ro_* 순서 필수)
      4. CCD 촬영 (StartAcquisition → 폴링 → GetAcquiredData)
      5. 레이저 OFF (성공/실패 무관하게 반드시 실행)

    Parameters
    ----------
    exposure : float or None
        CCD 노출 시간 [초]. None이면 현재 CCD 설정을 유지한다.
    power : float or None
        레이저 출력 [%], 0.004~100 (ND 필터 연속 조절).
        None이면 마지막으로 설정된 파워를 재적용한다(한 번도 없었다면 40%).
        레이저는 이 값을 항상 하드웨어에 적용한다 — 아래 주석 참고.
    stabilize_sec : float
        레이저 ON 후 안정화 대기 [초]. 기본 0.5.
    acq_mode : str or None
        'single' | 'accumulate' | 'kinetic'. None이면 현재 CCD 취득 모드를 유지한다.
    num_accumulations : int or None
        Accumulate/Kinetic 모드의 프레임당 누적 횟수. None이면 현재 값 유지.
    kinetic_count : int or None
        Kinetic 모드에서 수집할 총 프레임 수. None이면 현재 값 유지.
    kinetic_cycle_time : float or None
        Kinetic 프레임 간격 [초]. None이면 SDK가 자동 계산.
    read_mode : str or None
        'fvb' | 'single_track'. None이면 현재 CCD 읽기 모드를 유지한다.
        (CCD가 'image' 모드면 1D 스펙트럼을 조립할 수 없어 거부한다.)
    hbin : int or None
        수평 비닝 픽셀 수. None이면 현재 값 유지.
    single_track_center : int or None
        read_mode='single_track' 시 중심 픽셀 행. None이면 현재 설정된 트랙을 재사용.
    single_track_width : int or None
        read_mode='single_track' 시 트랙 폭 [픽셀]. None이면 현재 값 유지.
    trigger_mode : str or None
        'internal' | 'external' | 'external_start' | 'external_exposure' |
        'external_fvb_em' | 'software'. None이면 현재 트리거 모드를 유지한다.
    shutter : str or None
        'auto' | 'open' | 'close'. None이면 다른 파라미터와 같이 현재 설정을 유지한다 —
        단, 아무도 셔터를 지정한 적이 없으면 'auto'로 연다(CCD 초기화가 셔터를 close로
        두므로 그대로 두면 세션 첫 측정이 암전 프레임이 된다). 규약은 _effective_shutter
        참고. 다크/배경 프레임은 shutter='close'를 명시하거나 set_ccd_shutter('close')로
        걸어 둔다 — 후자는 이제 이후 측정에도 유지된다.

    Returns
    -------
    dict — Single / Accumulate 모드:
        ok, mode, length, max_intensity, sum_intensity, data,
        calibrated, exposure_time, laser_power_pct, shutter, trigger_mode,
        [num_accumulations,] [raman_shift_cm-1, wavelength_nm, laser_nm]
        exposure_time / laser_power_pct 등은 '요청값'이 아니라 장비에서 읽은 실제 적용값이다.

    dict — Kinetic 모드:
        ok, mode, num_frames, kinetic_count, exposure_time, laser_power_pct,
        frames: list of {frame_index, intensity, length, max_intensity,
                         sum_intensity, calibrated, [raman_shift_cm-1, ...]}
    """
    err = _ccd_ready()
    if err:
        return err
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}

    # ── 파라미터 해석 (하드웨어를 만지기 전에 전부 검증한다) ──
    # 레이저 파워만은 None이어도 반드시 하드웨어에 적용한다. laser_on()이 측정빔을 쏘려면
    # 드라이버의 _power_set 플래그가 True여야 하고, 그 플래그는 set_power()만이 세운다
    # (가이드빔 모드나 오토포커스 직후엔 False다). 같은 위치로의 재이동은 무해하다.
    if power is None:
        eff_power = float(getattr(_laser, "power_pct", None) or 40.0)
    else:
        eff_power = power                   # 검증은 _apply_laser_power 가 한다(단일 정책)

    eff_acq = acq_mode or getattr(_ccd, 'aq_mode', None) or 'single'
    if eff_acq not in _ACQ_MODES_1D:
        return {"ok": False, "error": (
            f"acquire_spectrum supports {list(_ACQ_MODES_1D)}, but the CCD is "
            f"currently in '{eff_acq}' mode. Pass acq_mode explicitly, or call "
            f"set_ccd_acquisition_mode first.")}

    # 현재 읽기 모드(드라이버 표기) → 이 함수의 인자 표기로 환산.
    eff_read = read_mode or _current_read_mode()
    if eff_read not in _READ_MODES_1D:
        return {"ok": False, "error": (
            f"acquire_spectrum assembles a 1D spectrum, but the CCD read mode is '{eff_read}'. "
            f"Pass read_mode='fvb' (or 'single_track'), or call set_ccd_read_mode first.")}

    eff_center = (single_track_center if single_track_center is not None
                  else getattr(_ccd, 'ro_single_track_center', None))
    if eff_read == 'single_track' and eff_center is None:
        return {"ok": False, "error": "single_track_center is required when read_mode='single_track'"}

    # 셔터·트리거는 레이저를 쏘기 전에 검증한다 — 잘못된 값으로 발사한 뒤 실패하면
    # 조사만 낭비되고 시료에는 이미 빔이 들어간 뒤다.
    eff_shutter = _effective_shutter(shutter)      # 규약은 _effective_shutter 단일 출처
    if eff_shutter not in _SHUTTER_MODES:
        return {"ok": False, "error": f"shutter must be one of {list(_SHUTTER_MODES)}."}

    # 트리거를 생략하면 드라이버가 기억하는 현재 모드를 그대로 쓴다(SDK엔 getter가 없다).
    eff_trigger = trigger_mode or getattr(_ccd, 'trigger_mode', None) or 'internal'
    if eff_trigger not in _TRIGGER_MODES:
        return {"ok": False, "error": f"trigger_mode must be one of {list(_TRIGGER_MODES)}."}

    # 파워 검증도 발사 전에 끝낸다(_apply_laser_power 가 실제 적용까지 한다).
    err = _apply_laser_power(eff_power, settle_s=stabilize_sec)
    if err:
        return err

    raw = None
    try:
        # 1. (파워는 위에서 이미 적용됨 — 검증 실패 시 레이저를 켜지 않기 위해 앞으로 뺐다)
        # 2. 레이저 ON + 안정화 대기
        _laser.laser_on()
        time.sleep(stabilize_sec)

        # 3-a. 취득 모드 설정 — 반드시 읽기 모드보다 먼저 (create_buffer가 aq_mode 의존).
        #      exposure/num_* 가 None이면 드라이버가 현재 값을 그대로 둔다.
        err = _apply_acq_mode(eff_acq, exposure=exposure,
                              num_accumulations=num_accumulations,
                              kinetic_count=kinetic_count,
                              kinetic_cycle_time=kinetic_cycle_time)
        if err:
            return err

        # 3-b. 읽기 모드 설정 — create_buffer()가 여기서 호출된다. aq_mode를 바꿨으면 버퍼
        #      모양도 다시 잡아야 하므로, read_mode를 생략했더라도 '현재 모드'로 한 번 다시
        #      적용한다(hbin/트랙 파라미터는 현재 값을 그대로 재사용).
        err = _apply_read_mode(eff_read, hbin=hbin, single_track_center=eff_center,
                               single_track_width=single_track_width)
        if err:
            return err

        # 3-c. 트리거 — 생략했으면 건드리지 않는다(현재 설정 유지).
        if trigger_mode is not None:
            err = _apply_trigger_mode(trigger_mode)
            if err:
                return err

        # 3-d. 셔터 — 생략 시 규약은 _effective_shutter 참고. 'close'면 다크 프레임이 된다.
        err = _apply_shutter(eff_shutter, explicit=(shutter is not None))
        if err:
            return err

        # 3-e. 이전 취득 버퍼 해제 (파라미터 변경 시 SDK 내부 메모리 단편화 방지)
        _ccd.free_internal_memory()

        # 실제로 장비에 걸린 값을 읽어 둔다 — 타임아웃 계산과 결과 보고에 모두 쓴다.
        # 요청값이 아니라 실측값을 보고해야 '노출을 바꿨는데 왜 결과가 같은가'를 알 수 있다.
        try:
            eff_exposure = float(_ccd.get_exposure_time())
        except Exception:
            eff_exposure = float(exposure) if exposure is not None else 0.0
        eff_num_acc = int(getattr(_ccd, 'num_acc', 1) or 1)
        eff_num_kin = int(getattr(_ccd, 'num_kin', 1) or 1)

        # 4. 촬영
        if eff_acq == 'kinetic':
            # Kinetic: 버퍼가 3D (num_kin, Ny_ro, Nx_ro) — start_acquisition_cycle() 직접 사용 불가
            _ccd.prepare_acquisition()        # 메모리 사전 할당 + 타이밍 초기화 (첫 프레임 지연 방지)
            _ccd.start_acquisition()
            if eff_trigger == 'software':     # software 트리거 발송 (없으면 ACQUIRING 상태에서 무한 대기)
                _ccd.send_software_trigger()

            # 외부 트리거 미도달 / 하드웨어 장애 시 무한 대기 방지.
            # 실제 적용된 노출·프레임수로 계산한다(요청값은 None일 수 있다).
            _cyc = kinetic_cycle_time if kinetic_cycle_time else (eff_exposure + 0.1)
            _timeout_s = _cyc * max(eff_num_kin, 1) * max(eff_num_acc, 1) * 2 + 15.0
            _deadline = time.time() + _timeout_s
            while _ccd.get_status() != 'IDLE':
                if time.time() > _deadline:
                    try:
                        _ccd.abort_acquisition()
                    except Exception:
                        pass
                    raise TimeoutError(
                        f"kinetic acquisition timeout: exceeded {_timeout_s:.1f} s "
                        f"(trigger={eff_trigger}, frames={eff_num_kin})"
                    )
                time.sleep(0.05)
            raw = _ccd.get_acquired_data()
        else:
            # Single / Accumulate: trigger_mode를 전달하여 software trigger 지원
            # internal 트리거면 무한 대기 허용, 외부/소프트웨어면 deadline 부여
            _timeout_ms = (
                None if eff_trigger == 'internal'
                else int((eff_exposure * max(eff_num_acc, 1) * 2 + 15) * 1000)
            )
            raw = _ccd.start_acquisition_cycle(
                trigger_mode_str=eff_trigger,
                timeout_ms=_timeout_ms,
            )

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        # 5. 레이저 OFF (성공/실패 무관 — 안전 보장). 위의 이른 return 들도 이 블록을
        #    거치므로, 검증 실패로 빠져나가도 레이저가 켜진 채 남지 않는다.
        _laser_off_quiet()

    if raw is None:
        return {"ok": False, "error": "CCD data acquisition failed"}

    # ── 결과 조립 ──
    # 보고하는 exposure_time / laser_power_pct / num_* 는 모두 '장비에 실제로 걸린 값'이다.
    if eff_acq == 'kinetic':
        # raw shape: (num_kin, Ny_ro, Nx_ro) — FVB이면 (num_kin, 1, Nx_ro)
        cal = getattr(_ccd, '_calibrator', None)
        frames = []
        for i in range(raw.shape[0]):
            frame_intensity = raw[i].flatten().tolist()
            frame = {
                "frame_index": i,
                "intensity": frame_intensity,
                "length": len(frame_intensity),
                "max_intensity": float(max(frame_intensity)),
                "sum_intensity": float(sum(frame_intensity)),
                "calibrated": False,
            }
            if cal is not None:
                pixels = range(len(frame_intensity))
                frame.update({
                    "calibrated": True,
                    "raman_shift_cm-1": [float(cal.pixel_to_raman_shift(p)) for p in pixels],
                    "wavelength_nm":    [float(cal.pixel_to_wavelength(p))   for p in pixels],
                    "laser_nm":         float(cal.laser_nm),
                })
            frames.append(frame)
        return _persist_spectrum({
            "ok": True,
            "mode": "kinetic",
            "num_frames": len(frames),
            "kinetic_count": eff_num_kin,
            "num_accumulations": eff_num_acc,
            "exposure_time": eff_exposure,
            "laser_power_pct": eff_power,
            "trigger_mode": eff_trigger,
            "shutter": eff_shutter,
            "frames": frames,
        })
    else:
        # Single / Accumulate: start_acquisition_cycle()이 calibration dict 반환
        data = raw
        if data.get("error"):
            return {"ok": False, "error": data["error"]}
        intensity = data["intensity"]
        result = {
            "ok": True,
            "mode": eff_acq,
            "length": len(intensity),
            "max_intensity": float(max(intensity)) if intensity else 0.0,
            "sum_intensity": float(sum(intensity)) if intensity else 0.0,
            "data": intensity,
            "calibrated": data.get("calibrated", False),
            "exposure_time": eff_exposure,
            "laser_power_pct": eff_power,
            "trigger_mode": eff_trigger,
            "shutter": eff_shutter,
        }
        if eff_acq == 'accumulate':
            result["num_accumulations"] = eff_num_acc
        if data.get("calibrated"):
            result["raman_shift_cm-1"] = data["raman_shift_cm-1"]
            result["wavelength_nm"]    = data["wavelength_nm"]
            result["laser_nm"]         = data["laser_nm"]
        return _persist_spectrum(result)

# ──────────────────────────────────────────
# CCD 파라미터 설정 툴
# ──────────────────────────────────────────

def get_ccd_info() -> dict:
    """현재 CCD 설정값 및 상태를 한 번에 조회한다."""
    err = _ccd_ready()
    if err:
        return err
    try:
        t           = _ccd.get_temperature()
        temp_status = _ccd.get_temperature_status()
        cam_status  = _ccd.get_status()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    def _attr(name):
        return getattr(_ccd, name, None)

    hs_conv = None
    if hasattr(_ccd, 'HSSpeeds_Conventional') and _ccd.HSSpeeds_Conventional:
        hs_conv = _ccd.HSSpeeds_Conventional[0]

    return {
        "ok":                      True,
        "camera_status":           cam_status,
        "temperature_C":           t,
        "temperature_status":      temp_status,
        "cooler_on":               _attr('cooler_on'),
        "exposure_time_s":         _attr('exposure_time'),
        "acquisition_mode":        _attr('aq_mode'),
        "read_mode":               _attr('ro_mode'),
        # 트리거/셔터도 함께 보고한다 — acquire_spectrum 이 이 값들을 생략 시 '유지'하므로,
        # 지금 무엇이 걸려 있는지 볼 수 없으면 다음 측정 조건을 예측할 수 없다.
        "trigger_mode":            _attr('trigger_mode'),
        "shutter_mode":            _attr('shutter_mode'),
        "num_accumulations":       _attr('num_acc'),
        "num_kinetics":            _attr('num_kin'),
        "em_mode":                 _attr('em_mode'),
        "em_gain":                 _attr('em_gain'),
        "output_amp":              _attr('output_amp'),
        "preamp_gain_index":       _attr('preamp_gain_i'),
        "preamp_gains_available":  _attr('preamp_gains'),
        "vs_speeds_us":            _attr('VSSpeeds'),
        "hs_speeds_conventional_mhz": hs_conv,
        "readout_pixels_Nx":       _attr('Nx_ro'),
        "readout_pixels_Ny":       _attr('Ny_ro'),
        "detector_Nx":             _attr('Nx'),
        "detector_Ny":             _attr('Ny'),
    }


@_serialized("set_ccd_exposure")
def set_ccd_exposure(exposure_time: float) -> dict:
    """CCD 노출 시간(초)을 설정한다."""
    err = _ccd_ready()
    if err:
        return err
    if exposure_time <= 0:
        return {"ok": False, "error": "Exposure time must be greater than 0."}
    try:
        actual = _ccd.set_exposure_time(exposure_time)
        return {"ok": True, "exposure_time_s": actual}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_ccd_acquisition_mode")
def set_ccd_acquisition_mode(
    mode: str,
    num_accumulations: int = None,
    num_kinetics: int = None,
) -> dict:
    """
    CCD 취득 모드를 설정한다(측정과 분리해서 미리 걸어 두고 싶을 때).

    mode: 'single' | 'accumulate' | 'kinetic' | 'run_till_abort'
    num_accumulations: accumulate/kinetic 모드에서 누적 횟수
    num_kinetics: kinetic 모드에서 총 프레임 수

    acquire_spectrum(acq_mode=..., num_accumulations=..., kinetic_count=...) 로도
    같은 설정을 걸 수 있고, 내부적으로 같은 코드를 지난다. 측정 직전에 정할 값이면
    acquire_spectrum 쪽 인자를 쓰는 편이 왕복이 하나 줄어든다.

    주의: 'run_till_abort' 는 이 도구로만 걸 수 있다. acquire_spectrum 은 1D 스펙트럼
    한 벌을 조립해 돌려주는 도구라 무한 연속 취득 모드를 다루지 못하고, 이 모드로 둔 채
    측정을 시도하면 acq_mode 를 명시하라는 에러를 받는다.
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        # 적용은 acquire_spectrum 과 완전히 같은 경로를 쓴다(허용값·순서 단일 출처).
        err = _apply_acq_mode(mode, num_accumulations=num_accumulations,
                              kinetic_count=num_kinetics)
        if err:
            return err
        out = {
            "ok":                True,
            "acquisition_mode":  mode,
            # 요청값이 아니라 장비에 실제로 걸린 값을 보고한다 — None 을 넘겨 '유지'한
            # 경우에도 현재 값을 알 수 있어야 다음 측정 조건을 예측할 수 있다.
            "num_accumulations": getattr(_ccd, 'num_acc', None),
            "num_kinetics":      getattr(_ccd, 'num_kin', None),
        }
        if mode == 'run_till_abort':
            out["note"] = ("acquire_spectrum cannot run in 'run_till_abort'. Switch back with "
                           "set_ccd_acquisition_mode('single') before measuring, or pass "
                           "acq_mode explicitly to acquire_spectrum.")
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_ccd_trigger_mode")
def set_ccd_trigger_mode(mode: str) -> dict:
    """
    CCD 트리거 모드를 설정한다.

    mode: 'internal' | 'external' | 'external_start' |
          'external_exposure' | 'external_fvb_em' | 'software'

    acquire_spectrum(trigger_mode=) 와 **같은 코드**를 지나므로 허용값이 항상 일치한다
    (예전에는 이쪽에만 external_fvb_em 이 빠져 있어 같은 값이 한쪽에서만 통했다).
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        err = _apply_trigger_mode(mode)
        if err:
            return err
        return {"ok": True, "trigger_mode": mode}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_ccd_read_mode")
def set_ccd_read_mode(
    mode: str,
    hbin: int = None,
    single_track_center: int = None,
    single_track_width: int = None,
) -> dict:
    """
    CCD 읽기 모드(readout mode)를 설정한다.

    mode:
      'fvb'          — Full Vertical Binning (1D 스펙트럼, 기본)
      'single_track' — 특정 수직 행 하나만 읽음. single_track_center(행 번호) 필수
      'image'        — 2D 이미지 전체 (acquire_spectrum 은 1D만 조립하므로 측정 전엔 부적합)

    hbin / single_track_center / single_track_width 는 acquire_spectrum 과 이름을 맞췄고,
    적용도 **같은 코드**를 지난다. None 이면 현재 설정값을 그대로 유지한다.

    'image' 는 이 도구로만 걸 수 있다 — acquire_spectrum 은 1D 스펙트럼을 조립하므로
    2D 이미지 모드로 두면 측정이 거부된다. 이미지 모드는 set_ccd_image_flip 처럼
    2D 전용 설정을 만질 때만 쓰고, 측정 전에 'fvb' 로 되돌린다.
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        err = _apply_read_mode(mode, hbin=hbin,
                               single_track_center=single_track_center,
                               single_track_width=single_track_width)
        if err:
            return err
        out = {
            "ok":          True,
            "read_mode":   mode,
            "hbin":        _ccd.get_current_hbin(),      # 실제 적용값
            "Nx_ro":       _ccd.Nx_ro,
            "Ny_ro":       _ccd.Ny_ro,
        }
        if mode == 'image':
            out["note"] = ("acquire_spectrum cannot assemble a 1D spectrum in 'image' mode. "
                           "Call set_ccd_read_mode('fvb') before measuring.")
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_ccd_preamp_gain")
def set_ccd_preamp_gain(index: int) -> dict:
    """
    프리앰프(PreAmp) 이득 인덱스를 설정한다.
    사용 가능한 이득 목록은 get_ccd_info()의 preamp_gains_available 참조.
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        _ccd.set_preamp_gain(index)
        gain_val = _ccd.preamp_gains[index] if _ccd.preamp_gains else None
        return {"ok": True, "preamp_gain_index": index, "gain_value": gain_val}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_ccd_em_gain")
def set_ccd_em_gain(gain: int) -> dict:
    """
    EM(Electron Multiplication) 이득을 설정한다.
    EM CCD 전용. get_ccd_info()의 em_gain_range 참조.
    """
    err = _ccd_ready()
    if err:
        return err
    if not getattr(_ccd, 'em_mode', False):
        return {"ok": False, "error": "This camera is not an EM CCD."}
    try:
        _ccd.set_EMCCD_gain(gain)
        return {"ok": True, "em_gain": gain}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# [제거됨 — set_mcp_gain / get_mcp_gain_range, 2026-07-30]
# 이 장비는 MCP(Micro-Channel Plate) 이득을 지원하지 않는다. 실측 로그에서 SDK 가
# GetMCPGainRange/SetMCPGain 에 DRV_NOT_SUPPORTED(20991) 를 반환한다(iStar ICCD 전용
# 기능이고 현재 카메라는 iDus 계열). 두 툴을 남겨 두면 에이전트가 "게인을 낮춰 포화를
# 해결하라"는 과제에서 반드시 실패하는 경로로 유인된다 — 실제로 그렇게 실패했다.
# 지원되는 대체 수단: set_ccd_preamp_gain(index) + get_ccd_info()의 preamp_gains_available.


@_serialized("set_ccd_output_amp")
def set_ccd_output_amp(amp: int) -> dict:
    """
    출력 앰프를 선택한다.
    0 = EMCCD 앰프, 1 = 일반(Conventional) 앰프.
    """
    err = _ccd_ready()
    if err:
        return err
    if amp not in (0, 1):
        return {"ok": False, "error": "amp must be 0 (EM) or 1 (Conventional)."}
    try:
        _ccd.set_output_amp(amp)
        return {"ok": True, "output_amp": amp, "mode": "EM" if amp == 0 else "Conventional"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_ccd_shift_speeds")
def set_ccd_shift_speeds(vs_index: int = None, hs_index: int = None) -> dict:
    """
    수직(VS) 및 수평(HS) 시프트 속도 인덱스를 설정한다.
    사용 가능한 속도 목록은 get_ccd_info()의 vs_speeds_us / hs_speeds_conventional_mhz 참조.
    둘 중 하나만 지정해도 됩니다.
    """
    err = _ccd_ready()
    if err:
        return err
    result = {"ok": True}
    try:
        if vs_index is not None:
            _ccd.set_vs_speed(vs_index)
            result["vs_speed_index"] = vs_index
            result["vs_speed_us"]    = (_ccd.VSSpeeds[vs_index]
                                        if vs_index < len(_ccd.VSSpeeds) else None)
        if hs_index is not None:
            _ccd.set_hs_speed_conventional(hs_index)
            result["hs_speed_index"] = hs_index
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return result


@_serialized("set_ccd_temperature")
def set_ccd_temperature(temp: int) -> dict:
    """
    CCD 냉각 목표 온도(°C)를 설정한다.
    실제 안정화는 시간이 걸리며, 상태는 get_ccd_info()로 확인한다.
    일반적 범위: -80 ~ 20°C.
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        _ccd.set_temperature(temp)
        return {"ok": True, "target_temperature_C": temp}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_ccd_cooler")
def set_ccd_cooler(on: bool) -> dict:
    """CCD 냉각기를 켜거나(True) 끈다(False)."""
    err = _ccd_ready()
    if err:
        return err
    try:
        _ccd.set_cooler(on)
        return {"ok": True, "cooler": "ON" if on else "OFF"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_ccd_shutter")
def set_ccd_shutter(mode: str) -> dict:
    """
    셔터 모드를 설정한다.
    'auto'  — 취득 시 자동 열고 닫음 (기본)
    'open'  — 강제로 열어둠
    'close' — 강제로 닫아둠 (다크/배경 측정 시)

    다른 CCD 설정 도구와 **같은 규약**이다(2026-07-31 수정): 여기서 한 번 걸어 두면
    이후 acquire_spectrum 이 shutter 인자를 생략해도 그 설정을 그대로 유지한다.
    따라서 다크/배경 프레임을 여러 장 찍을 때 set_ccd_shutter('close') 한 번이면 된다.

    예외는 '아무도 셔터를 지정한 적이 없는' 상태뿐이다 — CCD 초기화가 셔터를 close 로
    두기 때문에(hardware_manager._init_ccd, "촬영 직전까지 광원 차단"), 그 상태를 그대로
    유지하면 세션 첫 측정이 조용히 암전 프레임이 된다. 그래서 한 번도 지정되지 않았을
    때만 acquire_spectrum 이 'auto' 로 연다. 자세한 규약은 _effective_shutter 참고.
    """
    err = _ccd_ready()
    if err:
        return err
    try:
        # 허용값·적용 코드는 acquire_spectrum 과 공용. explicit=True 로 '직접 지정했음'을
        # 남겨, 이후 측정이 이 설정을 덮어쓰지 않게 한다.
        err = _apply_shutter(mode, explicit=True)
        if err:
            return err
        return {"ok": True, "shutter": mode,
                "note": ("This setting persists: later acquire_spectrum calls keep it unless you "
                         "pass their own shutter argument. Use 'close' for dark/background frames "
                         "and set it back to 'auto' before normal measurements.")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_ccd_image_flip")
def set_ccd_image_flip(hflip: bool, vflip: bool) -> dict:
    """
    이미지 반전을 설정한다. read_mode='image' 에서만 허용한다.
    hflip: 수평 좌우 반전 여부
    vflip: 수직 상하 반전 여부

    [왜 image 모드로 제한하는가]
    이 분광기는 pixel 0 이 고파장쪽에 맺혀서, 초기화가 FVB 기준으로 hflip=True 를 걸어
    둔다(hardware_manager._init_ccd, Config.ini [ANDOR_IDUS] Reverse=True). 그 상태에서
    캘리브레이션(raman_shift_cm-1 / wavelength_nm)이 픽셀 순서와 맞춰져 있다.
    1D 스펙트럼 모드에서 이 값을 뒤집으면 세기 배열만 좌우로 뒤집히고 축 배열은 그대로여서,
    스펙트럼이 조용히 파장축과 어긋난다(에러도 안 난다 — 그래서 더 위험하다). vflip 은
    FVB 가 수직을 전부 합산하므로 애초에 의미가 없다.
    2D 이미지 모드에서는 축 정합 문제가 없어 그때만 노출한다.
    """
    err = _ccd_ready()
    if err:
        return err
    ro = getattr(_ccd, 'ro_mode', None)
    if ro != 'IMG':
        return {"ok": False, "error": (
            f"Image flip is only allowed in the 'image' read mode (the CCD is currently in "
            f"'{ro}'). In 1D spectrum modes (fvb / single_track) flipping would silently "
            f"misalign the intensity array against the calibrated raman_shift_cm-1 / "
            f"wavelength_nm axes, and the factory orientation is already set at startup. "
            f"Call set_ccd_read_mode(mode='image') first if you really need a flipped 2D image.")}
    try:
        _ccd.set_image_flip(hflip=hflip, vflip=vflip)
        return {"ok": True, "hflip": hflip, "vflip": vflip, "read_mode": "image"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
@_serialized("start_camera_stream")
def start_camera_stream() -> dict:
    """
    카메라 실시간 스트리밍을 시작합니다.
    USE_camera_stream.py의 StreamingTUCam.start_stream()을 호출합니다.

    반환의 already_streaming: 이 호출 "이전에" 이미 스트리밍 중이었는지.
    호출자가 스트림 소유권을 판단하는 근거다 — 남이(예: 프론트 MJPEG 뷰) 켜 둔
    스트림을 내가 끄면 그쪽 화면이 죽는다. already_streaming=True면 끄지 말 것.
    """
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}

    try:
        # 이미 스트리밍 중인지 확인 (StreamingTUCam 내부 속성 활용)
        if getattr(_camera, 'is_streaming', False):
            return {"ok": True, "already_streaming": True,
                    "status": "Camera is already streaming."}

        _camera.start_stream()
        return {"ok": True, "already_streaming": False,
                "status": "Camera streaming started successfully."}

    except Exception as e:
        return {"ok": False, "error": f"Failed to start streaming: {str(e)}"}

@_serialized("stop_camera_stream")
def stop_camera_stream() -> dict:
    """
    카메라 실시간 스트리밍을 중지합니다.
    USE_camera_stream.py의 StreamingTUCam.stop_stream()을 호출합니다.
    """
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}

    try:
        if not getattr(_camera, 'is_streaming', False):
            return {"ok": True, "status": "Camera is not currently streaming."}

        _camera.stop_stream()
        return {"ok": True, "status": "Camera streaming stopped successfully."}

    except Exception as e:
        return {"ok": False, "error": f"Failed to stop streaming: {str(e)}"}


@_serialized("set_camera_exposure")
def set_camera_exposure(ms: float) -> dict:
    """카메라(TUCam) 노출 시간(ms)을 설정한다."""
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    if ms <= 0:
        return {"ok": False, "error": "Exposure time must be greater than 0."}
    try:
        _camera.set_exposure(ms)
        return {"ok": True, "exposure_ms": ms}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("set_camera_auto_exposure")
def set_camera_auto_exposure(enabled: bool) -> dict:
    """카메라 자동 노출을 활성화(True) 또는 비활성화(False)한다."""
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    try:
        from backend.TuCam.TUCam import TUCAM_Capa_SetValue, TUCAM_IDCAPA
        TUCAM_Capa_SetValue(
            _camera.TUCAMOPEN.hIdxTUCam,
            TUCAM_IDCAPA.TUIDC_ATEXPOSURE.value,
            1 if enabled else 0,
        )
        return {"ok": True, "auto_exposure": enabled}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# [제거됨 — capture_camera_frame, 2026-07-30]
# 최신 프레임 1장의 통계와 선명도 점수를 돌려주던 툴이다. analyze_microscope_image 와
# 같은 일(프레임 1장 가져오기)을 하면서 다음 문제가 있었다:
#   · uint16 프레임을 8bit 로 줄이지 않아 min/max/mean 이 다른 캡처 함수들과 다른 스케일로
#     나왔다(같은 화면을 두 툴로 보면 숫자가 안 맞는다).
#   · sharpness_score(라플라시안 분산)를 "오토포커스에 활용 가능"이라고 안내했지만,
#     run_autofocus 는 가이드빔 스팟 '면적 최소화'를 쓴다. 두 지표는 무관해서, 이걸로
#     직접 초점을 잡으면 run_autofocus 와 다른 Z 에 수렴한다.
# 통계와 선명도는 analyze_microscope_image 의 반환에 병합했다(같은 전처리 경로를 쓰므로
# 좌표계·스케일이 일치한다).


# ──────────────────────────────────────────
# 레이저 — 가이드빔
# ──────────────────────────────────────────

@_serialized("set_guide_beam_mode")
def set_guide_beam_mode() -> dict:
    """
    레이저를 가이드빔 대기 상태로 전환한다.
    - 빔 스플리터(축04) → 대기 위치
    - ND 필터(축02) → 메인 빔 차단 위치
    측정 레이저 미사용 시 시편 정렬·초점 확인에 활용한다.
    """
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    try:
        _laser.set_guide_beam()
        return {"ok": True, "status": "Switched to guide-beam mode"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 오토포커스 (카메라 선명도 기반 Z 스윕)
# ──────────────────────────────────────────

@_serialized("run_autofocus")
def run_autofocus(
    initial_z: float = None,
    step_size: float = 0.030,
    min_step: float = 0.001,
    max_steps: int = 100,
) -> dict:
    """
    가이드빔 레이저 스팟 면적 최소화 기반 힐클라이밍 오토포커스.
    USE_autofocus_local.py의 AutoFocusLocal 알고리즘을 헤드리스로 실행한다.

    동작 원리:
      각 Z 위치에서 레이저 OFF → 배경 프레임 취득 → 레이저 ON → 레이저 프레임 취득
      → clip 차분 → GaussianBlur → Otsu threshold → 스팟 픽셀 수(면적) 계산
      → 면적이 작을수록(레이저 스팟이 날카로울수록) 초점이 맞음
      → 적응형 힐클라이밍: 개선되면 같은 방향 전진, 나빠지면 방향 반전 + 보폭 절반
      → 역대 최솟값(global_best_z) 위치로 최종 귀환

    Parameters
    ----------
    initial_z : float, optional
        탐색 시작 Z 위치(mm). None이면 현재 Z 유지.
    step_size : float
        초기 Z 이동 보폭(mm). 기본 0.030mm (30µm).
    min_step : float
        최소 보폭(mm) — 이 이하면 탐색 종료. 기본 0.001mm (1µm).
    max_steps : int
        최대 스텝 수 — 초과 시 강제 종료. 기본 100.

    Returns
    -------
    dict
        optimal_z, best_area_px, step_count, z_scores, current_position
    """
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    if initial_z is not None:
        err = _check_stage_target(z=initial_z)
        if err:
            return err

    try:
        pos = _stage.get_position()
        cur_x, cur_y, cur_z = pos[0], pos[1], pos[2]
        cur_a = pos[3] if len(pos) > 3 else 0
        n_clamped = 0

        def _goto_z(z: float) -> float:
            """Z 이동. **허용 범위로 클리핑한다** — 힐클라이밍은 목표 Z 를 스스로 밀어
            올리는 루프라, 예전처럼 검사 없이 move_absolute 를 부르면 스테이지 한계를
            그대로 넘어서는 명령이 나갔다(이 함수만 범위 검증이 없었다). 여기서는
            거부가 아니라 클리핑이 맞다 — 한계에 닿으면 면적이 개선되지 않으므로
            알고리즘이 스스로 방향을 반전한다."""
            nonlocal n_clamped
            zc = max(STAGE_MIN_Z, min(STAGE_MAX_Z, float(z)))
            if zc != float(z):
                n_clamped += 1
            _stage.move_absolute(cur_x, cur_y, zc, cur_a)
            time.sleep(0.3)
            return zc

        if initial_z is not None:
            cur_z = _goto_z(initial_z)

        # 가이드빔 모드 + 카메라 스트리밍 보장
        _laser.set_guide_beam()
        if not getattr(_camera, 'is_streaming', False):
            _camera.start_stream()

        # 목적함수는 vision.guide_beam_spot_area 단일 출처다 — USE_autofocus_local 의
        # 대화형 오토포커스와 같은 함수를 쓰므로 두 경로가 같은 Z 로 수렴한다.
        def _capture_spot_area() -> int:
            return _vis.guide_beam_spot_area(_camera, _laser, n_avg=3)

        # 힐클라이밍 상태
        best_z = cur_z
        best_area = float('inf')
        direction = 1
        step_count = 0
        global_best_area = float('inf')
        global_best_z = cur_z
        z_scores: list = []
        sweep_state = 'init'

        while sweep_state != 'done':
            pos_now = _stage.get_position()
            z_now = pos_now[2] if pos_now else cur_z

            area = _capture_spot_area()
            z_scores.append({"z": round(z_now, 4), "area_px": area})

            if 0 < area < global_best_area:
                global_best_area = area
                global_best_z = z_now

            if sweep_state == 'init':
                best_area = area
                best_z = z_now
                _goto_z(best_z + direction * step_size)
                sweep_state = 'check'

            elif sweep_state == 'check':
                step_count += 1
                if area < best_area:
                    best_area = area
                    best_z = z_now
                    _goto_z(best_z + direction * step_size)
                else:
                    direction *= -1
                    step_size /= 2.0
                    if step_size < min_step or step_count >= max_steps:
                        sweep_state = 'done'
                    else:
                        _goto_z(best_z + direction * step_size)

        # 역대 최솟값 위치로 최종 귀환
        _goto_z(global_best_z)
        time.sleep(0.2)
        _laser_off_quiet()

        out = {
            "ok": True,
            "optimal_z": global_best_z,
            "best_area_px": global_best_area,
            "step_count": step_count,
            "z_scores": z_scores,
            "current_position": {"x": cur_x, "y": cur_y, "z": global_best_z},
        }
        if n_clamped:
            out["z_limit_hits"] = n_clamped
            out["note"] = (
                f"The search hit the Z travel limit ({STAGE_MIN_Z} to {STAGE_MAX_Z} mm) "
                f"{n_clamped} time(s), so the focus may lie outside the reachable range. "
                "If best_area_px is still large, the sample height is likely off - reposition "
                "the sample or the objective rather than repeating autofocus.")
        return out
    except Exception as e:
        _laser_off_quiet()
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 데이터 저장 / 로드
# ──────────────────────────────────────────

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ──────────────────────────────────────────
# 배경 제거 (IPBSA)
# ──────────────────────────────────────────

def _ipbsa(intensity, poly_order=5, max_iterations=100, threshold=0.001):
    import numpy as np
    y = np.array(intensity, dtype=np.float64)
    n = len(y)
    x = np.linspace(0.0, 1.0, n)
    working = y.copy()
    prev_bg = np.zeros(n, dtype=np.float64)
    converged = False
    for i in range(max_iterations):
        coeffs = np.polyfit(x, working, deg=poly_order)
        bg = np.polyval(coeffs, x)
        working = np.minimum(y, bg)
        denom = np.linalg.norm(prev_bg)
        if denom > 0 and np.linalg.norm(bg - prev_bg) / denom < threshold:
            converged = True
            break
        prev_bg = bg.copy()
    corrected = np.clip(y - bg, 0.0, None)
    return corrected.tolist(), bg.tolist(), i + 1, converged


def apply_background_subtraction(
    poly_order: int = 5,
    max_iterations: int = 100,
    threshold: float = 0.001,
    source: str = "last",
    version_label: str = "default",
    save_result: bool = False,
) -> dict:
    """IPBSA(반복 다항식 배경 제거)를 수행하고 결과를 이 세션의 버전 목록에 저장한다."""
    _st = _sstate()
    _last_spectrum = _st["last_spectrum"]

    if not (2 <= poly_order <= 10):
        return {"ok": False, "error": f"poly_order must be 2-10 (got: {poly_order})"}
    if not (10 <= max_iterations <= 500):
        return {"ok": False, "error": f"max_iterations must be 10-500 (got: {max_iterations})"}
    if not (0.001 <= threshold <= 1.0):
        return {"ok": False, "error": f"threshold must be 0.001-1.0 (got: {threshold})"}

    intensity: list = []
    raman_shift = None

    if source == "last":
        if _last_spectrum is None:
            return {
                "ok": False,
                "error": "No saved spectrum. Call acquire_spectrum() first.",
            }
        if "data" not in _last_spectrum:
            return {
                "ok": False,
                "error": "The last spectrum is in Kinetic mode. This applies only to Single/Accumulate spectra.",
            }
        intensity = _last_spectrum["data"]
        raman_shift = _last_spectrum.get("raman_shift_cm-1")
    else:
        filepath = Path(source)
        if not filepath.is_absolute():
            filepath = _DATA_DIR / source
        if not filepath.exists():
            return {"ok": False, "error": f"File not found: {filepath}"}
        try:
            if filepath.suffix.lower() == ".json":
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if "data" in loaded:
                    intensity = loaded["data"]
                elif "corrected_data" in loaded:
                    intensity = loaded["corrected_data"]
                else:
                    return {"ok": False, "error": "The JSON file has no 'data' or 'corrected_data' key."}
                raman_shift = loaded.get("raman_shift_cm-1")
            elif filepath.suffix.lower() == ".csv":
                # CSV 읽기는 load_spectrum 에 위임한다 — 여기서 csv.DictReader 를 직접
                # 쓰던 예전 코드는 '# key,value' 메타 주석행을 건너뛰지 않아서, 측정
                # 자동저장 CSV(data/results/...)를 source 로 주면 첫 주석행이 헤더로 잡혀
                # "intensity 열이 없다"로 실패했다(BOM 도 처리하지 않았다).
                loaded = load_spectrum(str(filepath))
                if not loaded.get("ok"):
                    return {"ok": False, "error": f"File load error: {loaded.get('error')}"}
                intensity = loaded["intensity"]
                raman_shift = loaded.get("raman_shift_cm-1")
            else:
                return {"ok": False, "error": "Unsupported file format (only JSON or CSV allowed)."}
        except Exception as e:
            return {"ok": False, "error": f"File load error: {e}"}

    if not intensity:
        return {"ok": False, "error": "The spectrum intensity array is empty."}
    if len(intensity) < poly_order + 1:
        return {
            "ok": False,
            "error": (
                f"Spectrum length ({len(intensity)}) is smaller than poly_order+1 ({poly_order + 1}). "
                "Lower the polynomial order."
            ),
        }

    try:
        corrected, background, iterations_run, converged = _ipbsa(
            intensity=intensity,
            poly_order=poly_order,
            max_iterations=max_iterations,
            threshold=threshold,
        )
    except Exception as e:
        return {"ok": False, "error": f"IPBSA algorithm error: {e}"}

    saved_path = None
    if save_result:
        # run_store 세션 폴더에 저장한다. 예전에는 data/ 최상위에 bg_corrected_<label>.csv 로
        # 떨궈서, 같은 라벨을 쓰면 이전 결과를 덮어썼고 어느 과제 산출물인지도 알 수 없었다
        # (save_spectrum 이 세션 폴더로 옮겨진 것과 같은 이유 — run_store.py 참고).
        try:
            from backend.agents import run_store
            save_filepath, rel = run_store.new_spectrum_path(f"bg_corrected_{version_label}")
            # 저장 포맷은 spectrum_store.write_spectrum_csv 단일 출처를 쓴다. 예전에는
            # 여기서 csv.writer 를 직접 돌리며 세기 열을 'corrected_intensity' 로 썼는데,
            # 같은 폴더에 run_analysis 의 save_result 가 'intensity' 로 쓰고 있어서
            # 한 폴더 안에 두 포맷이 섞였다(load_spectrum 이 열 이름을 추측해야 했던 이유).
            _store_write_csv(
                save_filepath,
                intensity=corrected,
                raman_shift=raman_shift,
                background=background,
                meta={"kind": "background_subtracted", "version_label": version_label,
                      "poly_order": poly_order, "iterations_run": iterations_run},
            )
            run_store.record(run_store.KIND_SPECTRA, rel, num_points=len(corrected),
                             kind_detail="background_subtracted", version_label=version_label)
            # 상대경로를 준다 — 이 문자열을 그대로 load_spectrum 에 넘겨 다시 읽을 수 있다.
            saved_path = rel
        except Exception:
            pass

    result = {
        "ok": True,
        "version_label": version_label,
        "poly_order": poly_order,
        "max_iterations": max_iterations,
        "threshold": threshold,
        "iterations_run": iterations_run,
        "converged": converged,
        "max_corrected_intensity": float(max(corrected)) if corrected else 0.0,
        "max_background_intensity": float(max(background)) if background else 0.0,
        "corrected_data": corrected,
        "background_data": background,
    }
    if raman_shift is not None:
        result["raman_shift_cm-1"] = raman_shift
    if saved_path is not None:
        result["saved_path"] = saved_path

    _st["bg_versions"][version_label] = result.copy()
    return result


def list_bg_versions() -> dict:
    """저장된 모든 배경 제거 결과 버전의 목록과 주요 통계를 반환한다."""
    _bg_versions = _sstate()["bg_versions"]
    if not _bg_versions:
        return {
            "ok": True,
            "count": 0,
            "versions": [],
            "message": "No saved versions. Call apply_background_subtraction() first.",
        }
    summaries = []
    for label, v in _bg_versions.items():
        summaries.append({
            "version_label":            label,
            "poly_order":               v.get("poly_order"),
            "max_iterations":           v.get("max_iterations"),
            "threshold":                v.get("threshold"),
            "iterations_run":           v.get("iterations_run"),
            "converged":                v.get("converged"),
            "max_corrected_intensity":  v.get("max_corrected_intensity"),
            "max_background_intensity": v.get("max_background_intensity"),
            "has_raman_shift":          "raman_shift_cm-1" in v,
            "data_length":              len(v.get("corrected_data", [])),
        })
    return {"ok": True, "count": len(summaries), "versions": summaries}


def get_bg_version(version_label: str) -> dict:
    """특정 버전의 배경 제거 결과 전체 데이터를 반환한다."""
    _bg_versions = _sstate()["bg_versions"]
    if version_label not in _bg_versions:
        return {
            "ok": False,
            "error": f"Version '{version_label}' not found.",
            "available_versions": list(_bg_versions.keys()),
        }
    return {"ok": True, **_bg_versions[version_label]}


# [제거됨 — save_spectrum 툴, 2026-07-30]
# "강도 배열을 인자로 받아 CSV 로 저장하는" 툴이었다. 세 가지 이유로 정당한 용례가 없다:
#
#  1) 에이전트는 그 배열을 애초에 갖고 있지 않다. 관측 축약기(_slim, single_agent_AILA.py)가
#     길이 32 초과 리스트를 통째로 버리므로 acquire_spectrum 의 data(1024~2048점)는 모델에
#     도달하지 않는다. 즉 이 툴을 호출하려면 배열을 지어내거나 잘라내야 한다.
#  2) 원측정 데이터는 이미 자동 저장된다(_persist_spectrum → data/results/<날짜>/<세션>/).
#  3) 가공한 배열은 run_analysis 안의 save_result 훅이 담당한다. 그 훅은 정확히 이 왕복을
#     없애려고 만들어졌다(analysis_sandbox.py 상단 주석: 1801점 스펙트럼이 컨텍스트를
#     2만 토큰씩 왕복해 생성이 잘리던 문제). 시스템 프롬프트도 이미 "print 해서
#     save_spectrum 에 넣지 말라"고 금지하고 있었다.
#
# 대체: 측정 결과 → 자동 저장 / 가공 결과 → run_analysis + save_result(...) /
#       다시 읽기 → load_spectrum(path) / 측정점 묶기 → save_measurement_point(...)


def load_spectrum(filename: str) -> dict:
    """
    저장된 스펙트럼 CSV 파일을 로드한다.
    절대 경로 또는 data/ 디렉토리 상대 경로 모두 허용.

    이 프로젝트가 쓰는 스펙트럼 CSV 는 모두 같은 포맷이다(spectrum_store.write_spectrum_csv):
        pixel_index, [raman_shift_cm-1,] [wavelength_nm,] intensity, [background_intensity]
    헤더 앞에는 '# key,value' 메타 주석행이 붙는다. 그걸 건너뛰지 않으면 첫 주석행이
    헤더로 잡혀 'intensity' 열을 못 찾는다 — 측정 결과를 다시 읽는 경로가 통째로 막힌다.
    encoding 은 utf-8-sig: 같은 파일이 BOM 을 달고 저장된다(엑셀 호환).

    측정 자동저장분(data/results/...), run_analysis 의 save_result 산출물, 배경 제거
    산출물(data/runs/...)을 모두 같은 방식으로 읽는다. 'corrected_intensity' 는 포맷
    통일 이전(2026-07-30 이전) 파일에만 남아 있는 옛 이름이라 하위호환으로만 인정한다.
    """
    try:
        if not filename.endswith(".csv"):
            filename += ".csv"
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = _DATA_DIR / filename
        if not filepath.exists():
            return {"ok": False, "error": f"File not found: {filepath}"}

        comments: dict = {}
        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            lines = f.read().splitlines()
        body = []
        for ln in lines:
            if not body and ln.lstrip().startswith("#"):
                bits = ln.lstrip().lstrip("#").strip().split(",", 1)
                if len(bits) == 2:
                    comments[bits[0].strip()] = bits[1].strip()
                continue
            body.append(ln)

        reader = csv.DictReader(body)
        headers = reader.fieldnames or []
        rows = list(reader)
        # 세기 열은 'intensity' 로 통일돼 있다. 'corrected_intensity' 는 포맷 통일
        # 이전에 배경 제거 툴이 쓰던 이름이라 옛 파일을 위해서만 남긴다.
        col = next((c for c in ("intensity", "corrected_intensity") if c in headers), None)
        if col is None:
            return {"ok": False,
                    "error": (f"No 'intensity' (or 'corrected_intensity') column in "
                              f"{filepath.name}. Columns: {headers}")}

        intensity = [float(r[col]) for r in rows]
        result: dict = {
            "ok": True,
            "filename": str(filepath),
            "num_points": len(intensity),
            "headers": headers,
            "intensity_column": col,
            "intensity": intensity,
        }
        if comments:
            result["metadata"] = comments        # laser_power_pct / exposure_time / mode 등
        if "raman_shift_cm-1" in headers:
            result["raman_shift_cm-1"] = [float(r["raman_shift_cm-1"]) for r in rows]
        if "wavelength_nm" in headers:
            result["wavelength_nm"] = [float(r["wavelength_nm"]) for r in rows]
        if "background_intensity" in headers:
            result["background_intensity"] = [float(r["background_intensity"]) for r in rows]
        return result
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────
# 측정점 기록
# ──────────────────────────────────────────
# [create_session / save_point_data 를 대체한다 — 2026-07-30]
# 옛 두 툴은 data/sessions/<id>/ 라는 '세 번째 세션 체계'를 만들었다. run_store
# (data/runs/<label>/) 와 spectrum_store (data/results/<날짜>/<label>/) 가 이미 있는데
# 서로 참조하지 않아서, create_session("exp1") 을 만들어도 acquire_spectrum 결과는
# 거기 들어가지 않고 data/results 로 갔다 — "하나의 측정점 기록"이라는 요구와 정면으로
# 어긋났다. 게다가 save_point_data 는 spectrum_data=[...] 로 강도 배열을 인자로 받았는데,
# 관측 축약(_slim)이 길이 32 초과 리스트를 버리므로 에이전트는 그 배열을 애초에 갖고
# 있지 않다(= 지어내야 호출된다). 그리고 create_session 은 스테이지 드라이버의 DLL 세션
# 메서드(TangoController.create_session)와 이름까지 겹쳤다.
#
# 새 툴은 배열을 받지 않는다. 스펙트럼과 현미경 이미지는 이미 자동 저장되어 있으므로,
# '직전 측정 + 직전 캡처 + 현재 좌표'를 하나의 레코드로 묶어 run_store 세션에 남긴다.


def save_measurement_point(point_id: str, note: str = None) -> dict:
    """이 지점의 측정 결과를 '측정점 기록' 1건으로 묶어 저장한다.

    직전 acquire_spectrum 의 저장물과 직전 capture_scene 의 이미지, 그리고 현재 스테이지
    좌표를 한 JSON 레코드로 만들어 data/runs/<세션>/points/NN_<point_id>.json 에 남기고
    세션 manifest 에 인덱싱한다. 강도 배열을 인자로 넘길 필요가 없다 — 파일은 이미
    저장되어 있고, 이 툴은 그 포인터를 모을 뿐이다.

    Parameters
    ----------
    point_id : str
        포인트 식별자 (예: 'P001'). 파일명에 쓰이므로 짧게.
    note : str, optional
        이 지점에 대한 메모(시료 부위, 관찰 내용 등).

    Returns
    -------
    dict — ok, point_id, path(data/ 기준 상대경로), position,
           spectrum(참조한 측정 파일), image(참조한 현미경 이미지 URL)
    """
    try:
        from backend.agents import run_store
    except Exception as e:
        return {"ok": False, "error": f"run_store unavailable: {type(e).__name__}: {e}"}

    pid = str(point_id or "").strip()
    if not pid:
        return {"ok": False, "error": "point_id is required (e.g. 'P001')."}

    # ── 현재 좌표 ──
    position = None
    try:
        if _stage is not None:
            p = _stage.get_position()
            position = {"x": round(float(p[0]), 4), "y": round(float(p[1]), 4),
                        "z": round(float(p[2]), 4)}
    except Exception:
        pass

    # ── 직전 측정의 저장물 ──
    spectrum = None
    _st = _sstate()
    _last_spectrum = _st["last_spectrum"]
    if _last_spectrum is not None:
        saved = _last_spectrum.get("saved") or {}
        files = saved.get("files") or {}
        spectrum = {
            "mode": _last_spectrum.get("mode"),
            "exposure_time": _last_spectrum.get("exposure_time"),
            "laser_power_pct": _last_spectrum.get("laser_power_pct"),
            "max_intensity": _last_spectrum.get("max_intensity"),
            "csv": files.get("csv"),
            "png": files.get("png"),
            "title": saved.get("title"),
        }
        if _last_spectrum.get("num_accumulations") is not None:
            spectrum["num_accumulations"] = _last_spectrum["num_accumulations"]

    # ── 직전 현미경 캡처 ──
    _last_scene = _st["last_scene"]
    image = dict(_last_scene) if _last_scene else None

    record = {
        "point_id": pid,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "position": position,
        "spectrum": spectrum,
        "image": image,
    }
    if note:
        record["note"] = str(note)

    # 무엇이 비었는지 명시해 준다 — "묶었다"고만 하고 실제로는 빈 레코드가 남는 것을 막는다.
    missing = [k for k in ("position", "spectrum", "image") if record.get(k) is None]

    try:
        filepath, rel = run_store.new_point_path(pid)
        filepath.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        run_store.record(run_store.KIND_POINT, rel, point_id=pid,
                         has_spectrum=spectrum is not None, has_image=image is not None)
    except Exception as e:
        return {"ok": False, "error": f"Failed to write the point record: {type(e).__name__}: {e}"}

    out = {"ok": True, "point_id": pid, "path": rel, "position": position,
           "spectrum": spectrum, "image": image}
    if missing:
        out["missing"] = missing
        out["note_to_caller"] = (
            "This point record is missing: " + ", ".join(missing) + ". "
            "Acquire a spectrum (acquire_spectrum) and/or capture the view (capture_scene) at this "
            "position BEFORE calling save_measurement_point, so the record can reference them.")
    return out


@_serialized("capture_scene")
def capture_scene() -> dict:
    """현재 현미경(카메라) 화면을 저장한다 — run_analysis 가 이 위에 피크맵을 오버레이한다.

    스테이지 위치와 보정된 시야(FOV)로 이미지의 스테이지 좌표 범위(extent, mm)를 계산해
    함께 저장하므로, 분석 코드에서 imshow(microscope_image, extent=image_extent) 후
    측정 (x,y)를 그 위에 정합해 찍을 수 있다. 카메라 스트리밍이 켜져 있어야 한다.

    [2026-07-30 수정 — 두 가지가 틀려 있었다]
    1) extent 를 LENS_*_UM/1000 으로 계산해 보정계수(CALIB_FACTOR)가 빠져 있었다.
       preview_grid_scan 이 쓰는 시야(0.427×0.296mm)와 달리 0.305×0.230mm 로 나와,
       같은 화면의 크기를 1.4배 다르게 보고했다. 이 extent 위에 측정 좌표를 찍으면
       그만큼 어긋난다(에러 없이 그림만 틀린다). 이제 optics_map.scene_extent 단일 출처.
    2) 프레임을 센서 네이티브 해상도 그대로 저장해서, 이 이미지에서 읽은 픽셀 좌표는
       move_to_pixel 에 넣을 수 없었다(analyze_microscope_image 와 다른 좌표계).
       이제 vision.to_view_bgr 로 뷰 해상도에 맞춘다 — 세 캡처 도구의 좌표계가 같다.
    """
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    import cv2
    frame = _camera.get_latest_frame()
    if frame is None:
        return {"ok": False, "error": "No camera frame. Start streaming first."}
    # 뷰 해상도로 정규화(도구 좌표계) 후 matplotlib 표시 기준인 RGB 로.
    img = cv2.cvtColor(_vis.to_view_bgr(frame), cv2.COLOR_BGR2RGB)

    extent = None
    try:
        if _stage is not None:
            pos = _stage.get_position()
            extent = _om.scene_extent(float(pos[0]), float(pos[1]))
    except Exception:
        pass

    saved = _store_save_scene(img, extent, {})
    if not saved.get("ok"):
        return saved

    # 마지막 캡처를 이 세션의 상태에 기억해 둔다 — save_measurement_point 가 '이 지점에서
    # 찍은 이미지'로 참조한다. 에이전트가 이미지 배열을 인자로 넘길 필요가 없게 하기 위한 것.
    _last_scene = {
        "image_url": saved["image_url"],
        "scene_npz": saved.get("scene_npz"),
        "extent": extent,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        if _stage is not None:
            _p = _stage.get_position()
            _last_scene["position"] = {"x": round(float(_p[0]), 4), "y": round(float(_p[1]), 4),
                                       "z": round(float(_p[2]), 4)}
    except Exception:
        pass
    _sstate()["last_scene"] = _last_scene

    return {"ok": True, "image_url": saved["image_url"], "extent": extent,
            "shape": list(img.shape),
            # saved 를 실으면 spectrum_event 배선을 타 캡처한 화면이 채팅에 바로 표시된다.
            "saved": {"title": "Microscope view capture", "image_url": saved["image_url"]},
            "note": "Usable later in run_analysis via microscope_image / image_extent, "
                    "and referenced automatically by save_measurement_point."}


@_serialized("analyze_microscope_image")
def analyze_microscope_image(question: str = "Find a specific object in the sample (e.g. a cell) and report its center-point pixel coordinates.") -> dict:
    """
    TuCam 현미경 카메라 화면을 PNG (Base64)로 캡처하여 반환.

    [왜 CAMERA_WIDTH×CAMERA_HEIGHT로 리사이즈하는가 — 두 가지 이유]

    1) 좌표계 일치 (기능 버그 수정)
       get_latest_frame()은 센서 네이티브 해상도를 그대로 준다(Config.ini의
       Width/Height는 뷰 기준 해상도일 뿐 프레임 크기가 아니다). 그런데 이 이미지를
       보고 vision LLM이 찍은 픽셀 좌표는 결국 move_to_pixel()로 들어가고,
       move_to_pixel은 이미지 중심을 (CAMERA_WIDTH/2, CAMERA_HEIGHT/2)로 가정해
       계산한다. 두 해상도가 다르면 스테이지가 엉뚱한 곳으로 이동한다.
       USE_scan.py도 같은 이유로 픽셀→스테이지 계산 전에 이 크기로 리사이즈한다.
       → 여기서 미리 맞춰 두면 이 함수의 출력 좌표계 = move_to_pixel의 입력 좌표계.

    2) API 이미지 제한
       Anthropic API는 이미지 1장당 base64 10MB, 긴 변 2576px가 상한이고 그보다 큰
       이미지는 어차피 서버에서 다운스케일된다. 네이티브 프레임을 무손실 PNG로 보내면
       12MB를 넘겨 요청 자체가 거부됐다(실제로 그랬다). 1060×800이면 ~2MB, 긴 변도
       상한 이하라 다운스케일이 없어 좌표가 보낸 그대로 유지된다.
    """
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    try:
        import base64
        import cv2
        frame = _camera.get_latest_frame()
        if frame is None:
            return {"ok": False, "error": "Failed to acquire frame (check whether streaming is active)"}

        # 뷰 기준 해상도로 정규화 — 위 docstring의 (1)(2). capture_scene /
        # preview_grid_scan 과 같은 vision.to_view_bgr 를 쓰므로 좌표계가 동일하다.
        frame_bgr = _vis.to_view_bgr(frame)
        height, width = frame_bgr.shape[:2]
        ret, buf = cv2.imencode('.png', frame_bgr)
        enhanced_question = f"{question}\n\n[The attached image has an original resolution of {width}px wide by {height}px tall. When returning pixel coordinates, give exact pixel values based on this resolution.][Note: you return pixel coordinates, which are NOT stage coordinates. To move the stage to that location, you must use the move_to_pixel(pixel_x, pixel_y) function.]"

        if not ret:
            return {"ok": False, "error": "PNG encoding failed"}
        img_b64 = base64.b64encode(buf).decode('utf-8')

        # 밝기 통계와 선명도 — 예전 capture_camera_frame 이 주던 값들을 여기로 합쳤다.
        # 리사이즈·8bit 정규화를 거친 '위에서 실제로 보낸 그 이미지'에서 계산하므로,
        # 모델이 보는 화면과 숫자가 일치한다(옛 툴은 uint16 원본을 그대로 재서 어긋났다).
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return {
            "ok": True,
            "image_base64": img_b64,
            "question": enhanced_question,
            "width": width,
            "height": height,
            "min_intensity": float(gray.min()),
            "max_intensity": float(gray.max()),
            "mean_intensity": float(gray.mean()),
            # 상대 비교용 지표다. run_autofocus 는 '가이드빔 스팟 면적'을 쓰므로 이 값과
            # 직접 비교하거나 이걸로 초점을 잡으려 하면 안 된다(다른 Z 에 수렴한다).
            "sharpness_score": _vis.sharpness_score(gray),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("move_to_pixel")
def move_to_pixel(pixel_x: int, pixel_y: int) -> dict:
    """
    카메라 이미지의 픽셀 좌표를 스테이지 mm 좌표로 변환해 이동한다.
    이미지 중심(CAMERA_WIDTH/2, CAMERA_HEIGHT/2)이 현재 스테이지 위치에 대응한다.

    입력 좌표계는 analyze_microscope_image / capture_scene / preview_grid_scan 이
    돌려주는 이미지와 동일하다(셋 다 vision.to_view_bgr 로 같은 해상도에 맞춘다).
    변환식은 optics_map 단일 출처 — 예전에는 이 함수, USE_scan.py, server.py 가
    각자 같은 식을 갖고 있었고 server.py 는 보정계수를 하드코딩까지 했다.
    """
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    try:
        pos = _stage.get_position()
        if pos is None:
            return {"ok": False, "error": "Failed to query stage position"}
        tx, ty = _om.pixel_to_stage(pixel_x, pixel_y, float(pos[0]), float(pos[1]))
        return move_stage(x=round(tx, 4), y=round(ty, 4))
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────
# 그리드 매핑 — 5×5 같은 격자 스캔을 도구 1~2개로 압축
# ──────────────────────────────────────────
# [배경] 예전엔 에이전트가 25점 격자를 move→autofocus→acquire "직접" 75회로 돌렸다.
# 매 스텝 커지는 히스토리를 매번 통째로 재전송해 토큰이 사실상 제곱으로 불고(1턴 ~8.5k
# 토큰), 스텝마다 LLM 추론이 끼어 633초가 걸렸다. 아래 두 도구는 그 루프를 파이썬으로
# 내려 토큰·지연·재전송을 없앤다:
#   preview_grid_scan : 격자 위치를 '현재 카메라 화면'에 원으로 오버레이해 에이전트에게
#                       보여주기만 한다(이동·조사 없음). 에이전트가 눈으로 보고 승인/수정.
#   run_grid_scan     : 승인 후 실제 격자 스캔을 내부 루프로 수행, 압축 요약 1개만 반환.

# 오버레이 원 크기 — 프론트 카메라 뷰의 '레이저 조사점' 빨간 링(CameraView.tsx: w-4 ≈ 16px)과
# 대략 같은 크기로 그린다. 물리적 빔 지름이 아니라 조사점 표식이다(사용자 지시: 대충 그 크기).
_GRID_SPOT_RADIUS_PX = 14

# run_grid_scan 한 번이 낼 수 있는 누적 조사량 상한(mJ)은 _GRID_MAX_DOSE_MJ 로 파일
# 상단에서 import 한다(backend.safety_limits). 에이전트 층의 per-turn 회로차단기는 도구
# 이름이 acquire_spectrum일 때만 도는데 run_grid_scan은 내부에서 N번 조사하므로, 여기서
# 독립적으로 한 번 더 막는다(방어적 이중화). 두 계층이 같은 상수·같은 공식을 쓰도록
# safety_limits 로 모았다 — 예전에는 1000.0 이 세 파일에 각각 박혀 있었다.

# 격자 점 개수 안전 상한(폭주 방지).
_GRID_MAX_POINTS = 400


# ── 그리드 스캔 사람-승인 게이트 (human-in-the-loop 하드 인터록) ──────────────────
# [왜 코드 게이트인가] 스키마/시스템 프롬프트로 "미리보기 후 멈추고 승인받아라"라고
# 지시해도 보장이 안 된다: ReAct(AILA)는 한 응답의 tool_call을 전부 실행하고, CoALA도
# preview와 run이 서로 다른 사이클이면 같은 턴 안에서 연속 실행될 수 있다. 레이저는
# 비가역이라, "실행 자체를 거부"하는 물리적 인터록이 프롬프트와 별개로 필요하다.
#
# [상태기계] state ∈ {"none","pending","armed"} + 승인된 격자 형상(geom).
#   preview_grid_scan 성공         → geom 저장, state="pending" (사람이 아직 못 봄)
#   grid_gate_begin_turn(대화)      → "pending"이면 "armed"(이번 턴 사람이 승인 가능),
#                                     "armed"인데 안 쓰였으면 만료 → "none"(재미리보기 필요)
#   run_grid_scan (enforce 시)      → "armed" + geom 일치일 때만 통과, 발사 직전 소비 → "none"
# 즉 미리보기와 실행 사이에 '사람 턴 경계'가 반드시 하나 끼도록 강제한다. 승인 창은 딱
# 한 턴(one-shot)이라, 취소하거나 무관한 요청을 한 뒤 stale 승인으로 실행되는 일이 없다.
#
# 벤치마크(run_experiment)는 사람이 없는 자율 평가이므로 grid_gate_begin_turn(interactive=
# False)로 게이트를 끈다(안 그러면 모든 벤치마크 격자 스캔이 승인 없이 거부된다).
#
# [왜 모듈 전역이 아니라 세션별인가 — 2026-07-31]
# 예전에는 "단일 장비·단일 사용자라 전역 dict로 충분하다"고 두었는데, 그러면 벤치마크
# 실행이 enforce를 끄는 순간 **동시에 열려 있던 대화 세션의 인터록까지 꺼졌다**.
# 즉 사람 승인 없이 레이저 격자 스캔이 나갈 수 있는 창이 열린다(테스트로 재현). 이제
# 세션 라벨로 키를 잡아, 한 세션이 게이트를 끄더라도 다른 세션은 영향받지 않는다.


def _gate() -> dict:
    """이 세션의 그리드 승인 게이트 상태."""
    return _sstate()["grid_gate"]


def _grid_gate_geom(rows, cols, spacing_mm, center_x, center_y) -> dict:
    """게이트 비교용 격자 형상 정규화 — 미리보기와 실행이 '같은 격자'인지 판정하는 기준.
    형상(rows/cols/spacing/center)만 본다. power/exposure/autofocus는 dose 회로차단기가
    따로 막으므로 승인 대상이 아니다(사람이 눈으로 승인한 것은 '어디에 몇 점을'이다).
    center는 모델이 넘긴 원값(None 포함)으로 비교한다 — 실측 위치로 해석하면 미리보기와
    실행 사이 스테이지 이동으로 값이 달라져 오탐이 난다."""
    return {
        "rows": int(rows), "cols": int(cols),
        "spacing_mm": round(float(spacing_mm), 4),
        "center_x": None if center_x is None else round(float(center_x), 4),
        "center_y": None if center_y is None else round(float(center_y), 4),
    }


def grid_gate_begin_turn(interactive: bool = True) -> None:
    """에이전트가 새 사용자 턴을 시작할 때 호출하는 훅(AILA/CoALA 공용).

    interactive=True (SSE 대화): 사람 승인 게이트 ON. 직전 턴에 만든 미리보기가 있으면
      '이제 사람이 보고 승인할 턴'이라는 뜻으로 pending→armed. 이미 armed였는데 실행에
      쓰이지 않았으면 승인 창이 만료된 것이라 none으로 되돌린다(재미리보기 강제).
    interactive=False (벤치마크 자율 실행): 게이트 OFF + 상태 초기화 — 사람이 없으므로
      미리보기 없이도 run_grid_scan을 허용한다."""
    _gate()["enforce"] = bool(interactive)
    if not interactive:
        _gate()["geom"] = None
        _gate()["state"] = "none"
        return
    if _gate()["state"] == "pending":
        _gate()["state"] = "armed"
    elif _gate()["state"] == "armed":
        # 지난 턴에 승인 가능(armed)했으나 실행하지 않았다 → 승인 창 만료.
        _gate()["state"] = "none"
        _gate()["geom"] = None


def _grid_gate_on_preview(geom: dict) -> None:
    """preview_grid_scan 성공 시 호출 — 승인 대기(pending) 상태로 만든다."""
    _gate()["geom"] = geom
    _gate()["state"] = "pending"


def _grid_gate_check(geom: dict):
    """run_grid_scan 진입 시 승인 여부 검사(부작용 없음). 통과면 None, 거부면 에러 dict를
    반환한다 — 에이전트는 이 관측을 읽고 '미리보기→턴 종료→대기'로 돌아가게 된다.
    실제 소비(승인 1회 사용)는 발사 직전 _grid_gate_consume()에서 한다 — 사전검증
    (범위/None) 실패로 승인이 헛되이 소모되지 않도록 검사와 소비를 분리한다."""
    if not _gate()["enforce"]:
        return None
    state = _gate()["state"]
    if state == "none" or _gate()["geom"] is None:
        return {"ok": False, "error": (
            "Human approval required: no approved grid preview is on record. Call preview_grid_scan "
            "first, show the user the preview, end your turn, and run only after the user approves.")}
    if state == "pending":
        return {"ok": False, "error": (
            "Human approval required: the grid preview has NOT been approved yet. Do not run in the "
            "same turn as the preview - end your turn now, let the user see the preview, and call "
            "run_grid_scan only after the user explicitly approves it in a new message.")}
    if geom != _gate()["geom"]:
        return {"ok": False, "error": (
            f"Approval mismatch: the user approved grid {_gate()['geom']} but this run requests "
            f"{geom}. Preview the exact grid again and get the user's approval before running.")}
    return None


def _grid_gate_consume() -> None:
    """승인을 1회 소비한다 — 발사 직전(모든 사전검증 통과 후) 호출. 게이트가 꺼져 있으면
    (벤치마크) 아무 것도 하지 않는다."""
    if not _gate()["enforce"]:
        return
    _gate()["geom"] = None
    _gate()["state"] = "none"


def _grid_stage_coords(center_x: float, center_y: float,
                       rows: int, cols: int, spacing_mm: float) -> list:
    """(center_x,center_y) 중심 대칭 rows×cols 격자의 스테이지 좌표를 행 우선(raster)으로.
    반환 각 항목: (index, row, col, x_mm, y_mm)."""
    pts = []
    idx = 0
    for i in range(rows):
        dy = (i - (rows - 1) / 2.0) * spacing_mm
        for j in range(cols):
            dx = (j - (cols - 1) / 2.0) * spacing_mm
            pts.append((idx, i, j, round(center_x + dx, 4), round(center_y + dy, 4)))
            idx += 1
    return pts


def _validate_grid_args(rows, cols, spacing_mm):
    """preview/run 공통 인자 검증 — 실패 시 error dict, 통과 시 None."""
    if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
        return {"ok": False, "error": "rows and cols must be integers >= 1."}
    if rows * cols > _GRID_MAX_POINTS:
        return {"ok": False, "error": f"Too many points ({rows*cols}); max {_GRID_MAX_POINTS}."}
    try:
        if float(spacing_mm) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return {"ok": False, "error": "spacing_mm must be a number > 0."}
    return None


@_serialized("preview_grid_scan")
def preview_grid_scan(rows: int, cols: int, spacing_mm: float,
                      center_x: float = None, center_y: float = None) -> dict:
    """격자 스캔 '미리보기'. 스테이지 이동·레이저 조사 없이, rows×cols 격자 위치를 현재
    카메라 화면에 원으로 오버레이한 이미지를 반환한다(에이전트가 보고 승인/수정하도록).

    center_* 미지정 시 현재 스테이지 위치를 격자 중심으로 쓴다. 화면 중심 = 현재 스테이지
    위치이므로, 중심을 현재 위치로 두면 격자가 화면 중앙에 대칭으로 그려진다. 카메라 시야(FOV)는
    좁아(≈0.43×0.30mm) 격자가 넓으면 일부 점은 화면 밖이다 — 화면 안 점만 세어 함께 알려준다.
    """
    err = _validate_grid_args(rows, cols, spacing_mm)
    if err:
        return err
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    if _camera is None:
        return {"ok": False, "error": "Camera is not initialized."}
    try:
        import base64
        import cv2

        spacing_mm = float(spacing_mm)
        pos = _stage.get_position()
        if pos is None:
            return {"ok": False, "error": "Failed to query stage position"}
        cur_x, cur_y = float(pos[0]), float(pos[1])          # 화면 중심 = 현재 위치
        cx = cur_x if center_x is None else float(center_x)  # 격자 중심
        cy = cur_y if center_y is None else float(center_y)

        frame = _camera.get_latest_frame()
        if frame is None:
            return {"ok": False, "error": "Failed to acquire frame (check whether streaming is active)"}
        frame_bgr = _vis.to_view_bgr(frame)     # 좌표계 정규화(다른 캡처 도구와 동일)

        pts = _grid_stage_coords(cx, cy, rows, cols, spacing_mm)
        n_in_view = 0
        for idx, i, j, sx, sy in pts:
            # optics_map.stage_to_pixel = move_to_pixel 의 정확한 역변환(단일 출처)
            px, py = _om.stage_to_pixel(sx, sy, cur_x, cur_y)
            ipx, ipy = int(round(px)), int(round(py))
            if 0 <= ipx < CAMERA_WIDTH and 0 <= ipy < CAMERA_HEIGHT:
                n_in_view += 1
            # 화면 밖 점도 cv2.circle이 자동 클립하므로 그냥 그린다(가장자리 힌트).
            cv2.circle(frame_bgr, (ipx, ipy), _GRID_SPOT_RADIUS_PX, (0, 255, 0), 2)      # green: 스캔 점
        # (요청) 미리보기에는 스캔 예정점(초록 원)만 표시한다. 화면 중심의 '현재 조사점'
        # 빨간 링은 스캔 계획과 무관해 혼동을 줄 수 있어 그리지 않는다.

        ret, buf = cv2.imencode('.png', frame_bgr)
        if not ret:
            return {"ok": False, "error": "PNG encoding failed"}
        img_b64 = base64.b64encode(buf).decode('utf-8')
        # 같은 오버레이를 파일로도 저장해 image_url을 실으면, spectrum_event 배선을 타고
        # 이 미리보기가 채팅창에 그대로 인라인 표시된다(프론트 수정 불필요). 에이전트는 위의
        # image_base64로 '보고' 판단하고, 사람은 채팅에서 같은 그림을 본다.
        saved_img = _store_save_preview(buf.tobytes(), tag="grid_preview")

        fov_x, fov_y = _om.fov_mm()             # capture_scene 의 extent 와 같은 출처
        span_x, span_y = (cols - 1) * spacing_mm, (rows - 1) * spacing_mm
        question = (
            f"Grid scan PREVIEW (not executed yet): {rows} rows x {cols} cols = {rows*cols} points "
            f"-> a {cols}-wide x {rows}-tall grid, {spacing_mm} mm spacing, "
            f"centered at stage (X={cx:.4f}, Y={cy:.4f}) mm. "
            f"Green circles mark the points to be scanned. "
            f"Grid span {span_x:.3f}x{span_y:.3f} mm vs camera view ~{fov_x:.3f}x{fov_y:.3f} mm; "
            f"{n_in_view}/{rows*cols} points fall within the current view (the rest are outside the frame "
            f"but will still be measured). "
            f"STOP HERE - do NOT run the scan yet. Show this preview to the user, confirm the "
            f"{cols}-wide x {rows}-tall orientation is what they asked for, then END YOUR TURN and WAIT "
            f"for their explicit approval. Only in a LATER turn, after the user approves, call run_grid_scan "
            f"with these SAME parameters. If the layout is wrong, call preview_grid_scan again with adjusted "
            f"rows/cols/spacing_mm/center."
        )
        out = {
            "ok": True,
            "rows": rows, "cols": cols, "spacing_mm": spacing_mm,
            "center": {"x": round(cx, 4), "y": round(cy, 4)},
            "n_points": rows * cols, "n_in_view": n_in_view,
            "fov_mm": {"x": round(fov_x, 4), "y": round(fov_y, 4)},
            "image_base64": img_b64,
            "question": question,
        }
        if saved_img.get("ok"):
            out["saved"] = {"title": f"Grid preview {rows}x{cols} @ {spacing_mm}mm",
                            "image_url": saved_img["image_url"]}
        # 사람-승인 게이트: 이 미리보기를 '승인 대기(pending)'로 등록한다. 이후 사용자
        # 턴 경계에서 armed로 올라가야 run_grid_scan이 통과한다(같은 턴 즉시 실행 차단).
        _grid_gate_on_preview(_grid_gate_geom(rows, cols, spacing_mm, center_x, center_y))
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_serialized("run_grid_scan")
def run_grid_scan(rows: int, cols: int, spacing_mm: float,
                  center_x: float = None, center_y: float = None,
                  autofocus: str = "each", exposure: float = 0.2, power: float = 40) -> dict:
    """격자 스캔 '실행'. rows×cols 격자를 내부 루프로 순회하며 각 점에서
    이동→(오토포커스)→스펙트럼 측정·자동저장하고, 압축 요약 1개만 반환한다.

    autofocus:
      "each"   — 매 점에서 오토포커스(가장 정확, 느림; 예전 수동 방식과 동일)
      "center" — 격자 중심에서 1회만 오토포커스 후 그 Z로 전체 측정(빠름, 평탄 시료용)
      "none"   — 오토포커스 없이 현재 Z로 측정
    레이저 조사가 실제로 일어나므로, 예상 누적 조사량이 상한을 넘으면 시작 전에 거부한다.
    """
    err = _validate_grid_args(rows, cols, spacing_mm)
    if err:
        return err
    if autofocus not in ("each", "center", "none"):
        return {"ok": False, "error": "autofocus must be one of: each, center, none."}
    # 사람-승인 게이트(하드 인터록) — 하드웨어를 만지기 전에 먼저 막는다. 승인된 미리보기
    # 없이 레이저 격자 스캔을 실행하지 않는다. 실제 소비는 발사 직전(_grid_gate_consume).
    gate_err = _grid_gate_check(_grid_gate_geom(rows, cols, spacing_mm, center_x, center_y))
    if gate_err:
        return gate_err
    if _stage is None:
        return {"ok": False, "error": "Stage is not initialized."}
    if _ccd is None:
        return {"ok": False, "error": "CCD is not initialized (cooling or not connected)."}
    if _laser is None:
        return {"ok": False, "error": "Laser is not initialized."}
    if autofocus != "none" and _camera is None:
        return {"ok": False, "error": "Camera is not initialized (required for autofocus). "
                                      "Use autofocus='none' to skip."}

    spacing_mm = float(spacing_mm)
    exposure, power = float(exposure), float(power)
    n = rows * cols
    # 조사량 공식은 safety_limits 단일 출처 — 에이전트 계층의 턴 누계와 같은 척도여야
    # 두 한계값이 같은 의미를 갖는다.
    dose_total = estimate_dose_mj(power, exposure, n)
    if dose_total > _GRID_MAX_DOSE_MJ:
        return {"ok": False, "error": (
            f"Safety block: estimated cumulative dose {dose_total:.1f} mJ exceeds the grid limit "
            f"({_GRID_MAX_DOSE_MJ} mJ). Reduce point count, power, or exposure.")}

    try:
        pos = _stage.get_position()
        if pos is None:
            return {"ok": False, "error": "Failed to query stage position"}
        cx = float(pos[0]) if center_x is None else float(center_x)
        cy = float(pos[1]) if center_y is None else float(center_y)

        pts = _grid_stage_coords(cx, cy, rows, cols, spacing_mm)
        # 범위 사전 검증 — 한 점이라도 벗어나면 시작조차 하지 않는다.
        for idx, i, j, sx, sy in pts:
            if not (0 <= sx <= STAGE_MAX_X and 0 <= sy <= STAGE_MAX_Y):
                return {"ok": False, "error": (
                    f"Point {idx} (row {i}, col {j}) at X={sx}, Y={sy} mm is outside the stage range "
                    f"(0..{STAGE_MAX_X} x 0..{STAGE_MAX_Y}). Adjust center/spacing/size.")}

        # center 모드: 격자 중심으로 이동 후 1회 오토포커스(이후 Z 유지).
        if autofocus == "center":
            mv = move_stage(x=cx, y=cy)
            if not mv.get("ok"):
                return {"ok": False, "error": f"Failed to move to grid center: {mv.get('error')}"}
            af = run_autofocus()
            if not af.get("ok"):
                return {"ok": False, "error": f"Autofocus at grid center failed: {af.get('error')}"}

        # 모든 사전검증(범위/None/오토포커스) 통과 — 이제 레이저를 쏜다. 승인을 여기서
        # 소비한다(1회용). 이 지점 이후 재실행하려면 다시 미리보기·승인이 필요하다.
        _grid_gate_consume()

        results = []
        n_ok = 0
        for idx, i, j, sx, sy in pts:
            mv = move_stage(x=sx, y=sy)
            if not mv.get("ok"):
                results.append({"i": idx, "row": i, "col": j, "x": sx, "y": sy,
                                "ok": False, "error": mv.get("error")})
                continue
            if autofocus == "each":
                run_autofocus()   # 실패해도 치명적이지 않다 — 현재 Z로 측정을 이어간다.
            res = _cache_and_return(acquire_spectrum(exposure=exposure, power=power))
            if res.get("ok"):
                n_ok += 1
                files = (res.get("saved") or {}).get("files") or {}
                ref = files.get("csv") or files.get("png") or ""
                fname = ref.replace("\\", "/").rsplit("/", 1)[-1]
                results.append({"i": idx, "x": sx, "y": sy,
                                "max_intensity": res.get("max_intensity"), "file": fname})
            else:
                results.append({"i": idx, "row": i, "col": j, "x": sx, "y": sy,
                                "ok": False, "error": res.get("error")})

        # 압축 반환 — 큰 격자에서 per-point 리스트가 에이전트 _slim(길이>32 리스트 폐기)에
        # 통째로 걸리지 않도록, 집계 통계는 항상 싣고 per-point는 32점 이하일 때만 인라인.
        oks = [r for r in results if "max_intensity" in r]
        fails = [r for r in results if r.get("ok") is False]
        inten = [r["max_intensity"] for r in oks if r.get("max_intensity") is not None]
        out = {
            "ok": True,
            "rows": rows, "cols": cols, "spacing_mm": spacing_mm,
            "center": {"x": round(cx, 4), "y": round(cy, 4)},
            "autofocus": autofocus, "exposure": exposure, "power": power,
            "n_points": n, "n_measured": n_ok, "n_failed": len(fails),
            "estimated_dose_mj": round(dose_total, 2),
            "intensity": ({"min": min(inten), "max": max(inten),
                           "mean": round(sum(inten) / len(inten), 1)} if inten else None),
            "note": ("Each point was auto-saved with its (x,y) tag. Use aggregate_spectra_csv / "
                     "combine_spectra / bundle_results / run_analysis to merge or inspect per-point data."),
        }
        if fails:
            out["failed_points"] = fails[:10]
        if n <= 32:
            out["points"] = oks
        return out
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ─────────────────────────────────────────
# tool dispatch 테이블 (agent loop에서 사용)
# ──────────────────────────────────────────

TOOL_DISPATCH = {
    # ── 스테이지 ─────────────────────────────────────────────────────────────
    "move_stage":               lambda a: move_stage(**a),
    "get_stage_position":       lambda a: get_stage_position(),
    "move_stage_relative":      lambda a: move_stage_relative(**a),
    "get_stage_speed":          lambda a: get_stage_speed(),
    "set_stage_speed":          lambda a: set_stage_speed(**a),
    # ── 하드웨어 연결 관리 ────────────────────────────────────────────────────
    "reconnect_hardware":       lambda a: reconnect_hardware(**a),
    "get_hardware_status":      lambda a: get_hardware_status(),
    # ── 레이저 ──────────────────────────────────────────────────────────────
    "laser_on":                 lambda a: laser_on(),
    "laser_off":                lambda a: laser_off(),
    "set_laser_power":          lambda a: set_laser_power(**a),
    "get_laser_status":         lambda a: get_laser_status(),
    "set_guide_beam_mode":      lambda a: set_guide_beam_mode(),
    # ── 스펙트럼 수집 ────────────────────────────────────────────────────────
    "acquire_spectrum":         lambda a: _cache_and_return(acquire_spectrum(**a)),
    # ── 카메라 ──────────────────────────────────────────────────────────────
    "start_camera_stream":      lambda a: start_camera_stream(),
    "stop_camera_stream":       lambda a: stop_camera_stream(),
    "set_camera_exposure":      lambda a: set_camera_exposure(**a),
    "set_camera_auto_exposure": lambda a: set_camera_auto_exposure(**a),
    "analyze_microscope_image": lambda a: analyze_microscope_image(**a),
    "move_to_pixel":            lambda a: move_to_pixel(**a),
    "capture_scene":            lambda a: capture_scene(),
    # ── 오토포커스 ───────────────────────────────────────────────────────────
    "run_autofocus":            lambda a: run_autofocus(**a),
    # ── 그리드 매핑(미리보기 + 실행) ─────────────────────────────────────────
    "preview_grid_scan":        lambda a: preview_grid_scan(**a),
    "run_grid_scan":            lambda a: run_grid_scan(**a),
    # ── CCD 설정 ─────────────────────────────────────────────────────────────
    "get_ccd_info":             lambda a: get_ccd_info(),
    "set_ccd_exposure":         lambda a: set_ccd_exposure(**a),
    "set_ccd_acquisition_mode": lambda a: set_ccd_acquisition_mode(**a),
    "set_ccd_trigger_mode":     lambda a: set_ccd_trigger_mode(**a),
    "set_ccd_read_mode":        lambda a: set_ccd_read_mode(**a),
    "set_ccd_preamp_gain":      lambda a: set_ccd_preamp_gain(**a),
    "set_ccd_em_gain":          lambda a: set_ccd_em_gain(**a),
    "set_ccd_output_amp":       lambda a: set_ccd_output_amp(**a),
    "set_ccd_shift_speeds":     lambda a: set_ccd_shift_speeds(**a),
    "set_ccd_temperature":      lambda a: set_ccd_temperature(**a),
    "set_ccd_cooler":           lambda a: set_ccd_cooler(**a),
    "set_ccd_shutter":          lambda a: set_ccd_shutter(**a),
    "set_ccd_image_flip":       lambda a: set_ccd_image_flip(**a),
    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    # (save_spectrum 은 제거했다 — 위 주석 참고. 저장은 자동저장/save_result/
    #  save_measurement_point 가 담당한다.)
    "load_spectrum":            lambda a: load_spectrum(**a),
    # ── 측정 결과 정리(자동 저장분 대상) ─────────────────────────────────────
    "list_results":             lambda a: {"ok": True, "items": [
                                    {k: it[k] for k in
                                     ("base", "session", "date", "title", "timestamp", "meta")}
                                    for it in _store_list_results(**a)]},
    "combine_spectra":          lambda a: _store_combine_spectra(**a),
    "aggregate_spectra_csv":    lambda a: _store_aggregate_csv(**a),
    "bundle_results":           lambda a: _store_bundle_results(**a),
    # ── 분석 전용 코드 샌드박스(하드웨어 미접근) ─────────────────────────────
    "run_analysis":             lambda a: _run_analysis(**a),
    # ── 외부 웹 검색(내부 지식에 없을 때) ────────────────────────────────────
    "web_search":               lambda a: _web_search(**a),
    # ── 측정점 기록 ──────────────────────────────────────────────────────────
    "save_measurement_point":   lambda a: save_measurement_point(**a),
    # ── 배경 제거 (IPBSA) ────────────────────────────────────────────────────
    "apply_background_subtraction": lambda a: apply_background_subtraction(**a),
    "list_bg_versions":             lambda a: list_bg_versions(),
    "get_bg_version":               lambda a: get_bg_version(**a),
}