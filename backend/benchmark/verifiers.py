"""
벤치마크 검증 로직.
각 verifier는 (verifier_spec, tool_trace, pre_state, post_state) → VerifyResult 반환.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class VerifyResult:
    passed: bool
    verifier_type: str
    detail: str
    is_human_only: bool = False


def run_verifiers(
    verifiers: list[dict],
    tool_trace: list[dict],
    pre_state: dict,
    post_state: dict,
) -> list[VerifyResult]:
    results = []
    for v in verifiers:
        results.append(_dispatch(v, tool_trace, pre_state, post_state))
    return results


def _dispatch(v: dict, tool_trace, pre_state, post_state) -> VerifyResult:
    vtype = v["type"]
    try:
        if vtype == "tool_called":
            return _tool_called(v, tool_trace)
        if vtype == "tool_called_any":
            return _tool_called_any(v, tool_trace)
        if vtype == "tool_result_ok":
            return _tool_result_ok(v, tool_trace)
        if vtype == "tool_args":
            return _tool_args(v, tool_trace)
        if vtype == "tool_arg_any":
            return _tool_arg_any(v, tool_trace)
        if vtype == "tool_sequence":
            return _tool_sequence(v, tool_trace)
        if vtype == "tool_call_count":
            return _tool_call_count(v, tool_trace)
        if vtype == "tool_args_sequence":
            return _tool_args_sequence(v, tool_trace)
        if vtype == "stage_position":
            return _stage_position(v, post_state)
        if vtype == "stage_velocity":
            return _stage_velocity(v, post_state)
        if vtype == "ccd_exposure":
            return _ccd_exposure(v, post_state)
        if vtype == "ccd_read_mode":
            return _ccd_read_mode(v, post_state)
        if vtype == "ccd_temperature":
            return _ccd_temperature(v, post_state)
        if vtype == "ccd_temperature_reported":
            return _ccd_temperature_reported(tool_trace)
        if vtype == "laser_state":
            return _laser_state(v, post_state)
        if vtype == "spectrum_valid":
            return _spectrum_valid(v, tool_trace)
        if vtype == "human_only":
            return VerifyResult(
                passed=True,
                verifier_type="human_only",
                detail=f"[인간 채점 필요] {v.get('note', '')}",
                is_human_only=True,
            )
        return VerifyResult(passed=False, verifier_type=vtype, detail=f"알 수 없는 verifier 타입: {vtype}")
    except Exception as e:
        return VerifyResult(passed=False, verifier_type=vtype, detail=f"검증 중 오류: {e}")


# ── tool_trace 헬퍼 ───────────────────────────────────────────────

def _calls_of(tool_trace: list[dict], tool_name: str) -> list[dict]:
    return [t for t in tool_trace if t["tool"] == tool_name]


# ── 열거형 문자열 정규화 ──────────────────────────────────────────
# 같은 뜻인데 표기만 다른 값(예: 툴 스키마 enum 'fvb'  ↔  하드웨어 ro_mode
# 'FULL_VERTICAL_BINNING')을 채점에서 동일하게 취급한다. 표기차이로 정답이
# 오답 처리되던 문제를 막는다. 대소문자/공백/언더스코어/하이픈은 무시.
_ENUM_ALIASES = {
    # CCD 읽기 모드: set_ccd_read_mode enum  ↔  andor 인터페이스 ro_mode
    "fvb": "fvb",
    "full_vertical_binning": "fvb",
    "single_track": "single_track",
    "multi_track": "multi_track",
    "random_track": "random_track",
    "image": "image",
    "img": "image",
}


def _norm_enum(s) -> str:
    """열거형/모드 문자열을 표기차이에 무관하게 비교하기 위한 정규 토큰으로."""
    t = str(s).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in t:
        t = t.replace("__", "_")
    return _ENUM_ALIASES.get(t, t)


# ── verifier 구현 ─────────────────────────────────────────────────

def _tool_called(v: dict, tool_trace: list[dict]) -> VerifyResult:
    missing = [t for t in v["tools"] if not _calls_of(tool_trace, t)]
    if missing:
        return VerifyResult(
            passed=False,
            verifier_type="tool_called",
            detail=f"호출되지 않은 tool: {missing}",
        )
    return VerifyResult(
        passed=True,
        verifier_type="tool_called",
        detail=f"모든 필수 tool 호출됨: {v['tools']}",
    )


def _tool_called_any(v: dict, tool_trace: list[dict]) -> VerifyResult:
    """허용된 tool 중 하나라도 호출되면 통과.

    같은 목표를 여러 툴 경로로 달성할 수 있을 때 공정하게 채점한다.
    예: 3x3 그리드 측정은 run_grid_scan 한 번 또는 move_stage+acquire_spectrum
        수동 루프 둘 다 정답이므로 tool_called_any=[run_grid_scan, acquire_spectrum].
    """
    tools = v["tools"]
    hit = [t for t in tools if _calls_of(tool_trace, t)]
    if hit:
        return VerifyResult(passed=True, verifier_type="tool_called_any",
                            detail=f"허용 tool 중 호출됨: {hit}")
    return VerifyResult(passed=False, verifier_type="tool_called_any",
                        detail=f"허용 tool 중 아무것도 호출되지 않음: {tools}")


def _tool_result_ok(v: dict, tool_trace: list[dict]) -> VerifyResult:
    calls = _calls_of(tool_trace, v["tool"])
    if not calls:
        return VerifyResult(passed=False, verifier_type="tool_result_ok",
                            detail=f"{v['tool']} 호출 없음")
    last = calls[-1]
    result = last.get("result", {})
    if result.get("ok") is True:
        return VerifyResult(passed=True, verifier_type="tool_result_ok",
                            detail=f"{v['tool']} 성공 반환")
    return VerifyResult(passed=False, verifier_type="tool_result_ok",
                        detail=f"{v['tool']} 결과 ok=False: {result.get('error', result)}")


def _tool_args(v: dict, tool_trace: list[dict]) -> VerifyResult:
    calls = _calls_of(tool_trace, v["tool"])
    if not calls:
        return VerifyResult(passed=False, verifier_type="tool_args",
                            detail=f"{v['tool']} 호출 없음")
    field = v["field"]
    expected = v["expected"]
    tolerance = v.get("tolerance")

    for call in calls:
        actual = call["args"].get(field)
        if actual is None:
            continue
        if tolerance is None:
            # 문자열/열거형 인자: 표기차이(대소문자·공백·별칭)를 무시하고 비교.
            if _norm_enum(actual) == _norm_enum(expected):
                return VerifyResult(passed=True, verifier_type="tool_args",
                                    detail=f"{v['tool']}.{field}={actual} (expected={expected})")
        else:
            if abs(float(actual) - float(expected)) <= tolerance:
                return VerifyResult(passed=True, verifier_type="tool_args",
                                    detail=f"{v['tool']}.{field}={actual} (expected={expected}±{tolerance})")

    last_val = calls[-1]["args"].get(field, "N/A")
    return VerifyResult(passed=False, verifier_type="tool_args",
                        detail=f"{v['tool']}.{field} 불일치: actual={last_val}, expected={expected}")


def _tool_arg_any(v: dict, tool_trace: list[dict]) -> VerifyResult:
    """여러 후보 (tool, field) 중 어느 하나라도 기대값과 일치하면 통과.

    같은 파라미터를 여러 경로로 줄 수 있을 때 공정하게 채점한다.
    예: 레이저 파워는 set_laser_power(percent=) 또는 acquire_spectrum(power=)
        어느 쪽으로 줘도 정답. acquire_spectrum 이 내부에서 파워설정+발사를
        수행하므로 사전 set_laser_power 호출을 강제하지 않는다.

    v = {type:"tool_arg_any", expected, tolerance?,
         candidates:[{tool, field}, ...]}
    """
    expected = v["expected"]
    tolerance = v.get("tolerance")
    seen = []
    for cand in v.get("candidates", []):
        for call in _calls_of(tool_trace, cand["tool"]):
            actual = call["args"].get(cand["field"])
            if actual is None:
                continue
            seen.append(f"{cand['tool']}.{cand['field']}={actual}")
            if tolerance is None:
                if _norm_enum(actual) == _norm_enum(expected):
                    return VerifyResult(passed=True, verifier_type="tool_arg_any",
                                        detail=f"{cand['tool']}.{cand['field']}={actual} (expected={expected})")
            else:
                try:
                    if abs(float(actual) - float(expected)) <= float(tolerance):
                        return VerifyResult(passed=True, verifier_type="tool_arg_any",
                                            detail=f"{cand['tool']}.{cand['field']}={actual} (expected={expected}±{tolerance})")
                except (TypeError, ValueError):
                    pass
    cand_str = " | ".join(f"{c['tool']}.{c['field']}" for c in v.get("candidates", []))
    return VerifyResult(passed=False, verifier_type="tool_arg_any",
                        detail=f"어느 후보도 expected={expected} 불일치 ({cand_str}). 관측={seen or '없음'}")


def _tool_sequence(v: dict, tool_trace: list[dict]) -> VerifyResult:
    ordered = v["tools_in_order"]
    tool_names = [t["tool"] for t in tool_trace]
    last_pos = -1
    for tool in ordered:
        found = False
        for i in range(last_pos + 1, len(tool_names)):
            if tool_names[i] == tool:
                last_pos = i
                found = True
                break
        if not found:
            return VerifyResult(
                passed=False,
                verifier_type="tool_sequence",
                detail=f"'{tool}'이 이전 단계 이후 tool_trace에 없음",
            )
    return VerifyResult(passed=True, verifier_type="tool_sequence",
                        detail=f"순서 정확: {ordered}")


def _tool_call_count(v: dict, tool_trace: list[dict]) -> VerifyResult:
    calls = _calls_of(tool_trace, v["tool"])
    count = len(calls)
    min_count = v.get("min_count", 1)
    max_count = v.get("max_count", None)
    ok = count >= min_count and (max_count is None or count <= max_count)
    constraint = f">={min_count}" if max_count is None else f"{min_count}~{max_count}"
    return VerifyResult(
        passed=ok,
        verifier_type="tool_call_count",
        detail=f"{v['tool']} 호출 횟수: {count} (조건: {constraint})",
    )


def _tool_args_sequence(v: dict, tool_trace: list[dict]) -> VerifyResult:
    calls = _calls_of(tool_trace, v["tool"])
    if not calls:
        return VerifyResult(passed=False, verifier_type="tool_args_sequence",
                            detail=f"{v['tool']} 호출 없음")
    field = v["field"]
    expected_seq = v["expected_sequence"]
    tolerance = v.get("tolerance")
    actual_seq = [c["args"].get(field) for c in calls]

    # 순서가 포함(subsequence)되는지 체크 (exact match 아님)
    matched = 0
    for val in actual_seq:
        if matched < len(expected_seq):
            exp = expected_seq[matched]
            if tolerance is not None:
                try:
                    if abs(float(val) - float(exp)) <= float(tolerance):
                        matched += 1
                except (TypeError, ValueError):
                    pass
            else:
                if val == exp:
                    matched += 1
    if matched == len(expected_seq):
        return VerifyResult(passed=True, verifier_type="tool_args_sequence",
                            detail=f"{v['tool']}.{field} 시퀀스 확인: {actual_seq}")
    return VerifyResult(
        passed=False,
        verifier_type="tool_args_sequence",
        detail=f"{v['tool']}.{field} 시퀀스 불일치: actual={actual_seq}, expected={expected_seq}",
    )


# ── 하드웨어 상태 verifier ────────────────────────────────────────

def _stage_position(v: dict, post_state: dict) -> VerifyResult:
    stage = (post_state or {}).get("stage")
    if not stage:
        return VerifyResult(passed=False, verifier_type="stage_position",
                            detail="post_state에 stage 정보 없음")
    expected = v["expected"]
    tol = v.get("tolerance", 0.05)
    errors = []
    for axis in ("x", "y", "z"):
        if axis not in expected:
            continue
        actual = stage.get(axis)
        if actual is None:
            errors.append(f"{axis}=없음")
        elif abs(float(actual) - float(expected[axis])) > tol:
            errors.append(f"{axis}: actual={actual:.4f}, expected={expected[axis]}±{tol}")
    if errors:
        return VerifyResult(passed=False, verifier_type="stage_position",
                            detail="위치 오차 초과: " + ", ".join(errors))
    actual_str = {ax: f"{stage.get(ax, '?'):.4f}" for ax in ("x", "y", "z") if ax in expected}
    return VerifyResult(passed=True, verifier_type="stage_position",
                        detail=f"위치 정확: {actual_str}")


def _stage_velocity(v: dict, post_state: dict) -> VerifyResult:
    stage = (post_state or {}).get("stage", {})
    velocity = stage.get("velocity", {})
    expected = v["expected"]
    tol = v.get("tolerance", 0.1)
    errors = []
    for axis in ("x", "y", "z"):
        if axis not in expected:
            continue
        actual = velocity.get(axis)
        if actual is None:
            errors.append(f"{axis} 속도 없음")
        elif abs(float(actual) - float(expected[axis])) > tol:
            errors.append(f"{axis}: actual={actual}, expected={expected[axis]}±{tol}")
    if errors:
        return VerifyResult(passed=False, verifier_type="stage_velocity",
                            detail="속도 오차: " + ", ".join(errors))
    return VerifyResult(passed=True, verifier_type="stage_velocity",
                        detail=f"속도 정확: {velocity}")


def _ccd_exposure(v: dict, post_state: dict) -> VerifyResult:
    ccd = (post_state or {}).get("ccd")
    if not ccd:
        return VerifyResult(passed=False, verifier_type="ccd_exposure",
                            detail="post_state에 CCD 정보 없음")
    actual = ccd.get("exposure_time")
    if actual is None:
        return VerifyResult(passed=False, verifier_type="ccd_exposure",
                            detail="exposure_time 필드 없음")
    tol = v.get("tolerance", 0.01)
    if abs(float(actual) - float(v["expected"])) <= tol:
        return VerifyResult(passed=True, verifier_type="ccd_exposure",
                            detail=f"CCD 노출시간: {actual}s (expected={v['expected']}s)")
    return VerifyResult(passed=False, verifier_type="ccd_exposure",
                        detail=f"CCD 노출시간 불일치: actual={actual}, expected={v['expected']}±{tol}")


def _ccd_read_mode(v: dict, post_state: dict) -> VerifyResult:
    ccd = (post_state or {}).get("ccd")
    if not ccd:
        return VerifyResult(passed=False, verifier_type="ccd_read_mode",
                            detail="post_state에 CCD 정보 없음")
    actual = ccd.get("ro_mode", "")
    expected = v["expected"]
    # 표기차이(fvb == FULL_VERTICAL_BINNING 등)는 정규화 후 비교.
    if _norm_enum(actual) == _norm_enum(expected):
        return VerifyResult(passed=True, verifier_type="ccd_read_mode",
                            detail=f"CCD 읽기 모드 일치: actual={actual} ≡ expected={expected}")
    return VerifyResult(passed=False, verifier_type="ccd_read_mode",
                        detail=f"CCD 읽기 모드 불일치: actual={actual}, expected={expected}")


def _ccd_temperature(v: dict, post_state: dict) -> VerifyResult:
    ccd = (post_state or {}).get("ccd")
    if not ccd:
        return VerifyResult(passed=False, verifier_type="ccd_temperature",
                            detail="post_state에 CCD 정보 없음")
    actual = ccd.get("temperature")
    if actual is None:
        return VerifyResult(passed=False, verifier_type="ccd_temperature",
                            detail="온도 정보 없음")
    expected = v["expected"]
    tol = v.get("tolerance", 2)
    if abs(float(actual) - float(expected)) <= tol:
        return VerifyResult(passed=True, verifier_type="ccd_temperature",
                            detail=f"CCD 온도: {actual}°C (expected={expected}°C)")
    return VerifyResult(passed=False, verifier_type="ccd_temperature",
                        detail=f"CCD 온도 불일치: actual={actual}°C, expected={expected}°C±{tol}")


def _ccd_temperature_reported(tool_trace: list[dict]) -> VerifyResult:
    calls = _calls_of(tool_trace, "get_ccd_info")
    if not calls:
        return VerifyResult(passed=False, verifier_type="ccd_temperature_reported",
                            detail="get_ccd_info 호출 없음")
    for call in calls:
        result = call.get("result", {})
        temp = result.get("temperature") if result.get("temperature") is not None else result.get("temperature_C")
        if result.get("ok") and temp is not None:
            return VerifyResult(passed=True, verifier_type="ccd_temperature_reported",
                                detail=f"온도 보고됨: {temp}°C")
    return VerifyResult(passed=False, verifier_type="ccd_temperature_reported",
                        detail="get_ccd_info 결과에 temperature 값 없음")


def _laser_state(v: dict, post_state: dict) -> VerifyResult:
    laser = (post_state or {}).get("laser")
    if not laser:
        return VerifyResult(passed=False, verifier_type="laser_state",
                            detail="post_state에 laser 정보 없음")
    actual = laser.get("is_on")
    expected = v["expected_on"]
    if bool(actual) == bool(expected):
        return VerifyResult(passed=True, verifier_type="laser_state",
                            detail=f"레이저 상태: {'ON' if actual else 'OFF'}")
    return VerifyResult(passed=False, verifier_type="laser_state",
                        detail=f"레이저 상태 불일치: actual={'ON' if actual else 'OFF'}, expected={'ON' if expected else 'OFF'}")


def _spectrum_valid(v: dict, tool_trace: list[dict]) -> VerifyResult:
    calls = _calls_of(tool_trace, v["tool"])
    if not calls:
        return VerifyResult(passed=False, verifier_type="spectrum_valid",
                            detail=f"{v['tool']} 호출 없음")
    for call in calls:
        result = call.get("result", {})
        if result.get("ok") and result.get("intensity"):
            intensity = result["intensity"]
            if isinstance(intensity, list) and len(intensity) > 0:
                max_val = result.get("max_intensity", max(intensity))
                if max_val > 0:
                    return VerifyResult(
                        passed=True,
                        verifier_type="spectrum_valid",
                        detail=f"스펙트럼 유효: {len(intensity)}픽셀, max={max_val:.1f}",
                    )
    return VerifyResult(passed=False, verifier_type="spectrum_valid",
                        detail="유효한 스펙트럼 데이터 없음 (intensity 배열 비어있거나 ok=False 또는 신호 없음)")
