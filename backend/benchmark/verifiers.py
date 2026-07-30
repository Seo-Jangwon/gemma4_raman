"""
벤치마크 검증 로직.
각 verifier는 (verifier_spec, tool_trace, pre_state, post_state, context) → VerifyResult.

[검증기의 두 계층]
  · 과정 검증 — 어떤 툴을 어떤 인자로 불렀는가(tool_called/tool_args/...). 트레이스만 본다.
  · 정답 검증 — 산출물이 정답과 맞는가(reference_match/answer_numeric/answer_contains).
    이쪽은 트레이스로는 알 수 없다. 툴 결과의 큰 배열은 _slim 이 버리므로(컨텍스트
    폭주 방지) 수치는 '에이전트가 저장한 파일' 또는 '답변 텍스트'에서만 얻을 수 있다.
    그래서 이 검증기들은 실행 레코드 전체(context)를 받는다.

context 는 run_bench 레코드다 — id/session_id/tool_calls/answer 를 쓴다. 없으면(구버전
호출) 정답 검증기는 '검증 불가'로 실패 처리하되, 과정 검증기는 그대로 동작한다.
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
    context: dict | None = None,
) -> list[VerifyResult]:
    results = []
    for v in verifiers:
        results.append(_dispatch(v, tool_trace, pre_state, post_state, context))
    return results


def _dispatch(v: dict, tool_trace, pre_state, post_state, context=None) -> VerifyResult:
    vtype = v["type"]
    try:
        if vtype == "reference_match":
            return _reference_match(v, context)
        if vtype == "answer_numeric":
            return _answer_numeric(v, context)
        if vtype == "answer_contains":
            return _answer_contains(v, context)
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


# ── 정답 검증 verifier (파일 / 답변 텍스트) ───────────────────────
#
# 여기부터는 '과정'이 아니라 '결과가 정답인가'를 본다. 판정 기준은 문항의
# grading_criteria 를 그대로 수치화한 것이다.

def _panel():
    """spectra_panel 을 늦게 import 한다 — CSV 읽기/산출물 탐색/오차계산을 재사용.
    (같은 로직을 두 벌 두면 리포트 그림과 채점 숫자가 갈라진다.)"""
    import spectra_panel
    return spectra_panel


def _need_context(vtype: str, context) -> VerifyResult | None:
    if not context:
        return VerifyResult(passed=False, verifier_type=vtype,
                            detail="실행 레코드(context)가 없어 정답 검증 불가")
    return None


def _spike_context(context, sp, ref_y):
    """스파이크 인덱스와 입력 세기를 돌려준다 — spike_aware 판정용.

    스파이크 위치는 task_files.json 의 ground_truth 를 1순위로, 없으면 '입력과
    레퍼런스의 차가 1000 을 넘는 점'으로 역산한다(스파이크 설계 진폭은 +5000).
    """
    import json as _json
    tid = str(context.get("id") or "")
    tf_path = sp._TASK_FILES
    try:
        tf = _json.loads(tf_path.read_text(encoding="utf-8"))
    except Exception:
        tf = {}
    files = (tf.get(tid) or {}).get("files") or [f"{tid}.csv"]
    ip = sp._find_upload(files[0])
    if ip is None:
        return [], None
    ic = sp.read_curves(ip, max_groups=1)
    if not ic or not ic[0]["y"]:
        return [], None
    xi, yi = ic[0]["x"], ic[0]["y"]
    n = min(len(yi), len(ref_y))
    pos = ((tf.get(tid) or {}).get("ground_truth") or {}).get("spike_positions_cm-1") or []
    if pos:
        idx = [min(range(len(xi)), key=lambda i: abs(xi[i] - float(p))) for p in pos]
    else:
        idx = [i for i in range(n) if abs(yi[i] - ref_y[i]) > 1000.0]
    return [i for i in idx if i < n], yi


def _reference_match(v: dict, context) -> VerifyResult:
    """에이전트가 저장한 스펙트럼 CSV 를 정답 레퍼런스와 점대점 비교한다.

    v = {type:"reference_match", reference:"T038_reference.csv",
         tolerance: 1e-5,          점당 허용 절대오차
         max_bad_points: 0,        tolerance 를 넘어도 되는 점 개수
         spike_aware: false,       True 면 '스파이크 계열' 기준으로 판정(아래)
         min_removal_pct: 99.0}    spike_aware 일 때 요구 제거율

    [spike_aware 가 필요한 이유]
    T039/T056/T099 의 레퍼런스는 '스파이크를 지운 결과'가 아니라 '스파이크를 넣기 전
    원본'이다(make_task_spectra: refs={...: (AXIS, ps)}). 스파이크는 단일 점에 +5000 을
    더한 것이라 원래 값이 소실됐고, 어떤 despike 알고리즘도 그 점을 정확히 복원할 수
    없다. 그래서 '전 구간 일치'를 요구하면 완벽한 답도 항상 실패한다 — 실측에서 비-
    스파이크 1796점이 max|Δ|=0.0(완전 동일)이고 스파이크를 100% 제거한 답이 전체
    max|Δ|=0.214 때문에 불일치로 찍혔다. 올바른 조건은 두 갈래다:
      · 스파이크가 아닌 점을 건드리지 않았는가 (tolerance 이내)
      · 스파이크 위치에서 초과분을 얼마나 제거했는가 (min_removal_pct 이상)
    자세한 근거는 backend/benchmark/answer_specs.py 의 SPECS 참고
    (review.py 가 채점 콘솔의 '정답 기준' 블록에 그대로 싣는다).

    에이전트가 후보를 여러 개 저장했으면(예: T038 은 3개) 그중 가장 잘 맞는 것으로
    판정하고, 후보 개수를 detail 에 남긴다 — tolerance 가 1e-5(= CSV 왕복 오차)라
    우연히 맞출 수는 없으므로 'best of N' 이 점수를 부풀리지 않는다. 대신 여러 개를
    저장했다는 사실 자체는 채점자가 볼 수 있게 적어 둔다.
    """
    bad = _need_context("reference_match", context)
    if bad:
        return bad
    sp = _panel()
    ref_name = v["reference"]
    ref_path = sp._TASK_REFS / ref_name
    if not ref_path.exists():
        return VerifyResult(passed=False, verifier_type="reference_match",
                            detail=f"레퍼런스 파일 없음: {ref_name}")
    rc = sp.read_curves(ref_path, max_groups=1)
    if not rc or not rc[0]["y"]:
        return VerifyResult(passed=False, verifier_type="reference_match",
                            detail=f"레퍼런스를 읽을 수 없음: {ref_name}")
    ref_y = rc[0]["y"]

    tol = float(v.get("tolerance", 1e-5))
    max_bad = int(v.get("max_bad_points", 0))

    csvs, _figs = sp.find_outputs(context)
    if not csvs:
        return VerifyResult(passed=False, verifier_type="reference_match",
                            detail="에이전트가 저장한 스펙트럼 파일이 없음 (비교 불가)")

    spike_aware = bool(v.get("spike_aware"))
    min_rem = float(v.get("min_removal_pct", 99.0))
    spike_idx, input_y = ([], None)
    if spike_aware:
        spike_idx, input_y = _spike_context(context, sp, ref_y)
        if not spike_idx or input_y is None:
            return VerifyResult(passed=False, verifier_type="reference_match",
                                detail="spike_aware 인데 스파이크 위치를 특정할 수 없음")

    best = None
    for p in csvs:
        oc = sp.read_curves(p, max_groups=1)
        if not oc or not oc[0]["y"]:
            continue
        a, b = oc[0]["y"], ref_y
        n = min(len(a), len(b))
        if n == 0:
            continue
        d = [abs(a[i] - b[i]) for i in range(n)]
        mx = max(d)
        rmse = (sum(t * t for t in d) / n) ** 0.5
        if spike_aware:
            sset = {i for i in spike_idx if i < n}
            ns = [d[i] for i in range(n) if i not in sset]
            ns_bad = sum(1 for t in ns if t > tol)
            ns_max = max(ns) if ns else 0.0
            rem = [100.0 * (1.0 - d[i] / abs(input_y[i] - b[i]))
                   for i in sorted(sset) if abs(input_y[i] - b[i]) > 1e-9]
            worst = min(rem) if rem else 0.0
            cand = {"key": (ns_bad, -worst), "name": p.name, "n": n, "max_abs": mx,
                    "rmse": rmse, "ns_bad": ns_bad, "ns_max": ns_max, "ns_n": len(ns),
                    "worst_rem": worst, "n_spike": len(sset)}
        else:
            n_bad = sum(1 for t in d if t > tol)
            cand = {"key": (n_bad, mx), "name": p.name, "n": n, "max_abs": mx,
                    "rmse": rmse, "n_bad": n_bad, "len_mismatch": len(a) != len(b)}
        if best is None or cand["key"] < best["key"]:
            best = cand
    if best is None:
        return VerifyResult(passed=False, verifier_type="reference_match",
                            detail=f"산출물 {len(csvs)}개를 모두 읽을 수 없음")

    extra = f", 후보 {len(csvs)}개 중 최선" if len(csvs) > 1 else ""
    if spike_aware:
        ok = best["ns_bad"] == 0 and best["worst_rem"] >= min_rem
        return VerifyResult(
            passed=ok, verifier_type="reference_match",
            detail=(f"{'일치' if ok else '불일치'}(스파이크 기준): {best['name']} — "
                    f"비-스파이크 {best['ns_n']}점 max|Δ|={best['ns_max']:.3e} "
                    f"(허용초과 {best['ns_bad']}점), 스파이크 {best['n_spike']}개 "
                    f"최저제거율 {best['worst_rem']:.2f}% (요구 {min_rem:g}%), "
                    f"참고 전체 max|Δ|={best['max_abs']:.3e}{extra}"))
    if best.get("len_mismatch"):
        extra += ", 길이 불일치"
    ok = best["n_bad"] <= max_bad
    return VerifyResult(
        passed=ok, verifier_type="reference_match",
        detail=(f"{'일치' if ok else '불일치'}: {best['name']} vs {ref_name} — "
                f"max|Δ|={best['max_abs']:.3e}, RMSE={best['rmse']:.3e}, "
                f"허용초과 {best['n_bad']}/{best['n']}점 "
                f"(tol={tol:g}, 허용 {max_bad}점{extra})"))


# 에이전트가 응답을 못 만들었을 때의 폴백 문구(두 에이전트 공통).
_NO_ANSWER_MARKERS = (
    "failed to generate a response",
    "the agent failed to generate a response",
)


def _no_answer(text: str) -> bool:
    t = str(text or "").strip().lower()
    return any(m in t for m in _NO_ANSWER_MARKERS) and len(t) < 120


def _extract_numbers(text: str) -> list[float]:
    """답변 텍스트에서 숫자를 뽑는다. 마크다운 강조·LaTeX·천단위 콤마를 견딘다."""
    import re
    t = str(text or "")
    t = t.replace("**", " ").replace("$", " ").replace("\\text", " ")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)          # 1,024 -> 1024
    out = []
    for m in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", t):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def _answer_numeric(v: dict, context) -> VerifyResult:
    """답변 텍스트에 기대 수치가 (허용오차 안에) 등장하는지.

    v = {type:"answer_numeric", expected: 12.0 또는 [1001,1602,1031],
         tolerance: 3.0            절대 허용오차
         rel_tolerance: 0.05       상대 허용오차(둘 중 하나만 주면 그것만 씀)
         ordered: true             리스트일 때 등장 순서까지 볼지
         note: "..."}

    한계를 분명히 해 둔다: 이건 '답변에 그 숫자가 있는가'를 보는 것이고,
    '그 숫자가 올바른 항목에 붙어 있는가'는 보지 못한다. 그래서 실패는 강한 신호이지만
    통과는 약한 신호다 — 그래서 이 검증기가 붙은 문항은 human_only 를 함께 남긴다.
    """
    bad = _need_context("answer_numeric", context)
    if bad:
        return bad
    text = context.get("answer") or context.get("final_report") or ""
    if not str(text).strip():
        return VerifyResult(passed=False, verifier_type="answer_numeric",
                            detail="답변이 비어 있음")
    if _no_answer(text):
        return VerifyResult(passed=False, verifier_type="answer_numeric",
                            detail="에이전트가 응답을 생성하지 못했다(빈 응답 폴백) - 답변 내용 없음")
    nums = _extract_numbers(text)
    exp = v["expected"]
    exp_list = list(exp) if isinstance(exp, (list, tuple)) else [exp]
    atol = v.get("tolerance")
    rtol = v.get("rel_tolerance")
    if atol is None and rtol is None:
        atol = 0.0

    def _tol_for(e) -> float:
        cands = []
        if atol is not None:
            cands.append(float(atol))
        if rtol is not None:
            cands.append(abs(float(e)) * float(rtol))
        return max(cands) if cands else 0.0

    if v.get("ordered") and len(exp_list) > 1:
        # 기대값이 답변에 '순서대로' 부분수열로 나타나는지
        pos = 0
        found = []
        for e in exp_list:
            t = _tol_for(e)
            hit = None
            for i in range(pos, len(nums)):
                if abs(nums[i] - float(e)) <= t:
                    hit = i
                    break
            if hit is None:
                return VerifyResult(
                    passed=False, verifier_type="answer_numeric",
                    detail=f"순서 불일치: {e}±{t:g} 를 앞선 항목 뒤에서 찾지 못함 "
                           f"(기대 순서 {exp_list}, 찾은 값 {found})")
            found.append(nums[hit])
            pos = hit + 1
        return VerifyResult(passed=True, verifier_type="answer_numeric",
                            detail=f"순서까지 일치: {found} (기대 {exp_list})")

    missing = []
    found = []
    for e in exp_list:
        t = _tol_for(e)
        hit = next((x for x in nums if abs(x - float(e)) <= t), None)
        if hit is None:
            missing.append(f"{e}±{t:g}")
        else:
            found.append(hit)
    if missing:
        return VerifyResult(
            passed=False, verifier_type="answer_numeric",
            detail=f"답변에서 찾지 못한 값: {missing} (찾은 값 {found}, "
                   f"답변 내 숫자 {len(nums)}개)")
    return VerifyResult(passed=True, verifier_type="answer_numeric",
                        detail=f"기대값 전부 등장: {found} (기대 {exp_list})")


# 답을 '선언'하는 문맥. 택일 문항에서 오답 라벨이 단순 언급된 것과, 실제로 그렇게
# 답한 것을 구별하기 위해 쓴다. 이게 없으면 "amorphous 다. 반면 crystalline 은
# 날카로운 피크를 갖는다" 같은 정상적인 대조 설명이 '여러 답을 말했다'로 오판된다
# (실측: T104 가 정확히 이 이유로 오탐 났다).
_DECL_MARKERS = (
    "classif", "classify", "conclusion", "answer is", "the answer", "determined",
    "identified as", "identific", "is most", "most similar", "most likely",
    "result:", "verdict", "therefore", "thus the", "final",
    "판정", "결론", "분류", "정답",
)


def _declaration_zones(text: str) -> str:
    """답을 선언하는 부분만 모아 돌려준다. 선언 표지가 있는 줄/문장 + 마지막 줄."""
    import re
    zones = []
    # 줄 단위(마크다운 불릿·헤딩), 그리고 문장 단위 둘 다 본다
    units = [u for u in re.split(r"[\n]+", text) if u.strip()]
    for u in units:
        low = u.lower()
        if any(m in low for m in _DECL_MARKERS):
            zones.append(u)
    for s in re.split(r"(?<=[.!?])\s+", text):
        low = s.lower()
        if any(m in low for m in _DECL_MARKERS):
            zones.append(s)
    if units:
        zones.append(units[-1])          # 마지막 줄은 결론일 가능성이 높다
    return "\n".join(zones)


def _answer_contains(v: dict, context) -> VerifyResult:
    """답변 텍스트에 기대 문자열(라벨/물질명 등)이 있는지. 대소문자 무시.

    v = {type:"answer_contains", expected:"polystyrene" 또는 ["a","b"],
         mode:"all"|"any",          기본 all
         forbidden:["crystalline"]} 택일 문항에서 '여러 답 말하기' 방지.
                                    단순 언급이 아니라 '선언 문맥'에서만 실패로 본다.
    """
    bad = _need_context("answer_contains", context)
    if bad:
        return bad
    text = str(context.get("answer") or context.get("final_report") or "")
    low = text.lower()
    if not text.strip():
        return VerifyResult(passed=False, verifier_type="answer_contains",
                            detail="답변이 비어 있음")
    # 에이전트가 빈 응답을 낸 경우의 폴백 문구. '표현을 못 찾음'이 아니라 '답 자체가
    # 없음'이라고 보고해야 채점자가 원인을 혼동하지 않는다.
    if _no_answer(text):
        return VerifyResult(passed=False, verifier_type="answer_contains",
                            detail="에이전트가 응답을 생성하지 못했다(빈 응답 폴백) - 답변 내용 없음")
    exp = v["expected"]
    exp_list = [str(x) for x in (exp if isinstance(exp, (list, tuple)) else [exp])]
    hits = [e for e in exp_list if e.lower() in low]
    forb = [str(x) for x in (v.get("forbidden") or [])]

    if forb:
        decl = _declaration_zones(text).lower()
        # 선언 문맥에 오답 라벨이 있고, 정답 라벨은 없을 때만 '여러 답'으로 본다.
        forb_decl = [f for f in forb if f.lower() in decl]
        exp_decl = [e for e in exp_list if e.lower() in decl]
        if forb_decl and not exp_decl:
            return VerifyResult(
                passed=False, verifier_type="answer_contains",
                detail=f"선언 문맥에서 오답 라벨을 답으로 말함: {forb_decl} (기대 {exp_list})")
    mode = v.get("mode", "all")
    ok = (len(hits) == len(exp_list)) if mode == "all" else bool(hits)
    if ok:
        return VerifyResult(passed=True, verifier_type="answer_contains",
                            detail=f"기대 표현 등장({mode}): {hits}")
    return VerifyResult(passed=False, verifier_type="answer_contains",
                        detail=f"기대 표현 없음({mode}): 기대={exp_list}, 등장={hits}")


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
