"""
HWManagerNode — 하드웨어 도구 래퍼 에이전트.

hw_tools/raman_tools.TOOL_DISPATCH를 LangChain tool로 래핑해 LLM에 bind.
실행 전 Critic C2 pre-check (power/dose), 실행 후 cumulative_dose_mj 갱신.

LLM: 미정 (Ollama 후보). 기본값 Claude claude-sonnet-4-6.
교체: 파일 상단 _llm 변수만 수정.
"""

from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from backend.agents.critic import check_c2_hardware_safety
from backend.agents.state import ExperimentState

# ── LLM 설정 (교체 포인트 — Ollama 사용 시 ChatOllama로 교체) ─────────────────
# from langchain_community.chat_models import ChatOllama
# _llm = ChatOllama(model="gemma4:31b", base_url="http://192.168.1.16:11434")
_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

_SYSTEM = """\
당신은 라만 분광기 하드웨어를 제어하는 전문 에이전트입니다.
사용 가능한 도구를 이용해 요청된 하드웨어 작업을 안전하게 수행하세요.
- 레이저를 켜기 전에 반드시 출력을 설정하세요
- 스펙트럼 획득 전 레이저가 켜져 있는지 확인하세요
- 모든 작업 후 결과를 명확히 보고하세요
- 오류 발생 시 즉시 레이저를 끄세요"""

# 허용할 하드웨어 도구 이름 집합
_lc_tools_cache: tuple[list, dict] | None = None

_HW_TOOL_NAMES = {
    "acquire_spectrum", "move_stage", "move_stage_relative", "get_stage_position",
    "set_stage_speed", "set_laser_power", "laser_on", "laser_off",
    "set_ccd_exposure", "get_ccd_info", "capture_camera_frame",
    "analyze_focus_quality", "start_camera_stream", "stop_camera_stream",
}


def _build_lc_tools():
    """hw_tools.TOOL_DISPATCH의 함수를 LangChain @tool로 래핑해 반환."""
    global _lc_tools_cache
    if _lc_tools_cache is not None:
        return _lc_tools_cache

    try:
        from backend.hw_tools.raman_tools import TOOL_DISPATCH
        from backend.hw_tools.raman_tool_schemas import RAMAN_TOOLS
    except ImportError:
        return [], {}

    lc_tools = []
    tool_map: dict[str, callable] = {}

    for schema in RAMAN_TOOLS:
        fn_info = schema.get("function", {})
        name = fn_info.get("name", "")
        if name not in _HW_TOOL_NAMES:
            continue
        raw_fn = TOOL_DISPATCH.get(name)
        if raw_fn is None:
            continue

        desc = fn_info.get("description", name)
        params_schema = fn_info.get("parameters", {"type": "object", "properties": {}})

        # 동적으로 LangChain tool 생성
        def _make_tool(fn, n, d, ps):
            @tool(n, description=d)
            def _t(**kwargs):
                return fn(**kwargs)
            return _t

        lc_tool = _make_tool(raw_fn, name, desc, params_schema)
        lc_tools.append(lc_tool)
        tool_map[name] = raw_fn

    _lc_tools_cache = (lc_tools, tool_map)
    return _lc_tools_cache


def hw_manager_node(state: ExperimentState) -> dict:
    plan = state.get("plan", [])
    idx  = state.get("current_step_idx", 0)
    step = plan[idx] if idx < len(plan) else None

    if step is None:
        return {"observations": [{"error": "hw_manager: plan step 없음"}]}

    params = step.get("params", {})
    power_pct  = float(params.get("power_pct",  0.0))
    exposure_s = float(params.get("exposure_s", 0.0))

    # ── C2 pre-check ──────────────────────────────────────────────────────────
    c2 = check_c2_hardware_safety(state, power_pct=power_pct, exposure_s=exposure_s)
    if c2["verdict"] == "ABORT":
        updated_plan = list(plan)
        updated_plan[idx] = {**step, "status": "failed", "result": {"c2_abort": c2["reason"]}}
        return {
            "critic_log": [c2],
            "abort_reason": c2["reason"],
            "plan": updated_plan,
            "next_node": "__end__",
        }

    # ── LLM + tool 실행 ───────────────────────────────────────────────────────
    lc_tools, tool_map = _build_lc_tools()
    if not lc_tools:
        # 하드웨어 미연결 — 시뮬레이션 결과 반환
        obs = {
            "tool": "hw_manager_sim",
            "action": step["action"],
            "result": {"ok": True, "simulated": True, "message": "하드웨어 미연결 — 시뮬레이션"},
            "step_id": step["step_id"],
        }
        updated_plan = list(plan)
        updated_plan[idx] = {**step, "status": "done", "result": obs["result"]}
        return {
            "observations": [obs],
            "plan": updated_plan,
            "current_step_idx": idx + 1,
        }

    llm_with_tools = _llm.bind_tools(lc_tools)
    prompt = (
        f"작업: {step['action']}\n"
        f"파라미터: {json.dumps(params, ensure_ascii=False)}\n"
        f"현재 스테이지 위치: {state.get('stage_position')}\n"
        "위 작업을 수행하세요."
    )

    messages = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=prompt),
    ]

    observations = []
    new_dose = state.get("cumulative_dose_mj", 0.0)
    last_position = state.get("stage_position")

    max_turns = 8
    for _ in range(max_turns):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            fn = tool_map.get(tc["name"])
            if fn is None:
                result = {"ok": False, "error": f"도구 없음: {tc['name']}"}
            else:
                try:
                    result = fn(**tc["args"])
                except Exception as e:
                    result = {"ok": False, "error": str(e)}

            obs = {
                "tool": tc["name"],
                "args": tc["args"],
                "result": result,
                "step_id": step["step_id"],
            }
            observations.append(obs)

            # dose 누적 (레이저 노출)
            if tc["name"] == "acquire_spectrum":
                args = tc["args"]
                p = float(args.get("power", power_pct))
                e = float(args.get("exposure", exposure_s))
                new_dose += p * e * 0.01

            # 스테이지 위치 갱신
            if tc["name"] in ("move_stage", "move_stage_relative"):
                if result.get("ok") and result.get("result"):
                    last_position = result["result"]

            messages.append(
                ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=tc["id"])
            )

    # plan 업데이트
    updated_plan = list(plan)
    updated_plan[idx] = {
        **step,
        "status": "done",
        "result": observations[-1]["result"] if observations else {},
    }

    result_dict: dict = {
        "observations": observations,
        "plan": updated_plan,
        "cumulative_dose_mj": new_dose,
        "current_step_idx": idx + 1,
    }
    if last_position:
        result_dict["stage_position"] = last_position

    return result_dict
