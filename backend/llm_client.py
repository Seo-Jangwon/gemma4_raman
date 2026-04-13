import ollama
import json

OLLAMA_HOST = "http://192.168.1.16:11434"
OLLAMA_MODEL = "gemma4:31b"

SYSTEM_PROMPT = """당신은 라만 분광기 제어 AI입니다.
사용 가능한 tool을 순서대로 호출해 사용자의 요청을 수행하세요.
- 스테이지 좌표 단위: mm (X: 0~75.3, Y: 0~50.2, Z: -1.0~1.0)
- 레이저를 켜기 전에 반드시 안전 여부를 확인하세요.
- 모든 작업이 끝나면 결과를 한국어로 요약해 주세요."""

client = ollama.Client(host=OLLAMA_HOST)


def generate(prompt: str) -> str:
    response = client.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response["response"]


def chat(messages: list) -> str:
    """messages: [{"role": "user"/"assistant", "content": "..."}]"""
    response = client.chat(model=OLLAMA_MODEL, messages=messages)
    return response["message"]["content"]


def run_agent(user_message: str, tools: list, tool_dispatch: dict,
              max_steps: int = 20, verbose: bool = True) -> str:
    """
    Tool calling agent loop.

    Parameters
    ----------
    user_message  : 사용자 자연어 명령
    tools         : raman_tool_schemas.RAMAN_TOOLS 같은 tool 스키마 리스트
    tool_dispatch : {"함수이름": callable(args_dict) -> dict} 매핑
    max_steps     : 무한루프 방지용 최대 tool 호출 횟수
    verbose       : True면 tool 호출/결과를 콘솔에 출력

    Returns
    -------
    LLM의 최종 텍스트 응답
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    for step in range(max_steps):
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=tools,
        )
        msg = response.message

        # tool call이 없으면 → 최종 답변
        if not msg.tool_calls:
            return msg.content

        # tool call 처리
        messages.append(msg)  # assistant 메시지(tool_calls 포함) 추가

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = tc.function.arguments or {}

            if verbose:
                print(f"[step {step+1}] tool: {fn_name}  args: {fn_args}")

            if fn_name not in tool_dispatch:
                result = {"ok": False, "error": f"알 수 없는 tool: {fn_name}"}
            else:
                result = tool_dispatch[fn_name](fn_args)

            if verbose:
                print(f"         result: {result}")

            messages.append({
                "role": "tool",
                "content": json.dumps(result, ensure_ascii=False),
            })

    return "최대 step 수에 도달했습니다. 작업이 완료되지 않았을 수 있습니다."


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # ── 기본 연결 테스트 ──────────────────────────────
    # print(generate("안녕하세요! 연결 테스트입니다."))

    # ── Tool calling mock 테스트 ──────────────────────
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from agents.raman_tool_schemas import RAMAN_TOOLS

    # 실제 하드웨어 없이 동작 확인용 mock
    def mock_dispatch(args):
        return {"ok": True, "mock": True, "args_received": args}

    mock_tool_dispatch = {
        "move_stage":          mock_dispatch,
        "get_stage_position":  mock_dispatch,
        "move_stage_relative": mock_dispatch,
        "laser_on":            mock_dispatch,
        "laser_off":           mock_dispatch,
        "set_laser_power":     mock_dispatch,
        "acquire_spectrum":    mock_dispatch,
    }

    result = run_agent(
        user_message="현재 스테이지 위치를 알려줘.",
        tools=RAMAN_TOOLS,
        tool_dispatch=mock_tool_dispatch,
    )
    print("\n[최종 답변]", result)
