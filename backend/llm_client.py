import ollama
import json
import sys
import atexit
from pathlib import Path

# hardware_manager 경로 확보
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware_manager import OLLAMA_MODEL

SYSTEM_PROMPT = """당신은 라만 분광기 제어 AI입니다.
사용 가능한 tool을 순서대로 호출해 사용자의 요청을 수행하세요.
- 스테이지 좌표 단위: mm (X: 0~75.3, Y: 0~50.2, Z: -1.0~1.0)
- 레이저를 켜기 전에 반드시 안전 여부를 확인하세요.
- 모든 작업이 끝나면 결과를 한국어로 요약해 주세요."""

# ── 클라이언트 (server.py lifespan에서 setup()으로 주입) ──
_client: "ollama.Client | None" = None


def setup(ollama_client: "ollama.Client") -> None:
    """
    server.py의 lifespan에서 HardwareManager 초기화 후 호출.
    ollama.Client 인스턴스를 모듈 전역 상태로 주입.
    """
    global _client
    _client = ollama_client


def _get_client() -> "ollama.Client":
    if _client is None:
        raise RuntimeError("llm_client.setup()이 먼저 호출되어야 합니다.")
    return _client


def generate(prompt: str) -> str:
    response = _get_client().generate(model=OLLAMA_MODEL, prompt=prompt)
    return response["response"]


def chat(messages: list) -> str:
    """messages: [{"role": "user"/"assistant", "content": "..."}]"""
    response = _get_client().chat(model=OLLAMA_MODEL, messages=messages)
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
    client = _get_client()
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
    # 단독 실행 시: HardwareManager를 직접 초기화 후 setup() 호출
    from hardware_manager import HardwareManager
    import atexit

    sys.stdout.reconfigure(encoding="utf-8")

    _hw = HardwareManager()
    _hw.startup()
    atexit.register(_hw.shutdown)
    setup(_hw.ollama)

    from backend.agents.raman_tool_schemas import RAMAN_TOOLS
    from backend.agents.raman_tools import TOOL_DISPATCH

    print("라만 분광기 AI 에이전트 시작 (종료: 'exit' 또는 Ctrl+C)")
    while True:
        try:
            user_input = input("\n명령 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in {"exit", "quit", "종료"}:
            break
        result = run_agent(
            user_message=user_input,
            tools=RAMAN_TOOLS,
            tool_dispatch=TOOL_DISPATCH,
        )
        print("\n[답변]", result)
