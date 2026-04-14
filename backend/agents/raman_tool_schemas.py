"""
LLM에게 전달할 tool 스키마 정의 (Ollama tool calling 포맷)
"""

RAMAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_stage",
            "description": "스테이지를 절대 좌표(mm)로 이동한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X축 위치 (mm, 0~75.3)"},
                    "y": {"type": "number", "description": "Y축 위치 (mm, 0~50.2)"},
                    "z": {"type": "number", "description": "Z축 위치 (mm, -1.0~1.0). 생략 가능"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stage_position",
            "description": "현재 스테이지의 X, Y, Z 위치(mm)를 읽어온다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_stage_relative",
            "description": "현재 위치 기준으로 스테이지를 상대 이동한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dx": {"type": "number", "description": "X 방향 이동량 (mm)"},
                    "dy": {"type": "number", "description": "Y 방향 이동량 (mm)"},
                    "dz": {"type": "number", "description": "Z 방향 이동량 (mm)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "laser_on",
            "description": "레이저를 켠다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "laser_off",
            "description": "레이저를 끈다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_laser_power",
            "description": "레이저 출력을 설정한다. 20, 40, 60, 80, 100 중 하나.",
            "parameters": {
                "type": "object",
                "properties": {
                    "percent": {
                        "type": "integer",
                        "description": "출력 퍼센트 (20/40/60/80/100)",
                        "enum": [20, 40, 60, 80, 100],
                    }
                },
                "required": ["percent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "acquire_spectrum",
            "description": (
                "현재 위치에서 라만 스펙트럼 1회를 수집한다. "
                "레이저 ON → 출력 안정화 → CCD 촬영 → 레이저 OFF 순서를 자동으로 처리한다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "exposure": {
                        "type": "number",
                        "description": "CCD 노출 시간 (초). 기본값 0.1",
                    },
                    "power": {
                        "type": "integer",
                        "description": "레이저 출력 (%). 기본값 100",
                        "enum": [20, 40, 60, 80, 100],
                    },
                    "stabilize_sec": {
                        "type": "number",
                        "description": "레이저 ON 후 출력 안정화 대기 시간 (초). 기본값 0.5",
                    },
                },
                "required": [],
            },
        },
    },
]
