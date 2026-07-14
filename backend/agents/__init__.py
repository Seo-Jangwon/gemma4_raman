# -*- coding: utf-8 -*-
"""
backend.agents 패키지 초기화.

여기서 프로젝트 루트의 .env를 로드하는 이유:
  모든 에이전트 모듈(translator/planner/critic/...)이 import 시점에
  `_llm = ChatAnthropic(model="claude-opus-4-8")`를 생성하고, ChatAnthropic은
  환경변수 ANTHROPIC_API_KEY에서 인증 키를 읽는다. 그런데 이 변수가 안 잡혀 있으면
  "Could not resolve authentication method..." 예외가 나고 translator가 fallback으로 빠진다.

  이 __init__은 `backend.agents.X`를 import할 때 가장 먼저(그리고 반드시 한 번) 실행되므로,
  server.py로 띄우든 벤치마크/테스트가 에이전트를 직접 import하든 항상 키가 먼저 채워진다.
  → 키 주입 지점을 이 한 곳으로 통일한다(각 에이전트 파일마다 넣지 않음).

  .env는 .gitignore에 있어 커밋되지 않는다(키가 저장소에 노출되지 않도록).
  실제 키는 프로젝트 루트에 .env 파일을 만들어 넣는다:  ANTHROPIC_API_KEY=sk-ant-...
"""

from pathlib import Path

try:
    from dotenv import load_dotenv

    # backend/agents/__init__.py → parents[2] == 프로젝트 루트
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _ENV_PATH = _PROJECT_ROOT / ".env"

    # override=False: 이미 셸/OS에 ANTHROPIC_API_KEY가 있으면 그걸 우선(파일이 덮어쓰지 않음).
    load_dotenv(dotenv_path=_ENV_PATH, override=False)
except Exception:
    # python-dotenv 미설치 등으로 실패해도 패키지 import 자체는 막지 않는다.
    # (OS 환경변수로 이미 키가 설정된 경우엔 .env 없이도 정상 동작하므로 치명적이지 않음)
    pass
