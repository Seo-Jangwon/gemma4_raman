# -*- coding: utf-8 -*-
"""
[역할] Ollama LLM 백엔드 설정(모델·호스트·컨텍스트·타임아웃)의 **단일 출처**.

═══════════════════════════════════════════════════════════════════════════════
모델을 바꾸려면 여기 딱 한 줄, 또는 환경변수 하나다.

    # 방법 1 — 파일을 고치지 않는다 (권장)
    RAMAN_OLLAMA_MODEL=gemma4:12b python -m backend.server
    RAMAN_OLLAMA_MODEL=gemma4:12b python -m vbench.run --agent CoALA

    # 방법 2 — 기본값 자체를 바꾼다
    아래 OLLAMA_MODEL 의 기본값 문자열 하나만 고친다.
═══════════════════════════════════════════════════════════════════════════════

[왜 별도 모듈인가 — 2026-08-09]
같은 모델명 "gemma4:31b" 와 같은 호스트가 **일곱 군데**에 각각 박혀 있었다:

    hardware_manager.py           OLLAMA_HOST / OLLAMA_MODEL   (정본 취급)
    single_agent_AILA.py          except 폴백으로 재정의
    single_agent_AILA_bench.py    except 폴백으로 재정의
    single_agent_CoALA.py         except 폴백으로 재정의
    single_agent_CoALA_bench.py   except 폴백으로 재정의
    agents/reason_log.py          except 폴백으로 재정의
    agents/knowledge.py           _ollama_host() 안에 호스트만
    agents/kb_ingest.py           캡션 호출에 모델명 직접 하드코딩

정본이 있는데도 사본이 다섯이나 생긴 이유는 **hardware_manager 가 장비 PC 의 Config.ini 와
Andor SDK 를 끌고 들어와 개발 PC 에서 import 자체가 실패**하기 때문이다. 그래서 각 파일이
try/except 로 감싸고 폴백에 같은 문자열을 다시 적었다 — 즉 하드웨어가 없는 환경에서는
**정본이 아니라 사본 다섯 개가 실제로 쓰이는** 구조였다. 모델을 바꾸려면 일곱 군데를 전부
고쳐야 하고, 하나라도 놓치면 그 경로만 다른 모델로 조용히 돌아간다. 하필 갈라지는 대상이
'두 에이전트가 같은 LLM 을 쓰는가'라서, 그게 어긋나면 AILA↔CoALA 비교 자체가 무너진다.

이 모듈은 backend.config(Config.ini)에도 장비 SDK 에도 의존하지 않으므로 **어디서든 항상
import 된다.** 따라서 try/except 폴백이 필요 없고, 사본이 생길 이유도 사라진다.
(같은 논리로 만들어진 backend/safety_limits.py 의 머리말 참고 — 그쪽은 조사량 상한이었다.)

[왜 server.py 가 아닌가]
server.py 한 곳에 두자는 것이 원래 요구였는데, 그러면 vbench 가 깨진다. vbench/run.py 는
장비 DLL 이 없어 서버를 띄울 이유가 없으므로 에이전트를 **직접 import** 한다 —
server.py 는 그 경로에서 한 번도 로드되지 않는다(vbench/episode.py 머리말 참고).
설정을 server.py 에 두면 벤치마크 A(서버 경유)와 벤치마크 B(인프로세스)가 서로 다른
모델로 돌 수 있다. 그래서 두 경로가 공통으로 import 하는 이 자리에 둔다.
"""
from __future__ import annotations

import os


def _env(*names: str, default: str) -> str:
    """환경변수를 순서대로 보고 처음 채워진 것을 쓴다. 공백만 있으면 없는 것으로 본다.

    빈 문자열을 '설정됨'으로 받으면 `RAMAN_OLLAMA_MODEL=` 로 띄웠을 때 모델명이
    빈 값이 되어, Ollama 가 404 를 돌려주는 것을 '모델이 죽었다'로 오해하게 된다.
    """
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    return default


# ── 모델 ──────────────────────────────────────────────────────────────────────
# 두 에이전트(AILA/CoALA)와 KB 캡션이 전부 이 하나를 쓴다.
# 비교 실험의 독립변수는 오케스트레이션 하나여야 하므로, 여기가 갈라지면 안 된다.
# gemma4:31b, gemma4:12b
OLLAMA_MODEL = _env("RAMAN_OLLAMA_MODEL", default="gemma4:12b")

# ── 호스트 ────────────────────────────────────────────────────────────────────
# OLLAMA_HOST 도 함께 보는 이유: ollama 공식 클라이언트가 쓰는 표준 변수명이라
# 이미 그걸로 띄워 둔 환경이 있을 수 있다. RAMAN_ 접두사 쪽이 우선한다.
OLLAMA_HOST = _env("RAMAN_OLLAMA_HOST", "OLLAMA_HOST",
                   default="http://192.168.1.15:11434")

# ── 컨텍스트 윈도우(토큰) ─────────────────────────────────────────────────────
# ChatOllama 에 명시하지 않으면 Ollama 가 호스트/모델 기본값(대개 작음)을 쓰고, 프롬프트가
# 넘치면 '경고 없이' 앞부분(시스템 프롬프트 + 도구 스키마)을 잘라내 모델이 빈 응답을 낸다
# ("Failed to generate a response." 의 실제 원인). 5×5 그리드처럼 한 턴에 도구 결과가
# 수십 개 쌓여도(≈9k 토큰) 여유가 남도록 넉넉히 잡는다.
#
# ★ 모델 크기를 바꿀 때 같이 봐야 하는 값이다. 31b → 12b 로 내리면 VRAM 이 남으므로
#   그대로 둬도 되지만, 반대로 VRAM 이 모자라 OOM 이 나면 이 값을 낮춘다.
NUM_CTX = int(_env("RAMAN_NUM_CTX", default="100000"))

# ── LLM HTTP 호출 상한(초) ────────────────────────────────────────────────────
# ChatOllama 1.1.0 에는 timeout 파라미터가 없고 밑단 ollama.Client(httpx) 기본값도
# 무제한이라, '연결은 살아 있는데 응답이 안 오는' 상태가 되면 invoke() 가 영원히 안
# 돌아온다. 스텝 카운터(_MAX_AGENT_STEPS)는 반복 횟수 가드일 뿐 벽시계 가드가 아니라
# 이 경우를 전혀 못 막는다. 정상 호출의 최악값(모델 로드 ~60s + 장문 생성)보다 넉넉히
# 크게 잡아, 멀쩡한 호출은 안 자르면서 무한 정지만 '명확한 에러'로 강등한다.
LLM_TIMEOUT_S = float(_env("RAMAN_LLM_TIMEOUT_S", default="600"))


def describe() -> str:
    """기동 로그·health 응답에 쓰는 한 줄 요약. 어떤 모델로 도는지 즉시 보이게 한다."""
    return f"{OLLAMA_MODEL} @ {OLLAMA_HOST} (num_ctx={NUM_CTX}, timeout={LLM_TIMEOUT_S:.0f}s)"
