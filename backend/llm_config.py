# -*- coding: utf-8 -*-
"""
[역할] 실행 설정의 **단일 출처** — 모델·연결·아키텍처·기억·세션 격리.

═══════════════════════════════════════════════════════════════════════════════
여기서 정하는 것 — 전부 '파일 한 줄' 또는 '환경변수 하나'다.

  ── LLM 연결 ──
    OLLAMA_MODEL           어떤 모델로 도는가            RAMAN_OLLAMA_MODEL
    OLLAMA_HOST            어디에 붙는가                 RAMAN_OLLAMA_HOST / OLLAMA_HOST
    NUM_CTX                컨텍스트 윈도우(토큰)          RAMAN_NUM_CTX
    LLM_TIMEOUT_S          LLM 호출 상한(초)             RAMAN_LLM_TIMEOUT_S

  ── 에이전트 ──
    AGENT_ARCH             CoALA 인가 AILA 인가          RAMAN_AGENT
    COALA_MEMORY_SCOPE     장기기억을 세션마다 비우는가   RAMAN_COALA_MEMORY_SCOPE
    COALA_EPISODIC_MEMORY  에피소딕 기억을 쓰는가         RAMAN_COALA_EPISODIC_MEMORY

  ── 데이터 경계 ──
    CHAT_SESSION_ISOLATED  대화가 남의 세션을 보는가      RAMAN_CHAT_ISOLATED

    # 방법 1 — 파일을 고치지 않는다 (권장)
    RAMAN_OLLAMA_MODEL=gemma4:12b python -m backend.web_controller.main
    RAMAN_AGENT=AILA python -m backend.web_controller.main         # 아키텍처 교체
    RAMAN_CHAT_ISOLATED=0 python -m backend.web_controller.main    # 세션 경계 없이

    # 방법 2 — 기본값 자체를 바꾼다
    아래 해당 상수의 default 문자열 하나만 고친다.

    python -m backend.llm_config      현재 설정 확인 + 자체 점검
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
(같은 논리로 만들어진 backend/util/safety_limits.py 의 머리말 참고 — 그쪽은 조사량 상한이었다.)

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


def _flag(*names: str, default: str) -> bool:
    """켜짐/꺼짐 설정. '0' 'false' 'no' 'off' 만 꺼짐이고 나머지는 전부 켜짐이다.

    bool 로 못 박는 이유: 이 값들은 그대로 다른 함수의 인자가 된다(예:
    begin_session(isolated=...)). 문자열 "0" 이 그냥 흘러가면 참으로 평가되어
    '껐는데 켜져 있는' 상태가 되고, 에러가 안 나서 발견이 늦다.
    """
    return _env(*names, default=default).lower() not in ("0", "false", "no", "off")


# ── 모델 ──────────────────────────────────────────────────────────────────────
# 두 에이전트(AILA/CoALA)와 KB 캡션이 전부 이 하나를 쓴다.
# 비교 실험의 독립변수는 오케스트레이션 하나여야 하므로, 여기가 갈라지면 안 된다.
# gemma4:31b, gemma4:12b
OLLAMA_MODEL = _env("RAMAN_OLLAMA_MODEL", default="gemma4:31b")

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

# ── 에이전트 아키텍처 ─────────────────────────────────────────────────────────
#: "CoALA" 또는 "AILA". 요청이 agent 를 명시하지 않으면 이 값으로 돈다.
#:
#: [왜 여기인가]
#: 기본값이 세 군데에 각각 적혀 있었고 서로 달랐다 — ExperimentRequest.agent 는 "CoALA",
#: select_agent_module 의 폴백은 "AILA", /api/agents/health 의 기본도 "AILA". 그래서
#: '아무것도 지정하지 않은' 호출이 어느 경로로 들어오느냐에 따라 다른 아키텍처로 돌았고,
#: 그게 두 아키텍처를 비교하는 실험의 독립변수 자체였다.
#:
#: 정규화는 select_agent_module 이 한다(대소문자·오타 폴백 규칙이 거기 하나로 모여 있다).
AGENT_ARCH = _env("RAMAN_AGENT", default="CoALA")

# ── CoALA 장기기억 ────────────────────────────────────────────────────────────
#: "global"  coala_memory/ 하나에 계속 축적된다. 세션을 넘어 노하우가 쌓인다(실사용 기본).
#: "session" coala_memory/sessions/<session_id>/ 로 갈라져, 새 세션은 빈 기억으로 시작한다.
#:
#: 벤치에서 global 이면 1번 문항은 경험 0건, 200번 문항은 199개가 남긴 경험으로 푸는 셈이
#: 되어 점수가 문항 순서에 의존한다. '삭제'가 아니라 '세션별 디렉터리'인 이유를 포함한
#: 자세한 설명은 backend/agents/memory/long_term_memory.py 참고.
#:
#: CHAT_SESSION_ISOLATED 와는 다른 축이다 — 그쪽은 '파일', 이쪽은 '기억'이다. 세션 격리를
#: 켜도 이 값이 global 이면 앞 세션의 경험은 그대로 조회된다.
COALA_MEMORY_SCOPE = _env("RAMAN_COALA_MEMORY_SCOPE", "RAMAN_MEMORY_SCOPE",
                          default="global").lower()

#: 에피소딕 기억(recall_experiences / record_experience) 사용 여부. 끄면 도구 2종이 액션
#: 공간에서 빠지고 프롬프트의 해당 지시문도 함께 빠져, "CoALA 에서 episodic 만 없앤"
#: ablation 이 된다. semantic(insights)은 그대로 남는다.
#:
#: 도구 쪽(long_term_memory)과 프롬프트 쪽(agents/prompts)이 각자 환경변수를 읽고 있었다.
#: 한쪽만 바뀌면 '프롬프트는 기록하라는데 도구가 없는' 상태가 되고, 모델은 없는 도구를
#: 계속 부른다. 한 곳에서 읽어 둘 다 같은 값을 보게 한다.
COALA_EPISODIC_MEMORY = _flag("RAMAN_COALA_EPISODIC_MEMORY", "RAMAN_EPISODIC_MEMORY",
                              default="1")

# ── 대화 세션 격리 ────────────────────────────────────────────────────────────
# True 면 대화 턴도 벤치와 같은 규칙으로 돈다 — **이 세션이 만든 파일만 읽는다.**
# 판정은 run_store.isolated_label() 한 곳이고, 이 값은 runtime.stream_turn 이
# begin_session(isolated=...) 으로 넘기는 인자일 뿐이다(규칙 자체는 여기 없다).
#
# 무엇이 달라지는가
#   결과 파일   results:<날짜>/<세션>/… 과 runs:<라벨>/… 은 자기 세션 것만 열린다.
#               남의 것을 열면 paths._accept 가 '어느 세션 소유인지'까지 적어 거부한다.
#   목록 도구   list_results 의 scope="all" 이 무시된다(모델이 명시해도 자기 세션만).
#               귀속 불명인 구버전 파일(날짜 폴더 직속)도 함께 배제된다.
#   첨부 파일   uploads: 는 그대로 열린다 — 사용자가 채팅에 붙인 것은 세션 소유가 아니다.
#
# 끄면(=0) 지난 세션 결과를 물어보는 사용이 가능해지는 대신, 세션 사이에 데이터가 그대로
# 보인다. 실제 실행에서 새 세션이 다른 세션의 측정 좌표·파워·노출을 그대로 읽어 갔다
# (2026-08-13). 기본값을 켜 둔 이유다.
#
# ※ CoALA 장기기억은 이 스위치와 별개다. 위의 COALA_MEMORY_SCOPE 가 따로 가른다 —
#   여기를 켜도 그쪽이 global 이면 앞 세션의 '경험'은 그대로 조회된다.
CHAT_SESSION_ISOLATED = _flag("RAMAN_CHAT_ISOLATED", default="1")


def describe() -> str:
    """기동 로그·health 응답에 쓰는 한 줄 요약. 어떤 설정으로 도는지 즉시 보이게 한다."""
    return (f"{AGENT_ARCH} · {OLLAMA_MODEL} @ {OLLAMA_HOST} "
            f"(num_ctx={NUM_CTX}, timeout={LLM_TIMEOUT_S:.0f}s, "
            f"chat_isolated={CHAT_SESSION_ISOLATED}, "
            f"memory={COALA_MEMORY_SCOPE}, episodic={COALA_EPISODIC_MEMORY})")


# ──────────────────────────────────────────────────────────────────────────────
# 자체 점검:  python -m backend.llm_config
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # _env 는 '빈 문자열 = 설정 안 함'이어야 한다. 아니면 RAMAN_OLLAMA_MODEL= 로 띄웠을 때
    # 모델명이 빈 값이 되어 Ollama 404 를 '모델이 죽었다'로 오해하게 된다.
    os.environ["_RAMAN_SELFCHECK"] = "   "
    assert _env("_RAMAN_SELFCHECK", default="fallback") == "fallback"
    os.environ["_RAMAN_SELFCHECK"] = "value"
    assert _env("_RAMAN_SELFCHECK", default="fallback") == "value"
    # 앞의 이름이 우선한다(RAMAN_ 접두가 표준 변수명을 이긴다).
    os.environ["_RAMAN_SELFCHECK_2"] = "second"
    assert _env("_RAMAN_SELFCHECK", "_RAMAN_SELFCHECK_2", default="x") == "value"
    del os.environ["_RAMAN_SELFCHECK"], os.environ["_RAMAN_SELFCHECK_2"]

    # 켜짐/꺼짐 설정은 전부 bool 이어야 한다 — 그대로 다른 함수의 인자가 되므로
    # 문자열 "0" 이 새면 참으로 평가되어 '껐는데 켜져 있는' 상태가 된다.
    assert isinstance(CHAT_SESSION_ISOLATED, bool)
    assert isinstance(COALA_EPISODIC_MEMORY, bool)
    for off in ("0", "false", "no", "off", "OFF", "False"):
        os.environ["_RAMAN_SELFCHECK_FLAG"] = off
        assert _flag("_RAMAN_SELFCHECK_FLAG", default="1") is False, off
    for on in ("1", "true", "yes", "anything"):
        os.environ["_RAMAN_SELFCHECK_FLAG"] = on
        assert _flag("_RAMAN_SELFCHECK_FLAG", default="0") is True, on
    del os.environ["_RAMAN_SELFCHECK_FLAG"]

    # 이 모듈은 backend.config(Config.ini)도 장비 SDK 도 끌지 않아야 한다. 하나라도 끌면
    # 장비 없는 PC 에서 import 가 깨지고, 그때마다 try/except 폴백 사본이 다시 태어난다.
    # 아래 소비자 검사가 에이전트를 로드하므로 그보다 **먼저** 본다.
    import sys as _sys
    _leaked = [m for m in _sys.modules
               if m.startswith(("backend.config", "backend.tools.hw_tools"))]
    assert not _leaked, f"장비 계층이 딸려 왔다: {_leaked}"

    # ── 소비자가 실제로 이 값들을 쓰는가 ──────────────────────────────────────
    # 상수만 검사하면 '여기 값은 맞는데 소비자는 옛 환경변수를 읽는' 상태를 못 잡는다.
    # 그게 정확히 이 모듈이 생긴 이유의 재발이므로, 소비자 쪽 값을 직접 대조한다.

    # 아키텍처: 입구가 셋(요청 body·health 쿼리·설정)이고 정규화는 select_agent_module
    # 한 곳이다. 명시하면 그 값, 미지정·오타면 설정 기본값으로 떨어져야 한다.
    from backend.web_controller.setups.agent_module import select_agent_module
    for given, want in (("CoALA", "CoALA"), ("coala", "CoALA"),
                        ("AILA", "AILA"), ("  aila  ", "AILA")):
        assert select_agent_module(given)[1] == want, given
    for blank in (None, "", "   ", "gpt-5"):
        assert select_agent_module(blank)[1].upper() == AGENT_ARCH.upper(), blank

    # 기억 스코프·에피소딕: 도구 계층과 프롬프트 계층이 같은 값을 봐야 한다. 갈라지면
    # '프롬프트는 기록하라는데 도구가 없는' 상태가 되고, 모델이 없는 도구를 계속 부른다.
    from backend.agents.memory import long_term_memory as _ltm
    from backend.agents import prompts as _prompts
    assert _ltm.SESSION_SCOPED is (COALA_MEMORY_SCOPE == "session")
    assert _ltm.EPISODIC_ENABLED is COALA_EPISODIC_MEMORY
    assert _prompts.EPISODIC_MEMORY is _ltm.EPISODIC_ENABLED
    # 도구가 빠지면 프롬프트의 지시문도 함께 빠져야 ablation 이 성립한다.
    _coala_prompt = _prompts.build_system_prompt("CoALA")
    assert ("record_experience" in _coala_prompt) is COALA_EPISODIC_MEMORY

    # 세션 격리: runtime 이 이 값을 begin_session 에 그대로 넘기는지.
    from backend.agents.runtime import runtime as _rt
    assert _rt.CHAT_SESSION_ISOLATED is CHAT_SESSION_ISOLATED

    print(describe())
    print("통과: 빈 값 폴백 · 이름 우선순위 · 켜짐꺼짐 bool · 장비 의존 없음 · "
          "소비자 정합(아키텍처 8 · 기억 4 · 격리)")
