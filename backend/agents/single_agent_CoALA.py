# -*- coding: utf-8 -*-
"""
SingleAgent (CoALA) — 라만 분광기를 gemma4(Ollama) 하나로 제어하되, Sumers et al.
2024 "Cognitive Architectures for Language Agents"(CoALA)의 아키텍처로 구조화한
단일 에이전트.

[이 파일은 왜 존재하는가 — single_agent_AILA와의 관계]
single_agent_AILA.py는 stateless ReAct baseline이다: 모델이 한 번에 emit한 모든
tool_call을 그대로 실행하고, 장기기억이 없으며, 오케스트레이션 로직이 사실상 없다.
이 파일은 그 baseline과 "완전히 독립적으로" 동작하는 두 번째 에이전트로, 비교
실험의 독립변수는 오직 오케스트레이션 아키텍처(ReAct vs CoALA)다. 따라서:
  - LLM 계층은 AILA와 100% 동일하다: ChatOllama(gemma4)만 쓰고 다른 LLM은 쓰지 않는다.
  - 공유 자원(RAMAN_TOOLS/TOOL_DISPATCH/knowledge)만 공유하고 AILA 코드는 import하지
    않는다. 얇은 유틸(_slim, _msg_text, dose 차단기 등)은 결합을 피하려 여기에 다시 둔다.

[CoALA 매핑 — 논문 §4]
  · Working memory  : WorkingMemory dataclass — LLM 호출 간 지속되는 자료구조
                      (goal / retrieved / observations / messages).
  · Semantic memory : 읽기 search_kb(knowledge.py), 쓰기 record_insight → insights.json.
  · Episodic memory : 읽기 recall_experiences, 쓰기 record_experience → experiences.json.
  · Procedural memory: 이 파일의 코드 + TOOL_DISPATCH + LLM 가중치(설계자가 초기화).
  · Action space    : external=grounding(하드웨어), internal=retrieval/reasoning/learning.
  · Decision cycle  : planning(propose·evaluate·select) → execution → observe (논문 §4.6).

═══════════════════════════════════════════════════════════════════════════════
[2026-07-22 수정 — planning stage 분리 (논문 Figure 4B / §4.6 정합)]

  이전 구현의 결함: retrieval·learning·grounding을 하나의 후보 리스트로 묶어
  evaluate/select가 나란히 점수 매겨 하나를 골랐다. 이는 논문에서 "planning의
  '수단'"이어야 할 retrieval을 "실행 대상"으로 격하시킨 것이었다.

  논문 §4.6 / Figure 4B의 올바른 구조:
    · planning stage에서 reasoning·retrieval을 '써서'(수단) 후보를 propose·
      evaluate·select 한다. 이 서브스테이지들은 interleave/iterate 하며 정보를
      쌓을 수 있다.
    · planning의 '산출물'은 하나의 grounding 또는 learning 액션이다.
    · execution stage가 그 선택된 grounding/learning을 실행하고 관찰한다.

  따라서 이 파일은 사이클 내부에 'planning 내부 루프'를 둔다:
    - 모델이 retrieval을 제안 → 즉시 실행해 working memory에 정보를 쌓고,
      사이클을 닫지 않고 다시 propose (정보 수집 반복).
    - 모델이 grounding/learning을 제안 → planning 종료. 그것들'만' evaluate·
      select·execute 대상이 된다.
    - retrieval과 grounding이 한 응답에 섞이면 retrieval만 먼저 실행하고
      grounding은 버려 재제안하게 한다 — 정보를 다 모은 '뒤'에 실행을 결정하도록.
      (라만 비가역성: 레이저는 planning이 끝난 뒤의 판단으로만 발사된다.)

  즉 retrieval은 이제 사이클을 닫지 않는 planning 도구이고, grounding/learning
  만이 propose→evaluate→select→execute의 대상이다.
═══════════════════════════════════════════════════════════════════════════════

[공개 API — server.py가 의존하는 계약 (AILA와 동일 시그니처)]
  ALL_TOOLS            : 바인딩된 도구 스키마 리스트 (/api/agents/health가 len() 호출)
  stream_experiment()  : SSE용 이벤트 제너레이터 (/api/experiment/stream)
  run_experiment()     : 동기 1회 실행 (/api/experiment/run, 벤치마크용)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from backend.agents.knowledge import search_kb
from backend.hw_tools.raman_tool_schemas import RAMAN_TOOLS

# hardware_manager는 장비 PC의 Config.ini를 읽으므로 개발 PC에서 import가 실패할 수
# 있다. 모델명/호스트는 실패해도 기본값으로 굴러가야 하므로 try로 감싼다.
# (AILA와 동일한 폴백 — LLM 백엔드를 완전히 일치시키기 위함.)
try:
    from backend.hardware_manager import OLLAMA_HOST, OLLAMA_MODEL
except Exception:
    OLLAMA_HOST = "http://192.168.1.16:11434"
    OLLAMA_MODEL = "gemma4:31b"

# ── 사이클/계획 예산 ──────────────────────────────────────────────────────────
_MAX_CYCLES = 20            # 최대 의사결정 사이클 수 (= commit 행동 실행 횟수)
_MAX_PLANNING_STEPS = 6     # 한 사이클 내 planning(정보수집) 라운드 상한
_SOFT_PLAN_LIMIT = 4        # 이 라운드부터 "이제 실행/보고서로" 진행 문구를 강화
_MAX_AGENT_STEPS = 40       # 턴 전체 propose() 호출 총량 가드(무한 루프 방지)

# 조사량 하드 상한 (대화 한 턴 기준). AILA와 동일한 물리적 회로차단기 —
# "판단"이 아니라 폭주 방지용 최후 안전장치.
_MAX_DOSE_MJ_PER_TURN = 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# 장기기억 저장소 (Episodic / Semantic) — 디스크 JSON
# ══════════════════════════════════════════════════════════════════════════════
#
# CoALA의 핵심은 "자기 생성 콘텐츠를 읽고 쓸 수 있는" 장기기억이다(논문 §4.5, §6).
# 세션이 끝나도 남아 다음 실험에서 조회되는 "노하우"를 여기 JSON에 축적한다.
# 단일 사용자 로컬 도구라 파일 락 없이 read-modify-write append로 충분하다.

_MEMORY_DIR = Path(__file__).resolve().parent / "coala_memory"
_EPISODIC_PATH = _MEMORY_DIR / "experiences.json"   # 실험 경험(에피소드)
_SEMANTIC_PATH = _MEMORY_DIR / "insights.json"      # 경험에서 증류한 일반 지식


def _load_json_list(path: Path) -> list[dict]:
    """저장소 파일을 리스트로 로드한다. 없거나 깨졌으면 빈 리스트."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _append_json_list(path: Path, item: dict) -> None:
    """저장소 파일에 항목 하나를 append한다(디렉터리 없으면 생성)."""
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    items = _load_json_list(path)
    items.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _match_score(query: str, entry: dict, fields: tuple[str, ...]) -> int:
    """질의 토큰이 항목의 지정 필드들에 몇 개나 부분매칭되는지 — 조회 랭킹용.

    임베딩 검색을 쓰지 않는 이유: episodic 저장소는 JSON 선택이므로 의존성 없는
    키워드 매칭으로 충분하고, 개발 PC에서도 임베딩 서버 없이 검사할 수 있다.
    """
    hay = " ".join(str(entry.get(fld, "")) for fld in fields).lower()
    toks = [t for t in re.split(r"\s+", query.lower().strip()) if t]
    return sum(1 for t in toks if t in hay)


def _recall_experiences(args: dict) -> dict:
    """recall_experiences 도구 구현 — episodic memory 읽기.

    과거 실험 경험 중 질의(시편/키워드)와 관련된 것을 top_k개 반환한다.
    비어 있으면 에러가 아니라 정상적인 "아직 축적된 경험 없음"으로 답한다 —
    그래야 모델이 재시도 루프에 빠지지 않고 스스로 판단하고 나중에 기록한다.
    """
    query = str(args.get("query", "")).strip()
    top_k = int(args.get("top_k", 3) or 3)
    episodes = _load_json_list(_EPISODIC_PATH)
    if not episodes:
        return {"ok": True, "results": [],
                "note": "축적된 과거 실험 경험이 아직 없습니다. 이번 측정을 마친 뒤 "
                        "record_experience로 경험을 남기면 다음 실험에서 조회됩니다."}
    if not query:
        # 질의가 없으면 최근 경험을 반환한다(그래도 유용한 컨텍스트).
        ranked = list(reversed(episodes))
    else:
        scored = [(e, _match_score(query, e, ("sample", "outcome", "lesson", "metrics")))
                  for e in episodes]
        ranked = [e for e, s in sorted(scored, key=lambda x: x[1], reverse=True) if s > 0]
        if not ranked:
            return {"ok": True, "results": [],
                    "note": f"'{query}'와 관련된 과거 경험을 찾지 못했습니다. 스스로 판단하세요."}
    return {"ok": True, "results": ranked[:top_k]}


def _record_experience(ctx: dict, args: dict) -> dict:
    """record_experience 도구 구현 — episodic memory 쓰기(학습 액션).

    실험 한 건의 경험을 experiences.json에 append한다. CoALA에서 학습은 스케줄이
    아니라 에이전트가 의사결정 사이클에서 "고르는" 액션이다 — 모델이 이 도구를
    호출하기로 결정했을 때만 기록된다.
    """
    sample = str(args.get("sample", "")).strip()
    if not sample:
        return {"ok": False, "error": "sample(시편 종류)은 필수입니다."}
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": ctx.get("session_id", ""),
        "sample": sample,
        "params": args.get("params", {}),
        "outcome": str(args.get("outcome", "")).strip(),
        "metrics": str(args.get("metrics", "")).strip(),
        "lesson": str(args.get("lesson", "")).strip(),
    }
    try:
        _append_json_list(_EPISODIC_PATH, entry)
    except OSError as e:
        return {"ok": False, "error": f"경험 저장 실패: {e}"}
    ctx["learned"] = True
    return {"ok": True, "recorded": entry["id"], "sample": sample}


def _record_insight(ctx: dict, args: dict) -> dict:
    """record_insight 도구 구현 — semantic memory 쓰기(학습 액션).

    경험에서 증류한 일반화 지식(예: "그래핀은 30%↑ 파워에서 포화")을 insights.json에
    남긴다. episodic(개별 사건)과 달리 semantic(재사용 가능한 규칙)을 구분해 저장한다.
    """
    topic = str(args.get("topic", "")).strip()
    insight = str(args.get("insight", "")).strip()
    if not topic or not insight:
        return {"ok": False, "error": "topic과 insight 모두 필요합니다."}
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topic": topic,
        "insight": insight,
    }
    try:
        _append_json_list(_SEMANTIC_PATH, entry)
    except OSError as e:
        return {"ok": False, "error": f"통찰 저장 실패: {e}"}
    ctx["learned"] = True
    return {"ok": True, "recorded": entry["id"], "topic": topic}


def _recall_insights(args: dict) -> dict:
    """recall_insights 도구 구현 — semantic memory(자기 생성분) 읽기.

    [이 도구가 추가된 이유 — write-only 갭 보완]
    record_insight로 쓴 일반화 지식을 '다시 읽는' 경로가 없으면 CoALA의 lifelong
    learning 루프(§4.5: self-generated 지식을 later episode에서 재사용)가 semantic
    쪽에서 반만 닫힌다. search_knowledge_base(큐레이션 KB) 와는 별개로, 에이전트가
    스스로 남긴 insights.json을 조회한다.
    """
    query = str(args.get("query", "")).strip()
    top_k = int(args.get("top_k", 3) or 3)
    insights = _load_json_list(_SEMANTIC_PATH)
    if not insights:
        return {"ok": True, "results": [],
                "note": "축적된 일반화 지식(insight)이 아직 없습니다. record_insight로 "
                        "남기면 이후 실험에서 조회됩니다."}
    if not query:
        ranked = list(reversed(insights))
    else:
        scored = [(e, _match_score(query, e, ("topic", "insight"))) for e in insights]
        ranked = [e for e, s in sorted(scored, key=lambda x: x[1], reverse=True) if s > 0]
        if not ranked:
            return {"ok": True, "results": [],
                    "note": f"'{query}'와 관련된 일반화 지식을 찾지 못했습니다. 스스로 판단하세요."}
    return {"ok": True, "results": ranked[:top_k]}


def _search_knowledge_base(args: dict) -> dict:
    """search_knowledge_base 도구 구현 — semantic memory 읽기(큐레이션 KB).

    다중/단일 에이전트와 "같은 파일을 같은 알고리즘으로" 검색해야 비교가 공정하므로
    검색 로직은 여기에 복제하지 않고 backend.agents.knowledge.search_kb에 위임한다.
    """
    query = str(args.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "query가 비어 있습니다. 시편/재료 키워드를 주세요."}
    hits = search_kb(query, top_k=3)
    if not hits:
        return {"ok": True, "results": [],
                "note": f"'{query}'에 해당하는 프로토콜이 지식베이스에 없습니다. "
                        "직접 판단해 파라미터를 정하고 보고서에 그 사실을 밝히세요."}
    return {"ok": True, "results": hits}


# ══════════════════════════════════════════════════════════════════════════════
# 도구 스키마 — internal action들도 tool로 노출한다 (CoALA §4.1: 모든 액션이 action space)
# ══════════════════════════════════════════════════════════════════════════════

_KB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "[semantic 기억 조회 · planning] 시편 종류(graphene, cell, exosome, silicon, "
            "CNT 등)로 라만 측정 프로토콜과 권장 파라미터(레이저 파워 %, 노출 시간 초, 주요 "
            "피크 위치)를 검색한다. 측정 파라미터를 정하기 전에 호출하라. 레이저를 켜지 않으므로 "
            "무해하며, 이 호출은 사이클을 끝내지 않는다(정보 수집 후 계속 계획한다)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "검색할 시편/재료 키워드. 예: 'graphene', 'silicon'."},
            },
            "required": ["query"],
        },
    },
}

_RECALL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recall_experiences",
        "description": (
            "[episodic 기억 조회 · planning] 과거에 수행한 유사 실험의 경험(사용한 파라미터, "
            "결과, 교훈)을 조회한다. 새 측정을 계획하기 전에 호출하면 지난 노하우를 재사용할 수 "
            "있다. 레이저를 켜지 않으므로 무해하며, 이 호출은 사이클을 끝내지 않는다. 아직 경험이 "
            "없으면 빈 결과가 온다(정상)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "시편/상황 키워드. 예: 'graphene 포화', 'exosome SNR'."},
                "top_k": {"type": "integer", "description": "가져올 경험 수(기본 3)."},
            },
            "required": ["query"],
        },
    },
}

_RECALL_INSIGHT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recall_insights",
        "description": (
            "[semantic 기억 조회 · planning] 과거에 record_insight로 남긴 일반화 지식(재사용 "
            "가능한 규칙/원리)을 조회한다. 큐레이션 KB(search_knowledge_base)와 달리 '내가 "
            "직접 축적한' 통찰을 본다. 레이저를 켜지 않으며 사이클을 끝내지 않는다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "주제 키워드. 예: 'graphene 광손상', '785nm 형광'."},
                "top_k": {"type": "integer", "description": "가져올 통찰 수(기본 3)."},
            },
            "required": ["query"],
        },
    },
}

_RECORD_EXP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "record_experience",
        "description": (
            "[episodic 기억 기록/학습 · execution] 방금 수행한 실험의 경험을 장기기억에 남긴다. "
            "측정을 마치고 보고서를 쓰기 직전에 호출하라 — 이 기록은 다음 세션의 "
            "recall_experiences로 조회되어 노하우로 재사용된다. 하드웨어를 만지지 않는다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sample": {"type": "string", "description": "시편 종류(필수). 예: 'graphene'."},
                "params": {"type": "object",
                           "description": "사용한 측정 파라미터. 예: {'power':20,'exposure':2.0}."},
                "outcome": {"type": "string", "description": "결과 요약. 예: 'G/2D 밴드 양호'."},
                "metrics": {"type": "string", "description": "정량 지표. 예: 'SNR 8.3, 포화 없음'."},
                "lesson": {"type": "string",
                           "description": "다음에 쓸 교훈. 예: '30%↑ 파워는 포화 위험'."},
            },
            "required": ["sample"],
        },
    },
}

_RECORD_INSIGHT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "record_insight",
        "description": (
            "[semantic 기억 기록/학습 · execution] 여러 경험에서 일반화한 재사용 가능한 지식을 "
            "남긴다. 개별 실험(record_experience)과 달리 '규칙/원리'를 저장한다. 하드웨어를 "
            "만지지 않는다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "주제. 예: 'graphene 광손상'."},
                "insight": {"type": "string",
                            "description": "일반화 지식. 예: '532nm에서 30%↑는 결함 유발'."},
            },
            "required": ["topic", "insight"],
        },
    },
}

# 모델에 바인딩되는 도구 전체. RAMAN_TOOLS(하드웨어 41종) + internal 액션 5종.
# server.py의 /api/agents/health가 len(ALL_TOOLS)를 읽는다.
_INTERNAL_TOOLS = [
    _KB_TOOL_SCHEMA,
    _RECALL_TOOL_SCHEMA,
    _RECALL_INSIGHT_TOOL_SCHEMA,
    _RECORD_EXP_TOOL_SCHEMA,
    _RECORD_INSIGHT_TOOL_SCHEMA,
]
ALL_TOOLS = RAMAN_TOOLS + _INTERNAL_TOOLS

# 어느 도구가 어느 CoALA 액션 범주인지 — 실행 디스패치와 planning/execution 분리에 쓴다.
#   retrieval  : planning 도구. 사이클을 닫지 않고 working memory에 정보를 쌓는다.
#   learning   : execution(commit) 액션. grounding과 함께 propose→evaluate→select 대상.
#   그 외(하드웨어): grounding = execution(commit) 액션.
_INTERNAL_RETRIEVAL = {"search_knowledge_base", "recall_experiences", "recall_insights"}
_INTERNAL_LEARNING = {"record_experience", "record_insight"}



# ══════════════════════════════════════════════════════════════════════════════
# 시스템 프롬프트 — CoALA 의사결정 사이클을 명시적으로 지시
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
당신은 라만 분광기를 제어하는 단일 AI 에이전트이며, CoALA(Cognitive Architectures for
Language Agents) 구조로 동작합니다. 당신은 명시적인 작업기억과 장기기억을 가지며,
아래의 의사결정 사이클(계획 → 실행)에 따라 행동합니다.

[기억 구조]
- 작업기억(working memory): 현재 목표, 조회한 지식, 최근 관측이 프롬프트에 함께 제공됩니다.
- semantic 기억: search_knowledge_base(큐레이션 프로토콜)·recall_insights(내가 남긴 통찰)로
  읽고, record_insight로 씁니다(재사용 규칙).
- episodic 기억: recall_experiences로 과거 실험을 읽고, record_experience로 씁니다(노하우).

[의사결정 사이클 — 계획(planning)과 실행(execution)을 구분하라]
당신의 행동은 두 종류입니다. 이 둘의 성격이 완전히 다릅니다.

  · 계획 행동(정보 수집): search_knowledge_base, recall_experiences, recall_insights.
    - 이 행동들은 레이저를 켜지 않고 '근거를 모으는' 행동입니다. 필요한 만큼 여러 번
      연달아 호출해 작업기억을 충분히 채우세요. 이 행동은 아무것도 되돌릴 수 없게 만들지
      않으므로 마음껏 쓰되, 같은 것을 반복 조회하지는 마세요.

  · 실행 행동(commit): 하드웨어 도구(스테이지 이동, 레이저, acquire_spectrum, 카메라 등)와
    기록 도구(record_experience, record_insight).
    - 이 행동들은 실제로 세상이나 장기기억을 바꿉니다. 특히 acquire_spectrum은 시편에
      레이저를 조사하며, 광손상은 되돌릴 수 없습니다.
    - 반드시 '한 번에 하나만' 실행합니다. 실행이 필요하면 지금 상황에서 가장 가치 있는
      단 하나의 실행 행동을 고르세요. 그 결과(관측)를 본 뒤 다음 사이클에서 다음 실행을
      다시 정합니다.

  · 원칙: 실행 행동을 고르기 '전에' 필요한 계획 행동(정보 수집)을 먼저 끝내세요. 레이저는
    근거를 충분히 모은 뒤에만 발사되어야 합니다.

  · finish: 더 호출할 도구가 없으면 도구 없이 한국어 최종 보고서를 작성하면 이번 턴이 끝납니다.

[측정 절차 — 사이클을 거치며 스스로 진행]
1. 시편/기판/목표 위치를 모르면 레이저를 켜기 전에 사용자에게 먼저 묻습니다.
2. 시편을 알면 먼저 recall_experiences로 과거 노하우를, recall_insights로 축적된 규칙을,
   search_knowledge_base로 프로토콜을 조회해 작업기억을 채웁니다(계획 행동). 파라미터는
   추측하지 말고 이 근거로 정합니다.
3. 위치를 모르면 analyze_microscope_image로 화면을 보고 좌표를 읽어 move_to_pixel로 이동,
   필요하면 run_autofocus로 초점을 맞춥니다(실행 행동, 한 번에 하나).
4. 목표 신호를 기판 배경과 구분하려면 목표와 '완전히 동일한' 파워·노출로 빈 영역을 한 번
   측정해 기준선을 잡고, 원래 위치로 되돌아옵니다.
5. SNR·포화·신호대배경비를 평가해 필요하면 위치/파라미터를 바꿔 재측정합니다. 단 1~2회
   재시도해도 개선이 없으면 기존 데이터로 진행하고 한계를 보고서에 명시합니다.
6. 측정을 마치면 보고서를 쓰기 전에 record_experience로 이번 경험을(가능하면 record_insight로
   일반화 지식도) 장기기억에 남깁니다 — 이것이 다음 실험을 위한 노하우 축적입니다.
7. 마지막으로 도구 호출 없이 한국어 최종 보고서를 작성합니다:
   실험 목적 / 측정 조건(조정 내역 포함) / 결과 요약(목표 vs 배경) / 스펙트럼의 물리적 분석
   (피크 위치·귀속, SNR, 포화; 배경과 겹치는 피크는 기판 유래로 제외) / 도메인 해석과 결론 /
   진행 중 문제와 대처 / 결론 및 권고.

[안전 규칙]
- 도구가 오류나 안전 차단을 반환하면 우회하지 말고 상황을 그대로 사용자에게 보고합니다.
- 모르는 것을 추측하지 않습니다 — 도구로 확인하거나 사용자에게 묻습니다.
- 인사/잡담/능력 질문은 도구 없이 즉시 한국어로 답합니다."""


# ══════════════════════════════════════════════════════════════════════════════
# LLM 로딩 — AILA와 동일하게 ChatOllama만 사용 (다른 LLM 금지)
# ══════════════════════════════════════════════════════════════════════════════

# tool-bound(propose/execute용)과 plain(evaluate/reasoning용) 두 Runnable을 캐시한다.
# 둘 다 같은 ChatOllama(같은 모델·호스트)라 LLM 백엔드는 AILA와 완전히 동일하다.
_llm_tools_cache = None
_llm_plain_cache = None


def _get_llm_tools():
    """ALL_TOOLS를 바인딩한 ChatOllama Runnable(실패 시 None) — propose/execute용."""
    global _llm_tools_cache
    if _llm_tools_cache is not None:
        return _llm_tools_cache
    try:
        from langchain_ollama import ChatOllama
        _llm_tools_cache = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
        ).bind_tools(ALL_TOOLS)
    except Exception:
        return None
    return _llm_tools_cache


def _get_llm_plain():
    """도구 없이 텍스트만 내는 ChatOllama Runnable(실패 시 None) — evaluate/reasoning용.

    evaluate 단계는 도구를 '호출'하는 게 아니라 후보들을 '점수화'하는 순수 추론이라
    tool 바인딩이 없는 편이 JSON 출력을 방해받지 않아 안정적이다. 모델·호스트는 동일.
    """
    global _llm_plain_cache
    if _llm_plain_cache is not None:
        return _llm_plain_cache
    try:
        from langchain_ollama import ChatOllama
        _llm_plain_cache = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
    except Exception:
        return None
    return _llm_plain_cache


def _get_dispatch():
    """raman_tools.TOOL_DISPATCH 로드. 하드웨어 모듈이 없으면 None.

    ImportError만이 아니라 Exception 전체를 잡는 이유: raman_tools가 import하는
    config.py는 장비 PC의 Config.ini를 읽는데, 개발 PC에서는 NoSectionError가 난다.
    """
    try:
        from backend.hw_tools.raman_tools import TOOL_DISPATCH
        return TOOL_DISPATCH
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 유틸 (AILA와 의도적으로 동일 — 결합을 피하려 여기 자체 구현)
# ══════════════════════════════════════════════════════════════════════════════

_INJECTED_IMAGE = "_injected_image"     # 이미지 주입용 HumanMessage 표시


def _slim(result):
    """대용량 배열(스펙트럼 원본 등)은 컨텍스트에 싣지 않는다 — 토큰 낭비/혼란 방지.
    길이 32 초과 리스트를 버린다. 기억 조회 결과는 짧아 걸리지 않는다."""
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if not (isinstance(v, list) and len(v) > 32)}
    return result


def _msg_text(msg) -> str:
    """AIMessage content에서 순수 텍스트를 뽑는다(str 또는 콘텐츠 블록 리스트 모두 처리)."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def _call_tool(ctx: dict, name: str, args: dict) -> dict:
    """단일 도구 실행 관문. internal 액션은 여기서, grounding은 TOOL_DISPATCH로.
    유일한 비-LLM 판단은 조사량 회로차단기뿐(AILA와 동일)."""

    # ── internal actions (하드웨어 dispatch와 무관하게 항상 처리) ──────────────
    if name == "search_knowledge_base":
        return _search_knowledge_base(args)
    if name == "recall_experiences":
        return _recall_experiences(args)
    if name == "recall_insights":
        return _recall_insights(args)
    if name == "record_experience":
        return _record_experience(ctx, args)
    if name == "record_insight":
        return _record_insight(ctx, args)

    # ── external grounding actions ────────────────────────────────────────────
    dispatch = ctx["dispatch"]
    if dispatch is None:
        return {"ok": False, "error": "하드웨어가 연결되어 있지 않습니다."}
    fn = dispatch.get(name)
    if fn is None:
        return {"ok": False, "error": f"알 수 없는 도구: {name}"}

    if name == "acquire_spectrum":
        power = float(args.get("power", 40.0))
        exposure = float(args.get("exposure", 0.2))
        dose_inc = power * exposure * 0.01
        if ctx["dose"] + dose_inc > _MAX_DOSE_MJ_PER_TURN:
            return {"ok": False,
                    "error": (f"안전 차단: 이번 턴 누적 조사량이 상한"
                              f"({_MAX_DOSE_MJ_PER_TURN}mJ)을 초과합니다. "
                              "측정을 마무리하거나 새 요청으로 다시 시작하세요.")}
        try:
            result = fn(dict(args))
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if isinstance(result, dict) and result.get("ok"):
            ctx["dose"] += dose_inc   # 실패한 조사는 누계에 넣지 않는다
        return result

    try:
        return fn(dict(args))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# Working memory (CoALA §4.1) — LLM 호출 간 지속되는 자료구조
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkingMemory:
    """현재 의사결정 사이클의 활성 상태. 매 propose 프롬프트에 직렬화되어 주입된다.

    messages는 tool-call ↔ ToolMessage 쌍을 담는 LangChain 메시지 로그로, 모델이
    관측을 '보는' 실제 컨텍스트다. 나머지 필드(goal/retrieved/observations)는 사람이
    읽는 요약이자 planning 프롬프트의 상단 컨텍스트가 된다.
    """
    goal: str = ""
    retrieved: list = field(default_factory=list)      # 조회한 semantic/episodic 지식 요약
    observations: list = field(default_factory=list)   # 최근 grounding 관측 요약
    messages: list = field(default_factory=list)       # LangChain 메시지 로그

    def render(self) -> str:
        """작업기억을 planning 프롬프트에 넣을 텍스트 블록으로 직렬화한다."""
        lines = ["[작업기억]"]
        lines.append(f"- 현재 목표: {self.goal or '(아직 명확한 측정 목표 없음)'}")
        if self.retrieved:
            lines.append("- 조회한 지식(기억):")
            lines += [f"    · {s}" for s in self.retrieved[-6:]]
        else:
            lines.append("- 조회한 지식(기억): 아직 없음 (필요하면 계획 행동으로 조회하세요)")
        if self.observations:
            lines.append("- 최근 관측:")
            lines += [f"    · {s}" for s in self.observations[-6:]]
        return "\n".join(lines)


def _update_working_memory(wm: WorkingMemory, name: str, args: dict,
                           result: dict, action: str) -> None:
    """실행 결과를 작업기억 요약(retrieved/observations)에 반영한다."""
    if not isinstance(result, dict):
        return
    if action == "retrieval":
        hits = result.get("results", [])
        if hits:
            for h in hits[:3]:
                title = h.get("title") or h.get("sample") or h.get("topic") or "?"
                rp = h.get("recommended_params") or h.get("params")
                extra = f" 권장 {rp}" if rp else ""
                wm.retrieved.append(f"[{name}] {title}{extra}")
        else:
            note = result.get("note", "해당 없음")
            wm.retrieved.append(f"[{name}] {note}")
    else:
        ok = result.get("ok", True)
        if not ok:
            wm.observations.append(f"⚠️ {name} 실패: {result.get('error', '')}")
        elif name == "acquire_spectrum":
            wm.observations.append(f"스펙트럼 획득 (max {result.get('max_intensity', 0)} ADU)")
        elif action == "learning":
            wm.observations.append(f"기억 기록: {name} → {result.get('sample') or result.get('topic', '')}")
        else:
            wm.observations.append(f"{name} 실행 완료")


# ══════════════════════════════════════════════════════════════════════════════
# 의사결정 사이클 요소: 후보 분류 / 단일 실행 / propose / evaluate+select
# ══════════════════════════════════════════════════════════════════════════════

def _candidate_label(tc: dict) -> str:
    """tool_call 후보를 사람이 읽는 한 줄로."""
    args = tc.get("args") or {}
    compact = ", ".join(f"{k}={v}" for k, v in list(args.items())[:4])
    return f"{tc.get('name', '?')}({compact})"


def _partition_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """제안된 tool_call들을 (planning 행동, commit 행동)으로 나눈다.

    · planning = retrieval(정보 수집). 사이클을 닫지 않고 working memory를 채운다.
    · commit   = grounding(하드웨어) + learning(기록). propose→evaluate→select→execute
                 의 대상이며, 하나가 실행되면 사이클이 닫힌다.

    이 분리가 이 파일의 핵심이다(논문 §4.6): retrieval은 planning의 '수단'이지
    실행 대상이 아니다.
    """
    planning: list[dict] = []
    commit: list[dict] = []
    for tc in candidates:
        name = tc.get("name", "")
        if name in _INTERNAL_RETRIEVAL:
            planning.append(tc)
        else:
            # 하드웨어(grounding) 또는 record_*(learning) → 실행(commit) 액션
            commit.append(tc)
    return planning, commit


def _execute_one(ctx: dict, name: str, args: dict, tool_call_id: str) -> dict:
    """tool_call 하나를 실제로 실행한다(dose 가드 포함). 메시지/이벤트는 호출자가 처리.

    반환: {name, args, result, action, img_b64, question, tool_call_id}
    """
    ctx["tool_call_order"].append(name)
    raw = _call_tool(ctx, name, args)
    result = _slim(raw) if isinstance(raw, dict) else raw

    # 이미지 반환 도구는 base64를 tool 메시지에 싣지 않고 별도 user 이미지 블록으로 전달.
    img_b64 = result.pop("image_base64", None) if isinstance(result, dict) else None
    question = result.pop("question", None) if isinstance(result, dict) else None

    if name in _INTERNAL_RETRIEVAL:
        action = "retrieval"
    elif name in _INTERNAL_LEARNING:
        action = "learning"
    else:
        action = "grounding"

    return {"name": name, "args": args, "result": result, "action": action,
            "img_b64": img_b64, "question": question, "tool_call_id": tool_call_id}


def _plan_progress_note(round_no: int, retrieval_count: int, repeated: bool) -> str:
    """planning 진행 상황을 모델에게 알리는 문구 — 매 propose마다 새로 계산된다.

    [이 문구가 왜 필요한가]
    모델이 정보 수집(retrieval)만 반복하다 계획 예산을 소진하면 측정 한 번 못 하고
    턴이 끝난다. 하드 넛지 한 번 대신, "지금 이 사이클에서 몇 번째 계획 중인지"를
    매번 알려줘 모델이 스스로 '이제 실행/보고서로 넘어갈 때'를 판단하게 한다.
    영구 히스토리(wm.messages)가 아니라 매 호출 새로 만드는 SystemMessage에만 실려
    세션에 남지 않는다(다음 턴 오염 없음).
    """
    left = _MAX_PLANNING_STEPS - round_no
    note = (f"[계획 진행] 이번 사이클에서 {round_no}번째 계획(정보 수집) 라운드입니다"
            f"(연속 조회 {retrieval_count}회, 남은 계획 라운드 {left}회).")
    if repeated:
        note += (" 방금 직전과 같은 조회를 반복하고 있습니다 — 같은 조회를 또 하지 말고, "
                 "지금까지 모은 근거로 실행 행동(grounding/learning) 하나를 고르거나 보고서를 쓰세요.")
    elif round_no >= _SOFT_PLAN_LIMIT:
        note += " 정보는 충분해 보입니다. 이제 실행 행동 하나를 고르거나 최종 보고서를 작성하세요."
    if left <= 0:
        note += " 이번이 마지막 계획 라운드입니다 — 다음엔 반드시 실행하거나 보고서를 써야 합니다."
    return note


def _propose(llm_tools, wm: WorkingMemory, plan_note: str = "") -> AIMessage:
    """Propose — 작업기억(+계획 진행 문구)을 주입해 다음 행동 후보(tool_calls)를 생성한다.

    반환된 AIMessage.tool_calls가 후보 목록이다(0개면 finish = 최종 답변).
    시스템 프롬프트/작업기억/진행 문구는 세션 히스토리에 남기지 않고 매 호출마다 새로
    붙인다(중복·오염 방지).
    """
    content = SYSTEM_PROMPT + "\n\n" + wm.render()
    if plan_note:
        content += "\n\n" + plan_note
    return llm_tools.invoke([SystemMessage(content=content)] + wm.messages)


def _evaluate_and_select(llm_plain, wm: WorkingMemory, candidates: list[dict],
                         dose: float) -> tuple[dict, dict]:
    """Evaluate + Select — '실행(commit) 후보'가 여럿이면 점수화 후 argmax, 하나면 그대로.

    여기 들어오는 candidates는 grounding/learning(실행 대상)뿐이다 — retrieval은
    planning 단계에서 이미 처리되어 이 지점에 오지 않는다.

    Returns (선택된 tool_call, {"scores": [...], "reason": str}) — 두 번째는 UI/로그용.

    후보 1개면 LLM 호출 없이 통과시킨다(논문 §4.6: 단순 상황은 평가 생략). 여럿이면
    plain LLM으로 유용성·안전(dose/광손상)·근거성을 0~1 점수화한다. 파싱 실패 시 첫
    후보로 폴백해 사이클이 멈추지 않게 한다.
    """
    if len(candidates) == 1:
        return candidates[0], {"scores": [1.0], "reason": "실행 후보 단일 — 평가 생략"}

    listing = "\n".join(
        f"{i}. {_candidate_label(c)}" for i, c in enumerate(candidates)
    )
    dose_note = (f"현재 누적 조사량 {dose:.1f}/{_MAX_DOSE_MJ_PER_TURN}mJ. "
                 "레이저 조사(acquire_spectrum)는 이 예산과 광손상(비가역)을 함께 고려하라.")
    prompt = (
        "당신은 라만 실험 에이전트의 '실행 행동 평가자'입니다. 아래 작업기억과 실행 후보들을 "
        "보고, 지금 실행하기에 각 후보가 얼마나 가치 있는지 0.0~1.0으로 점수화하세요. "
        "유용성(목표 진전), 안전성(광손상/조사량), 근거성(작업기억·기억에 부합)을 함께 보세요.\n\n"
        f"{wm.render()}\n\n{dose_note}\n\n[실행 후보]\n{listing}\n\n"
        "반드시 다음 JSON만 출력하세요(설명 금지): "
        '{"scores": [숫자, ...], "reason": "가장 높은 후보를 고른 이유 한 문장"}'
    )
    scores = None
    reason = ""
    try:
        resp = llm_plain.invoke([HumanMessage(content=prompt)])
        text = _msg_text(resp)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            raw = data.get("scores", [])
            scores = [float(x) for x in raw][:len(candidates)]
            reason = str(data.get("reason", ""))
    except Exception:
        scores = None

    if not scores or len(scores) != len(candidates):
        # 평가 실패 → 첫 후보로 폴백(사이클 진행 보장).
        return candidates[0], {"scores": [], "reason": "평가 파싱 실패 — 첫 후보 선택"}

    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_idx], {"scores": scores, "reason": reason}


# ══════════════════════════════════════════════════════════════════════════════
# Planning stage — retrieval을 반복 호출해 근거를 쌓고, commit 후보가 나오면 종료
# ══════════════════════════════════════════════════════════════════════════════

def _planning_stage(llm_tools, ctx: dict, wm: WorkingMemory,
                    propose_state: list, outcome: dict) -> Iterator[dict]:
    """한 사이클의 planning 단계(논문 §4.6 / Figure 4B).

    reasoning(모델의 내재 CoT)·retrieval을 써서 근거를 쌓다가, 실행(grounding/learning)
    후보가 제안되면 planning을 끝낸다. retrieval 호출은 사이클을 닫지 않고 working
    memory만 갱신하며, 필요한 만큼 반복(interleave)된다.

    결과는 `outcome` dict에 기록한다:
      outcome["kind"] ∈ {"commit", "finish", "stuck"}
      outcome["commit"]      = [tool_call, ...]   (kind=="commit"일 때, 실행 후보들)
      outcome["commit_text"] = str                (commit 제안에 딸린 모델 텍스트)
      outcome["final_text"]  = str                (kind=="finish"일 때, 최종 보고서)

    propose_state는 [남은_propose_예산]을 공유하는 가변 리스트다(턴 전체 총량 가드).
    """
    # 이번 사이클에서 실행한 retrieval의 시그니처(name+args) 이력 — 반복 조회 감지용.
    prior_sigs: list = []

    for step in range(_MAX_PLANNING_STEPS):
        if propose_state[0] <= 0:
            outcome["kind"] = "stuck"
            return

        # 지금까지의 조회 이력으로 "같은 조회를 또 하고 있나"를 판단해 진행 문구를 만든다.
        # (직전 조회가 그 이전에도 나왔던 것이면 반복으로 본다.)
        repeated = bool(prior_sigs) and prior_sigs[-1] in prior_sigs[:-1]
        plan_note = _plan_progress_note(step + 1, len(prior_sigs), repeated)

        try:
            ai_msg = _propose(llm_tools, wm, plan_note)
        except Exception as e:
            outcome["kind"] = "error"
            outcome["detail"] = f"LLM 호출 실패(propose): {type(e).__name__}: {e}"
            return
        propose_state[0] -= 1

        candidates = list(ai_msg.tool_calls or [])

        # 후보 없음 = finish. 모델이 최종 보고서를 낸 것 → 이번 턴 종료.
        if not candidates:
            wm.messages.append(ai_msg)
            outcome["kind"] = "finish"
            outcome["final_text"] = _msg_text(ai_msg).strip() or "응답을 생성하지 못했습니다."
            return

        planning_actions, commit_actions = _partition_candidates(candidates)

        # ── retrieval(planning)이 있으면: 그것만 실행하고 계속 계획 ──────────────
        if planning_actions:
            # 이 응답에 섞여 나온 commit_actions는 '버린다' — 근거를 더 모은 뒤
            # 다음 propose에서 재제안하게 해서, 실행은 항상 최신 작업기억으로 결정.
            planning_ai = AIMessage(content=_msg_text(ai_msg), tool_calls=planning_actions)
            wm.messages.append(planning_ai)

            # 이번 라운드 조회 시그니처를 이력에 남긴다(다음 라운드 반복 감지용).
            prior_sigs.append(tuple(sorted(
                f"{c.get('name')}:{json.dumps(c.get('args') or {}, sort_keys=True, ensure_ascii=False)}"
                for c in planning_actions)))

            yield {"type": "phase", "phase": "plan",
                   "message": "정보 수집(retrieval): "
                              + " / ".join(_candidate_label(c) for c in planning_actions)}

            for tc in planning_actions:
                ex = _execute_one(ctx, tc["name"], dict(tc.get("args") or {}), tc.get("id") or "")
                yield {"type": "tool", "name": ex["name"], "args": ex["args"],
                       "result": ex["result"], "action": ex["action"]}
                wm.messages.append(ToolMessage(
                    content=json.dumps(ex["result"], ensure_ascii=False, default=str),
                    tool_call_id=ex["tool_call_id"],
                ))
                _update_working_memory(wm, ex["name"], ex["args"], ex["result"], ex["action"])
                # retrieval은 이미지를 반환하지 않지만, 방어적으로 주입 경로를 둔다.
                if ex["img_b64"]:
                    wm.messages.append(HumanMessage(
                        content=[
                            {"type": "text", "text": ex["question"] or "이미지:"},
                            {"type": "image_url",
                             "image_url": f"data:image/png;base64,{ex['img_b64']}"},
                        ],
                        additional_kwargs={_INJECTED_IMAGE: True},
                    ))
            continue  # 같은 사이클 안에서 다시 propose (planning 반복)

        # ── retrieval이 없고 commit 후보만 있으면: planning 종료 ────────────────
        outcome["kind"] = "commit"
        outcome["commit"] = commit_actions
        outcome["commit_text"] = _msg_text(ai_msg)
        return

    # planning 예산 소진 — 실행/종료 결정을 못 내림.
    outcome["kind"] = "stuck"


# ══════════════════════════════════════════════════════════════════════════════
# 에이전트 루프 (CoALA decision cycle): [planning] → evaluate → select → execute
# ══════════════════════════════════════════════════════════════════════════════

def run_stream(llm_tools, llm_plain, history: list, user_message: str,
               session_id: str = "") -> Iterator[dict]:
    """CoALA 의사결정 사이클 루프(논문 §4.6 / Figure 4B).

    각 사이클:
      1) planning stage — reasoning·retrieval로 근거를 쌓는다(retrieval은 사이클을
         닫지 않고 working memory만 채운다). 실행 후보(grounding/learning)가 나오면 종료.
      2) evaluate + select — 실행 후보가 여럿이면 점수화 후 하나 선택.
      3) execute + observe — 선택된 grounding/learning '하나'만 실행하고 관측을 남긴다.
    그리고 다음 사이클로. 실행 후보만 담은 AIMessage([chosen])를 히스토리에 남겨
    tool_call↔ToolMessage 짝을 항상 유효하게 유지한다.

    yield 이벤트:
      {"type": "phase", "phase": str, "message": str}          사이클 단계 진행
      {"type": "tool",  "name": str, "args": dict, "result": dict, "action": str}
      {"type": "error", "detail": str}
      {"type": "final", "text": str, "ctx": dict, "messages": list}
    """
    if llm_tools is None or llm_plain is None:
        yield {"type": "error",
               "detail": "Ollama LLM이 연결되지 않았습니다. "
                         "(langchain-ollama 설치 및 Ollama 서버 상태를 확인하세요)"}
        return

    ctx = {"dispatch": _get_dispatch(), "dose": 0.0, "session_id": session_id,
           "tool_call_order": [], "learned": False}
    wm = WorkingMemory(messages=list(history) + [HumanMessage(content=user_message)])
    wm.goal = user_message.strip()

    propose_state = [_MAX_AGENT_STEPS]   # 턴 전체 propose 호출 총량 가드(가변 공유)

    for _cycle in range(_MAX_CYCLES):
        # ── 1) PLANNING STAGE ────────────────────────────────────────────────
        outcome: dict = {}
        yield from _planning_stage(llm_tools, ctx, wm, propose_state, outcome)

        kind = outcome.get("kind")

        if kind == "error":
            yield {"type": "error", "detail": outcome.get("detail", "planning 실패")}
            return

        if kind == "finish":
            yield {"type": "final", "text": outcome["final_text"],
                   "ctx": ctx, "messages": wm.messages}
            return

        if kind != "commit":
            # stuck — 계획만 반복하고 실행/종료를 못 정함. 안전하게 턴을 닫는다.
            yield {"type": "final",
                   "text": "계획 단계 예산에 도달해 이번 턴을 마칩니다. "
                           "진행 상황을 확인하고 다시 요청해 주세요.",
                   "ctx": ctx, "messages": wm.messages}
            return

        commit_candidates = outcome["commit"]

        # ── 2) EVALUATE + SELECT (grounding/learning 후보에 한해서) ────────────
        if len(commit_candidates) > 1:
            yield {"type": "phase", "phase": "evaluate",
                   "message": f"실행 후보 {len(commit_candidates)}개 평가 중…"}
        try:
            chosen, meta = _evaluate_and_select(llm_plain, wm, commit_candidates, ctx["dose"])
        except Exception as e:
            yield {"type": "error",
                   "detail": f"LLM 호출 실패(evaluate): {type(e).__name__}: {e}"}
            return

        yield {"type": "phase", "phase": "select",
               "message": f"선택 → {_candidate_label(chosen)}"
                          + (f"  ({meta['reason']})" if meta.get("reason") else "")}

        # ── 3) EXECUTE + OBSERVE ─────────────────────────────────────────────
        # 선택된 실행 후보 '하나만' 담은 AIMessage를 히스토리에 남긴다(나머지는 버림).
        selected_ai = AIMessage(content=outcome.get("commit_text", ""), tool_calls=[chosen])
        wm.messages.append(selected_ai)

        ex = _execute_one(ctx, chosen["name"], dict(chosen.get("args") or {}),
                          chosen.get("id") or "")
        yield {"type": "tool", "name": ex["name"], "args": ex["args"],
               "result": ex["result"], "action": ex["action"]}

        wm.messages.append(ToolMessage(
            content=json.dumps(ex["result"], ensure_ascii=False, default=str),
            tool_call_id=ex["tool_call_id"],
        ))
        _update_working_memory(wm, ex["name"], ex["args"], ex["result"], ex["action"])

        if ex["img_b64"]:
            wm.messages.append(HumanMessage(
                content=[
                    {"type": "text", "text": ex["question"] or "현미경 카메라 이미지:"},
                    {"type": "image_url",
                     "image_url": f"data:image/png;base64,{ex['img_b64']}"},
                ],
                additional_kwargs={_INJECTED_IMAGE: True},
            ))
        # 다음 사이클로 (관측을 반영해 다시 planning)

    yield {"type": "final",
           "text": f"최대 사이클({_MAX_CYCLES}회)에 도달해 중단했습니다. "
                   "진행 상황을 확인하고 다시 요청해 주세요.",
           "ctx": ctx, "messages": wm.messages}


# ══════════════════════════════════════════════════════════════════════════════
# 세션 관리 + SSE 진입점 (공개 API — 서버 계약)
# ══════════════════════════════════════════════════════════════════════════════

# 세션별 LangChain 메시지 히스토리(대화 기억). cross-session 노하우는 별도로 JSON에 축적.
_SESSIONS: dict[str, list] = {}
_HISTORY_MAX_TURNS = 20


def _is_user_turn(msg) -> bool:
    """이 메시지가 '사람이 친 사용자 턴'인지 — 트리밍 경계 판정용.

    이미지 주입용 HumanMessage는 사용자 턴이 아니므로 제외한다. 포함시키면 한 턴이
    여러 턴으로 세어져 히스토리가 과하게 잘린다. (계획 진행 문구는 히스토리에 남기지
    않고 매 propose의 SystemMessage에만 실으므로 여기서 걸러낼 대상 자체가 없다.)
    """
    if not isinstance(msg, HumanMessage):
        return False
    return not msg.additional_kwargs.get(_INJECTED_IMAGE, False)


def _trim_history(messages: list) -> list:
    """마지막 _HISTORY_MAX_TURNS번째 사용자 메시지 지점부터 보존 —
    도구호출↔응답 쌍이 중간에서 끊기지 않도록 '사용자 턴' 단위로 자른다."""
    user_idx = [i for i, m in enumerate(messages) if _is_user_turn(m)]
    if len(user_idx) <= _HISTORY_MAX_TURNS:
        return messages
    start = user_idx[-_HISTORY_MAX_TURNS]
    return messages[start:]


def _describe_tool(name: str, args: dict, result: dict, action: str) -> str:
    """tool 호출 1건을 사람이 읽는 한 줄로 — SSE "node" 이벤트 메시지."""
    ok = result.get("ok", True) if isinstance(result, dict) else True
    if not ok:
        return f"⚠️ {name} 실패: {result.get('error', '')}"
    if name == "acquire_spectrum":
        return f"📈 스펙트럼 획득 (max {result.get('max_intensity', 0):.0f} ADU)"
    if name in ("move_stage", "move_stage_relative", "move_to_pixel"):
        pos = result.get("position", {})
        return f"🧭 이동 → ({pos.get('x', '?')}, {pos.get('y', '?')})"
    if name == "analyze_microscope_image":
        return "👁️ 현미경 이미지 확인"
    if name == "run_autofocus":
        return "🔬 오토포커스 완료"
    if name == "apply_background_subtraction":
        return "🧹 형광 배경 제거 적용"
    if name == "search_knowledge_base":
        hits = result.get("results", [])
        titles = ", ".join(h.get("title", "?") for h in hits) if hits else "해당 없음"
        return f"📚 지식(semantic) 조회 → {titles}"
    if name == "recall_experiences":
        hits = result.get("results", [])
        return f"🧠 경험(episodic) 조회 → {len(hits)}건"
    if name == "recall_insights":
        hits = result.get("results", [])
        return f"🔎 통찰(semantic) 조회 → {len(hits)}건"
    if name == "record_experience":
        return f"💾 경험 기록(episodic) → {result.get('sample', '')}"
    if name == "record_insight":
        return f"💡 통찰 기록(semantic) → {result.get('topic', '')}"
    return f"🔧 {name} 호출"


def stream_experiment(user_message: str, session_id: str = "") -> Iterator[dict]:
    """CoALA 에이전트를 이벤트 제너레이터로 실행한다 (프론트엔드 SSE용).

    yield하는 이벤트 — 모두 "type"과 "session_id"를 포함(서버 계약과 동일 vocabulary):
      {"type": "node",  "node": str, "message": str}   사이클 단계/도구 호출 진행상황
      {"type": "chat",  "reply": str}                  측정 없이 끝난 턴
      {"type": "done",  "final_report": str}           측정을 포함한 턴 완료
      {"type": "error", "detail": str}
    """
    sid = session_id or str(uuid.uuid4())

    def ev(d: dict) -> dict:
        d["session_id"] = sid
        return d

    try:
        llm_tools = _get_llm_tools()
        llm_plain = _get_llm_plain()
        history = _SESSIONS.get(sid, [])

        final_text = None
        final_ctx = None
        final_messages = history

        for event in run_stream(llm_tools, llm_plain, history, user_message, session_id=sid):
            etype = event["type"]
            if etype == "phase":
                yield ev({"type": "node", "node": f"cycle:{event['phase']}",
                          "message": event["message"]})
            elif etype == "tool":
                yield ev({"type": "node", "node": event["name"],
                          "message": _describe_tool(event["name"], event["args"],
                                                    event["result"], event["action"])})
            elif etype == "error":
                yield ev({"type": "error", "detail": event["detail"]})
                return
            elif etype == "final":
                final_text = event["text"]
                final_ctx = event["ctx"]
                final_messages = event["messages"]

        if final_ctx is None:
            yield ev({"type": "error", "detail": "에이전트가 응답을 생성하지 못했습니다."})
            return

        _SESSIONS[sid] = _trim_history(final_messages)

        # 측정(레이저 조사)이 실제로 있었는지로 "실험 보고서" vs "일반 대화"를 가른다.
        used_measurement = "acquire_spectrum" in final_ctx.get("tool_call_order", [])
        if used_measurement:
            yield ev({"type": "done", "final_report": final_text})
        else:
            yield ev({"type": "chat", "reply": final_text})

    except Exception as e:
        yield ev({"type": "error", "detail": str(e)})


def run_experiment(user_message: str, session_id: str = "") -> dict:
    """동기 1회 실행 — 벤치마크/레거시용 (세션 히스토리 없이 매번 새로 시작)."""
    llm_tools = _get_llm_tools()
    llm_plain = _get_llm_plain()
    final_text = ""
    for event in run_stream(llm_tools, llm_plain, [], user_message, session_id=session_id):
        if event["type"] == "final":
            final_text = event["text"]
        elif event["type"] == "error":
            final_text = f"[오류] {event['detail']}"
    return {"final_report": final_text}