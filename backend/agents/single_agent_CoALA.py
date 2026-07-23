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

from backend.agents.detail_log import new_turn
from backend.agents.knowledge import search_kb
from backend.hw_tools.raman_tool_schemas import RAMAN_TOOLS
from backend.spectrum_store import spectrum_event

# hardware_manager는 장비 PC의 Config.ini를 읽으므로 개발 PC에서 import가 실패할 수
# 있다. 모델명/호스트는 실패해도 기본값으로 굴러가야 하므로 try로 감싼다.
# (AILA와 동일한 폴백 — LLM 백엔드를 완전히 일치시키기 위함.)
try:
    from backend.hardware_manager import OLLAMA_HOST, OLLAMA_MODEL
except Exception:
    OLLAMA_HOST = "http://192.168.1.15:11434"
    OLLAMA_MODEL = "gemma4:31b"

# ── 사이클/계획 예산 ──────────────────────────────────────────────────────────
_MAX_CYCLES = 150            # 최대 의사결정 사이클 수 (= commit 행동 실행 횟수)
_MAX_PLANNING_STEPS = 6     # 한 사이클 내 planning(정보수집) 라운드 상한
_SOFT_PLAN_LIMIT = 4        # 이 라운드부터 "이제 실행/보고서로" 진행 문구를 강화
_MAX_AGENT_STEPS = 150       # 턴 전체 propose() 호출 총량 가드(무한 루프 방지)

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
                "note": "No past experiments accumulated yet. After finishing this measurement, "
                        "leave one with record_experience and it will be retrievable in future experiments."}
    if not query:
        # 질의가 없으면 최근 경험을 반환한다(그래도 유용한 컨텍스트).
        ranked = list(reversed(episodes))
    else:
        scored = [(e, _match_score(query, e, ("sample", "outcome", "lesson", "metrics")))
                  for e in episodes]
        ranked = [e for e, s in sorted(scored, key=lambda x: x[1], reverse=True) if s > 0]
        if not ranked:
            return {"ok": True, "results": [],
                    "note": f"No past experience related to '{query}'. Decide on your own."}
    return {"ok": True, "results": ranked[:top_k]}


def _record_experience(ctx: dict, args: dict) -> dict:
    """record_experience 도구 구현 — episodic memory 쓰기(학습 액션).

    실험 한 건의 경험을 experiences.json에 append한다. CoALA에서 학습은 스케줄이
    아니라 에이전트가 의사결정 사이클에서 "고르는" 액션이다 — 모델이 이 도구를
    호출하기로 결정했을 때만 기록된다.
    """
    sample = str(args.get("sample", "")).strip()
    if not sample:
        return {"ok": False, "error": "sample (sample type) is required."}
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
        return {"ok": False, "error": f"Failed to save experience: {e}"}
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
        return {"ok": False, "error": "Both topic and insight are required."}
    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topic": topic,
        "insight": insight,
    }
    try:
        _append_json_list(_SEMANTIC_PATH, entry)
    except OSError as e:
        return {"ok": False, "error": f"Failed to save insight: {e}"}
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
                "note": "No generalized knowledge (insights) accumulated yet. Leave one with "
                        "record_insight and it will be retrievable in future experiments."}
    if not query:
        ranked = list(reversed(insights))
    else:
        scored = [(e, _match_score(query, e, ("topic", "insight"))) for e in insights]
        ranked = [e for e, s in sorted(scored, key=lambda x: x[1], reverse=True) if s > 0]
        if not ranked:
            return {"ok": True, "results": [],
                    "note": f"No generalized knowledge related to '{query}'. Decide on your own."}
    return {"ok": True, "results": ranked[:top_k]}


def _search_knowledge_base(args: dict) -> dict:
    """search_knowledge_base 도구 구현 — semantic memory 읽기(큐레이션 KB).

    다중/단일 에이전트와 "같은 파일을 같은 알고리즘으로" 검색해야 비교가 공정하므로
    검색 로직은 여기에 복제하지 않고 backend.agents.knowledge.search_kb에 위임한다.
    """
    query = str(args.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "query is empty. Provide a sample/material keyword."}
    hits = search_kb(query, top_k=3)
    if not hits:
        return {"ok": True, "results": [],
                "note": f"No protocol matching '{query}' in the knowledge base. "
                        "Decide the parameters yourself and state that in the report."}
    return {"ok": True, "results": hits}


# ══════════════════════════════════════════════════════════════════════════════
# 도구 스키마 — internal action들도 tool로 노출한다 (CoALA §4.1: 모든 액션이 action space)
# ══════════════════════════════════════════════════════════════════════════════

_KB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "[semantic memory read - planning] Search the Raman measurement protocol and recommended "
            "parameters (laser power %, exposure time in seconds, main peak positions) by sample type "
            "(graphene, cell, exosome, silicon, CNT, etc.). Call it before deciding measurement parameters. "
            "It does not turn on the laser, so it is harmless, and this call does not end the cycle "
            "(keep planning after gathering information)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Sample/material keyword to search. e.g. 'graphene', 'silicon'."},
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
            "[episodic memory read - planning] Query experiences from past similar experiments (parameters "
            "used, results, lessons). Calling it before planning a new measurement lets you reuse past "
            "know-how. It does not turn on the laser, so it is harmless, and this call does not end the cycle. "
            "If there is no experience yet, an empty result is returned (normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Sample/situation keyword. e.g. 'graphene saturation', 'exosome SNR'."},
                "top_k": {"type": "integer", "description": "Number of experiences to fetch (default 3)."},
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
            "[semantic memory read - planning] Query generalized knowledge (reusable rules/principles) "
            "previously left with record_insight. Unlike the curated KB (search_knowledge_base), this views "
            "insights 'I accumulated myself'. It does not turn on the laser and does not end the cycle."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Topic keyword. e.g. 'graphene photodamage', '785nm fluorescence'."},
                "top_k": {"type": "integer", "description": "Number of insights to fetch (default 3)."},
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
            "[episodic memory write/learning - execution] Record the experience of the experiment just "
            "performed into long-term memory. Call it after finishing the measurement and just before writing "
            "the report - this record is retrieved via recall_experiences in the next session and reused as "
            "know-how. It does not touch hardware."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sample": {"type": "string", "description": "Sample type (required). e.g. 'graphene'."},
                "params": {"type": "object",
                           "description": "Measurement parameters used. e.g. {'power':20,'exposure':2.0}."},
                "outcome": {"type": "string", "description": "Result summary. e.g. 'G/2D bands good'."},
                "metrics": {"type": "string", "description": "Quantitative metrics. e.g. 'SNR 8.3, no saturation'."},
                "lesson": {"type": "string",
                           "description": "Lesson for next time. e.g. 'power above 30% risks saturation'."},
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
            "[semantic memory write/learning - execution] Leave reusable knowledge generalized from several "
            "experiences. Unlike an individual experiment (record_experience), this stores 'rules/principles'. "
            "It does not touch hardware."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic. e.g. 'graphene photodamage'."},
                "insight": {"type": "string",
                            "description": "Generalized knowledge. e.g. 'above 30% at 532nm induces defects'."},
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
You are a single AI agent that controls a Raman spectrometer, operating under the CoALA (Cognitive
Architectures for Language Agents) architecture. You have explicit working memory and long-term
memory, and you act according to the decision cycle below (planning -> execution).

[Memory structure]
- Working memory: the current goal, retrieved knowledge, and recent observations are provided in the prompt.
- Semantic memory: read with search_knowledge_base (curated protocols) and recall_insights (insights I left),
  and write with record_insight (reusable rules).
- Episodic memory: read past experiments with recall_experiences and write with record_experience (know-how).

[Decision cycle - distinguish planning from execution]
Your actions are of two kinds, and their nature is completely different.

  · Planning actions (information gathering): search_knowledge_base, recall_experiences, recall_insights.
    - These do not turn on the laser; they 'gather evidence'. Call them several times in a row as needed
      to fill working memory sufficiently. They make nothing irreversible, so use them freely, but do not
      repeat the same lookup.

  · Execution actions (commit): hardware tools (stage move, laser, acquire_spectrum, camera, etc.) and
    recording tools (record_experience, record_insight).
    - These actually change the world or long-term memory. In particular, acquire_spectrum irradiates the
      sample with the laser, and photodamage is irreversible.
    - Always execute 'only one at a time'. When execution is needed, choose the single most valuable
      execution action in the current situation. After seeing its result (observation), decide the next
      execution in the following cycle.

  · Principle: finish the necessary planning actions (information gathering) 'before' choosing an execution
    action. The laser should only be fired after enough evidence is gathered.

  · finish: if there are no more tools to call, write the final report in English without tools and this turn ends.

[Measurement procedure - proceed on your own through the cycles]
1. If you do not know the sample/substrate/target location, ask the user first before turning on the laser.
2. Once you know the sample, first query past know-how with recall_experiences, accumulated rules with
   recall_insights, and protocols with search_knowledge_base to fill working memory (planning actions).
   Do not guess parameters - set them from this evidence.
3. If you do not know the location, view the image with analyze_microscope_image, read the coordinates,
   move with move_to_pixel, and focus with run_autofocus if needed (execution actions, one at a time).
4. To distinguish the target signal from the substrate background, measure a blank area once with the
   'exactly identical' power and exposure as the target to set a baseline, then return to the original position.
5. Evaluate SNR, saturation, and signal-to-background ratio, and re-measure by changing position/parameters
   if needed. But if 1-2 retries show no improvement, proceed with the existing data and state the limitation
   in the report.
6. When the measurement is done, before writing the report, record this experience with record_experience
   (and, if possible, generalized knowledge with record_insight) into long-term memory - this is how know-how
   accumulates for the next experiment.
7. Finally, write the final report in English without tool calls:
   Experiment objective / measurement conditions (including adjustments) / results summary (target vs background) /
   physical analysis of the spectrum (peak positions and assignments, SNR, saturation; peaks overlapping the
   background are excluded as substrate-derived) / domain interpretation and conclusion / problems and handling
   during the process / conclusion and recommendations.

[Safety rules]
- If a tool returns an error or safety block, do not bypass it; report the situation to the user as is.
- Do not guess what you do not know - verify with a tool or ask the user.
- Answer greetings/small talk/capability questions immediately in English without tools.
- Stage coordinate units: mm (X: 0-75.3, Y: 0-50.2, Z: -1.0-1.0; origin at x=37.8759, y=25.24805, z n/a)
"""


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
        return {"ok": False, "error": "Hardware is not connected."}
    fn = dispatch.get(name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    if name == "acquire_spectrum":
        power = float(args.get("power", 40.0))
        exposure = float(args.get("exposure", 0.2))
        dose_inc = power * exposure * 0.01
        if ctx["dose"] + dose_inc > _MAX_DOSE_MJ_PER_TURN:
            return {"ok": False,
                    "error": (f"Safety block: this turn's cumulative dose would exceed the limit "
                              f"({_MAX_DOSE_MJ_PER_TURN} mJ). "
                              "Wrap up the measurement or start again with a new request.")}
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
        lines = ["[Working memory]"]
        lines.append(f"- Current goal: {self.goal or '(no clear measurement goal yet)'}")
        if self.retrieved:
            lines.append("- Retrieved knowledge (memory):")
            lines += [f"    · {s}" for s in self.retrieved[-6:]]
        else:
            lines.append("- Retrieved knowledge (memory): none yet (query with a planning action if needed)")
        if self.observations:
            lines.append("- Recent observations:")
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
                extra = f" recommended {rp}" if rp else ""
                wm.retrieved.append(f"[{name}] {title}{extra}")
        else:
            note = result.get("note", "no match")
            wm.retrieved.append(f"[{name}] {note}")
    else:
        ok = result.get("ok", True)
        if not ok:
            wm.observations.append(f"⚠️ {name} failed: {result.get('error', '')}")
        elif name == "acquire_spectrum":
            wm.observations.append(f"Spectrum acquired (max {result.get('max_intensity', 0)} ADU)")
        elif action == "learning":
            wm.observations.append(f"Memory recorded: {name} → {result.get('sample') or result.get('topic', '')}")
        else:
            wm.observations.append(f"{name} executed")


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
    note = (f"[Planning progress] This is planning (information gathering) round {round_no} in this cycle "
            f"({retrieval_count} consecutive lookups, {left} planning rounds left).")
    if repeated:
        note += (" You are repeating the same lookup as just before - do not repeat the same lookup; "
                 "with the evidence gathered so far, choose one execution action (grounding/learning) or write the report.")
    elif round_no >= _SOFT_PLAN_LIMIT:
        note += " Information looks sufficient. Now choose one execution action or write the final report."
    if left <= 0:
        note += " This is the last planning round - next you must execute or write the report."
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
        return candidates[0], {"scores": [1.0], "reason": "single execution candidate - evaluation skipped"}

    listing = "\n".join(
        f"{i}. {_candidate_label(c)}" for i, c in enumerate(candidates)
    )
    dose_note = (f"Current cumulative dose {dose:.1f}/{_MAX_DOSE_MJ_PER_TURN} mJ. "
                 "For laser irradiation (acquire_spectrum), consider this budget together with photodamage (irreversible).")
    prompt = (
        "You are the 'execution-action evaluator' of a Raman experiment agent. Looking at the working memory "
        "and execution candidates below, score how valuable each candidate is to execute right now from 0.0 to 1.0. "
        "Consider usefulness (progress toward the goal), safety (photodamage/dose), and groundedness "
        "(consistency with working memory/memory).\n\n"
        f"{wm.render()}\n\n{dose_note}\n\n[Execution candidates]\n{listing}\n\n"
        "Output only the following JSON (no explanation): "
        '{"scores": [number, ...], "reason": "one sentence on why the highest candidate was chosen"}'
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
        return candidates[0], {"scores": [], "reason": "evaluation parse failed - first candidate chosen"}

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
            outcome["detail"] = f"LLM call failed (propose): {type(e).__name__}: {e}"
            return
        propose_state[0] -= 1

        candidates = list(ai_msg.tool_calls or [])

        # 후보 없음 = finish. 모델이 최종 보고서를 낸 것 → 이번 턴 종료.
        if not candidates:
            wm.messages.append(ai_msg)
            outcome["kind"] = "finish"
            outcome["final_text"] = _msg_text(ai_msg).strip() or "Failed to generate a response."
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
                   "message": "Information gathering (retrieval): "
                              + " / ".join(_candidate_label(c) for c in planning_actions),
                   "candidates": [_candidate_label(c) for c in planning_actions]}

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
                            {"type": "text", "text": ex["question"] or "Image:"},
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
               "detail": "Ollama LLM is not connected. "
                         "(Check that langchain-ollama is installed and the Ollama server is running)"}
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
            yield {"type": "error", "detail": outcome.get("detail", "planning failed")}
            return

        if kind == "finish":
            yield {"type": "final", "text": outcome["final_text"],
                   "ctx": ctx, "messages": wm.messages}
            return

        if kind != "commit":
            # stuck — 계획만 반복하고 실행/종료를 못 정함. 안전하게 턴을 닫는다.
            yield {"type": "final",
                   "text": "Reached the planning-stage budget, ending this turn. "
                           "Please check the progress and request again.",
                   "ctx": ctx, "messages": wm.messages}
            return

        commit_candidates = outcome["commit"]

        # ── 2) EVALUATE + SELECT (grounding/learning 후보에 한해서) ────────────
        commit_labels = [_candidate_label(c) for c in commit_candidates]
        if len(commit_candidates) > 1:
            yield {"type": "phase", "phase": "evaluate",
                   "message": f"Evaluating {len(commit_candidates)} execution candidates…",
                   "candidates": commit_labels}
        try:
            chosen, meta = _evaluate_and_select(llm_plain, wm, commit_candidates, ctx["dose"])
        except Exception as e:
            yield {"type": "error",
                   "detail": f"LLM call failed (evaluate): {type(e).__name__}: {e}"}
            return

        # select 이벤트에 propose→evaluate→select의 전 과정을 실어 벤치마크 로그
        # ("planning evaluation process")가 후보/점수/이유/선택을 다 담게 한다.
        yield {"type": "phase", "phase": "select",
               "message": f"Selected → {_candidate_label(chosen)}"
                          + (f"  ({meta['reason']})" if meta.get("reason") else ""),
               "candidates": commit_labels,
               "scores": meta.get("scores") or None,
               "reason": meta.get("reason") or None,
               "chosen": _candidate_label(chosen)}

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
                    {"type": "text", "text": ex["question"] or "Microscope camera image:"},
                    {"type": "image_url",
                     "image_url": f"data:image/png;base64,{ex['img_b64']}"},
                ],
                additional_kwargs={_INJECTED_IMAGE: True},
            ))
        # 다음 사이클로 (관측을 반영해 다시 planning)

    yield {"type": "final",
           "text": f"Stopped after reaching the maximum number of cycles ({_MAX_CYCLES}). "
                   "Please check the progress and request again.",
           "ctx": ctx, "messages": wm.messages}


# ══════════════════════════════════════════════════════════════════════════════
# 세션 관리 + SSE 진입점 (공개 API — 서버 계약)
# ══════════════════════════════════════════════════════════════════════════════

# 세션별 LangChain 메시지 히스토리(대화 기억). cross-session 노하우는 별도로 JSON에 축적.
_SESSIONS: dict[str, list] = {}
_HISTORY_MAX_TURNS = 100


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
        return f"⚠️ {name} failed: {result.get('error', '')}"
    if name == "acquire_spectrum":
        return f"📈 Spectrum acquired (max {result.get('max_intensity', 0):.0f} ADU)"
    if name in ("move_stage", "move_stage_relative", "move_to_pixel"):
        pos = result.get("position", {})
        return f"🧭 Moved → ({pos.get('x', '?')}, {pos.get('y', '?')})"
    if name == "analyze_microscope_image":
        return "👁️ Microscope image checked"
    if name == "run_autofocus":
        return "🔬 Autofocus complete"
    if name == "apply_background_subtraction":
        return "🧹 Fluorescence background subtraction applied"
    if name == "search_knowledge_base":
        hits = result.get("results", [])
        titles = ", ".join(h.get("title", "?") for h in hits) if hits else "no match"
        return f"📚 Knowledge (semantic) lookup → {titles}"
    if name == "recall_experiences":
        hits = result.get("results", [])
        return f"🧠 Experience (episodic) lookup → {len(hits)} item(s)"
    if name == "recall_insights":
        hits = result.get("results", [])
        return f"🔎 Insight (semantic) lookup → {len(hits)} item(s)"
    if name == "record_experience":
        return f"💾 Experience recorded (episodic) → {result.get('sample', '')}"
    if name == "record_insight":
        return f"💡 Insight recorded (semantic) → {result.get('topic', '')}"
    return f"🔧 {name} called"


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

    # 벤치마크 로그: resolved sid를 넘겨 세션별로 파일이 갈리게 한다. run_stream 소비 전에
    # 만들어 phase/tool 이벤트를 전부 관측한다(로깅 실패는 detail_log가 내부에서 삼킨다).
    turn = new_turn("CoALA", sid, user_message)

    try:
        llm_tools = _get_llm_tools()
        llm_plain = _get_llm_plain()
        history = _SESSIONS.get(sid, [])

        final_text = None
        final_ctx = None
        final_messages = history

        for event in run_stream(llm_tools, llm_plain, history, user_message, session_id=sid):
            turn.observe(event)
            etype = event["type"]
            if etype == "phase":
                yield ev({"type": "node", "node": f"cycle:{event['phase']}",
                          "message": event["message"]})
            elif etype == "tool":
                yield ev({"type": "node", "node": event["name"],
                          "message": _describe_tool(event["name"], event["args"],
                                                    event["result"], event["action"])})
                sp = spectrum_event(event["result"])   # 측정이면 스펙트럼 이미지도 전달
                if sp:
                    yield ev(sp)
            elif etype == "error":
                turn.fail(event["detail"])
                yield ev({"type": "error", "detail": event["detail"]})
                return
            elif etype == "final":
                final_text = event["text"]
                final_ctx = event["ctx"]
                final_messages = event["messages"]

        if final_ctx is None:
            turn.fail("The agent failed to generate a response.")
            yield ev({"type": "error", "detail": "The agent failed to generate a response."})
            return

        _SESSIONS[sid] = _trim_history(final_messages)

        # 측정(레이저 조사)이 실제로 있었는지로 "실험 보고서" vs "일반 대화"를 가른다.
        used_measurement = "acquire_spectrum" in final_ctx.get("tool_call_order", [])
        turn.complete("done" if used_measurement else "chat", final_text, final_ctx)
        if used_measurement:
            yield ev({"type": "done", "final_report": final_text})
        else:
            yield ev({"type": "chat", "reply": final_text})

    except Exception as e:
        turn.fail(str(e))
        yield ev({"type": "error", "detail": str(e)})


def run_experiment(user_message: str, session_id: str = "") -> dict:
    """동기 1회 실행 — 벤치마크/레거시용 (세션 히스토리 없이 매번 새로 시작)."""
    # 벤치마크 로그: 빈 session_id면 실행마다 uuid를 새로 만들어 실행 1회 = 파일 1개로
    # 분리한다(안 그러면 모든 벤치마크 질의가 'nosession' 한 파일에 뭉친다).
    sid = session_id or str(uuid.uuid4())
    turn = new_turn("CoALA", sid, user_message)
    llm_tools = _get_llm_tools()
    llm_plain = _get_llm_plain()
    final_text = ""
    final_ctx = None
    error_detail = None
    for event in run_stream(llm_tools, llm_plain, [], user_message, session_id=sid):
        turn.observe(event)
        if event["type"] == "final":
            final_text = event["text"]
            final_ctx = event["ctx"]
        elif event["type"] == "error":
            error_detail = event["detail"]
            final_text = f"[Error] {event['detail']}"
    if error_detail is not None:
        turn.fail(error_detail, final_ctx)
    else:
        used_measurement = "acquire_spectrum" in (final_ctx or {}).get("tool_call_order", [])
        turn.complete("done" if used_measurement else "chat", final_text, final_ctx)
    return {"final_report": final_text}