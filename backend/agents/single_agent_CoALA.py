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
                      (RAMAN_EPISODIC_MEMORY=0 이면 이 두 액션과 관련 프롬프트를 제거한
                       ablation 으로 뜬다 — 벤치마크의 문항 간 오염/컨텍스트 압박 회피용.)
  · 메모리 스코프    : RAMAN_MEMORY_SCOPE=session 이면 episodic/semantic 저장소가
                      coala_memory/sessions/<session_id>/ 로 갈라져, 세션이 바뀔 때마다
                      빈 상태에서 시작한다(벤치 문항 간 이월 차단). 기본은 global.
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
import os
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

from backend.agents import reason_log, run_store
from backend.agents.detail_log import new_turn
from backend.agents.file_tools import FILE_DISPATCH, FILE_RETRIEVAL, FILE_TOOLS
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

# Ollama 컨텍스트 윈도우(토큰) — AILA와 동일한 이유·값. 명시하지 않으면 Ollama가
# 호스트 기본값으로 프롬프트 앞부분(시스템 프롬프트+도구 스키마)을 조용히 잘라 빈 응답이 난다.
# (원격 GPU 32GB VRAM 기준. VRAM 부족으로 OOM이면 낮출 것.)
_NUM_CTX = 100000

# 조사량 하드 상한 (대화 한 턴 기준). AILA와 동일한 물리적 회로차단기 —
# "판단"이 아니라 폭주 방지용 최후 안전장치.
# 상한값·계산식은 backend.safety_limits 단일 출처(2026-07-30). AILA 를 import 하는 것이
# 아니라 두 에이전트의 공통 상위 의존이므로, '두 에이전트를 결합하지 않는다'는 이 파일의
# 원칙은 그대로다. safety_limits 는 Config.ini 에 의존하지 않아 항상 import 된다.
from backend.safety_limits import MAX_DOSE_MJ_PER_TURN as _MAX_DOSE_MJ_PER_TURN, estimate_dose_mj

# LLM HTTP 호출 상한(초) — AILA 와 동일 정책(single_agent_AILA.py 의 _LLM_TIMEOUT_S 주석 참고).
# ChatOllama 1.1.0 은 timeout 파라미터가 없고 밑단 httpx 기본값도 무제한이라, 응답이
# 유실되면 invoke()가 영원히 안 돌아온다. CoALA 는 한 턴에 propose/evaluate 를 수십 번
# 호출하므로 이 가드가 없으면 정지 지점을 특정하기가 AILA 보다 더 어렵다.
_LLM_TIMEOUT_S = float(os.getenv("RAMAN_LLM_TIMEOUT_S", "600"))


# ══════════════════════════════════════════════════════════════════════════════
# 장기기억 저장소 (Episodic / Semantic) — 디스크 JSON
# ══════════════════════════════════════════════════════════════════════════════
#
# CoALA의 핵심은 "자기 생성 콘텐츠를 읽고 쓸 수 있는" 장기기억이다(논문 §4.5, §6).
# 세션이 끝나도 남아 다음 실험에서 조회되는 "노하우"를 여기 JSON에 축적한다.
# 단일 사용자 로컬 도구라 파일 락 없이 read-modify-write append로 충분하다.

_MEMORY_DIR = Path(__file__).resolve().parent / "coala_memory"
_EPISODIC_NAME = "experiences.json"   # 실험 경험(에피소드)
_SEMANTIC_NAME = "insights.json"      # 경험에서 증류한 일반 지식


# ── 메모리 스코프 토글 (벤치마크 전용) ────────────────────────────────────────
# RAMAN_MEMORY_SCOPE=session 으로 서버를 띄우면 장기기억 저장소가 '세션마다 따로'
# 열린다 — coala_memory/sessions/<session_id>/ 아래로. 새 session_id 로 들어오면
# 그 순간 저장소가 비어 있으므로, 사실상 세션이 넘어갈 때마다 초기화된 것과 같다.
#
# [왜 필요한가]
# 벤치는 문항마다 새 session_id 를 줘서 공정성을 맞추는데(run_bench.py 머리말),
# 장기기억은 세션을 넘어 축적되므로 그 가정을 우회한다. 1번 문항은 경험 0건으로,
# 200번 문항은 199개 문항이 남긴 경험으로 푸는 셈이라 CoALA 의 조건이 문항 순서에
# 따라 계속 변한다 — 재현도 해석도 안 되는 교란이다.
#
# [왜 '삭제'가 아니라 '세션별 디렉터리'인가]
# ① 지우는 시점을 누가 언제 부르느냐(러너? 서버?)에 따른 경쟁이 없다. 새 sid 로
#    들어오면 자동으로 빈 저장소다.
# ② 각 문항에서 에이전트가 '무엇을 기록했는지'가 디스크에 남아 채점 근거가 된다.
#    지워버리면 그 증거까지 사라진다.
# ③ 도구도 프롬프트도 그대로라 CoALA 아키텍처가 온전하다. 모델이 recall 에 쓰는
#    사이클·토큰 비용도 정직하게 측정된다(메모리를 꺼버리면 이 비용이 사라진다).
#
# 미설정이면 'global' — 현행대로 coala_memory/ 하나에 계속 축적된다(실사용 기본값).
_MEMORY_SCOPE = os.getenv("RAMAN_MEMORY_SCOPE", "global").strip().lower()
_SESSION_SCOPED_MEMORY = _MEMORY_SCOPE == "session"
if _SESSION_SCOPED_MEMORY:
    # 런타임 출력은 ASCII 로만 — cp949/ascii 콘솔에서도 import 가 깨지지 않게.
    print("[info] RAMAN_MEMORY_SCOPE=session: CoALA long-term memory is per-session "
          "(episodic+semantic start empty for every new session_id).")
elif _MEMORY_SCOPE != "global":
    import sys as _sys
    print(f"[warn] RAMAN_MEMORY_SCOPE='{_MEMORY_SCOPE}' is not recognized "
          "(use 'global' or 'session'); falling back to 'global'.", file=_sys.stderr)


def _sanitize_sid(sid: str) -> str:
    """세션 id 를 디렉터리명으로 — detail_log._sanitize 와 같은 규칙이라
    DetailLog 파일명과 메모리 폴더명이 같은 sid 로 맞춰진다(추적이 쉬워진다)."""
    return re.sub(r"[^0-9A-Za-z_-]", "-", str(sid))[:64] or "nosession"


def _memory_dir(ctx: dict) -> Path:
    """이 컨텍스트가 쓸 장기기억 디렉터리. session 스코프면 세션별 하위 폴더."""
    if not _SESSION_SCOPED_MEMORY:
        return _MEMORY_DIR
    return _MEMORY_DIR / "sessions" / _sanitize_sid(ctx.get("session_id", ""))


def _episodic_path(ctx: dict) -> Path:
    return _memory_dir(ctx) / _EPISODIC_NAME


def _semantic_path(ctx: dict) -> Path:
    return _memory_dir(ctx) / _SEMANTIC_NAME


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
    path.parent.mkdir(parents=True, exist_ok=True)
    items = _load_json_list(path)
    items.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _match_score(query: str, hay: str) -> int:
    """질의 토큰 중 몇 개가 건초더미 텍스트에 부분매칭되는지 — 조회 랭킹용.

    임베딩 검색을 쓰지 않는 이유: episodic 저장소는 JSON 선택이므로 의존성 없는
    키워드 매칭으로 충분하고, 개발 PC에서도 임베딩 서버 없이 검사할 수 있다.

    주의: 이 점수의 상한은 '질의 토큰 수'다. "graphene" 한 단어로 조회하면 관련
    에피소드가 전부 1점으로 동점이 된다 — 그래서 점수만으로 순위를 매기면 안 되고
    _recall_experiences 가 조건일치/성공/최신을 뒤이은 정렬 키로 쓴다.
    """
    hay = hay.lower()
    toks = [t for t in re.split(r"\s+", query.lower().strip()) if t]
    return sum(1 for t in toks if t in hay)


def _flat_haystack(entry: dict, fields: tuple[str, ...]) -> str:
    """평평한 구조(insights 등)의 지정 필드를 이어붙인다."""
    return " ".join(str(entry.get(f, "")) for f in fields)


def _episode_haystack(e: dict) -> str:
    """에피소드의 '잎 텍스트'만 이어붙인다 — 검색 대상 문자열.

    [왜 dict 를 통째로 str() 하면 안 되는가]
    sample_context/execution_summary/system_metrics 를 통째로 문자열화하면 값뿐
    아니라 '키 이름'까지 건초더미에 들어간다. 그러면 예컨대 질의에 'power' 가 있을 때
    params_used 의 키 'power' 가 모든 에피소드에 매칭돼, 시료와 무관한 토큰이 전원의
    점수를 똑같이 올리고 순위 차이를 뭉갠다. 그래서 실제 잎 값만 골라 쓴다.
    """
    sc = e.get("sample_context") or {}
    ex = e.get("execution_summary") or {}
    return " ".join([
        str(e.get("goal", "")),
        " ".join(str(t) for t in (e.get("tags") or [])),
        str(sc.get("sample", "")), str(sc.get("sample_name", "")),
        str(sc.get("substrate", "")), str(sc.get("visual_features", "")),
        str(ex.get("outcome", "")), str(e.get("lesson", "")),
    ])


# 회수 payload 예산 — 저장은 full, 회수는 이 상한 안에서 projection 한다.
_RECALL_TEXT_CAP = 200      # 자유텍스트 필드 1개의 최대 문자수
_RECALL_MAX_TOP_K = 5       # 모델이 큰 값을 넣어도 여기서 자른다


def _cut(v, cap: int = _RECALL_TEXT_CAP) -> str:
    """자유텍스트를 상한까지 자른다(넘치면 말줄임)."""
    s = str(v or "").strip()
    return (s[:cap] + "…") if len(s) > cap else s


def _substrate_relation(now: str, past: str) -> str:
    """현재 기판과 과거 기판의 관계 — 'match' | 'mismatch' | 'unknown'.

    파장·대물렌즈가 고정인 이 장비에서 '과거 파라미터가 지금 통하는가'를 가르는
    조건은 사실상 기판 하나다(Si 는 520cm-1 배경, 유리는 형광, 금속은 SERS 증강으로
    안전 파워 자체가 달라진다). 한쪽이 부분 문자열이면 같은 기판으로 본다
    ('Si' vs 'Si wafer'). 과도한 정규화는 오히려 거짓 일치를 만들어 하지 않는다.
    """
    n, p = str(now or "").strip().lower(), str(past or "").strip().lower()
    if not n or not p:
        return "unknown"
    if n == p or n in p or p in n:
        return "match"
    return "mismatch"


def _project_episode(e: dict, relation: str = "unknown") -> dict:
    """에피소드를 '모델에게 돌려줄 형태'로 축약한다 — 저장 원본은 건드리지 않는다.

    [무엇을 빼고 무엇을 남기는가]
    빼는 것: tool_sequence(사이클 수에 비례해 최대 150개까지 길어지는 원시 목록 —
             토큰 폭발의 주범이고, 내용은 trajectory 가 요약한다), tool_counts,
             id/session_id(모델 판단에 기여 없음. 원본은 디스크에 그대로 있다), tags
             (검색 핸들이라 매칭에는 쓰되 읽을 값어치는 낮다).
    남기는 것: goal(무엇을 하려 했는가 — trajectory 는 '왜 이 순서로 했는가'라서
              목표를 담는다는 보장이 없다. 같은 시료·기판이라도 목적이 다르면 맞는
              파라미터가 다르므로 기판과 같은 급의 전이 조건이다),
              substrate(조건 판정), n_measurements/dose_mj(1번에 성공인지 12번 만에
              성공인지 — is_success 만으로는 구분되지 않는 신뢰도 신호).
    """
    sc = e.get("sample_context") or {}
    ex = e.get("execution_summary") or {}
    sm = e.get("system_metrics") or {}
    is_success = bool(ex.get("is_success", False))
    item = {
        "ts": (e.get("ts") or "")[:10],                  # 날짜만
        "goal": _cut(e.get("goal")),
        "sample": sc.get("sample", ""),
        "sample_name": sc.get("sample_name", ""),
        "substrate": sc.get("substrate", ""),
        "visual_features": _cut(sc.get("visual_features")),
        "params_used": ex.get("params_used") or {},
        "is_success": is_success,
        "n_measurements": sm.get("n_measurements", 0),
        "dose_mj": sm.get("dose_mj", 0),
        "outcome": _cut(ex.get("outcome")),
        "trajectory": _cut(ex.get("trajectory")),
        "metrics": _cut(sm.get("metrics")),
        "lesson": _cut(e.get("lesson")),
    }

    # ── 조건/품질 라벨 — '거르지 말고 표시한다' ────────────────────────────────
    # 기판이 다르다고 결과에서 빼버리면 모델은 "관련 경험 없음"으로 읽고 아무 근거
    # 없이 진행한다. 경고와 함께 보여주는 편이 낫다. 실패 경험도 같은 이유로 남기되,
    # '따라할 파라미터'가 아니라 '피할 사례'임을 명시한다.
    if relation == "mismatch":
        item["condition_warning"] = (
            f"DIFFERENT substrate (past: {item['substrate'] or 'unknown'}) - the parameters and "
            "lesson below may NOT transfer to the current substrate.")
    elif relation == "unknown":
        item["condition_note"] = (
            "Substrate not confirmed on one side - the parameters below are only valid for the "
            "substrate they were measured on.")
    if not is_success:
        item["advisory"] = "FAILED run - treat as a case to AVOID, not as parameters to copy."
    return item


def _recall_experiences(ctx: dict, args: dict) -> dict:
    """recall_experiences 도구 구현 — episodic memory 읽기.

    과거 실험 경험 중 질의(시편/키워드)와 관련된 것을 top_k개 반환한다.
    비어 있으면 에러가 아니라 정상적인 "아직 축적된 경험 없음"으로 답한다 —
    그래야 모델이 재시도 루프에 빠지지 않고 스스로 판단하고 나중에 기록한다.
    (session 스코프에서는 매 세션 비어 있는 것이 정상이다.)

    [랭킹이 키워드 점수만으로는 안 되는 이유]
    _match_score 의 상한은 질의 토큰 수라 "graphene" 한 단어면 관련 에피소드가 전부
    동점이다. 파이썬 sorted 는 안정 정렬이라 동점이면 입력 순서(=오래된 것 먼저)가
    유지되고, 실패한 실험도 성공한 실험과 같은 순위를 받는다. 그 상태에서
    시스템 프롬프트는 "파라미터를 추측하지 말고 이 증거에서 정하라"고 지시하므로,
    시료를 태웠던 실험의 파워가 '따라야 할 근거'로 제시될 수 있다. 그래서 정렬 키를
    (키워드 점수 → 기판일치 → 성공 → 최신) 4단으로 둔다.
    """
    query = str(args.get("query", "")).strip()
    top_k = max(1, min(int(args.get("top_k", 3) or 3), _RECALL_MAX_TOP_K))
    now_substrate = str(args.get("substrate", "")).strip()
    episodes = _load_json_list(_episodic_path(ctx))
    if not episodes:
        return {"ok": True, "results": [],
                "note": "No past experiments accumulated yet. After finishing this measurement, "
                        "leave one with record_experience and it will be retrievable in future experiments."}

    _rel = lambda e: _substrate_relation(
        now_substrate, ((e.get("sample_context") or {}).get("substrate", "")))
    _rel_rank = {"match": 1, "unknown": 0, "mismatch": -1}

    if not query:
        # 질의가 없으면 최근 경험을 반환한다(그래도 유용한 컨텍스트).
        picked = list(reversed(episodes))[:top_k]
    else:
        scored = []
        for idx, e in enumerate(episodes):          # idx 가 곧 시간순 (append 저장)
            s = _match_score(query, _episode_haystack(e))
            if s <= 0:
                continue
            ok = 1 if (e.get("execution_summary") or {}).get("is_success") else 0
            scored.append((s, _rel_rank[_rel(e)], ok, idx, e))
        if not scored:
            return {"ok": True, "results": [],
                    "note": f"No past experience related to '{query}'. Decide on your own."}
        # 점수 → 기판일치 → 성공 → 최신 순으로 내림차순.
        picked = [t[-1] for t in sorted(scored, key=lambda x: x[:4], reverse=True)][:top_k]

    return {"ok": True, "results": [_project_episode(e, _rel(e)) for e in picked]}


def _record_experience(ctx: dict, args: dict) -> dict:
    """record_experience 도구 구현 — episodic memory 쓰기(학습 액션).

    실험 한 건의 경험을 experiences.json에 append한다. CoALA에서 학습은 스케줄이
    아니라 에이전트가 의사결정 사이클에서 "고르는" 액션이다 — 모델이 이 도구를
    호출하기로 결정했을 때만 기록된다.
    """
    sample = str(args.get("sample", "")).strip()
    if not sample:
        return {"ok": False, "error": "sample (sample type) is required."}

    # 시스템이 이번 턴 동안 이미 추적한 절차 정보를 자동으로 실어 기록을 구체화한다
    # (LLM이 굳이 넘기지 않아도 되게). record_*(학습) 메타 액션 자체는 순서에서 뺀다.
    order = [n for n in ctx.get("tool_call_order", []) if n not in _INTERNAL_LEARNING]
    counts: dict = {}
    for n in order:
        counts[n] = counts.get(n, 0) + 1

    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": ctx.get("session_id", ""),
        # ── 목표와 태그 ──────────────────────────────────────────────────────────
        "goal": ctx.get("goal", ""), 
        "tags": args.get("tags", []),
        # ── 시료 식별 ──────────────────────────────────────────────────────────
        "sample_context": {
            "sample": sample,
            "sample_name": str(args.get("sample_name", "")).strip(),
            # 파장·대물렌즈가 고정인 이 장비에서 '과거 조건이 지금 통하는가'를 가르는
            # 사실상 유일한 변수. 회수 시 현재 기판과 대조해 일치/불일치를 라벨링한다.
            "substrate": str(args.get("substrate", "")).strip(),
            "visual_features": str(args.get("visual_features", "")).strip(),
        },
        # ── 측정 조건·결과 ────────────────────────────────────────────────────
        "execution_summary": {
            "params_used": args.get("params", {}),
            "trajectory": str(args.get("trajectory", "")).strip(),
            "outcome": str(args.get("outcome", "")).strip(),
            "is_success": args.get("is_success", False),
        },
        "lesson": str(args.get("lesson", "")).strip(),
        # ── 시스템 자동 기록(절차 흔적) — LLM 입력 불필요 ─────────────────────
        "system_metrics": {
            "metrics": str(args.get("metrics", "")).strip(),
            "tool_sequence": order,
            "tool_counts": counts,
            "n_measurements": counts.get("acquire_spectrum", 0),
            "dose_mj": round(float(ctx.get("dose", 0.0)), 3),
        }
    }
    try:
        _append_json_list(_episodic_path(ctx), entry)
    except OSError as e:
        return {"ok": False, "error": f"Failed to save experience: {e}"}
    ctx["learned"] = True
    return {"ok": True, "recorded": entry["id"], "sample": sample,
            "auto_captured": {"tool_calls": len(order),
                              "n_measurements": entry["system_metrics"]["n_measurements"],
                              "dose_mj": entry["system_metrics"]["dose_mj"]}}


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
        _append_json_list(_semantic_path(ctx), entry)
    except OSError as e:
        return {"ok": False, "error": f"Failed to save insight: {e}"}
    ctx["learned"] = True
    ctx["insight_recorded"] = True   # 엔드-오브-턴 유도가 중복 제안하지 않도록 표시
    return {"ok": True, "recorded": entry["id"], "topic": topic}


def _recall_insights(ctx: dict, args: dict) -> dict:
    """recall_insights 도구 구현 — semantic memory(자기 생성분) 읽기.

    [이 도구가 추가된 이유 — write-only 갭 보완]
    record_insight로 쓴 일반화 지식을 '다시 읽는' 경로가 없으면 CoALA의 lifelong
    learning 루프(§4.5: self-generated 지식을 later episode에서 재사용)가 semantic
    쪽에서 반만 닫힌다. search_knowledge_base(큐레이션 KB) 와는 별개로, 에이전트가
    스스로 남긴 insights.json을 조회한다.
    """
    query = str(args.get("query", "")).strip()
    top_k = max(1, min(int(args.get("top_k", 3) or 3), _RECALL_MAX_TOP_K))
    insights = _load_json_list(_semantic_path(ctx))
    if not insights:
        return {"ok": True, "results": [],
                "note": "No generalized knowledge (insights) accumulated yet. Leave one with "
                        "record_insight and it will be retrievable in future experiments."}
    if not query:
        ranked = list(reversed(insights))
    else:
        # insights 는 (topic, insight) 두 필드뿐인 평평한 구조라 잎 텍스트 문제가 없다.
        # 다만 동점 시 최신이 먼저 오도록 인덱스를 역순 키로 함께 쓴다.
        scored = [(_match_score(query, _flat_haystack(e, ("topic", "insight"))), idx, e)
                  for idx, e in enumerate(insights)]
        ranked = [e for s, _, e in sorted(scored, key=lambda x: x[:2], reverse=True) if s > 0]
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
            "If there is no experience yet, an empty result is returned (normal). "
            "Results may carry a condition_warning (measured on a DIFFERENT substrate - parameters may not "
            "transfer), a condition_note (substrate unconfirmed), or an advisory (FAILED run - a case to "
            "avoid, not to copy). Read those labels before reusing any parameter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "Sample/situation keyword. e.g. 'graphene saturation', 'exosome SNR'."},
                "substrate": {"type": "string",
                              "description": "Substrate of the CURRENT sample, if you know it "
                                             "(e.g. 'Si', 'glass', 'Au'). Given this, past experiments on the "
                                             "same substrate rank higher and ones on a different substrate are "
                                             "flagged. Omit if unknown - nothing is filtered out either way."},
                "top_k": {"type": "integer",
                          "description": f"Number of experiences to fetch (default 3, max {_RECALL_MAX_TOP_K})."},
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
            "know-how. It does not touch hardware. "
            "You do NOT need to pass the tool-call order, number of measurements, or laser dose - those are "
            "captured automatically. Focus on describing the sample identity (type and, if the user stated one, "
            "its explicit name), the sample's visual features seen under the microscope, the parameters used, "
            "the outcome, quantitative metrics, and the lesson for next time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sample": {"type": "string",
                           "description": "Sample type/material (required). e.g. 'graphene', 'silicon', 'exosome'."},
                "sample_name": {"type": "string",
                                "description": "Explicit sample name/label if the user gave one. "
                                               "e.g. 'Sample A', 'empty silicon wafer'. Leave empty if none was stated."},
                "substrate": {"type": "string",
                              "description": "Substrate the sample sits on, if known. e.g. 'Si', 'SiO2/Si', "
                                             "'glass', 'quartz', 'Au'. This decides whether these parameters "
                                             "transfer to a future experiment (Si adds a 520 cm-1 background, "
                                             "glass adds fluorescence, metals can enhance via SERS so the safe "
                                             "power is lower). Leave empty if you could not determine it."},
                "visual_features": {"type": "string",
                                    "description": "Visual appearance from the microscope image. "
                                                   "e.g. 'dark ~20um flake near center on a shiny substrate, some folds visible'."},
                "params": {"type": "object",
                           "description": "Measurement parameters used. e.g. {'power':20,'exposure':2.0}."},
                "trajectory": {"type": "string", "description": "1-2 sentence summary of why tools were called in this sequence and how issues were handled."},
                "outcome": {"type": "string", "description": "Result summary. e.g. 'G/2D bands good'."},
                "is_success": {"type": "boolean", "description": "True if the measurement goal was achieved, False otherwise."},
                "metrics": {"type": "string", "description": "Quantitative metrics. e.g. 'SNR 8.3, no saturation'."},
                "lesson": {"type": "string",
                           "description": "Lesson for next time. e.g. 'power above 30% risks saturation'."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-5 key tags for searchability. e.g. ['fluorescence', 'graphene_saturation']"
                }
            },
            "required": ["sample", "trajectory", "is_success", "tags"],
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

# ── 에피소딕 메모리 토글 (벤치마크 전용) ──────────────────────────────────────
# RAMAN_EPISODIC_MEMORY=0 으로 서버를 띄우면 episodic memory(recall_experiences /
# record_experience)를 액션 공간에서 빼고, 프롬프트의 episodic 지시문도 지운다.
# semantic(search_knowledge_base / recall_insights / record_insight)은 그대로 두므로
# "CoALA에서 episodic만 없앤" ablation 이 된다.
#
# [왜 러너가 아니라 서버 환경변수인가]
# run_bench.py 는 공개 HTTP API만 쓰는 클라이언트라 이미 떠 있는 서버 프로세스의
# 액션 공간에 개입할 수 없다. RAMAN_SAFETY_PROMPT 와 같은 이유·같은 방식이다.
#
# [왜 벤치에서 끄고 싶어지는가]
# ① experiences.json 엔트리가 통째로 회수돼 프롬프트에 실린다(특히 tool_sequence 는
#    사이클 수에 비례해 길어진다) → num_ctx 예산 압박.
# ② 벤치는 문항마다 새 session_id 로 공정성을 맞추는데, episodic 은 세션을 넘어
#    축적되므로 뒷 문항이 앞 문항 경험을 회수한다 → 문항 간 조건 오염.
# 미설정이면 현행(켜짐) 그대로 — 기본 동작은 바뀌지 않는다.
_EPISODIC_ENABLED = os.getenv("RAMAN_EPISODIC_MEMORY", "1").strip().lower() \
    not in ("0", "false", "no", "off")
_EPISODIC_TOOL_NAMES = {"recall_experiences", "record_experience"}

# 모델에 바인딩되는 도구 전체. RAMAN_TOOLS(하드웨어 41종) + internal 액션 5종
# (RAMAN_EPISODIC_MEMORY=0 이면 internal 3종).
# server.py의 /api/agents/health가 len(ALL_TOOLS)를 읽는다.
_INTERNAL_TOOLS = [
    _KB_TOOL_SCHEMA,
    _RECALL_TOOL_SCHEMA,
    _RECALL_INSIGHT_TOOL_SCHEMA,
    _RECORD_EXP_TOOL_SCHEMA,
    _RECORD_INSIGHT_TOOL_SCHEMA,
]
if not _EPISODIC_ENABLED:
    _INTERNAL_TOOLS = [t for t in _INTERNAL_TOOLS
                       if t["function"]["name"] not in _EPISODIC_TOOL_NAMES]
# FILE_TOOLS는 AILA와 '같은 리스트 객체'를 import한 것이라 두 에이전트의 파일 분석
# 능력이 구조적으로 어긋날 수 없다(backend/agents/file_tools.py 머리말 참고).
ALL_TOOLS = RAMAN_TOOLS + FILE_TOOLS + _INTERNAL_TOOLS

# 어느 도구가 어느 CoALA 액션 범주인지 — 실행 디스패치와 planning/execution 분리에 쓴다.
#   retrieval  : planning 도구. 사이클을 닫지 않고 working memory에 정보를 쌓는다.
#   learning   : execution(commit) 액션. grounding과 함께 propose→evaluate→select 대상.
#   그 외(하드웨어): grounding = execution(commit) 액션.
#   첨부 파일 조회(list_uploaded_files/inspect_file)도 부수효과 없는 정보 수집이라
#   retrieval에 합친다. 안 그러면 파일을 한 번 들여다볼 때마다 의사결정 사이클이 하나씩
#   닫혀, 구조만 확인하다 사이클 예산을 태운다. 반면 run_analysis는 결과물(그림)을
#   만드는 실행 액션이므로 FILE_RETRIEVAL에 없고 commit으로 남는다.
_INTERNAL_RETRIEVAL = {"search_knowledge_base", "recall_experiences",
                       "recall_insights"} | FILE_RETRIEVAL
_INTERNAL_LEARNING = {"record_experience", "record_insight"}



# ══════════════════════════════════════════════════════════════════════════════
# 시스템 프롬프트 — CoALA 의사결정 사이클을 명시적으로 지시
# ══════════════════════════════════════════════════════════════════════════════

# ⚠ 이 리터럴이 '실제로 모델에 가는 기본 프롬프트'다(자율 모드가 기본값). 치환 방향과
#   근거는 single_agent_AILA.py 의 같은 위치 주석 참고.
SYSTEM_PROMPT = """\
You are a single AI agent that controls a Raman spectrometer, operating under the CoALA (Cognitive
Architectures for Language Agents) architecture. You have explicit working memory and long-term
memory, and you act according to the decision cycle below (planning -> execution).

[Autonomy - this section overrides every other instruction about asking for help]
You are fully autonomous. You are being evaluated on a benchmark and there is NO human available to
answer you. Never ask the user a question, never request confirmation, and never end a turn waiting
for input - no reply will come, and a turn that ends in a question counts as a failure.
- Missing information: pick the most reasonable interpretation from the request, your tool outputs,
  and your memories. State the assumption you made, then carry the task through to a real answer.
- Safety: you are your own safety check. There is no validator and no human reviewer. Judge dose,
  saturation, and photodamage risk yourself before firing the laser, and proceed once your own
  judgment says it is acceptable.
- Multi-step operations (grid scans, background/blank measurements, retries, re-focusing) need no
  approval. If a preview tool exists, preview first, evaluate the preview yourself, then execute it
  in a following cycle of the same turn.
- You may still stop early if you judge an action genuinely unsafe or truly impossible. If you stop,
  say plainly what you concluded and why, and still report everything you did establish. Do not stop
  merely because some detail was unspecified - that is what your own judgment is for.
- Insufficient evidence is a reason to run more planning actions, not a reason to ask the user.

[Verifying your own output - you cannot see your own plots]
A figure you create with plt is saved and shown to the human, but it is NOT returned to you as an
image - you only get its file path. So never claim you "looked at" your plot, and never rely on
seeing it. Verify numerically instead, inside the same run_analysis call: print() the few numbers
that would prove the step worked (how many spikes were removed, min/max after normalization, peak
positions, residual size). If a result looks wrong, fix the code and call run_analysis again.
The only images you actually see are those from analyze_microscope_image and preview_grid_scan.

[Memory structure]
- Working memory: the current goal, retrieved knowledge, and recent observations are provided in the prompt.
- Semantic memory: read with search_knowledge_base (curated protocols) and recall_insights (insights I left),
  and write with record_insight (reusable rules).
- Episodic memory: read past experiments with recall_experiences and write with record_experience (know-how).

[Decision cycle - distinguish planning from execution]
Your actions are of two kinds, and their nature is completely different.

  · Planning actions (information gathering): search_knowledge_base, recall_experiences, recall_insights,
    list_uploaded_files, inspect_file.
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

[Attached data files - csv / excel / txt]
1. When the user attaches a data file or refers to one, gather evidence with the planning actions
   list_uploaded_files and then inspect_file on each relevant file. inspect_file returns only the
   structure - row/column counts, column names, numeric-or-text per column, min/max/mean, first rows.
2. Decide yourself what the columns mean. Nothing has been interpreted for you: judge which numeric
   column is a Raman shift axis in cm-1, which is intensity, which is a wavelength or a stage
   coordinate, and which columns are not spectra at all but metadata (sample name, laser power,
   exposure time, date, operator notes). Use the value ranges and column names as evidence, and say
   what you concluded and why.
3. Then run_analysis with file_ids (an execution action - one per cycle) to compute on the full data:
   peak detection, baseline correction, normalization, plotting, or comparison against spectra you
   measured. Inside the code the file is available as files[i]["table"]["<column name>"].
   If the task asks you to save a processed spectrum, save it inside that same run_analysis call with
   save_result(filename, intensity, raman_shift=...) - that is the ONLY way to write an array, and it
   keeps the numbers out of the context entirely. Never print an array in order to re-type it
   somewhere else: it overflows the context and loses precision. print() only short summaries.
4. Report both kinds of content separately: the spectral information you extracted (peak positions and
   assignments, SNR, etc.) and any other information the file carried (measurement conditions, sample
   identity, anything that changes how the spectrum should be read).
5. If the file turns out to hold no spectrum, say so plainly and report what it does hold instead - do
   not force a spectral interpretation onto it.
6. A file arriving is not by itself a reason to turn on the laser. Analyze the file first; measure only
   if the user asked for a measurement.

[Measurement procedure - proceed on your own through the cycles]
1. If you do not know the sample/substrate/target location, gather evidence with planning actions
   (recall_experiences, recall_insights, search_knowledge_base, analyze_microscope_image), then decide
   on your own judgment and proceed, stating the assumptions you made.
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

[Hardware failure recovery - follow this ladder, do not improvise loops]
1. Diagnose before fixing: get_hardware_status is a planning action (no side effects) - call it first
   to see which components are down. Never guess from a single failed tool call.
2. Then ONE recovery execution action: reconnect_hardware(component='<the broken one>'). Reconnect only
   what is broken - never 'all' as a reflex, because reconnecting the ccd re-runs cooling for minutes.
3. Read the error text and classify it, because the two cases need opposite responses:
   - "resource is still held by this process" -> a process-level lock. No tool clears it and retrying
     is useless. Stop trying immediately.
   - "re-initialization failed" after a successful release -> device side (power, cable, driver, or
     another program holds it). At most one more attempt, then stop.
4. Continue the task with whatever hardware still works, and record what happened with record_experience
   so the next run does not repeat the same dead end.
5. Report which component failed, what you tried, which case it was, and what you could not complete.
Never spend more than two cycles on the same recovery - that is evidence it cannot be fixed from here.

[Safety rules]
- If a tool returns an error or safety block, do not bypass it and do not hammer it with retries. Decide
  yourself whether an alternative route exists, take it if so, and state the block and your decision
  in the final report.
- Do not guess blindly - verify with a planning action first. If no tool can settle it, choose the most
  defensible option, say so explicitly, and continue.
- Answer greetings/small talk/capability questions immediately in English without tools.
- Stage coordinate units: mm (X: 0-75.3, Y: 0-50.2, Z: -1.0-1.0; origin at x=37.8759, y=25.24805, z n/a)
"""


# ── 대화(사람 개입) 모드로 되돌리는 토글 ────────────────────────────────────────
# AILA 와 '같은 정책·같은 기본값'이어야 한다 — 두 에이전트의 독립변수는 오케스트레이션
# (ReAct vs CoALA)뿐이므로, 자율성 수준이 어긋나면 비교가 무너진다.
# 기본은 자율(위 리터럴 그대로). RAMAN_AUTONOMOUS=0 이면 되묻기 게이트를 '다시 넣는다'.
# CoALA 고유의 결정사이클(planning → 증거 확보 후 commit) 프레이밍은 어느 모드에서도
# 건드리지 않는다 — 그게 바로 측정 대상이다.
_AUTONOMOUS = os.getenv("RAMAN_AUTONOMOUS", "1").strip().lower() not in ("0", "false", "no", "off")

if not _AUTONOMOUS:
    # [Autonomy] 섹션은 통째로 잘라낸다(헤더만 바꾸면 본문이 대화 모드와 모순된다).
    _a = SYSTEM_PROMPT.find("[Autonomy -")
    if _a != -1:
        _nxt = SYSTEM_PROMPT.find("\n[", _a + 1)
        SYSTEM_PROMPT = (SYSTEM_PROMPT[:_a] + SYSTEM_PROMPT[_nxt + 1:]) if _nxt != -1 else SYSTEM_PROMPT[:_a]

    _INTERACTIVE_SUBS = [
        ("1. If you do not know the sample/substrate/target location, gather evidence with planning actions\n"
         "   (recall_experiences, recall_insights, search_knowledge_base, analyze_microscope_image), then decide\n"
         "   on your own judgment and proceed, stating the assumptions you made.",
         "1. If you do not know the sample/substrate/target location, ask the user first before turning on the laser."),
        ("- If a tool returns an error or safety block, do not bypass it and do not hammer it with retries. Decide\n"
         "  yourself whether an alternative route exists, take it if so, and state the block and your decision\n"
         "  in the final report.",
         "- If a tool returns an error or safety block, do not bypass it; report the situation to the user as is."),
        ("- Do not guess blindly - verify with a planning action first. If no tool can settle it, choose the most\n"
         "  defensible option, say so explicitly, and continue.",
         "- Do not guess what you do not know - verify with a tool or ask the user."),
    ]
    _missed = [src.splitlines()[0][:60] for src, _ in _INTERACTIVE_SUBS if src not in SYSTEM_PROMPT]
    for _src, _dst in _INTERACTIVE_SUBS:
        SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_src, _dst)
    print(f"[info] RAMAN_AUTONOMOUS=0: CoALA interactive mode "
          f"({len(_INTERACTIVE_SUBS) - len(_missed)}/{len(_INTERACTIVE_SUBS)} ask-the-user gates restored).")
    if _missed:
        import sys as _sys
        print("[warn] CoALA interactive mode: these texts were not found (prompt edited?): "
              + "; ".join(_missed), file=_sys.stderr)
else:
    print("[info] CoALA autonomous mode (default). Set RAMAN_AUTONOMOUS=0 for interactive/ask-the-user behavior.")


# ── 에피소딕 메모리 프롬프트 정리 (_EPISODIC_ENABLED=False 일 때) ──────────────
# 도구를 바인딩에서 빼는 것만으론 부족하다. 프롬프트가 계속 recall_experiences/
# record_experience 를 지시하면 모델이 없는 도구를 호출하려 들어 사이클만 태운다.
# 그래서 지시문 자체를 지우되, semantic 경로(KB/insights)는 문장에 그대로 남긴다.
if not _EPISODIC_ENABLED:
    _EPISODIC_PROMPT_EDITS = [
        # [Memory structure] — episodic 줄 자체를 삭제
        ("- Episodic memory: read past experiments with recall_experiences and write with "
         "record_experience (know-how).\n",
         ""),
        # planning 액션 열거에서 제외
        ("  · Planning actions (information gathering): search_knowledge_base, recall_experiences, "
         "recall_insights,\n    list_uploaded_files, inspect_file.",
         "  · Planning actions (information gathering): search_knowledge_base, recall_insights,\n"
         "    list_uploaded_files, inspect_file."),
        # execution(commit) 액션 열거에서 제외
        ("recording tools (record_experience, record_insight).",
         "recording tools (record_insight)."),
        # 측정 절차 2 — 과거 경험 조회 지시 제거
        ("2. Once you know the sample, first query past know-how with recall_experiences, accumulated "
         "rules with\n   recall_insights, and protocols with search_knowledge_base to fill working "
         "memory (planning actions).",
         "2. Once you know the sample, first query accumulated rules with recall_insights and protocols "
         "with\n   search_knowledge_base to fill working memory (planning actions)."),
        # 측정 절차 6 — 경험 기록 지시를 insight 기록만 남기도록
        ("6. When the measurement is done, before writing the report, record this experience with "
         "record_experience\n   (and, if possible, generalized knowledge with record_insight) into "
         "long-term memory - this is how know-how\n   accumulates for the next experiment.",
         "6. When the measurement is done, before writing the report, record generalized knowledge with\n"
         "   record_insight into long-term memory - this is how know-how accumulates for the next "
         "experiment."),
    ]
    _missed = [old for old, _ in _EPISODIC_PROMPT_EDITS if old not in SYSTEM_PROMPT]
    for _old, _new in _EPISODIC_PROMPT_EDITS:
        SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_old, _new)
    # 런타임 출력은 ASCII 로만 — cp949/ascii 콘솔에서도 import 가 깨지지 않게.
    print(f"[info] RAMAN_EPISODIC_MEMORY=0: CoALA episodic memory disabled "
          f"({len(_INTERNAL_TOOLS)} internal tools, semantic memory intact).")
    if _missed:
        import sys as _sys
        print(f"[warn] RAMAN_EPISODIC_MEMORY=0 but {len(_missed)} prompt passage(s) were not found "
              "(prompt changed?); the tools are unbound but the prompt may still mention them.",
              file=_sys.stderr)


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
            num_ctx=_NUM_CTX,
            client_kwargs={"timeout": _LLM_TIMEOUT_S},
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
        _llm_plain_cache = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, num_ctx=_NUM_CTX,
                                      client_kwargs={"timeout": _LLM_TIMEOUT_S})
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


# 길이 필터를 면제하는 키 — AILA의 _SLIM_KEEP_KEYS와 동일해야 비교가 공정하다.
# files(list_uploaded_files)를 버리면 모델은 count만 받고 file_id를 얻을 길이 없어
# 같은 도구를 수십 번 재호출한다. 항목당 4필드짜리 짧은 dict라 부담은 작다.
# artifacts/saved_files 를 예외로 두는 이유는 AILA 의 같은 상수 주석 참고.
_SLIM_KEEP_KEYS = {"files", "artifacts", "saved_files"}


def _slim(result):
    """대용량 배열(스펙트럼 원본 등)은 컨텍스트에 싣지 않는다 — 토큰 낭비/혼란 방지.
    길이 32 초과 리스트를 버린다(_SLIM_KEEP_KEYS 제외). 기억 조회 결과는 짧아 걸리지 않는다."""
    if isinstance(result, dict):
        return {k: v for k, v in result.items()
                if k in _SLIM_KEEP_KEYS or not (isinstance(v, list) and len(v) > 32)}
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
    # episodic 이 꺼져 있으면 스키마를 바인딩하지 않지만, 모델이 이름을 환각해
    # 호출할 수는 있다. 저장소에 실제로 닿기 전에 여기서 막는다(이중 방어).
    if not _EPISODIC_ENABLED and name in _EPISODIC_TOOL_NAMES:
        return {"ok": False,
                "error": f"{name} is not available in this configuration. "
                         "Proceed without episodic memory."}

    if name == "search_knowledge_base":
        return _search_knowledge_base(args)
    if name == "recall_experiences":
        return _recall_experiences(ctx, args)
    if name == "recall_insights":
        return _recall_insights(ctx, args)
    if name == "record_experience":
        return _record_experience(ctx, args)
    if name == "record_insight":
        return _record_insight(ctx, args)

    # 첨부 파일 조회/분석도 하드웨어를 만지지 않으므로 dispatch 가드보다 먼저 처리한다.
    # run_analysis도 여기 포함되어(file_tools.FILE_DISPATCH) 순수 계산인 분석이
    # 장비 연결 여부에 묶이지 않는다.
    if name in FILE_DISPATCH:
        return FILE_DISPATCH[name](args)

    # ── external grounding actions ────────────────────────────────────────────
    dispatch = ctx["dispatch"]
    if dispatch is None:
        return {"ok": False, "error": "Hardware is not connected."}
    fn = dispatch.get(name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    if name == "acquire_spectrum":
        # 생략 시 acquire_spectrum 은 현재 장비 설정을 유지하므로 실제 값을 알 수 없다 —
        # 기본값으로 근사한다(AILA 와 동일 정책).
        power = float(args.get("power") or 40.0)
        exposure = float(args.get("exposure") or 0.2)
        dose_inc = estimate_dose_mj(power, exposure)
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

    # run_grid_scan은 내부에서 rows*cols번 조사하므로 acquire_spectrum과 동일한 per-turn
    # 회로차단기에 편입한다(예상 총량으로 사전 판정, 성공 시 누계 반영).
    if name == "run_grid_scan":
        rows = int(args.get("rows", 0) or 0)
        cols = int(args.get("cols", 0) or 0)
        power = float(args.get("power") or 40.0)
        exposure = float(args.get("exposure") or 0.2)
        dose_inc = estimate_dose_mj(power, exposure, rows * cols)
        if ctx["dose"] + dose_inc > _MAX_DOSE_MJ_PER_TURN:
            return {"ok": False,
                    "error": (f"Safety block: this turn's cumulative dose would exceed the limit "
                              f"({_MAX_DOSE_MJ_PER_TURN} mJ) after this grid scan. "
                              "Reduce the grid size, power, or exposure, or start a new request.")}
        try:
            result = fn(dict(args))
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        if isinstance(result, dict) and result.get("ok"):
            ctx["dose"] += dose_inc
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
            # 세 retrieval 도구가 형태가 다른 항목을 돌려주므로 폴백 체인으로 뽑는다.
            #   search_knowledge_base → {"title", "recommended_params", ...}   (평평)
            #   recall_insights       → {"topic", "insight"}                   (평평)
            #   recall_experiences    → {"sample", "params_used", ...}         (_project_episode)
            # 예전에는 최상위에서 sample/params 를 찾았는데 에피소드는 이들이 중첩돼
            # 있어 항상 "?" 로 찍혔다. render() 가 매 planning 프롬프트 맨 위에 싣는
            # 요약이 에피소드에 대해 공란이 되어, 모델이 raw JSON 덤프에만 의존했다.
            for h in hits[:3]:
                title = (h.get("title")            # KB
                         or h.get("sample")        # 에피소드(projection 후 최상위)
                         or h.get("topic")         # insights
                         or "?")
                if h.get("substrate"):
                    title += f" on {h['substrate']}"
                rp = h.get("recommended_params") or h.get("params_used") or h.get("params")
                extra = f" params {rp}" if rp else ""
                # 성공/실패와 조건 불일치는 요약 단계에서부터 눈에 띄어야 한다 —
                # 그래야 모델이 raw 덤프를 안 읽어도 '따라할 것/피할 것'을 구분한다.
                # 마커는 ASCII 로 — 이 문자열은 프롬프트 본문에 들어가고 콘솔에도
                # 찍힐 수 있다(cp949 콘솔에서 비ASCII 는 인코딩 에러를 낸다).
                if h.get("is_success") is True:
                    extra += " [OK]"
                elif h.get("is_success") is False:
                    extra += " [FAILED-avoid]"
                if h.get("condition_warning"):
                    extra += " [different-substrate]"
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
        elif name == "run_grid_scan":
            wm.observations.append(
                f"Grid scan: {result.get('n_measured', 0)}/{result.get('n_points', 0)} points measured")
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
    _t0 = time.time()
    raw = _call_tool(ctx, name, args)
    _elapsed_ms = (time.time() - _t0) * 1000.0
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
            "img_b64": img_b64, "question": question, "tool_call_id": tool_call_id,
            "elapsed_ms": _elapsed_ms}


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


def _propose(llm_tools, wm: WorkingMemory, plan_note: str = "",
             rlog=None, stage: str = "CoALA propose", step: int = 0) -> AIMessage:
    """Propose — 작업기억(+계획 진행 문구)을 주입해 다음 행동 후보(tool_calls)를 생성한다.

    반환된 AIMessage.tool_calls가 후보 목록이다(0개면 finish = 최종 답변).
    시스템 프롬프트/작업기억/진행 문구는 세션 히스토리에 남기지 않고 매 호출마다 새로
    붙인다(중복·오염 방지).
    """
    content = SYSTEM_PROMPT + "\n\n" + wm.render()
    # 세션 요약(내 세션 라벨 + 지금까지 저장한 산출물)을 매 호출마다 새로 만든다 —
    # 작업기억에 넣으면 낡은 목록이 누적돼 모델이 지난 상태를 현재로 오인한다.
    _sess = run_store.summary_for_prompt()
    if _sess:
        content += f"\n\n[This session]\n{_sess}\n"
    if plan_note:
        content += "\n\n" + plan_note
    # rlog.invoke 는 llm_tools.invoke 를 그대로 부르고 프롬프트·생각·본문·토큰통계를
    # 남긴다. 로거가 없거나 로깅이 꺼져 있으면 llm_tools.invoke(...) 와 완전히 같다.
    rlog = rlog or reason_log.NULL
    return rlog.invoke([SystemMessage(content=content)] + wm.messages,
                       llm=llm_tools, stage=stage, step=step)


def _evaluate_and_select(llm_plain, wm: WorkingMemory, candidates: list[dict],
                         dose: float, rlog=None) -> tuple[dict, dict]:
    """Evaluate + Select — '실행(commit) 후보'가 여럿이면 점수화 후 argmax, 하나면 그대로.

    여기 들어오는 candidates는 grounding/learning(실행 대상)뿐이다 — retrieval은
    planning 단계에서 이미 처리되어 이 지점에 오지 않는다.

    Returns (선택된 tool_call, {"scores": [...], "reason": str}) — 두 번째는 UI/로그용.

    후보 1개면 LLM 호출 없이 통과시킨다(논문 §4.6: 단순 상황은 평가 생략). 여럿이면
    plain LLM으로 유용성·안전(dose/광손상)·근거성을 0~1 점수화한다. 파싱 실패 시 첫
    후보로 폴백해 사이클이 멈추지 않게 한다.
    """
    rlog = rlog or reason_log.NULL
    if len(candidates) == 1:
        rlog.phase("evaluate", "후보 1개 — 평가 생략(논문 4.6: 단순 상황)",
                   _candidate_label(candidates[0]))
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
    rlog.phase("evaluate", f"{len(candidates)}개 후보를 점수화 (누적 dose {dose:.1f} mJ 반영)",
               listing)
    try:
        resp = rlog.invoke([HumanMessage(content=prompt)], llm=llm_plain,
                           stage="CoALA evaluate (도구 없는 평가자 호출)")
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
        rlog.phase("evaluate", "점수 JSON 파싱 실패 → 첫 후보로 폴백")
        return candidates[0], {"scores": [], "reason": "evaluation parse failed - first candidate chosen"}

    best_idx = max(range(len(candidates)), key=lambda i: scores[i])
    rlog.phase("evaluate", "점수 " + ", ".join(
        f"{_candidate_label(c)}={s:.2f}" for c, s in zip(candidates, scores)), reason)
    return candidates[best_idx], {"scores": scores, "reason": reason}


# ══════════════════════════════════════════════════════════════════════════════
# Planning stage — retrieval을 반복 호출해 근거를 쌓고, commit 후보가 나오면 종료
# ══════════════════════════════════════════════════════════════════════════════

def _planning_stage(llm_tools, ctx: dict, wm: WorkingMemory,
                    propose_state: list, outcome: dict,
                    rlog=None, cycle: int = 0) -> Iterator[dict]:
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
    rlog = rlog or reason_log.NULL

    for step in range(_MAX_PLANNING_STEPS):
        if propose_state[0] <= 0:
            rlog.phase("plan", f"cycle {cycle} · propose 예산 소진 → stuck")
            outcome["kind"] = "stuck"
            return

        # 지금까지의 조회 이력으로 "같은 조회를 또 하고 있나"를 판단해 진행 문구를 만든다.
        # (직전 조회가 그 이전에도 나왔던 것이면 반복으로 본다.)
        repeated = bool(prior_sigs) and prior_sigs[-1] in prior_sigs[:-1]
        plan_note = _plan_progress_note(step + 1, len(prior_sigs), repeated)

        rlog.phase("plan", f"cycle {cycle} · 계획 라운드 {step + 1}/{_MAX_PLANNING_STEPS}"
                           + (" · 같은 조회 반복 감지" if repeated else ""), plan_note)
        try:
            ai_msg = _propose(llm_tools, wm, plan_note, rlog=rlog,
                              stage=f"CoALA propose (cycle {cycle}, plan {step + 1})",
                              step=_MAX_AGENT_STEPS - propose_state[0] + 1)
        except Exception as e:
            outcome["kind"] = "error"
            outcome["detail"] = f"LLM call failed (propose): {type(e).__name__}: {e}"
            return
        propose_state[0] -= 1

        candidates = list(ai_msg.tool_calls or [])

        # 후보 없음 = finish. 모델이 최종 보고서를 낸 것 → 이번 턴 종료.
        if not candidates:
            rlog.phase("plan", f"cycle {cycle} · 도구 후보 없음 → 최종 보고서로 턴 종료")
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

            rlog.phase("retrieval",
                       f"cycle {cycle} · 정보수집 {len(planning_actions)}건 실행 (사이클을 닫지 않음)",
                       " / ".join(_candidate_label(c) for c in planning_actions))
            if commit_actions:
                rlog.phase("retrieval",
                           f"cycle {cycle} · 같은 응답에 섞여 나온 실행 후보 "
                           f"{len(commit_actions)}건은 버림 (다음 propose 에서 재제안시킴)",
                           " / ".join(_candidate_label(c) for c in commit_actions))

            yield {"type": "phase", "phase": "plan",
                   "message": "Information gathering (retrieval): "
                              + " / ".join(_candidate_label(c) for c in planning_actions),
                   "candidates": [_candidate_label(c) for c in planning_actions]}

            for tc in planning_actions:
                ex = _execute_one(ctx, tc["name"], dict(tc.get("args") or {}), tc.get("id") or "")
                rlog.observed(ex["name"], ex["result"], ex.get("elapsed_ms", 0.0), ex["action"])
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
        rlog.phase("plan", f"cycle {cycle} · 정보수집 종료 · 실행 후보 {len(commit_actions)}건 확보 "
                           f"→ evaluate/select 로", " / ".join(_candidate_label(c) for c in commit_actions))
        outcome["kind"] = "commit"
        outcome["commit"] = commit_actions
        outcome["commit_text"] = _msg_text(ai_msg)
        return

    # planning 예산 소진 — 실행/종료 결정을 못 내림.
    rlog.phase("plan", f"cycle {cycle} · 계획 라운드 {_MAX_PLANNING_STEPS}회를 다 쓰고도 "
                       f"실행/종료를 못 정함 → stuck")
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
    # 추론 로그(results/<run_id>/<문항>.log). AILA 와 달리 여기는 session_id 를 인자로
    # 받으므로 그대로 넘긴다. 로깅이 꺼져 있으면 무동작 대역이라 경로가 전혀 안 바뀐다.
    rlog = reason_log.open_turn("CoALA", user_message, session_id=session_id)
    try:
        if llm_tools is None or llm_plain is None:
            rlog.failed("Ollama LLM is not connected.")
            yield {"type": "error",
                   "detail": "Ollama LLM is not connected. "
                             "(Check that langchain-ollama is installed and the Ollama server is running)"}
            return

        ctx = {"dispatch": _get_dispatch(), "dose": 0.0, "session_id": session_id,
               "tool_call_order": [], "learned": False, "goal": user_message.strip()}
        wm = WorkingMemory(messages=list(history) + [HumanMessage(content=user_message)])
        wm.goal = user_message.strip()

        propose_state = [_MAX_AGENT_STEPS]   # 턴 전체 propose 호출 총량 가드(가변 공유)

        for _cycle in range(_MAX_CYCLES):
            cycle = _cycle + 1
            rlog.phase("cycle", f"────── 사이클 {cycle} 시작 "
                                f"(propose 예산 {propose_state[0]}/{_MAX_AGENT_STEPS}, "
                                f"누적 dose {ctx['dose']:.1f} mJ) ──────")
            # ── 1) PLANNING STAGE ────────────────────────────────────────────────
            outcome: dict = {}
            yield from _planning_stage(llm_tools, ctx, wm, propose_state, outcome,
                                       rlog=rlog, cycle=cycle)

            kind = outcome.get("kind")

            if kind == "error":
                rlog.failed(outcome.get("detail", "planning failed"))
                yield {"type": "error", "detail": outcome.get("detail", "planning failed")}
                return

            if kind == "finish":
                rlog.final(outcome["final_text"], ctx)
                yield {"type": "final", "text": outcome["final_text"],
                       "ctx": ctx, "messages": wm.messages}
                return

            if kind != "commit":
                # stuck — 계획만 반복하고 실행/종료를 못 정함. 안전하게 턴을 닫는다.
                _stuck = ("Reached the planning-stage budget, ending this turn. "
                          "Please check the progress and request again.")
                rlog.final(_stuck, ctx)
                yield {"type": "final", "text": _stuck,
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
                chosen, meta = _evaluate_and_select(llm_plain, wm, commit_candidates,
                                                    ctx["dose"], rlog=rlog)
            except Exception as e:
                rlog.failed(f"LLM call failed (evaluate): {type(e).__name__}: {e}")
                yield {"type": "error",
                       "detail": f"LLM call failed (evaluate): {type(e).__name__}: {e}"}
                return

            rlog.phase("select", f"cycle {cycle} · 선택 → {_candidate_label(chosen)}",
                       (meta.get("reason") or "") +
                       (f"\n(후보 {len(commit_labels)}개: " + " / ".join(commit_labels) + ")"
                        if len(commit_labels) > 1 else ""))

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

            rlog.executing(chosen["name"], dict(chosen.get("args") or {}))
            ex = _execute_one(ctx, chosen["name"], dict(chosen.get("args") or {}),
                              chosen.get("id") or "")
            rlog.observed(ex["name"], ex["result"], ex.get("elapsed_ms", 0.0), ex["action"])
            yield {"type": "tool", "name": ex["name"], "args": ex["args"],
                   "result": ex["result"], "action": ex["action"]}

            wm.messages.append(ToolMessage(
                content=json.dumps(ex["result"], ensure_ascii=False, default=str),
                tool_call_id=ex["tool_call_id"],
            ))
            _update_working_memory(wm, ex["name"], ex["args"], ex["result"], ex["action"])

            if ex["img_b64"]:
                rlog.phase("observe", f"cycle {cycle} · 이미지 1장을 모델에게 주입 "
                                      f"(base64 {len(ex['img_b64'])}자, 로그에는 싣지 않음)")
                wm.messages.append(HumanMessage(
                    content=[
                        {"type": "text", "text": ex["question"] or "Microscope camera image:"},
                        {"type": "image_url",
                         "image_url": f"data:image/png;base64,{ex['img_b64']}"},
                    ],
                    additional_kwargs={_INJECTED_IMAGE: True},
                ))
            # 다음 사이클로 (관측을 반영해 다시 planning)

        _capped = (f"Stopped after reaching the maximum number of cycles ({_MAX_CYCLES}). "
                   "Please check the progress and request again.")
        rlog.final(_capped, ctx)
        yield {"type": "final", "text": _capped, "ctx": ctx, "messages": wm.messages}
    finally:
        # 벤치 러너가 중단하면 server.py 가 gen.close() 를 부른다(GeneratorExit).
        # 그때도 로그 꼬리가 닫히도록 finally 에 둔다.
        rlog.close()


# ══════════════════════════════════════════════════════════════════════════════
# 세션 관리 + SSE 진입점 (공개 API — 서버 계약)
# ══════════════════════════════════════════════════════════════════════════════

# 세션별 LangChain 메시지 히스토리(대화 기억). cross-session 노하우는 별도로 JSON에 축적.
_SESSIONS: dict[str, list] = {}
# 보존할 최대 사용자 턴 수. AILA와 동일한 이유·값(30). 이건 서버 RAM이 아니라 매 호출
# 프롬프트 토큰 수(=num_ctx 예산)를 좌우한다 — 100은 문항 맥락 누적으로 컨텍스트를
# 폭주시켜 무응답을 냈고, 30이면 num_ctx(32768) 아래에 들어오면서 되묻기 맥락도 유지된다.
_HISTORY_MAX_TURNS = 30


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
    if name == "preview_grid_scan":
        return (f"🔲 Grid preview {result.get('rows', '?')}×{result.get('cols', '?')} "
                f"({result.get('n_in_view', '?')}/{result.get('n_points', '?')} in view)")
    if name == "run_grid_scan":
        return (f"🗺️ Grid scan done "
                f"({result.get('n_measured', '?')}/{result.get('n_points', '?')} points)")
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
    if name == "list_uploaded_files":
        files = result.get("files", [])
        if not files:
            return "📎 Attached files — none"
        return f"📎 Attached files → {', '.join(f.get('filename', '?') for f in files)}"
    if name == "inspect_file":
        return (f"🔍 Inspected {result.get('filename', '?')} "
                f"({result.get('n_rows', '?')} rows × {result.get('n_cols', '?')} cols)")
    if name == "run_analysis":
        return f"🧮 Analysis code executed ({result.get('image_count', 0)} figure(s))"
    return f"🔧 {name} called"


def _grid_gate_begin_turn(interactive: bool) -> None:
    """raman_tools의 그리드 사람-승인 게이트에 턴 시작을 알린다(대화=강제 ON, 벤치마크=OFF).
    하드웨어 모듈 import가 실패하는 개발 PC에서는 조용히 무시한다 — 그 경우 grid scan 자체가
    'Hardware not connected'로 막히므로 게이트는 무의미하다. (AILA와 동일한 훅.)"""
    try:
        from backend.hw_tools.raman_tools import grid_gate_begin_turn
        grid_gate_begin_turn(interactive=interactive)
    except Exception:
        pass


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
    # 이 턴의 산출물이 갈 세션 폴더를 연다(data/runs/<sid>/) — AILA 와 동일.
    run_store.begin_session(sid, "CoALA")

    try:
        llm_tools = _get_llm_tools()
        llm_plain = _get_llm_plain()
        history = _SESSIONS.get(sid, [])
        # 새 사용자 턴 시작 — 그리드 사람-승인 게이트를 이번 턴 상태로 맞춘다.
        # 자율 모드(_AUTONOMOUS, 기본 ON)에서는 대화 경로에서도 끈다 — AILA 와 동일 정책.
        # (근거는 single_agent_AILA.py 의 같은 위치 주석 참고.)
        _grid_gate_begin_turn(interactive=not _AUTONOMOUS)

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
        used_measurement = bool({"acquire_spectrum", "run_grid_scan"}
                                 & set(final_ctx.get("tool_call_order", [])))
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
    # 이 문항의 산출물이 갈 세션 폴더를 연다 — AILA 와 동일.
    run_store.begin_session(sid, "CoALA")
    # 벤치마크는 사람이 없는 자율 평가 — 그리드 승인 게이트를 끈다(안 끄면 모든 격자
    # 스캔이 승인 없이 거부된다).
    _grid_gate_begin_turn(interactive=False)
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
        used_measurement = bool({"acquire_spectrum", "run_grid_scan"}
                                 & set((final_ctx or {}).get("tool_call_order", [])))
        turn.complete("done" if used_measurement else "chat", final_text, final_ctx)
    return {"final_report": final_text}