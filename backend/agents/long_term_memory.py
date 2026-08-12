# -*- coding: utf-8 -*-
"""
CoALA 장기기억 — episodic(경험) / semantic(증류한 규칙) 저장소와 그 도구 4종.

CoALA(Sumers et al. 2024) §4.5·§6 의 핵심은 "자기 생성 콘텐츠를 읽고 쓸 수 있는" 장기기억이다.
세션이 끝나도 남아 다음 실험에서 조회되는 '노하우'를 디스크 JSON 에 축적한다.
단일 사용자 로컬 도구라 파일 락 없이 read-modify-write append 로 충분하다.

    coala_memory/experiences.json   episodic — 실험 한 건의 경험
    coala_memory/insights.json      semantic — 경험에서 증류한 일반 규칙

여기 있는 것은 '저장과 조회'뿐이다. 언제 기록하고 언제 조회할지를 고르는 것은
의사결정 사이클(single_agent_CoALA.py)의 몫이다 — CoALA 에서 학습은 스케줄이 아니라
에이전트가 사이클에서 **고르는 액션**이기 때문이다.

[스코프 토글 — RAMAN_MEMORY_SCOPE=session]
저장소가 coala_memory/sessions/<session_id>/ 로 갈라져, 새 session_id 로 들어오면 그 순간
빈 저장소가 된다. 벤치는 문항마다 새 session_id 로 공정성을 맞추는데 장기기억은 세션을
넘어 축적되므로, 1번 문항은 경험 0건으로 200번 문항은 199개가 남긴 경험으로 푸는 셈이
된다 — 재현도 해석도 안 되는 교란이다.
'삭제'가 아니라 '세션별 디렉터리'인 이유: ① 지우는 시점 경쟁이 없다 ② 각 문항에서 무엇을
기록했는지가 디스크에 남아 채점 근거가 된다 ③ 도구도 프롬프트도 그대로라 아키텍처가
온전하고, 모델이 recall 에 쓰는 사이클·토큰 비용도 정직하게 측정된다.
미설정이면 'global' — coala_memory/ 하나에 계속 축적된다(실사용 기본값).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

_MEMORY_DIR = Path(__file__).resolve().parent / "coala_memory"
_EPISODIC_NAME = "experiences.json"
_SEMANTIC_NAME = "insights.json"

_SCOPE = os.getenv("RAMAN_MEMORY_SCOPE", "global").strip().lower()
SESSION_SCOPED = _SCOPE == "session"
if SESSION_SCOPED:
    # 런타임 출력은 ASCII 로만 — cp949/ascii 콘솔에서도 import 가 깨지지 않게.
    print("[info] RAMAN_MEMORY_SCOPE=session: CoALA long-term memory is per-session "
          "(episodic+semantic start empty for every new session_id).")
elif _SCOPE != "global":
    print(f"[warn] RAMAN_MEMORY_SCOPE='{_SCOPE}' is not recognized "
          "(use 'global' or 'session'); falling back to 'global'.", file=sys.stderr)

#: 에피소딕 메모리 사용 여부. 끄면 도구 2종을 액션 공간에서 빼고, 프롬프트의 episodic
#: 지시문도 함께 빠진다(backend.agents.prompts 참고) — semantic 은 그대로라
#: "CoALA 에서 episodic 만 없앤" ablation 이 된다.
EPISODIC_ENABLED = os.getenv("RAMAN_EPISODIC_MEMORY", "1").strip().lower() \
    not in ("0", "false", "no", "off")
EPISODIC_TOOL_NAMES = {"recall_experiences", "record_experience"}

#: 회수 payload 예산 — 저장은 full, 회수는 이 상한 안에서 projection 한다.
_RECALL_TEXT_CAP = 200      # 자유텍스트 필드 1개의 최대 문자수
_RECALL_MAX_TOP_K = 5       # 모델이 큰 값을 넣어도 여기서 자른다


# ══════════════════════════════════════════════════════════════════════════════
# 저장소 파일
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize_sid(sid: str) -> str:
    """세션 id 를 디렉터리명으로 — detail_log._sanitize 와 같은 규칙이라
    DetailLog 파일명과 메모리 폴더명이 같은 sid 로 맞춰진다(추적이 쉬워진다)."""
    return re.sub(r"[^0-9A-Za-z_-]", "-", str(sid))[:64] or "nosession"


def _dir(ctx: dict) -> Path:
    if not SESSION_SCOPED:
        return _MEMORY_DIR
    return _MEMORY_DIR / "sessions" / _sanitize_sid(ctx.get("session_id", ""))


def _load(path: Path) -> list[dict]:
    """저장소 파일을 리스트로 로드한다. 없거나 깨졌으면 빈 리스트."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _append(path: Path, item: dict) -> None:
    """저장소 파일에 항목 하나를 append 한다(디렉터리 없으면 생성)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = _load(path)
    items.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# 검색 랭킹
# ══════════════════════════════════════════════════════════════════════════════

def _match_score(query: str, hay: str) -> int:
    """질의 토큰 중 몇 개가 건초더미 텍스트에 부분매칭되는지 — 조회 랭킹용.

    임베딩 검색을 쓰지 않는 이유: episodic 저장소는 JSON 하나라 의존성 없는 키워드
    매칭으로 충분하고, 개발 PC 에서도 임베딩 서버 없이 검사할 수 있다.

    주의: 이 점수의 상한은 '질의 토큰 수'다. "graphene" 한 단어로 조회하면 관련 에피소드가
    전부 1점으로 동점이 된다 — 그래서 점수만으로 순위를 매기면 안 되고, recall_experiences
    가 조건일치/성공/최신을 뒤이은 정렬 키로 쓴다.
    """
    hay = hay.lower()
    return sum(1 for t in re.split(r"\s+", query.lower().strip()) if t and t in hay)


def _episode_haystack(e: dict) -> str:
    """에피소드의 '잎 텍스트'만 이어붙인다 — 검색 대상 문자열.

    [왜 dict 를 통째로 str() 하면 안 되는가]
    sample_context/execution_summary/system_metrics 를 통째로 문자열화하면 값뿐 아니라
    '키 이름'까지 건초더미에 들어간다. 그러면 질의에 'power' 가 있을 때 params_used 의 키
    'power' 가 모든 에피소드에 매칭돼, 시료와 무관한 토큰이 전원의 점수를 똑같이 올리고
    순위 차이를 뭉갠다. 그래서 실제 잎 값만 골라 쓴다.
    """
    sc, ex = e.get("sample_context") or {}, e.get("execution_summary") or {}
    return " ".join([
        str(e.get("goal", "")),
        " ".join(str(t) for t in (e.get("tags") or [])),
        str(sc.get("sample", "")), str(sc.get("sample_name", "")),
        str(sc.get("substrate", "")), str(sc.get("visual_features", "")),
        str(ex.get("outcome", "")), str(e.get("lesson", "")),
    ])


def _substrate_relation(now: str, past: str) -> str:
    """현재 기판과 과거 기판의 관계 — 'match' | 'mismatch' | 'unknown'.

    파장·대물렌즈가 고정인 이 장비에서 '과거 파라미터가 지금 통하는가'를 가르는 조건은
    사실상 기판 하나다(Si 는 520cm-1 배경, 유리는 형광, 금속은 SERS 증강으로 안전 파워
    자체가 달라진다). 한쪽이 부분 문자열이면 같은 기판으로 본다('Si' vs 'Si wafer').
    과도한 정규화는 오히려 거짓 일치를 만들어 하지 않는다.
    """
    n, p = str(now or "").strip().lower(), str(past or "").strip().lower()
    if not n or not p:
        return "unknown"
    return "match" if (n == p or n in p or p in n) else "mismatch"


def _cut(v, cap: int = _RECALL_TEXT_CAP) -> str:
    s = str(v or "").strip()
    return (s[:cap] + "…") if len(s) > cap else s


def _project_episode(e: dict, relation: str = "unknown") -> dict:
    """에피소드를 '모델에게 돌려줄 형태'로 축약한다 — 저장 원본은 건드리지 않는다.

    빼는 것: tool_sequence(사이클 수에 비례해 최대 150개까지 길어지는 원시 목록 — 토큰
             폭발의 주범이고 내용은 trajectory 가 요약한다), tool_counts, id/session_id
             (모델 판단에 기여 없음. 원본은 디스크에 그대로 있다), tags(검색 핸들이라
             매칭에는 쓰되 읽을 값어치는 낮다).
    남기는 것: goal(같은 시료·기판이라도 목적이 다르면 맞는 파라미터가 다르므로 기판과
              같은 급의 전이 조건이다), substrate(조건 판정), n_measurements/dose_mj
              (1번에 성공인지 12번 만에 성공인지 — is_success 만으로는 안 보이는 신뢰도).
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
    # 기판이 다르다고 결과에서 빼버리면 모델은 "관련 경험 없음"으로 읽고 아무 근거 없이
    # 진행한다. 경고와 함께 보여주는 편이 낫다. 실패 경험도 같은 이유로 남기되,
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


# ══════════════════════════════════════════════════════════════════════════════
# 도구 구현 — 시그니처는 전부 (ctx, args) -> dict
# ══════════════════════════════════════════════════════════════════════════════

def recall_experiences(ctx: dict, args: dict) -> dict:
    """episodic memory 읽기 — 질의와 관련된 과거 실험 경험 top_k 개.

    비어 있으면 에러가 아니라 정상적인 "아직 축적된 경험 없음"으로 답한다 — 그래야 모델이
    재시도 루프에 빠지지 않고 스스로 판단하고 나중에 기록한다(session 스코프에서는 매
    세션 비어 있는 것이 정상이다).

    [랭킹이 키워드 점수만으로는 안 되는 이유]
    _match_score 의 상한은 질의 토큰 수라 "graphene" 한 단어면 관련 에피소드가 전부
    동점이다. 파이썬 sorted 는 안정 정렬이라 동점이면 입력 순서(=오래된 것 먼저)가 유지되고,
    실패한 실험도 성공한 실험과 같은 순위를 받는다. 그 상태에서 시스템 프롬프트는
    "파라미터를 추측하지 말고 이 증거에서 정하라"고 지시하므로, 시료를 태웠던 실험의 파워가
    '따라야 할 근거'로 제시될 수 있다. 그래서 정렬 키를 4단으로 둔다:
    키워드 점수 → 기판 일치 → 성공 → 최신.
    """
    query = str(args.get("query", "")).strip()
    top_k = max(1, min(int(args.get("top_k", 3) or 3), _RECALL_MAX_TOP_K))
    now_substrate = str(args.get("substrate", "")).strip()
    episodes = _load(_dir(ctx) / _EPISODIC_NAME)
    if not episodes:
        return {"ok": True, "results": [],
                "note": "No past experiments accumulated yet. After finishing this measurement, "
                        "leave one with record_experience and it will be retrievable in future experiments."}

    def rel(e):
        return _substrate_relation(now_substrate,
                                   (e.get("sample_context") or {}).get("substrate", ""))
    rank = {"match": 1, "unknown": 0, "mismatch": -1}

    if not query:
        picked = list(reversed(episodes))[:top_k]      # 질의가 없으면 최근 경험
    else:
        scored = []
        for idx, e in enumerate(episodes):             # idx 가 곧 시간순 (append 저장)
            s = _match_score(query, _episode_haystack(e))
            if s <= 0:
                continue
            ok = 1 if (e.get("execution_summary") or {}).get("is_success") else 0
            scored.append((s, rank[rel(e)], ok, idx, e))
        if not scored:
            return {"ok": True, "results": [],
                    "note": f"No past experience related to '{query}'. Decide on your own."}
        picked = [t[-1] for t in sorted(scored, key=lambda x: x[:4], reverse=True)][:top_k]

    return {"ok": True, "results": [_project_episode(e, rel(e)) for e in picked]}


def record_experience(ctx: dict, args: dict) -> dict:
    """episodic memory 쓰기(학습 액션) — 실험 한 건의 경험을 append 한다.

    시스템이 이번 턴 동안 이미 추적한 절차 정보(도구 순서·측정 횟수·조사량)를 자동으로
    실어 기록을 구체화한다 — LLM 이 굳이 넘기지 않아도 되게. record_*(학습) 메타 액션
    자체는 순서에서 뺀다.
    """
    sample = str(args.get("sample", "")).strip()
    if not sample:
        return {"ok": False, "error": "sample (sample type) is required."}

    order = [n for n in ctx.get("tool_call_order", []) if n not in _LEARNING_TOOL_NAMES]
    counts: dict = {}
    for n in order:
        counts[n] = counts.get(n, 0) + 1

    entry = {
        "id": uuid.uuid4().hex[:12],
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": ctx.get("session_id", ""),
        "goal": ctx.get("goal", ""),
        "tags": args.get("tags", []),
        "sample_context": {
            "sample": sample,
            "sample_name": str(args.get("sample_name", "")).strip(),
            # 파장·대물렌즈가 고정인 이 장비에서 '과거 조건이 지금 통하는가'를 가르는 사실상
            # 유일한 변수. 회수 시 현재 기판과 대조해 일치/불일치를 라벨링한다.
            "substrate": str(args.get("substrate", "")).strip(),
            "visual_features": str(args.get("visual_features", "")).strip(),
        },
        "execution_summary": {
            "params_used": args.get("params", {}),
            "trajectory": str(args.get("trajectory", "")).strip(),
            "outcome": str(args.get("outcome", "")).strip(),
            "is_success": args.get("is_success", False),
        },
        "lesson": str(args.get("lesson", "")).strip(),
        # 시스템 자동 기록(절차 흔적) — LLM 입력 불필요
        "system_metrics": {
            "metrics": str(args.get("metrics", "")).strip(),
            "tool_sequence": order,
            "tool_counts": counts,
            "n_measurements": counts.get("acquire_spectrum", 0),
            "dose_mj": round(float(ctx.get("dose", 0.0)), 3),
        },
    }
    try:
        _append(_dir(ctx) / _EPISODIC_NAME, entry)
    except OSError as e:
        return {"ok": False, "error": f"Failed to save experience: {e}"}
    ctx["learned"] = True
    return {"ok": True, "recorded": entry["id"], "sample": sample,
            "auto_captured": {"tool_calls": len(order),
                              "n_measurements": entry["system_metrics"]["n_measurements"],
                              "dose_mj": entry["system_metrics"]["dose_mj"]}}


def recall_insights(ctx: dict, args: dict) -> dict:
    """semantic memory(자기 생성분) 읽기.

    [이 도구가 있는 이유 — write-only 갭 보완]
    record_insight 로 쓴 일반화 지식을 '다시 읽는' 경로가 없으면 CoALA 의 lifelong learning
    루프(§4.5: self-generated 지식을 later episode 에서 재사용)가 semantic 쪽에서 반만
    닫힌다. search_knowledge_base(큐레이션 KB)와는 별개로, 에이전트가 스스로 남긴
    insights.json 을 조회한다.
    """
    query = str(args.get("query", "")).strip()
    top_k = max(1, min(int(args.get("top_k", 3) or 3), _RECALL_MAX_TOP_K))
    insights = _load(_dir(ctx) / _SEMANTIC_NAME)
    if not insights:
        return {"ok": True, "results": [],
                "note": "No generalized knowledge (insights) accumulated yet. Leave one with "
                        "record_insight and it will be retrievable in future experiments."}
    if not query:
        ranked = list(reversed(insights))
    else:
        # insights 는 (topic, insight) 두 필드뿐인 평평한 구조라 잎 텍스트 문제가 없다.
        # 동점 시 최신이 먼저 오도록 인덱스를 역순 키로 함께 쓴다.
        scored = [(_match_score(query, f"{e.get('topic','')} {e.get('insight','')}"), idx, e)
                  for idx, e in enumerate(insights)]
        ranked = [e for s, _, e in sorted(scored, key=lambda x: x[:2], reverse=True) if s > 0]
        if not ranked:
            return {"ok": True, "results": [],
                    "note": f"No generalized knowledge related to '{query}'. Decide on your own."}
    return {"ok": True, "results": ranked[:top_k]}


def record_insight(ctx: dict, args: dict) -> dict:
    """semantic memory 쓰기(학습 액션) — 경험에서 증류한 일반 규칙을 남긴다.

    episodic(개별 사건)과 달리 semantic(재사용 가능한 규칙)을 구분해 저장한다.
    """
    topic = str(args.get("topic", "")).strip()
    insight = str(args.get("insight", "")).strip()
    if not topic or not insight:
        return {"ok": False, "error": "Both topic and insight are required."}
    entry = {"id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
             "topic": topic, "insight": insight}
    try:
        _append(_dir(ctx) / _SEMANTIC_NAME, entry)
    except OSError as e:
        return {"ok": False, "error": f"Failed to save insight: {e}"}
    ctx["learned"] = True
    ctx["insight_recorded"] = True   # 엔드-오브-턴 유도가 중복 제안하지 않도록 표시
    return {"ok": True, "recorded": entry["id"], "topic": topic}


def _blocked(name: str):
    """episodic 이 꺼졌을 때의 이중 방어.

    스키마를 바인딩하지 않아도 모델이 이름을 환각해 호출할 수는 있다. 저장소에 실제로
    닿기 전에 여기서 막는다.
    """
    def _handler(ctx: dict, args: dict) -> dict:
        return {"ok": False,
                "error": f"{name} is not available in this configuration. "
                         "Proceed without episodic memory."}
    return _handler


_LEARNING_TOOL_NAMES = {"record_experience", "record_insight"}

#: 정보 수집(retrieval) 액션 — 사이클을 닫지 않고 working memory 만 채운다.
RETRIEVAL_TOOL_NAMES = {"recall_experiences", "recall_insights"}
#: 학습(learning) 액션 — grounding 과 함께 propose→evaluate→select 의 대상이다.
LEARNING_TOOL_NAMES = _LEARNING_TOOL_NAMES

#: runtime.call_tool 에 넘길 내부 액션 핸들러. episodic 이 꺼져 있으면 차단 핸들러로 바뀐다.
HANDLERS = {
    "recall_experiences": recall_experiences if EPISODIC_ENABLED else _blocked("recall_experiences"),
    "record_experience":  record_experience if EPISODIC_ENABLED else _blocked("record_experience"),
    "recall_insights":    recall_insights,
    "record_insight":     record_insight,
}


# ══════════════════════════════════════════════════════════════════════════════
# 도구 스키마 — CoALA 는 internal action 도 tool 로 노출한다 (논문 §4.1: 전부 action space)
# ══════════════════════════════════════════════════════════════════════════════
#
# 설명문의 "does not end the cycle" / "planning" 표기가 중요하다: 이것이 없으면 모델이
# 조회 한 번에 사이클이 닫히는 줄 알고 정보 수집을 아낀다.

_SCHEMA_RECALL_EXPERIENCES = {
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

_SCHEMA_RECALL_INSIGHTS = {
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

_SCHEMA_RECORD_EXPERIENCE = {
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
                "trajectory": {"type": "string",
                               "description": "1-2 sentence summary of why tools were called in this sequence "
                                              "and how issues were handled."},
                "outcome": {"type": "string", "description": "Result summary. e.g. 'G/2D bands good'."},
                "is_success": {"type": "boolean",
                               "description": "True if the measurement goal was achieved, False otherwise."},
                "metrics": {"type": "string", "description": "Quantitative metrics. e.g. 'SNR 8.3, no saturation'."},
                "lesson": {"type": "string",
                           "description": "Lesson for next time. e.g. 'power above 30% risks saturation'."},
                "tags": {"type": "array", "items": {"type": "string"},
                         "description": "3-5 key tags for searchability. e.g. ['fluorescence', 'graphene_saturation']"},
            },
            "required": ["sample", "trajectory", "is_success", "tags"],
        },
    },
}

_SCHEMA_RECORD_INSIGHT = {
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

#: CoALA 액션 공간에 실릴 장기기억 도구 스키마. episodic 이 꺼져 있으면 2종만 남는다.
SCHEMAS = [_SCHEMA_RECALL_INSIGHTS, _SCHEMA_RECORD_INSIGHT]
if EPISODIC_ENABLED:
    SCHEMAS = [_SCHEMA_RECALL_EXPERIENCES, _SCHEMA_RECALL_INSIGHTS,
               _SCHEMA_RECORD_EXPERIENCE, _SCHEMA_RECORD_INSIGHT]
