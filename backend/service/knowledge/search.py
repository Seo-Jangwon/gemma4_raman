# -*- coding: utf-8 -*-
"""
지식베이스 검색 — knowledge_base.json 을 키워드로 훑는다. LLM 은 쓰지 않는다.

┌─────────────────────────────────────────────────────────────────────────────┐
│ 이 모듈의 공개 API                                                           │
│   search_kb(query, top_k)   → 텍스트 지식 검색   [시그니처 불변]             │
│   kb_status()               → 지금 KB 에 뭐가 몇 개 들었는지 진단            │
│   reload_kb()               → 캐시 비우기 (/api/kb/reload)                   │
└─────────────────────────────────────────────────────────────────────────────┘

[왜 벡터 검색(Chroma)을 걷어냈는가 — 2026-08-12]
붙어 있던 Chroma 인덱스가 **문서 0개**였다. 즉 도입한 이래 한 번도 답한 적이 없고,
모든 검색이 조용히 아래 키워드 매칭으로 처리되고 있었다. kb_status() 로 확인한 값:

    {"docs_count": 0, "spectra_count": 0, "retriever": "keyword", "kb_json_entries": 5}

로그에도 같은 증거가 남아 있다 — search_knowledge_base 응답이 2 ms 다. 벡터 검색은
/api/embed 로 임베딩 서버에 HTTP 왕복을 해야 해서 2 ms 에 끝날 수 없다.

그래서 없앤 것은 '기능'이 아니라 '한 번도 실행된 적 없는 경로'다. 함께 사라진 것:
chromadb + onnxruntime(~200MB) 의존, bge-m3 임베딩 왕복, 컬렉션 차원 분리 안전장치,
색인기(ingest.py)와 /api/kb/{upload,reindex}.

[언제 되돌려야 하는가]
매뉴얼 PDF 처럼 **수백 쪽짜리 원본을 넣기로 하면** 그때다. 항목마다 keywords 를 손으로
달 수 없는 규모가 되면 임베딩이 값을 한다. 지금 KB 는 손 큐레이션 5항목이라, 동의어를
keywords 배열에 직접 적는 편이 더 정확하고 디버깅도 된다.

[의도적으로 남긴 것 — _retriever 필드]
결과마다 "_retriever": "keyword" 를 계속 박는다. 지금은 검색기가 하나라 상수지만,
과거 벤치마크 로그가 이 필드로 검색기를 추적하고 있어 모양을 깨지 않는다.

[이 모듈에 채팅 LLM import 가 없어야 하는 이유]
KB 는 '있으면 더 잘하는' 참고자료이지 필수 의존이 아니다. 여기서 무거운 것을 import
하면 그게 죽는 순간 검색이 아니라 측정 자체가 막힌다. 이 파일은 표준 라이브러리만 쓴다.

═══════════════════════════════════════════════════════════════════════════════
[저장 구조]

  backend/service/knowledge/
  ├── search.py             ← 이 파일. 검색만 한다(읽기 전용).
  └── kb_sources/
      └── knowledge_base.json   ← 손 큐레이션 지식 (git 추적). 이게 KB 전부다.

지식을 늘리려면 knowledge_base.json 에 항목을 추가하고 POST /api/kb/reload 를 부른다
(서버 재기동 불필요). 항목 형식:

    {"title": ..., "content": ..., "keywords": [...], "recommended_params": {...}}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

_HERE = Path(__file__).parent

# 손 큐레이션 원본. KB 는 이 파일 하나다.
_KB_JSON_PATH = _HERE / "knowledge_base.json"

# 원본을 드랍하는 폴더. knowledge_base.json 이 실제로 사는 곳이기도 하다.
KB_SOURCES_DIR = _HERE / "kb_sources"

_kb_cache: list[dict] | None = None


# ══════════════════════════════════════════════════════════════════════════════
# 경고 (프로세스당 1회만)
# ══════════════════════════════════════════════════════════════════════════════

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """같은 경고를 도구 호출마다 반복해 로그를 도배하지 않도록 1회만 찍는다.

    [머리표가 ASCII 인 이유 — 이모지를 쓰면 안 된다]
    윈도우 콘솔의 stderr 는 cp949 이고 파이썬은 인코딩 불가 문자에 backslashreplace 를
    적용한다. "⚠"를 쓰면 실제로 "\\u26a0"이라는 글자가 찍힌다(확인함). 이 경고는
    "KB 가 비었다"는 실험 유효성 경보라서 어떤 콘솔에서도 읽혀야 한다.
    """
    if key in _warned:
        return
    _warned.add(key)
    print(f"[knowledge][WARNING] {msg}", file=sys.stderr, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 적재
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_kb_json() -> Path | None:
    """knowledge_base.json 이 실제로 있는 자리. 없으면 None.

    두 자리를 다 보는 이유: 색인 대상으로 쓰려고 kb_sources/ 로 옮겨 두는 일이 잦다.
    한 자리만 보면 파일이 멀쩡히 있는데 KB 가 조용히 비고, 그게 벤치마크 최악의
    시나리오다.
    """
    return next((p for p in (_KB_JSON_PATH, KB_SOURCES_DIR / _KB_JSON_PATH.name)
                 if p.exists()), None)


def load_kb() -> list[dict]:
    """knowledge_base.json 을 읽어 항목 리스트를 반환한다(실패 시 빈 리스트).

    [주의 — 프로세스 수명 동안 캐시된다]
    파일을 고쳐도 서버를 재시작해야 반영된다. 핫 리로드는 reload_kb() 또는
    POST /api/kb/reload.

    [왜 예외를 삼키는가]
    KB 는 참고자료이지 필수 의존이 아니다. JSON 이 깨졌다고 측정을 막으면 안 된다.
    다만 조용히 비면 벤치마크가 무효가 되므로 경고는 반드시 남긴다.
    """
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    path = _resolve_kb_json()
    try:
        if path is not None:
            with open(path, encoding="utf-8") as f:
                _kb_cache = json.load(f)
        else:
            _kb_cache = []
            _warn_once("no_kb_json", f"{_KB_JSON_PATH.name} not found in "
                                     f"{_HERE.name}/ or {KB_SOURCES_DIR.name}/ -> KB is empty.")
    except Exception as e:
        # JSON 문법 오류(트레일링 콤마 등)가 여기로 온다. 예전엔 완전히 조용해서
        # KB 가 통째로 사라져도 아무도 몰랐다 — 그게 벤치마크 최악의 시나리오다.
        _kb_cache = []
        _warn_once("kb_json_broken", f"{_KB_JSON_PATH.name} failed to parse -> KB is empty: {e}")
    return _kb_cache


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API
# ══════════════════════════════════════════════════════════════════════════════

def search_kb(query: str, top_k: int = 3) -> list[dict]:
    """질의 단어가 걸리는 지식 항목을 최대 top_k 개 반환한다.

    질의를 공백으로 쪼갠 각 단어가 (title + content + keywords) 를 소문자로 이어붙인
    텍스트에 들어 있으면 1점. 점수 내림차순 상위 top_k 개. 0점 항목은 제외.

    한계: "탄소나노튜브"처럼 keywords 에 없는 표현은 놓친다. 한/영 교차 검색이 안 되어
    keywords 에 양쪽을 손으로 다 적어야 한다. 항목이 수백 개로 늘면 이게 아프고,
    그때가 임베딩을 다시 붙일 시점이다(모듈 머리말 참고).

    Returns
    -------
    list[dict] — knowledge_base.json 원본 구조 + 추적용 언더스코어 필드:
        {"title": str, "content": str,
         "keywords": [str],                 (있을 때만)
         "recommended_params": {...},       (있을 때만)
         "_retriever": "keyword",
         "_score": int,                     ← 매칭된 질의 단어 수
         "_source": str}
    """
    query = (query or "").strip()
    if not query:
        return []

    kb = load_kb()
    if not kb:
        return []

    query_words = set(query.lower().split())
    scored: list[tuple[int, dict]] = []

    for entry in kb:
        text = (
            entry.get("title", "") + " "
            + entry.get("content", "") + " "
            + " ".join(entry.get("keywords", []))
        ).lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[dict] = []
    for score, entry in scored[:top_k]:
        e = dict(entry)
        e["_retriever"] = "keyword"
        e["_score"] = score
        e["_source"] = _KB_JSON_PATH.name
        out.append(e)
    return out


def kb_status() -> dict:
    """KB 진단 — 항목이 몇 개 실려 있는지.

    ⚠ 벤치마크 전에 확인할 것. kb_json_entries 가 0 이면 모든 search_knowledge_base 가
    빈 결과를 돌려주고, 모델은 파라미터를 전부 스스로 추측하게 된다.
    """
    entries = load_kb()
    path = _resolve_kb_json()
    return {
        # 찾아본 자리가 아니라 **실제로 읽은 자리**를 준다. 둘을 헷갈리면
        # "파일은 있는데 KB 가 비었다"를 진단할 수가 없다.
        "kb_json_path": str(path) if path else None,
        "kb_sources_dir": str(KB_SOURCES_DIR),
        "kb_json_entries": len(entries),
        "retriever": "keyword",
    }


def reload_kb() -> None:
    """캐시를 비워 다음 조회 때 디스크에서 다시 읽게 한다(/api/kb/reload용)."""
    global _kb_cache
    _kb_cache = None
    _warned.clear()


# ──────────────────────────────────────────────────────────────────────────────
# 자체 점검:  python -m backend.service.knowledge.search
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _kb_cache = [
        {"title": "Graphene", "content": "D band and G band.",
         "keywords": ["graphene", "carbon"], "recommended_params": {"laser_power_pct": 20}},
        {"title": "Silicon", "content": "520 cm-1 peak.", "keywords": ["silicon", "si"]},
    ]

    hits = search_kb("graphene carbon")
    assert len(hits) == 1 and hits[0]["title"] == "Graphene", hits
    assert hits[0]["_score"] == 2, hits                      # 두 단어 다 맞음
    assert hits[0]["_retriever"] == "keyword", hits
    assert hits[0]["recommended_params"] == {"laser_power_pct": 20}, hits

    # 원본을 오염시키지 않는다(호출부가 결과를 고쳐도 KB 는 그대로여야 한다).
    hits[0]["title"] = "MUTATED"
    assert _kb_cache[0]["title"] == "Graphene", _kb_cache

    assert search_kb("") == []
    assert search_kb("   ") == []
    assert search_kb("nothing matches here") == []

    # 점수 높은 것이 먼저.
    ranked = search_kb("silicon graphene carbon")
    assert [h["title"] for h in ranked] == ["Graphene", "Silicon"], ranked

    assert len(search_kb("silicon graphene carbon", top_k=1)) == 1

    _kb_cache = []
    assert search_kb("graphene") == []                        # 빈 KB 는 조용히 빈 결과

    print("search.py self-check passed")
