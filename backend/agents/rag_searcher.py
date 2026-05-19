"""
RAGSearcherNode — Raman 문헌/프로토콜 지식 검색.

MVP: knowledge_base.json 키워드 매칭.
V1: chromadb + sentence-transformers 벡터 검색으로 교체.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from backend.agents.state import ExperimentState

_KB_PATH = Path(__file__).parent / "knowledge_base.json"
_kb_cache: list[dict] | None = None


def _load_kb() -> list[dict]:
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    if _KB_PATH.exists():
        with open(_KB_PATH, encoding="utf-8") as f:
            _kb_cache = json.load(f)
    else:
        _kb_cache = []
    return _kb_cache


def _keyword_search(query: str, kb: list[dict], top_k: int = 3) -> list[dict]:
    if not kb:
        return []
    query_words = set(query.lower().split())
    scored = []
    for entry in kb:
        text = (entry.get("title", "") + " " + entry.get("content", "") + " " +
                " ".join(entry.get("keywords", []))).lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def rag_searcher_node(state: ExperimentState) -> dict:
    intent = state.get("intent") or {}
    query = intent.get("primary_objective", state.get("user_message", ""))
    sample = intent.get("sample_type", "")
    full_query = f"{query} {sample}".strip()

    kb = _load_kb()
    results = _keyword_search(full_query, kb)

    return {"rag_results": results}
