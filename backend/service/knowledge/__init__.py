# -*- coding: utf-8 -*-
"""지식베이스 — 손 큐레이션 JSON 을 키워드로 검색한다.

    search.py                   질의 → 항목. 표준 라이브러리만 쓴다.
    kb_sources/
      knowledge_base.json       KB 원본 전부 (git 추적)

[색인기가 없는 이유 — 2026-08-12]
예전에는 ingest.py 가 pdf/txt/스펙트럼을 Chroma 에 색인했다. 그런데 그 인덱스가
문서 0개였다 — 도입 이래 한 번도 답한 적이 없고 모든 검색이 키워드 매칭으로
처리되고 있었다. 그래서 Chroma 와 색인기를 함께 걷어냈다(search.py 머리말 참고).

지식을 늘리는 방법은 이제 하나다: knowledge_base.json 에 항목을 추가하고
POST /api/kb/reload. 서버를 재기동할 필요는 없다.
"""
from backend.service.knowledge.search import (  # noqa: F401
    KB_SOURCES_DIR,
    kb_status,
    reload_kb,
    search_kb,
)
