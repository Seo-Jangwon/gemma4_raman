# -*- coding: utf-8 -*-
"""지식베이스 — 읽기(search)와 쓰기(ingest)를 나눠 둔 패키지.

    search.py    질의 → 청크. Chroma 벡터 검색, 실패하면 키워드 폴백.
    ingest.py    pdf/txt/json/스펙트럼 → 색인. 무거운 의존(pymupdf 등)이 여기 있다.
    kb_sources/  색인 대상 원본을 드랍하는 폴더 (git 추적)
    knowledge_base/  Chroma 영속 디렉터리 (git 무시, 언제든 재생성되는 파생물)

[여기서 search 만 재수출하는 이유]
호출부(에이전트 · /api/kb)는 검색만 쓴다. ingest 는 pymupdf 를 끌어오므로, 패키지를
import 하는 것만으로 색인용 의존이 따라 들어오면 서버 기동이 그만큼 무거워지고
그 라이브러리가 없는 PC 에서는 검색까지 같이 죽는다. 색인은 부를 때만 import 한다:

    python -m backend.service.knowledge.ingest
"""
from backend.service.knowledge.search import (  # noqa: F401
    KB_SOURCES_DIR,
    kb_status,
    reload_kb,
    search_kb,
    search_spectra,
)
