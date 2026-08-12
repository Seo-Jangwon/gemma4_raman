# -*- coding: utf-8 -*-
"""외부 검색. 지식베이스(service/knowledge)가 로컬 색인을 보는 것과 달리
이쪽은 인터넷으로 나간다 — 그래서 오프라인에서 실패하는 것이 정상이고,
호출부는 search_kb 로 폴백한다."""
from backend.service.search.web_search import web_search  # noqa: F401
