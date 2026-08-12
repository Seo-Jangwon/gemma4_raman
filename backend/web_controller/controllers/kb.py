"""
지식베이스(KB) 라우터 — 운영·진단용.

  GET  /api/kb/status    KB 에 항목이 몇 개 실려 있는지
  GET  /api/kb/search    에이전트와 똑같은 경로로 검색해 보기 (디버깅)
  POST /api/kb/reload    캐시 비우기 (knowledge_base.json 수정 후)

에이전트는 이 HTTP API 를 쓰지 않는다 — search_knowledge_base 도구로 knowledge.search_kb()
를 직접 호출한다. 여기 있는 건 사람용이다:
  운영: knowledge_base.json 을 고친 뒤 reload
  디버깅: 에이전트가 뭘 검색해 오는지 눈으로 확인
  실험 전 점검: KB 가 비어 있지 않은지 확인 (가장 중요)

[upload / reindex 가 없어진 이유 — 2026-08-12]
Chroma 벡터 색인을 걷어냈다(backend/service/knowledge/search.py 머리말 참고).
색인기가 없으니 재색인할 대상이 없고, kb_sources/ 에 PDF 를 올려도 읽는 코드가
없다. '조용히 아무 일도 안 하는 엔드포인트'를 남기느니 지운다.
지식을 늘리는 경로는 이제 하나다 — knowledge_base.json 편집 → POST /api/kb/reload.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["kb"])


@router.get("/kb/status")
async def kb_status_endpoint():
    """KB 진단 — 항목이 몇 개 실려 있는지.

    ⚠ 벤치마크를 돌리기 전에 반드시 확인할 것. kb_json_entries 가 0 이면 모든
    search_knowledge_base 가 빈 결과를 돌려주고, 모델이 파라미터를 전부 스스로
    추측한다 — "KB 를 붙인 결과"라고 쓴 게 전부 거짓이 된다.
    """
    from backend.service.knowledge import kb_status
    return kb_status()


@router.get("/kb/search")
async def kb_search_endpoint(q: str, top_k: int = 3):
    """에이전트와 똑같은 경로로 KB를 검색해 본다(디버깅용).

    에이전트가 이상한 파라미터를 고를 때, 프롬프트 탓인지 검색 탓인지 가르는 데 쓴다.
    """
    from backend.service.knowledge import search_kb
    if not q.strip():
        raise HTTPException(status_code=400, detail="q가 비어 있습니다")
    hits = search_kb(q, top_k=top_k)
    return {
        "query": q,
        "count": len(hits),
        # 항목별 _retriever와 별개로, 응답 수준에서도 한 번 더 노출한다.
        "retriever": hits[0].get("_retriever") if hits else None,
        "results": hits,
    }


@router.post("/kb/reload")
async def kb_reload_endpoint():
    """캐시를 비워 다음 조회 때 디스크를 다시 읽게 한다.

    knowledge_base.json을 고쳤을 때 서버를 껐다 켜지 않아도 되게 하는 용도.
    """
    from backend.service.knowledge import kb_status, reload_kb
    reload_kb()
    return {"ok": True, "message": "KB 캐시를 비웠습니다", "status": kb_status()}
