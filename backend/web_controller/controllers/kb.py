"""
지식베이스(KB) 라우터 — RAG 검색기 운영·진단용.

  GET  /api/kb/status    어느 검색기가 살아있고 인덱스에 몇 개 들었는지
  GET  /api/kb/search    에이전트와 똑같은 경로로 검색해 보기 (디버깅)
  POST /api/kb/reload    캐시만 비우기 (knowledge_base.json 수정 후)
  POST /api/kb/upload    문서를 kb_sources/에 저장 (색인은 안 함)
  POST /api/kb/reindex   kb_sources/를 다시 읽어 Chroma 재색인

에이전트는 이 HTTP API 를 쓰지 않는다 — search_knowledge_base 도구로 knowledge.search_kb()
를 직접 호출한다. 여기 있는 건 사람용이다:
  운영: 문서 업로드 → 재색인
  디버깅: 에이전트가 뭘 검색해 오는지 눈으로 확인
  실험 전 점검: 지금 벡터 검색인지 키워드 폴백인지 확인 (가장 중요)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.web_controller.setups.state import StateDep, run_in_worker

router = APIRouter(prefix="/api", tags=["kb"])

# KB 문서로 받아들이는 확장자. 표 데이터(csv/xlsx)는 여기가 아니라 /api/files/upload 로 간다.
ALLOWED_DOC_SUFFIXES = {".pdf", ".txt", ".md", ".json"}


@router.get("/kb/status")
async def kb_status_endpoint():
    """KB 진단 — 어느 검색기가 살아있고 인덱스에 몇 개 들었는지.

    ⚠ 벤치마크를 돌리기 전에 반드시 확인할 것. retriever가 "keyword"인데 그걸
    모른 채 실험하면 "벡터 RAG를 붙인 결과"라고 쓴 게 전부 거짓이 된다.
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
    (Chroma 인덱스 자체를 갱신하려면 /api/kb/reindex가 필요하다 — 이건 캐시만 비운다.)
    """
    from backend.service.knowledge import kb_status, reload_kb
    reload_kb()
    return {"ok": True, "message": "KB 캐시를 비웠습니다", "status": kb_status()}


@router.post("/kb/upload")
async def kb_upload_endpoint(file: UploadFile = File(...)):
    """문서를 kb_sources/에 저장한다. 색인은 하지 않는다.

    [왜 업로드와 색인을 분리하나]
    색인은 문서 수에 비례해 임베딩을 돌리므로 수 초~수 분이 걸린다. 업로드 요청을
    그동안 붙잡아 두면 프론트가 타임아웃난다. 여러 파일을 올린 뒤 /api/kb/reindex를
    한 번 부르는 게 임베딩 왕복도 줄인다.
    """
    from backend.service.knowledge import KB_SOURCES_DIR

    name = Path(file.filename or "").name          # 경로 탈출 방지 — 파일명만 취한다
    if not name:
        raise HTTPException(status_code=400, detail="파일명이 없습니다")
    if Path(name).suffix.lower() not in ALLOWED_DOC_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 형식입니다. 가능: {', '.join(sorted(ALLOWED_DOC_SUFFIXES))}",
        )

    KB_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    dest = KB_SOURCES_DIR / name
    try:
        dest.write_bytes(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장 실패: {e}")

    return {
        "ok": True,
        "filename": name,
        "bytes": dest.stat().st_size,
        "message": "저장됨. 검색에 반영하려면 POST /api/kb/reindex 를 호출하세요.",
    }


@router.post("/kb/reindex")
async def kb_reindex_endpoint(state: StateDep, caption: bool = False):
    """kb_sources/와 knowledge_base.json을 다시 읽어 Chroma를 재색인한다.

    caption=true면 PDF 페이지를 VLM으로 캡션한다(페이지당 1회 — 매우 느림).
    색인은 블로킹 작업(임베딩 HTTP 왕복 다수)이라 워커 스레드에서 돌린다.
    """
    from backend.service.knowledge.ingest import ingest
    from backend.service.knowledge import kb_status, reload_kb

    try:
        result = await run_in_worker(state, lambda: ingest(caption=caption, with_spectra=True))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"색인 실패: {e}")

    # 색인이 컬렉션을 지웠다 다시 만들었으므로, 검색 쪽이 들고 있는 낡은 핸들을 버린다.
    reload_kb()
    return {"ok": True, "indexed": result, "status": kb_status()}
