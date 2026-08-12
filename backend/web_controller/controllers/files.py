"""
첨부 데이터 파일 라우터.

  POST /api/files/upload   채팅창에 붙인 측정 데이터(csv/excel/txt)를 저장하고 file_id 발급
  GET  /api/files          해당 날짜에 올라온 첨부 목록 (프론트 표시·디버깅용)

[kb 라우터와 목적이 다르다]
  /api/kb/upload     문서(pdf/md/…)를 지식베이스에 색인 — '프로토콜을 가르친다'
  /api/files/upload  표 데이터를 에이전트가 분석 — '이 데이터를 봐 달라'

에이전트는 이 HTTP API 를 쓰지 않는다. list_uploaded_files / inspect_file 도구로
backend.service.store.upload_store 를 직접 부른다(파일 위치 규칙은 upload_store 머리말 참고).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/api", tags=["files"])


@router.post("/files/upload")
async def files_upload(file: UploadFile = File(...)):
    """데이터 파일을 data/uploads/<날짜>/에 저장하고 file_id를 돌려준다.

    파싱은 여기서 하지 않는다 — 큰 파일이면 업로드 응답이 그만큼 늦어지고, 애초에
    '어떻게 읽을지'는 에이전트가 inspect_file 로 판단할 몫이다. 여기는 저장만 한다.
    """
    from backend.service.store.upload_store import ALLOWED_SUFFIXES, save_upload

    name = Path(file.filename or "").name       # 경로 탈출 방지 — 파일명만 취한다
    if not name:
        raise HTTPException(status_code=400, detail="파일명이 없습니다")
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(f"지원하지 않는 형식입니다. 가능: {', '.join(sorted(ALLOWED_SUFFIXES))} "
                    f"(논문·프로토콜 문서는 /api/kb/upload 로)"),
        )
    try:
        data = await file.read()
        return save_upload(name, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"저장 실패: {e}")


@router.get("/files")
async def files_list(date: Optional[str] = None):
    """해당 날짜(기본 오늘)에 올라온 첨부 파일 목록 — 프론트 표시/디버깅용."""
    from backend.service.store.upload_store import list_uploads
    return {"ok": True, "files": list_uploads(date)}
