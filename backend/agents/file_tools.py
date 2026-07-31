# -*- coding: utf-8 -*-
"""
[역할] 채팅에 첨부된 데이터 파일(csv/excel/txt)을 에이전트가 다루기 위한 도구 계층.

  스키마(FILE_TOOLS)와 실행부(FILE_DISPATCH)를 한곳에 모아, AILA·CoALA 두 에이전트가
  '같은 객체'를 import 해서 쓴다. 각 에이전트 파일에서 고치는 것은 2줄뿐이다:

      ALL_TOOLS = RAMAN_TOOLS + FILE_TOOLS + [...]          # 스키마 바인딩
      if name in FILE_DISPATCH: return FILE_DISPATCH[name](args)   # _call_tool 안

  [왜 각 에이전트에 복붙하지 않고 공용 모듈인가]
  두 에이전트의 도구 능력이 어긋나는 순간 AILA↔CoALA 비교의 독립변수가 오케스트레이션이
  아니라 '어느 쪽에 도구를 더 줬는가'가 되어 실험이 무너진다(single_agent_AILA 파일
  머리말의 search_knowledge_base 추가 사유와 같은 논리). 같은 리스트·같은 dict 를
  양쪽이 import 하면 비대칭이 구조적으로 불가능해진다.

  [왜 RAMAN_TOOLS/TOOL_DISPATCH 가 아니라 여기인가]
  raman_tools.py 는 config.py 를 통해 Config.ini 를 읽으므로, 그 파일이 없는 환경에서는
  import 가 통째로 실패할 수 있다(에이전트 _get_dispatch() 의 docstring 참고). 그러면
  _get_dispatch() 가 None 이 되어 그 뒤의 모든 도구가 "Hardware is not connected." 로
  막히는데, 파일 분석은 하드웨어와 무관한 작업이라 거기 함께 묶일 이유가 없다.
  그래서 각 에이전트의 _call_tool 에서 '하드웨어 가드보다 먼저' 처리한다 —
  search_knowledge_base 가 이미 같은 이유로 같은 자리에 있다.

  [run_analysis 를 여기서 가로채는 이유]
  run_analysis 의 실체인 backend.analysis_sandbox 는 하드웨어를 import 하지 않는다
  (numpy/matplotlib + spectrum_store + upload_store 뿐). 그런데 실행 경로가
  raman_tools.TOOL_DISPATCH 를 거치기 때문에, 위 상황에서는 '순수 계산'인 분석까지
  하드웨어와 함께 막힌다. 여기서 가로채면 장비 연결 여부와 무관하게 분석이 돌고,
  모델에게 보이는 도구 이름도 하나로 유지된다.
  (스키마는 그대로 raman_tool_schemas.RAMAN_TOOLS 에 있고, 실행 경로만 이쪽으로 온다.
   따라서 TOOL_DISPATCH 의 run_analysis 항목은 이제 이 경로에 가려 쓰이지 않는다.)
"""
from __future__ import annotations

from backend.upload_store import ALLOWED_SUFFIXES, inspect_upload, list_uploads

# ══════════════════════════════════════════════════════════════════════════════
# 스키마 — RAMAN_TOOLS 와 동일한 OpenAI function 포맷
# ══════════════════════════════════════════════════════════════════════════════

_LIST_FILES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_uploaded_files",
        "description": (
            "List the data files the user attached to the chat (csv, excel, txt, etc.). "
            "Returns each file's file_id, filename and size - not its contents. "
            "Call this first when the user mentions an attached/uploaded file, or when you are "
            "asked to analyze data you did not measure yourself. "
            "It touches no hardware and is free of side effects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Upload date 'YYYY-MM-DD'. If omitted, today.",
                },
            },
            "required": [],
        },
    },
}

_INSPECT_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "inspect_file",
        "description": (
            "Inspect the STRUCTURE of one uploaded file: number of rows/columns, column names, "
            "whether each column is numeric or text, min/max/mean per numeric column, the first few "
            "rows, and (for Excel) the sheet names. It does NOT return the full data and does NOT "
            "interpret anything for you. "
            "You decide from this summary what the file contains - e.g. whether some numeric column "
            "is a Raman shift axis in cm-1, an intensity axis, a wavelength, a stage coordinate, or "
            "unrelated metadata such as sample names and measurement conditions. "
            "Then use run_analysis with file_ids to compute on the full data. "
            "It touches no hardware and is free of side effects, so call it for every attached file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "file_id from list_uploaded_files, e.g. '2026-07-24/113045_123_sample.csv'.",
                },
                "sheet": {
                    "type": "string",
                    "description": "Excel only: sheet name to inspect. If omitted, the first sheet.",
                },
                "max_rows": {
                    "type": "integer",
                    "description": "How many leading rows to preview (1-20). Default 5.",
                    "minimum": 1,
                    "maximum": 20,
                },
            },
            "required": ["file_id"],
        },
    },
}

_LIST_SESSION_ARTIFACTS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_session_artifacts",
        "description": (
            "List the files YOU have produced in this session (processed spectra you saved with "
            "save_result inside run_analysis, measurement-point records, and figures), in the order "
            "you saved them. "
            "Each entry has a data/-relative `path` you can read back with load_spectrum('<path>'). "
            "Use it when a task builds on something you saved earlier in this conversation, when you "
            "need to confirm a save actually happened, or when you must report where your output went. "
            "It touches no hardware and has no side effects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": ("Filter by kind: 'spectra' for saved spectra, 'figure' for plots. "
                                    "Omit to list everything."),
                    "enum": ["spectra", "figure", "measurement"],
                },
            },
            "required": [],
        },
    },
}

# 두 에이전트가 ALL_TOOLS 에 그대로 이어 붙이는 리스트.
# run_analysis 는 여기 없다 — 스키마는 이미 RAMAN_TOOLS 에 있고 실행만 가로채기 때문.
FILE_TOOLS = [_LIST_FILES_SCHEMA, _INSPECT_FILE_SCHEMA, _LIST_SESSION_ARTIFACTS_SCHEMA]

# CoALA 전용: 부수효과 없는 '정보 수집'이라 planning(retrieval) 액션으로 분류돼야 한다.
# 이 집합에 없으면 사이클을 닫는 실행(commit) 액션이 되어, 파일 한 번 들여다볼 때마다
# 의사결정 사이클을 하나씩 소모한다. run_analysis 는 결과물(그림)을 만드는
# 실행 액션이므로 여기 넣지 않는다.
FILE_RETRIEVAL = {"list_uploaded_files", "inspect_file", "list_session_artifacts"}


# ══════════════════════════════════════════════════════════════════════════════
# 실행부
# ══════════════════════════════════════════════════════════════════════════════

def _t_list_uploaded_files(args: dict) -> dict:
    date = (args.get("date") or "").strip() or None
    try:
        items = list_uploads(date)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not items:
        # 빈 목록을 에러로 만들지 않는다 — '첨부가 없다'는 정상 상태이고, 모델은 이때
        # 사용자에게 파일을 요청하거나 측정 경로로 가야 한다. 에러로 주면 재시도 루프에 빠진다.
        return {"ok": True, "files": [],
                "note": "No files have been attached to this chat. Ask the user to attach one, "
                        "or proceed with instrument measurement instead."}
    return {"ok": True, "count": len(items), "files": items}


def _t_inspect_file(args: dict) -> dict:
    file_id = str(args.get("file_id") or "").strip()
    if not file_id:
        return {"ok": False, "error": "file_id is empty. Get it from list_uploaded_files first."}
    sheet = (args.get("sheet") or "").strip() or None
    try:
        max_rows = int(args.get("max_rows") or 5)
    except (TypeError, ValueError):
        max_rows = 5
    try:
        return inspect_upload(file_id, sheet=sheet, max_rows=max_rows)
    except FileNotFoundError as e:
        return {"ok": False, "error": str(e),
                "hint": "Check the exact file_id with list_uploaded_files."}
    except ValueError as e:
        return {"ok": False, "error": str(e),
                "hint": f"Supported file types: {', '.join(sorted(ALLOWED_SUFFIXES))}."}
    except Exception as e:
        return {"ok": False, "error": f"Failed to read the file: {type(e).__name__}: {e}"}


def _t_run_analysis(args: dict) -> dict:
    """분석 샌드박스 호출. 하드웨어를 타지 않으므로 장비 미연결에서도 동작한다.

    지연 import 인 이유: analysis_sandbox 는 spectrum_store 를 통해 matplotlib 을 끌어오는데,
    이 모듈은 에이전트 import 시점에 항상 로드되므로 서버 기동을 무겁게 만들 이유가 없다.

    인자 정규화(문자열→리스트, 빈 문자열→None, code 검증)는 run_analysis 안에서 한다 —
    raman_tools.TOOL_DISPATCH 경로로 들어와도 동작이 같아야 하므로(2026-07-30).
    """
    from backend.analysis_sandbox import run_analysis

    try:
        return run_analysis(
            code=args.get("code"),
            date=args.get("date"),
            names=args.get("names"),
            title=args.get("title"),
            file_ids=args.get("file_ids"),
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _t_list_session_artifacts(args: dict) -> dict:
    """이 세션에서 에이전트가 만든 산출물 목록(run_store.manifest 조회)."""
    from backend.agents import run_store

    kind = (args.get("kind") or "").strip() or None
    cur = run_store.current()
    try:
        arts = run_store.list_artifacts(kind)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if not arts:
        # 빈 목록은 에러가 아니다 — '아직 아무것도 저장하지 않았다'는 정상 상태.
        return {"ok": True, "session": cur["label"], "artifacts": [],
                "note": ("You have not saved anything in this session yet. "
                         "Save computed spectra with save_result inside run_analysis.")}
    return {"ok": True, "session": cur["label"], "session_dir": cur["rel_dir"],
            "count": len(arts), "artifacts": arts}


# 이름 → 실행 함수. 각 에이전트의 _call_tool 이 하드웨어 가드보다 '먼저' 이 dict 를 본다.
FILE_DISPATCH = {
    "list_uploaded_files":     _t_list_uploaded_files,
    "inspect_file":            _t_inspect_file,
    "run_analysis":            _t_run_analysis,
    "list_session_artifacts":  _t_list_session_artifacts,
}
