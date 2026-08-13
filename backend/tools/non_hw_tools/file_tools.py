# -*- coding: utf-8 -*-
"""
[역할] 채팅에 첨부된 데이터 파일(csv/excel/txt)과 세션 산출물을 에이전트가 다루기 위한
       도구 계층. **하드웨어를 import 하지 않는다.**

  [왜 하드웨어와 분리되어 있는가]
  하드웨어 도구 모듈은 config.py 를 통해 Config.ini 를 읽으므로, 그 파일이 없는 환경에서는
  import 가 통째로 실패할 수 있다(runtime.get_tool_dispatch() 의 docstring 참고). 그러면
  TOOL_DISPATCH 가 None 이 되어 그 뒤의 모든 도구가 "Hardware is not connected." 로 막히는데,
  파일 분석은 하드웨어와 무관한 작업이라 거기 함께 묶일 이유가 없다. 그래서 runtime._dispatch
  가 '하드웨어 가드보다 먼저' 이 모듈의 FILE_DISPATCH 를 본다.

  이 모듈의 import 문에 backend.hw_tools 가 등장하는 순간 그 보호가 사라진다. 늘리지 말 것.

  [run_analysis 를 여기서 가로채는 이유]
  run_analysis 의 실체인 backend.service.analysis_sandbox 는 하드웨어를 import 하지 않는다
  (numpy/matplotlib + spectrum_store + upload_store 뿐). 그런데 스키마와 어댑터는 도구 계층
  (backend.tools.data_tools)에 있고 그쪽은 TOOL_DISPATCH 경로로 들어온다. 여기서 가로채면
  장비 연결 여부와 무관하게 분석이 돌고, 모델에게 보이는 도구 이름도 하나로 유지된다.
  (선언은 그대로 data_tools.run_analysis 에 있고, 실행 경로만 이쪽으로 온다.)
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import Field

from backend.service.store.upload_store import ALLOWED_SUFFIXES, inspect_upload, list_uploads
from backend.tools.result import fail, ok
from backend.tools.schema import call_with, tool_schema

# ══════════════════════════════════════════════════════════════════════════════
# 도구
#   인자의 이름·타입·범위·설명은 아래 시그니처에 직접 적는다. 스키마는 tool_schema() 가
#   그것을 읽어 만든다 — 선언이 한 곳뿐이라 스키마와 함수가 어긋날 수가 없다.
# ══════════════════════════════════════════════════════════════════════════════


def list_uploaded_files(
    date: Annotated[Optional[str], Field(description="Upload date 'YYYY-MM-DD'. If omitted, today.")] = None,
) -> dict:
    """첨부 파일 목록. 내용이 아니라 file_id·이름·종류·크기만 돌려준다."""
    date = (date or "").strip() or None
    try:
        items = list_uploads(date)
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")
    if not items:
        # 빈 목록을 에러로 만들지 않는다 — '첨부가 없다'는 정상 상태이고, 모델은 이때
        # 사용자에게 파일을 요청하거나 측정 경로로 가야 한다. 에러로 주면 재시도 루프에 빠진다.
        return ok(files=[],
                  note="No files have been attached to this chat. Ask the user to attach one, "
                       "or proceed with instrument measurement instead.")
    return ok(count=len(items), files=items)


#: 파일 '내용 자체'가 결과에 실리는 종류(image·json)에만 거는 크기 상한(바이트).
#: 현미경 캡처가 1060x800 에서 ~2MB 라 4MB 면 넉넉하고, 실수로 거대한 파일을 지목했을 때
#: base64(원본의 4/3배)가 프로세스 메모리와 HTTP 페이로드를 밀어내는 것을 막는다.
#:
#: 표(table)에는 걸지 않는다 — 합치기 전 inspect_file 에도 상한이 없었다. 표는 파일이
#: 아무리 커도 나가는 것은 컬럼·통계·head 몇 줄뿐이라 크기와 페이로드가 무관하고,
#: 매핑 스캔 결과처럼 수십 MB 인 csv/xlsx 가 실제로 들어온다. 여기 상한을 걸면
#: "너무 큽니다" 로 막혀 볼 방법이 사라진다.
_MAX_OPEN_BYTES = 4 * 1024 * 1024


def open_file(
    file_id: Annotated[str, Field(description="A file_id exactly as a listing or an earlier tool result gave it, e.g. 'uploads:2026-07-24/113045_sample.csv', 'results:2026-08-12/_microscope_153726.png', 'runs:<session>/spectra/01_corrected.csv'.")],
    question: Annotated[Optional[str], Field(description="Images only: what you want to check in the picture (optional).")] = None,
    sheet: Annotated[Optional[str], Field(description="Excel only: sheet name to read. If omitted, the first sheet.")] = None,
    max_rows: Annotated[Optional[int], Field(ge=1, le=20, description="Tables only: how many leading rows to preview (1-20). Default 5.")] = None,
) -> dict:
    """저장된 파일 하나를 열어 **종류에 맞는 것**을 돌려준다.

    [왜 도구 하나인가 — 2026-08-12]
    예전에는 view_image / inspect_file / load_spectrum 셋이었다. 모델은 id 하나를 들고
    '이게 어느 도구 것인가'를 먼저 맞혀야 했고, 틀리면 "File not found" 를 받았다.
    그 실패가 실제로 2026-08-07 벤치에서 여섯 문항을 날렸다(paths.py 머리말).

    확장자→종류 판정은 코드가 이미 확실히 할 수 있는 일이다(paths.kind_of). 모델에게
    떠넘길 이유가 없어서 여기서 분기한다. 그러면 '도구를 잘못 골랐다'는 실패 자체가
    존재하지 않게 된다 — 좋은 에러 메시지로 안내하는 것보다 낫다.

    반환은 종류마다 다르지만 `kind` 가 항상 붙는다(image / table / json):
        image  → image_base64  … execute_tool 이 떼어내 모델에게 그림으로 주입한다
        table  → 구조 요약(컬럼·통계·head). 전체 데이터는 run_analysis(file_ids=…) 담당.
        json   → 저장된 레코드의 필드 그대로(측정 결과 sidecar, 측정점 기록)
    """
    import base64
    import json as _json

    from backend.service.store.paths import kind_of, try_resolve

    # 해석·경로탈출 방어·'어디를 뒤졌는지' 안내는 전부 paths.resolve 한 곳이 한다.
    path, why = try_resolve(file_id)
    if path is None:
        return fail(why)

    file_id = str(file_id or "").strip()
    kind = kind_of(path)
    size = path.stat().st_size
    if kind != "table" and size > _MAX_OPEN_BYTES:
        return fail(f"'{file_id}' is too large to open ({size / 1048576:.1f} MB, "
                    f"limit {_MAX_OPEN_BYTES // 1048576} MB).")

    if kind == "image":
        try:
            raw = path.read_bytes()
        except Exception as e:
            return fail(f"Failed to read the image: {type(e).__name__}: {e}")
        question = str(question or "").strip() or "Stored image."
        return ok(kind="image",
                  file_id=file_id,
                  bytes=size,
                  image_base64=base64.b64encode(raw).decode("utf-8"),
                  question=f"{question}\n\n[This is a stored image ({file_id}), not a live view. "
                           f"The stage may have moved since it was captured.]")

    if kind == "json":
        try:
            doc = _json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return fail(f"Failed to read the JSON record: {type(e).__name__}: {e}")
        # dict 가 아닌 JSON(배열 등)도 있을 수 있으므로 record 아래에 담아 모양을 고정한다.
        return ok(kind="json", file_id=file_id, bytes=size,
                  record=doc if isinstance(doc, dict) else {"value": doc})

    if kind == "table":
        sheet = (sheet or "").strip() or None
        try:
            max_rows = int(max_rows or 5)
        except (TypeError, ValueError):
            max_rows = 5
        try:
            out = inspect_upload(file_id, sheet=sheet, max_rows=max_rows)
        except FileNotFoundError as e:
            return fail(str(e), hint="Check the exact file_id with list_uploaded_files.")
        except ValueError as e:
            return fail(str(e))
        except Exception as e:
            return fail(f"Failed to read the file: {type(e).__name__}: {e}")
        out["kind"] = "table"        # inspect_upload 의 kind 는 text/excel — 계약을 덮어쓴다
        return out

    return fail(f"'{file_id}' has an extension this tool cannot open ({path.suffix or 'none'}). "
                f"Openable types: {', '.join(sorted(ALLOWED_SUFFIXES | {'.json'}))}.")


def list_session_artifacts(
    kind: Annotated[Optional[Literal["spectra", "figure", "measurement"]], Field(description="Filter by kind: 'spectra' for saved spectra, 'figure' for plots. Omit to list everything.")] = None,
) -> dict:
    """이 세션에서 에이전트가 만든 산출물 목록(run_store.manifest 조회)."""
    from backend.service.store import run_store

    kind = (kind or "").strip() or None
    cur = run_store.current()
    try:
        arts = run_store.list_artifacts(kind)
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")
    if not arts:
        # 빈 목록은 에러가 아니다 — '아직 아무것도 저장하지 않았다'는 정상 상태.
        return ok(session=cur["label"],
                  artifacts=[],
                  note="You have not saved anything in this session yet. "
                       "Save computed spectra with save_result inside run_analysis.")
    return ok(session=cur["label"], session_dir=cur["rel_dir"], count=len(arts), artifacts=arts)


def _run_analysis(args: dict) -> dict:
    """분석 샌드박스 호출. 하드웨어를 타지 않으므로 장비 미연결에서도 동작한다.

    선언(인자 설명·스키마)은 data_tools.run_analysis 에 있다. 여기 있는 것은 실행 경로뿐이다 —
    그 모듈은 하드웨어와 같은 디스패치 표에 실리므로, 장비가 없을 때 순수 계산인 분석까지
    함께 막히지 않도록 이름만 가로챈다.

    지연 import 인 이유: analysis_sandbox 는 spectrum_store 를 통해 matplotlib 을 끌어오는데,
    이 모듈은 에이전트 import 시점에 항상 로드되므로 서버 기동을 무겁게 만들 이유가 없다.

    인자 정규화(문자열→리스트, 빈 문자열→None, code 검증)는 run_analysis 안에서 한다 —
    data_tools 경로로 들어와도 동작이 같아야 하므로(2026-07-30).
    """
    from backend.service.analyse.analysis_sandbox import run_analysis

    try:
        return run_analysis(
            code=args.get("code"),
            date=args.get("date"),
            names=args.get("names"),
            title=args.get("title"),
            file_ids=args.get("file_ids"),
        )
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 바인딩
# ══════════════════════════════════════════════════════════════════════════════

#: 모델에게 노출할 스키마. run_analysis 는 여기 없다 — 선언은 data_tools 에 있고
#: 실행만 가로채기 때문이다.
FILE_TOOLS = [
    tool_schema(
        list_uploaded_files,
        "List the files the user attached to the chat (csv, excel, txt, and images). Returns each file's file_id, filename, `kind` and size - not its contents. Open any of them with open_file - it works out the file kind for you. Call this first when the user mentions an attached/uploaded file, or when you are asked to analyze data you did not measure yourself. It touches no hardware and is free of side effects.",
    ),
    tool_schema(
        open_file,
        "Open any stored file by its file_id and get back whatever that file can tell you. "
        "One tool for every kind - you do not have to work out which reader applies:\n"
        "- an IMAGE (.png/.jpg) is shown to you, so you can read pixel coordinates off it "
        "exactly as when it was first captured;\n"
        "- a TABLE (.csv/.xlsx/.txt/...) comes back as a STRUCTURE summary: row/column counts, "
        "column names, numeric-or-text per column, min/max/mean, the first few rows, and (for "
        "Excel) the sheet names. It does NOT return the full data and interprets nothing for "
        "you - you decide whether some numeric column is a Raman shift in cm-1, an intensity, a "
        "stage coordinate, or unrelated metadata, then use run_analysis with file_ids to compute "
        "on the full data;\n"
        "- a JSON record (a saved measurement or measurement point) comes back as its fields.\n"
        "The reply always carries `kind` so you know which of these you got. "
        "Use it for files the user attached AND for anything produced earlier in this session: "
        "a microscope view you captured in an earlier turn (the picture itself is dropped from "
        "the conversation when a turn ends, but the file stays), a grid preview, a saved "
        "spectrum, a measurement record. "
        "Opening an image does NOT take a new picture - it shows the sample as it was at capture "
        "time, so if the stage has moved since, call analyze_microscope_image instead. "
        "It touches no hardware and has no side effects, so call it freely.",
    ),
    tool_schema(
        list_session_artifacts,
        "List the files YOU have produced in this session (processed spectra you saved with "
        "save_result inside run_analysis, measurement-point records, and figures), in the order "
        "you saved them. "
        "Each entry has a `path` you can read back with open_file('<path>'). "
        "Use it when a task builds on something you saved earlier in this conversation, when you "
        "need to confirm a save actually happened, or when you must report where your output went. "
        "It touches no hardware and has no side effects.",
    ),
]

# CoALA 전용: 부수효과 없는 '정보 수집'이라 planning(retrieval) 액션으로 분류돼야 한다.
# 이 집합에 없으면 사이클을 닫는 실행(commit) 액션이 되어, 파일 한 번 들여다볼 때마다
# 의사결정 사이클을 하나씩 소모한다. run_analysis 는 결과물(그림)을 만드는
# 실행 액션이므로 여기 넣지 않는다.
# open_file 은 디스크에서 읽기만 하므로 여기 속한다. 이미지를 열어도 주입 경로는
# execute_tool 하나라 retrieval 로 분류해도 그림은 그대로 모델에게 간다.
FILE_RETRIEVAL = {"list_uploaded_files", "open_file", "list_session_artifacts"}

#: 이름 → 실행 함수. runtime._dispatch 가 하드웨어 가드보다 '먼저' 이 dict 를 본다.
FILE_DISPATCH = {
    "list_uploaded_files":     lambda a: call_with(list_uploaded_files, a),
    "open_file":               lambda a: call_with(open_file, a),
    "run_analysis":            _run_analysis,
    "list_session_artifacts":  lambda a: call_with(list_session_artifacts, a),
}
