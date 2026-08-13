# -*- coding: utf-8 -*-
"""구현이 service 계층에 있는 도구들의 얇은 어댑터.

    list_results           저장된 측정 목록
    combine_spectra        여러 측정을 격자 이미지 한 장으로
    aggregate_spectra_csv  측정당 1행 요약 CSV
    bundle_results         결과 파일을 zip 으로 묶어 다운로드 링크
    run_analysis           분석 코드 샌드박스 실행
    web_search             외부 웹 검색
    search_knowledge_base  큐레이션 KB 검색

[왜 선언만 여기 있는가]
계산과 저장은 service 계층이 한다. 그런데 모델에게 보여줄 인자 설명은 도구 계층의 것이다 —
spectrum_store 같은 모듈은 우리 코드도 부르는 순수 유틸이라, 거기에 프롬프트 문장이 섞이면
계층이 무너진다. 그래서 선언(시그니처 + Field 설명)은 이쪽에, 구현은 저쪽에 남긴다.

[None 을 걸러 넘기는 이유]
이 툴셋의 규약은 '인자를 생략하면 현재 설정 유지'다. 모델이 안 보낸 인자를 그대로 None 으로
넘기면 service 쪽 기본값이 죽고 '명시적 None'이 되어 규약이 조용히 깨진다. 그래서 각
어댑터는 ``locals()`` 에서 None 이 아닌 것만 골라 전달한다.

[이 모듈은 하드웨어를 import 하지 않는다]
Config.ini 가 없는 PC 에서는 하드웨어 도구 모듈의 import 가 통째로 실패하는데, 분석·검색·
KB 조회는 그 상황에서도 살아 있어야 한다. 그래서 세션 캐시나 장비 핸들이 필요한 도구
(save_measurement_point 등)는 여기 두지 않는다. import 문에 hw_tools 를 늘리지 말 것.
"""
from __future__ import annotations

# 실제 구현들. 별칭(_ 접두)으로 받아 같은 이름의 어댑터와 헷갈리지 않게 한다.
from backend.service.analyse.analysis_sandbox import run_analysis as _run_analysis
from backend.service.search.web_search import web_search as _web_search
from backend.service.store.spectrum_store import aggregate_spectra_csv as _store_aggregate_csv, bundle_results as _store_bundle_results, combine_spectra as _store_combine_spectra, list_results as _store_list_results
from backend.tools.result import fail, ok
from backend.tools.schema import INTERNAL      # 스키마에서 감출 인자 표식
from pydantic import Field
from typing import Annotated, Literal, Optional


# ══════════════════════════════════════════════════════════════════════════════
# 측정 결과 조회·정리·분석 — 구현은 전부 service 계층
# ══════════════════════════════════════════════════════════════════════════════

def list_results(
    date: Annotated[Optional[str], Field(description="Date to query 'YYYY-MM-DD'. If omitted, today.")] = None,
    scope: Annotated[Optional[Literal['session', 'all']], Field(description="Which measurements to consider. 'session' (default) = only the ones measured in THIS session, which is almost always what you want. 'all' = every session saved that day; use it only when the request is explicitly about combining work from earlier, separate sessions.")] = None,
) -> dict:
    """자동 저장된 측정 목록을 조회한다. 구현은 spectrum_store.list_results."""
    given = {k: v for k, v in locals().items() if v is not None}    # 생략 인자는 전달하지 않는다
    return _store_list_results(**given)


def combine_spectra(
    date: Annotated[Optional[str], Field(description="Target date 'YYYY-MM-DD'. If omitted, today.")] = None,
    names: Annotated[Optional[list[str]], Field(description='List of measurement bases to combine (check with list_results). If omitted, the whole date.')] = None,
    max_cols: Annotated[Optional[int], Field(ge=1, description='Number of grid columns. Default 4.')] = None,
    scope: Annotated[Optional[Literal['session', 'all']], Field(description="Which measurements to consider. 'session' (default) = only the ones measured in THIS session, which is almost always what you want. 'all' = every session saved that day; use it only when the request is explicitly about combining work from earlier, separate sessions.")] = None,
    out_name: Annotated[Optional[str], Field(json_schema_extra=INTERNAL, description='Internal: output file stem. Defaults to combined_<HHMMSS>.')] = None,
) -> dict:
    """여러 측정을 격자 이미지 한 장으로 렌더한다. 구현은 spectrum_store.combine_spectra."""
    given = {k: v for k, v in locals().items() if v is not None}
    return _store_combine_spectra(**given)


def aggregate_spectra_csv(
    date: Annotated[Optional[str], Field(description="Target date 'YYYY-MM-DD'. If omitted, today.")] = None,
    names: Annotated[Optional[list[str]], Field(description='List of measurement bases to organize. If omitted, all of yours from that date.')] = None,
    scope: Annotated[Optional[Literal['session', 'all']], Field(description="Which measurements to consider. 'session' (default) = only the ones measured in THIS session, which is almost always what you want. 'all' = every session saved that day; use it only when the request is explicitly about combining work from earlier, separate sessions.")] = None,
    out_name: Annotated[Optional[str], Field(json_schema_extra=INTERNAL, description='Internal: output file stem. Defaults to summary_<HHMMSS>.')] = None,
) -> dict:
    """측정당 한 행짜리 요약 CSV 를 만든다. 구현은 spectrum_store.aggregate_spectra_csv."""
    given = {k: v for k, v in locals().items() if v is not None}
    return _store_aggregate_csv(**given)


# ══════════════════════════════════════════════════════════════════════════════
# 분석·검색
# ══════════════════════════════════════════════════════════════════════════════

def web_search(
    query: Annotated[str, Field(description="Search query. e.g. 'raman baseline correction asymmetric least squares'")],
    max_results: Annotated[Optional[int], Field(ge=1, le=10, description='Number of results to fetch (1-10). Default 5.')] = None,
) -> dict:
    """외부 웹을 검색한다. 구현은 service.search.web_search."""
    given = {k: v for k, v in locals().items() if v is not None}
    return _web_search(**given)


def run_analysis(
    code: Annotated[str, Field(description="Python analysis code to run. Use spectra, np, plt directly. e.g. compute each spectrum's peak intensity and draw a peak map as an (x,y) scatter. If the task asks you to save a computed spectrum, call save_result('name', corrected_intensity, raman_shift=x) at the end of this code rather than printing the array. KEEP EACH CALL SHORT - aim for 40 lines or fewer. This code travels to the sandbox as a single JSON string, and a long block (many escaped quotes and newlines) is the most common way for a call to be lost in transit: the call silently never arrives and the task ends with no answer. Do not write one large end-to-end script. Split the work and call this tool several times - e.g. (1) load the data and print its shape and column names, (2) do one computation step and print a short summary, (3) produce the final numbers. Nothing carries over between calls: every call runs in a fresh process, so each one must rebuild what it needs from spectra/files, or read back an intermediate you wrote earlier with save_result (pass the path it returned as a file_id in file_ids).")],
    date: Annotated[Optional[str], Field(description="Measurement date to analyze 'YYYY-MM-DD'. If omitted, today.")] = None,
    names: Annotated[Optional[list[str]], Field(description='List of measurement bases to analyze (check with list_results). If omitted, the whole date.')] = None,
    file_ids: Annotated[Optional[list[str]], Field(description='file_ids of attached files to load into the `files` variable (get them from list_uploaded_files). If omitted, no attached file is loaded.')] = None,
    title: Annotated[Optional[str], Field(description='Title to attach to the result figure.')] = None,
    timeout_sec: Annotated[Optional[int], Field(json_schema_extra=INTERNAL, description='Internal: sandbox wall-clock limit (s). Deliberately not model-settable - a runaway script must not be able to raise its own timeout.')] = None,
) -> dict:
    """분석 코드를 샌드박스에서 실행한다. 구현은 analysis_sandbox.run_analysis.

    Notes
    -----
    이 함수는 스키마(모델이 보는 인자 설명)를 위한 선언이다. 실행은 file_tools 가 같은
    이름으로 가로챈다 — 순수 계산인 분석이 장비 유무에 묶이지 않게 하기 위해서다.
    """
    given = {k: v for k, v in locals().items() if v is not None}
    return _run_analysis(**given)


def bundle_results(
    date: Annotated[Optional[str], Field(description="Target date 'YYYY-MM-DD'. If omitted, today.")] = None,
    names: Annotated[Optional[list[str]], Field(description='List of measurement bases to bundle. If omitted, all of yours from that date.')] = None,
    scope: Annotated[Optional[Literal['session', 'all']], Field(description="Which measurements to bundle. 'session' (default) = only the ones measured in THIS session. 'all' = every session saved that day; use it only when the request is explicitly about earlier, separate sessions.")] = None,
    include: Annotated[Optional[list[str]], Field(json_schema_extra=INTERNAL, description='Internal: which file kinds to bundle. Defaults to png+csv+json.')] = None,
) -> dict:
    """저장된 결과 파일을 zip 하나로 묶는다. 구현은 spectrum_store.bundle_results."""
    given = {k: v for k, v in locals().items() if v is not None}
    return _store_bundle_results(**given)


# ══════════════════════════════════════════════════════════════════════════════
# 지식베이스 검색
# ══════════════════════════════════════════════════════════════════════════════
#
# 선언도 구현도 여기 한 벌뿐이다. 두 에이전트(AILA·CoALA)가 "같은 KB 를 같은 알고리즘으로"
# 검색해야 아키텍처 비교가 공정한데, 선언이 두 벌이면 한쪽만 고쳐져도 아무도 모른다.
# 도구 설명문만 아키텍처별로 갈라지고(tools.py 의 KB_TOOL / KB_TOOL_COALA), 부르는 함수는
# 이것 하나다.


def search_knowledge_base(
    query: Annotated[str, Field(description="Sample/material keyword to search. e.g. 'graphene', 'exosome cell', 'silicon'. English keywords match better (keyword substring matching).")],
) -> dict:
    """시편 종류로 측정 프로토콜·권장 파라미터를 찾는다.

    Parameters
    ----------
    query : str
        시편/물질 키워드. 부분 문자열 매칭이라 영어 키워드가 잘 맞는다.

    Returns
    -------
    dict
        ``{"ok": True, "results": [...]}``. 매칭이 없어도 성공이며 ``note`` 로 알린다.

    Notes
    -----
    검색 알고리즘은 service.knowledge.search 한 곳에 있다. 여기 복제하지 말 것 —
    두 에이전트가 다른 알고리즘으로 같은 KB 를 뒤지면 비교가 성립하지 않는다.
    """
    # 지연 import: KB 파일 적재를 이 도구가 실제로 불릴 때까지 미룬다.
    from backend.service.knowledge.search import search_kb

    query = str(query or "").strip()           # 모델이 필수 인자를 빠뜨려도 죽지 않게 정규화
    if not query:
        return fail("query is empty. Provide a sample/material keyword.")

    hits = search_kb(query, top_k=3)
    if not hits:
        # 빈 결과는 에러가 아니다. 'KB 에 없는 시편'은 정상 상황이고, 이때 모델은 스스로
        # 파라미터를 정해야 한다. 에러로 주면 재시도 루프에 빠지거나 측정을 포기한다.
        return ok(results=[],
                  note=f"No protocol matching '{query}' in the knowledge base. "
                       "Decide the parameters yourself and state that in the report.")
    return ok(results=hits)
