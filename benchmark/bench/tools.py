# -*- coding: utf-8 -*-
"""실존하는 도구 이름 — 오탈자를 실행 전에 잡는다.

문항 파일이 chk.called(run, "move_stge") 처럼 이름을 틀리면 그 판정은 영원히 0회로
읽혀 조용히 오답이 된다. 러너가 시작할 때 모든 문항의 참조 이름을 이 집합과 대조한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent    # benchmark/bench → 루트


def _load() -> set:
    if str(PROJ) not in sys.path:
        sys.path.insert(0, str(PROJ))
    try:
        from backend.agents.file_tools import FILE_TOOLS
        from backend.hw_tools.raman_tool_schemas import RAMAN_TOOLS
    except Exception as e:                       # 스키마를 못 읽으면 검증을 포기하지 않는다
        raise RuntimeError(f"도구 스키마를 읽지 못했습니다: {type(e).__name__}: {e}") from e
    return ({t["function"]["name"] for t in RAMAN_TOOLS + FILE_TOOLS}
            # 지식베이스 검색은 에이전트 모듈이 각자 등록한다(스키마 파일에 없다).
            | {"search_knowledge_base"})


TOOL_NAMES: set = _load()

# 도구별 인자 이름 — chk.arg("acquire_spectrum", "expsure", ...) 같은 오타를 잡는다.
def _params() -> dict:
    if str(PROJ) not in sys.path:
        sys.path.insert(0, str(PROJ))
    from backend.agents.file_tools import FILE_TOOLS
    from backend.hw_tools.raman_tool_schemas import RAMAN_TOOLS
    out = {}
    for t in RAMAN_TOOLS + FILE_TOOLS:
        f = t["function"]
        out[f["name"]] = set((f.get("parameters") or {}).get("properties", {}))
    return out


TOOL_PARAMS: dict = _params()
