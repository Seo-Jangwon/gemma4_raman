"""
라우트 유실 방지 검사 — server.py(1188줄)를 controllers/ 로 쪼갤 때 만든 안전망.

controllers/*.py 를 AST 로 훑어 (메서드, 경로) 집합을 만들고, 분할 전 server.py 가 갖고 있던
기대 목록과 정확히 일치하는지 확인한다. 하나라도 빠지거나 경로가 바뀌면 실패한다.

앱을 import 하지 않으므로(= 하드웨어 SDK·삭제된 모듈과 무관) 어느 PC 에서든 그냥 돈다:

    python backend/web_controller/test_routes.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_ROUTERS_DIR = Path(__file__).resolve().parent / "controllers"

_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}

# server.py 분할 후 남은 라우트 전부. 이 목록은 손대지 말 것 — 여기를 고쳐서 검사를
# 통과시키는 건 '기능이 사라졌다'를 '기대값이 바뀌었다'로 덮는 것이다.
# (분할 당시엔 32개였다. /api/bench/* 11개는 벤치 코드를 걷어내면서 함께 사라졌다.)
EXPECTED_ROUTES = {
    # 장비 연결·상태·조작
    ("POST", "/api/camera/connect"),
    ("POST", "/api/stage/connect"),
    ("POST", "/api/laser/connect"),
    ("GET",  "/api/ccd/status"),
    ("POST", "/api/ccd/connect"),
    ("GET",  "/api/camera/stream"),
    ("POST", "/api/stage/move-pixel"),
    ("GET",  "/api/hardware/state"),
    ("POST", "/api/stage/speed"),
    ("POST", "/api/spectrum/acquire"),
    ("GET",  "/api/health"),
    # 에이전트
    ("POST", "/api/experiment/run"),
    ("POST", "/api/experiment/stream"),
    ("GET",  "/api/agents/health"),
    # 첨부 파일
    ("POST", "/api/files/upload"),
    ("GET",  "/api/files"),
    # 지식베이스
    ("GET",  "/api/kb/status"),
    ("GET",  "/api/kb/search"),
    ("POST", "/api/kb/reload"),
    # (kb/upload · kb/reindex 는 Chroma 색인기와 함께 제거 — 2026-08-12.
    #  knowledge/search.py 머리말 참고. 지식 추가는 knowledge_base.json 편집 + reload.)
}


def _router_prefix(tree: ast.Module) -> str:
    """`router = APIRouter(prefix="/api", ...)` 에서 prefix 를 꺼낸다."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if getattr(func, "id", None) != "APIRouter" and getattr(func, "attr", None) != "APIRouter":
            continue
        for kw in node.value.keywords:
            if kw.arg == "prefix" and isinstance(kw.value, ast.Constant):
                return kw.value.value
    return ""


def collect_routes() -> set[tuple[str, str]]:
    """controllers/*.py 에 선언된 (메서드, 전체경로) 전부."""
    found: set[tuple[str, str]] = set()
    for path in sorted(_ROUTERS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        prefix = _router_prefix(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in node.decorator_list:
                # @router.post("/경로", ...) 형태만 본다
                if not isinstance(deco, ast.Call) or not isinstance(deco.func, ast.Attribute):
                    continue
                if getattr(deco.func.value, "id", None) != "router":
                    continue
                method = deco.func.attr.lower()
                if method not in _HTTP_METHODS or not deco.args:
                    continue
                if isinstance(deco.args[0], ast.Constant):
                    found.add((method.upper(), prefix + deco.args[0].value))
    return found


def main() -> int:
    found = collect_routes()
    missing = EXPECTED_ROUTES - found
    extra = found - EXPECTED_ROUTES

    for method, path in sorted(missing):
        print(f"  [사라짐] {method:6} {path}")
    for method, path in sorted(extra):
        print(f"  [추가됨] {method:6} {path}")

    if missing or extra:
        print(f"\n실패: 기대 {len(EXPECTED_ROUTES)}개, 실제 {len(found)}개 "
              f"(사라짐 {len(missing)}, 추가됨 {len(extra)})")
        return 1

    print(f"통과: 라우트 {len(found)}개가 분할 전과 동일합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
