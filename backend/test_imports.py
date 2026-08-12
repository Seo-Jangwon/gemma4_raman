# -*- coding: utf-8 -*-
"""import 경로 검사 — 파일을 옮기면서 만든 안전망.

backend 안의 모든 `from backend.X import a, b` / `import backend.X` 를 AST 로 모아
두 가지를 본다:

  ① X 가 실제 파일(또는 패키지)로 존재하는가
  ② a, b 가 그 모듈의 최상위 이름으로 존재하는가

왜 이 검사가 필요한가: 리팩터로 파일이 옮겨질 때마다 경로가 어긋났고, 그게 전부
**서버를 띄워야만** 드러났다. 게다가 이 프로젝트는 하드웨어 SDK 와 Config.ini 가
없는 PC 에서는 아예 import 가 안 돼서, 띄워 보는 것 자체가 개발 중에는 불가능하다.
AST 로 보면 어디서든 돈다.

지연 import(함수 안의 `from backend...`)도 똑같이 본다 — 오히려 그쪽이 더 위험하다.
모듈 로드 때는 멀쩡하다가 그 함수를 부르는 순간에야 터지기 때문이다.

    python backend/test_imports.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
_ROOT = _BACKEND.parent

# 검사에서 빼는 것.
#   _OLDS   옮기기 전 원본 보관소. 옛 경로를 가리키는 게 정상이다.
#   SDKs    장비 벤더가 준 코드. 우리가 경로를 정하지 않는다.
_SKIP_DIRS = {"_OLDS", "SDKs", "__pycache__"}


def _module_file(dotted: str) -> Path | None:
    """'backend.service.store.run_store' → 실제 파일 경로. 없으면 None."""
    rel = Path(*dotted.split("."))
    for cand in (_ROOT / rel.with_suffix(".py"), _ROOT / rel / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _top_level_names(path: Path) -> set[str]:
    """그 모듈이 밖으로 내주는 최상위 이름들(정의 + 재수출된 import).

    try 블록 안도 본다 — 벤더 SDK 의 __init__.py 는 하위 모듈마다
    `try: from .x import Y / except: print(...)` 로 재수출하는 형태다. 그걸 못 보면
    실제로 있는 이름을 '없다'고 잡는다.
    """
    names: set[str] = set()

    def walk(body: list) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.Try):
                walk(node.body)
            elif isinstance(node, ast.If):
                walk(node.body)
                walk(node.orelse)

    walk(ast.parse(path.read_text(encoding="utf-8")).body)
    return names


def _guarded_lines(tree: ast.Module) -> set[int]:
    """try/except 로 감싼 import 의 줄 번호.

    이 프로젝트는 '없어도 되는 의존'을 전부 이 형태로 쓴다 — 장비 SDK, chromadb,
    벤치 전용 사본. 없는 것이 **정상 동작**이고 호출부가 폴백을 갖고 있으므로,
    모듈이 없다고 실패로 잡으면 안 된다.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not node.handlers:
            continue
        for stmt in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                lines.add(stmt.lineno)
    return lines


def _sources() -> list[Path]:
    return [p for p in _BACKEND.rglob("*.py")
            if not _SKIP_DIRS & set(p.relative_to(_BACKEND).parts)]


def check() -> list[str]:
    problems: list[str] = []
    # 모듈별 최상위 이름 캐시 — 같은 모듈을 여러 곳에서 import 하므로.
    names_of: dict[Path, set[str]] = {}

    for src in _sources():
        rel = src.relative_to(_ROOT).as_posix()
        try:
            tree = ast.parse(src.read_text(encoding="utf-8"), filename=str(src))
        except SyntaxError as e:
            problems.append(f"{rel}:{e.lineno} 문법 오류: {e.msg}")
            continue

        guarded = _guarded_lines(tree)
        for node in ast.walk(tree):
            # import backend.X  /  import backend.X as Y
            if isinstance(node, ast.Import):
                for a in node.names:
                    if (a.name.startswith("backend.") and node.lineno not in guarded
                            and _module_file(a.name) is None):
                        problems.append(f"{rel}:{node.lineno} 없는 모듈: {a.name}")
                continue

            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("backend") or node.lineno in guarded:
                continue

            target = _module_file(node.module)
            if target is None:
                problems.append(f"{rel}:{node.lineno} 없는 모듈: {node.module}")
                continue

            # `from backend.pkg import 하위모듈` 도 정당하다 — 이름이 아니라 파일이다.
            if target not in names_of:
                names_of[target] = _top_level_names(target)
            exported = names_of[target]
            for a in node.names:
                if a.name == "*" or a.name in exported:
                    continue
                if _module_file(f"{node.module}.{a.name}") is not None:
                    continue                       # 패키지 아래의 하위 모듈
                problems.append(
                    f"{rel}:{node.lineno} {node.module} 에 '{a.name}' 없음")
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print(f"  [실패] {p}")
    if problems:
        print(f"\n실패 {len(problems)}건")
        return 1
    print(f"통과: backend 내부 import 전부 해소됨 (파일 {len(_sources())}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
