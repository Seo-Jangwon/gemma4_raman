# -*- coding: utf-8 -*-
"""
라만_벤치마크.xlsx → tasks_raw.json 변환기.

xlsx(단일 시트 'Task_채점기준')의 129개 태스크를 구조화 JSON으로 뽑는다.
에이전트 코드는 건드리지 않는다 — 이 파일은 순수 입력 전처리다.

시트 열:
  Category / Complexity / Required capability / Task / Grading criteria / Presetup

산출 tasks_raw.json 의 각 항목:
  {
    "id": "T001",
    "category": "1. Single action",
    "complexity": "Basic",
    "capability": "Absolute move",
    "task": "Move the stage once to ...",     # ← 에이전트에 넣을 질문(프롬프트)
    "grading_criteria": "Passes if ...",       # ← 수동/자동 채점의 근거(프로즈)
    "presetup": "Presetup: ... / Required files: ..."
  }

실행:
  python -m backend.benchmark.xlsx_to_tasks
  python -m backend.benchmark.xlsx_to_tasks --xlsx 라만_벤치마크.xlsx --out backend/benchmark/tasks_raw.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 프로젝트 루트(=xlsx 기본 위치). __file__ = backend/benchmark/xlsx_to_tasks.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_XLSX = _PROJECT_ROOT / "라만_벤치마크.xlsx"
_DEFAULT_OUT = Path(__file__).resolve().parent / "tasks_raw.json"

# 기대 헤더(순서 무관, 이름으로 매핑). 시트 헤더가 바뀌면 여기만 고친다.
_COLS = {
    "category": "Category",
    "complexity": "Complexity",
    "capability": "Required capability",
    "task": "Task",
    "grading_criteria": "Grading criteria",
    "presetup": "Presetup / required files",
}


def load_tasks(xlsx_path: Path, sheet: str | None = None) -> list[dict]:
    """xlsx를 읽어 태스크 dict 리스트로. 빈 Task 행은 건너뛴다."""
    import openpyxl  # 지연 import — 이 스크립트 실행 시에만 필요

    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    # 헤더명 → 열 인덱스
    idx: dict[str, int] = {}
    for key, colname in _COLS.items():
        if colname in header:
            idx[key] = header.index(colname)
    missing = [c for k, c in _COLS.items() if k not in idx]
    if missing:
        raise ValueError(f"xlsx 헤더에서 못 찾은 열: {missing}  (실제 헤더: {header})")

    def cell(row, key):
        v = row[idx[key]]
        return str(v).strip() if v is not None else ""

    tasks: list[dict] = []
    n = 0
    for row in rows[1:]:
        task_text = cell(row, "task")
        if not task_text:
            continue  # Task 비어 있으면 스킵(구분행 등)
        n += 1
        tasks.append({
            "id": f"T{n:03d}",
            "category": cell(row, "category"),
            "complexity": cell(row, "complexity"),
            "capability": cell(row, "capability"),
            "task": task_text,
            "grading_criteria": cell(row, "grading_criteria"),
            "presetup": cell(row, "presetup"),
        })
    return tasks


def main():
    ap = argparse.ArgumentParser(description="라만_벤치마크.xlsx → tasks_raw.json")
    ap.add_argument("--xlsx", default=str(_DEFAULT_XLSX), help="입력 xlsx 경로")
    ap.add_argument("--sheet", default=None, help="시트 이름(기본: 첫 시트)")
    ap.add_argument("--out", default=str(_DEFAULT_OUT), help="출력 json 경로")
    args = ap.parse_args()

    tasks = load_tasks(Path(args.xlsx), args.sheet)
    out = Path(args.out)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {len(tasks)}개 태스크 추출 → {out}")
    # 카테고리별 개수 요약
    from collections import Counter
    cats = Counter(t["category"] for t in tasks)
    for c, k in cats.items():
        print(f"    {k:3d}  {c}")


if __name__ == "__main__":
    main()
