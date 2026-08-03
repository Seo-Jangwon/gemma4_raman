# -*- coding: utf-8 -*-
"""평가.json 쓰기와 전체 성적 취합.

[문항 하나의 점수]
    획득 = 배점 × Σ(check.score × weight) / Σ(weight)
blocked 표시가 붙은 판정은 분모에서 **뺀다**. 그건 장비 설정이 못 미쳐 판정 자체가
불가능했던 것이라, 에이전트 실력이 아니다.

[전체 총점]
분모는 '실행한 문항'이 아니라 **문항 목록 전체의 배점**이다. 중간에 끊긴 실행이 오히려
높은 백분율로 보이면 안 된다 — 안 돈 문항은 0점으로 계상하고 그 사실을 함께 적는다.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def score_task(task, checks) -> dict:
    """문항 하나의 채점 결과 — 그대로 평가.json 이 된다."""
    live = [c for c in checks if not c.blocked]
    skipped = [c for c in checks if c.blocked]
    total_w = sum(c.weight for c in live)
    frac = (sum(c.score * c.weight for c in live) / total_w) if total_w else 0.0
    return {
        "task": task.id,
        "axis": task.axis,
        "mode": task.mode,
        "score_max": task.score,
        "score": round(task.score * frac, 4),
        "fraction": round(frac, 4),
        "passed": sum(1 for c in live if c.passed),
        "checks_total": len(live),
        "checks": [c.as_dict() for c in checks],
        "blocked": [c.name for c in skipped],
        # 판정 항목이 하나도 없으면 만점이 되어 버린다. 그런 문항은 만들면 안 되고,
        # 실수로 생기면 여기서 드러나야 한다.
        "no_checks": not live,
    }


def write_eval(out_dir: Path, task_id: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{task_id}.평가.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def summarize(results: list, all_tasks: list, meta: dict = None) -> dict:
    """전체 취합. results 는 score_task() 결과들, all_tasks 는 Task 전체."""
    done = {r["task"] for r in results}
    total_max = sum(t.score for t in all_tasks)
    earned = sum(r["score"] for r in results)
    missing = [t.id for t in all_tasks if t.id not in done]

    by_axis = defaultdict(lambda: {"score": 0.0, "max": 0.0, "n": 0})
    for t in all_tasks:
        by_axis[t.axis]["max"] += t.score
    for r in results:
        a = by_axis[r["axis"]]
        a["score"] += r["score"]
        a["n"] += 1

    problems = {
        "판정 항목 없음": [r["task"] for r in results if r["no_checks"]],
        "장비 설정 미달로 일부 제외": [r["task"] for r in results if r["blocked"]],
        "인프라 오류": [r["task"] for r in results if r.get("errors")],
        "미실행": missing,
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "meta": meta or {},
        "total": {"score": round(earned, 2), "max": round(total_max, 2),
                  "percent": round(earned / total_max * 100, 2) if total_max else 0.0,
                  "tasks_run": len(results), "tasks_total": len(all_tasks)},
        "by_axis": {k: {"score": round(v["score"], 2), "max": round(v["max"], 2),
                        "percent": round(v["score"] / v["max"] * 100, 1) if v["max"] else 0.0,
                        "tasks": v["n"]}
                    for k, v in sorted(by_axis.items())},
        "by_task": sorted(
            [{"task": r["task"], "axis": r["axis"], "score": r["score"],
              "max": r["score_max"], "fraction": r["fraction"],
              "passed": r["passed"], "of": r["checks_total"]} for r in results],
            key=lambda r: r["task"]),
        "problems": {k: v for k, v in problems.items() if v},
    }


def print_summary(s: dict, stream=None) -> None:
    import sys
    w = stream or sys.stdout
    t = s["total"]
    print("=" * 68, file=w)
    print(f"총점  {t['score']:.1f} / {t['max']:.0f}점  ({t['percent']:.1f}%)   "
          f"문항 {t['tasks_run']}/{t['tasks_total']}", file=w)
    print("-" * 68, file=w)
    for axis, v in s["by_axis"].items():
        bar = "█" * int(v["percent"] / 5)
        print(f"  {axis:12s} {v['score']:6.1f}/{v['max']:5.0f}  {v['percent']:5.1f}% {bar}",
              file=w)
    if s["problems"]:
        print("-" * 68, file=w)
        for k, v in s["problems"].items():
            print(f"  [{k}] {len(v)}개: {', '.join(v[:14])}"
                  f"{' ...' if len(v) > 14 else ''}", file=w)
    print("=" * 68, file=w)
