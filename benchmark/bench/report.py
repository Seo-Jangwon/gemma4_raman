# -*- coding: utf-8 -*-
"""결과 파일 쓰기와 전체 성적 취합.

[결과 파일 하나만 보고 무엇을 틀렸는지 알 수 있어야 한다]
그래서 문항마다 다음을 한 파일에 같이 싣는다.
    question   에이전트에게 실제로 보낸 문장
    criteria   무엇을 맞다고 보는지
    answer     에이전트가 낸 본문과 JSON 블록
    checks     판정 하나하나의 기대값·실제값·배점 비중
    why_wrong  떨어뜨린 판정만 점수 손실 큰 순으로
나중에 이 파일만 열어도 "왜 이 점수인지"를 사람이 재구성할 수 있다.

[영어로 쓰는 이유]
결과는 사람만 읽는 게 아니라 표·스크립트·외부 협업자를 거친다. 한글이 섞이면 인코딩
사고(cp949)가 나고, 파일명이 한글이면 그대로 깨지는 환경이 있다.

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

MAX_ANSWER_CHARS = 12000       # 모델이 장문을 쏟아도 결과 파일을 못 열 정도로 커지지 않게


def score_task(task, checks, run=None) -> dict:
    """문항 하나의 채점 결과 — 그대로 <문항>.json 이 된다."""
    live = [c for c in checks if not c.blocked]
    skipped = [c for c in checks if c.blocked]
    total_w = sum(c.weight for c in live)
    frac = (sum(c.score * c.weight for c in live) / total_w) if total_w else 0.0

    rows = []
    for c in checks:
        d = c.as_dict()
        # 이 판정 때문에 몇 점을 잃었는지. '무엇을 고치면 몇 점이 오르는가'가 바로 보인다.
        d["points_lost"] = (0.0 if (c.blocked or not total_w) else
                            round(task.score * (1 - c.score) * c.weight / total_w, 3))
        rows.append(d)

    out = {
        "task": task.id,
        "axis": task.axis,
        "mode": task.mode,
        "score": round(task.score * frac, 4),
        "score_max": task.score,
        "fraction": round(frac, 4),
        "passed": sum(1 for c in live if c.passed),
        "checks_total": len(live),
        "question": task.prompt.strip(),
        "criteria": task.criteria.strip(),
        "checks": rows,
        "why_wrong": sorted(
            [{"check": c.name, "reason": c.detail, "points_lost": r["points_lost"]}
             for c, r in zip(checks, rows) if not c.passed and not c.blocked],
            key=lambda r: -r["points_lost"]),
        "blocked": [c.name for c in skipped],
        # 판정 항목이 하나도 없으면 만점이 되어 버린다. 그런 문항은 만들면 안 되고,
        # 실수로 생기면 여기서 드러나야 한다.
        "no_checks": not live,
    }
    if run is not None:
        out["input_files"] = list(task.inputs)
        out["answer_text"] = (run.text or "")[:MAX_ANSWER_CHARS]
        out["answer_json"] = run.answer
        out["tool_calls"] = [
            {"name": c.get("name"), "args": c.get("args"),
             "ok": (c.get("result") or {}).get("ok")
                   if isinstance(c.get("result"), dict) else None}
            for c in run.calls]
        out["artifacts"] = list(run.artifacts)
        out["session_id"] = run.session_id
        out["elapsed_s"] = round(run.elapsed_s, 2)
        out["errors"] = list(run.errors)
    return out


def write_eval(out_dir: Path, task_id: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{task_id}.json"
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
        "no_checks_defined": [r["task"] for r in results if r["no_checks"]],
        "partially_excluded": [r["task"] for r in results if r["blocked"]],
        "infrastructure_errors": [r["task"] for r in results if r.get("errors")],
        "not_run": missing,
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
              "passed": r["passed"], "of": r["checks_total"],
              "failed": [w["check"] for w in r["why_wrong"]]} for r in results],
            key=lambda r: r["task"]),
        "problems": {k: v for k, v in problems.items() if v},
    }


def print_summary(s: dict, stream=None) -> None:
    """콘솔 요약 — 여기는 사람이 바로 읽는 화면이라 한국어로 둔다."""
    import sys
    w = stream or sys.stdout
    t = s["total"]
    print("=" * 72, file=w)
    print(f"총점  {t['score']:.1f} / {t['max']:.0f}점  ({t['percent']:.1f}%)   "
          f"문항 {t['tasks_run']}/{t['tasks_total']}", file=w)
    print("-" * 72, file=w)
    for axis, v in s["by_axis"].items():
        bar = "█" * int(v["percent"] / 5)
        print(f"  {axis:20s} {v['score']:6.1f}/{v['max']:5.0f}  {v['percent']:5.1f}% {bar}",
              file=w)
    if s["problems"]:
        LABEL = {"no_checks_defined": "판정 항목 없음",
                 "partially_excluded": "장비 설정 미달로 일부 제외",
                 "infrastructure_errors": "인프라 오류", "not_run": "미실행"}
        print("-" * 72, file=w)
        for k, v in s["problems"].items():
            print(f"  [{LABEL.get(k, k)}] {len(v)}개: {', '.join(v[:14])}"
                  f"{' ...' if len(v) > 14 else ''}", file=w)
    print("=" * 72, file=w)
