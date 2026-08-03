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
(콘솔 요약만 한국어다 — 그건 돌리는 사람이 그 자리에서 읽는 화면이다.)

[문항 하나의 결과 — 부분점은 없다 (2026-08-03)]
    pass     채점 가능한 판정이 **전부** 통과
    fail     하나라도 떨어짐
    blocked  판정이 전부 '채점 불가'(장비 설정 미달) — 에이전트 실력이 아니다
    error    실행 자체가 실패(인프라·LLM 무응답) — 답을 받아 보지도 못했다

예전에는 배점 × Σ(score×weight)/Σ(weight) 로 부분점을 줬다. 그게 결함을 가렸다:
plan_order 가 프롬프트에 이미 있는 단계를 요구하는 버그로 T063 이 1.91/3(64%)을 받으니
'대체로 맞았다'로 보여 아무도 파보지 않았다. 이진은 그걸 0 으로 만들어 소리를 크게 낸다.
대신 판정 하나만 깐깐해도 문항 전체가 죽으므로, 판정은 명세(프롬프트)가 실제로 요구한
것만 봐야 한다.

blocked 와 error 를 fail 과 한 칸에 넣으면 안 된다. 설정 실수·장비 사고를 에이전트
실력으로 기록하는 것이 이 프레임워크에서 가장 나쁜 고장이다. 해결률의 분모에서 뺀다.

[전체 집계]
    solve_rate = pass / (pass + fail)          blocked·error 는 분모에서 제외
분모는 '실행한 문항'이 아니라 문항 목록 전체를 기준으로 센다 — 중간에 끊긴 실행이
오히려 높은 비율로 보이면 안 되므로, 안 돈 문항은 not_run 으로 따로 밝힌다.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MAX_ANSWER_CHARS = 12000       # 모델이 장문을 쏟아도 결과 파일을 못 열 정도로 커지지 않게

# 에이전트 모듈이 '모델이 빈 응답을 냈다'는 뜻으로 쓰는 문장. 이건 오답이 아니라 실행
# 실패다 — 2026-08-03 실행에서 7 문항이 이 상태로 조용히 0 점 처리됐다.
EMPTY_REPLY = "Failed to generate a response."

PASS, FAIL, BLOCKED, ERROR = "pass", "fail", "blocked", "error"


def _outcome(live, skipped, run) -> tuple:
    """(결과, 사유). 판정 결과보다 '채점이 성립했는가'를 먼저 본다."""
    if run is not None and run.errors:
        return ERROR, "; ".join(str(e) for e in run.errors)[:300]
    if run is not None and (run.text or "").strip() == EMPTY_REPLY:
        return ERROR, "the agent returned an empty reply (LLM produced no text)"
    if not live:
        if skipped:
            return BLOCKED, "; ".join(c.detail for c in skipped)[:300]
        # 판정이 아예 없는 문항은 만점이 되어 버린다. 만들면 안 되고, 실수로 생기면
        # 여기서 드러나야 한다.
        return ERROR, "the task defines no check - it cannot be graded"
    bad = [c for c in live if not c.passed]
    if bad:
        return FAIL, f"{bad[0].name}: {bad[0].detail}"
    return PASS, ""


def score_task(task, checks, run=None) -> dict:
    """문항 하나의 채점 결과 — 그대로 <문항>.json 이 된다."""
    live = [c for c in checks if not c.blocked]
    skipped = [c for c in checks if c.blocked]
    result, reason = _outcome(live, skipped, run)
    failures = [{"check": c.name, "kind": c.kind, "reason": c.detail}
                for c in live if not c.passed]

    out = {
        "task": task.id,
        "axis": task.axis,
        "mode": task.mode,
        # 최종 채점은 이것 하나다. 부분점은 없다.
        "result": result,
        "reason": reason,
        # 아래는 '무엇이 틀렸는지'를 재구성하기 위한 상세다. 점수를 만들지 않는다.
        "checks_passed": sum(1 for c in live if c.passed),
        "checks_total": len(live),
        "question": task.prompt.strip(),
        "criteria": task.criteria.strip(),
        "checks": [c.as_dict() for c in checks],
        "failures": failures,
        "blocked": [c.name for c in skipped],
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
        # 상태 스냅샷도 싣는다. 상태 판정이 왜 떨어졌는지는 그때의 장비 값을 봐야 알 수
        # 있는데, 예전 결과 파일에는 이게 없어서 사후에 확인할 방법이 없었다(문항을 다시
        # 돌리는 수밖에). 재채점·감사에 필요하다.
        out["state_before"] = dict(run.state_before or {})
        out["state_after"] = dict(run.state_after or {})
        out["session_id"] = run.session_id
        out["elapsed_s"] = round(run.elapsed_s, 2)
        out["errors"] = list(run.errors)
    return out


def write_eval(out_dir: Path, task_id: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{task_id}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def _rate(solved, gradable) -> float:
    return round(solved / gradable * 100, 1) if gradable else 0.0


def summarize(results: list, all_tasks: list, meta: dict = None) -> dict:
    """전체 취합. results 는 score_task() 결과들, all_tasks 는 Task 전체."""
    done = {r["task"] for r in results}
    missing = [t.id for t in all_tasks if t.id not in done]

    def n(rs, kind):
        return sum(1 for r in rs if r["result"] == kind)

    by_axis = defaultdict(list)
    for r in results:
        by_axis[r["axis"]].append(r)

    solved, failed = n(results, PASS), n(results, FAIL)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "meta": meta or {},
        # 해결률의 분모는 '채점이 성립한 문항'이다. blocked·error 는 따로 밝힌다.
        "total": {
            "solved": solved,
            "failed": failed,
            "gradable": solved + failed,
            "solve_rate_pct": _rate(solved, solved + failed),
            "blocked": n(results, BLOCKED),
            "errors": n(results, ERROR),
            "tasks_run": len(results),
            "tasks_total": len(all_tasks),
        },
        "by_axis": {
            a: {"solved": n(rs, PASS), "failed": n(rs, FAIL),
                "gradable": n(rs, PASS) + n(rs, FAIL),
                "solve_rate_pct": _rate(n(rs, PASS), n(rs, PASS) + n(rs, FAIL)),
                "blocked": n(rs, BLOCKED), "errors": n(rs, ERROR),
                "tasks": len(rs)}
            for a, rs in sorted(by_axis.items())},
        "by_task": sorted(
            [{"task": r["task"], "axis": r["axis"], "result": r["result"],
              "reason": r["reason"],
              "checks_passed": r["checks_passed"], "checks_total": r["checks_total"],
              "failed_checks": [f["check"] for f in r["failures"]]} for r in results],
            key=lambda r: r["task"]),
        "problems": {k: v for k, v in {
            "blocked": [r["task"] for r in results if r["result"] == BLOCKED],
            "errors": [r["task"] for r in results if r["result"] == ERROR],
            "not_run": missing,
        }.items() if v},
    }


def write_markdown(out_dir: Path, s: dict) -> Path:
    """사람이 읽는 한 장짜리 요약 — SUMMARY.md.

    외부 협업자가 받는 것이 이 파일이므로 전부 영어다. 실행끼리 git diff 로 견줄 수
    있도록 표의 행 순서를 문항 번호로 고정한다.
    """
    t, m = s["total"], s.get("meta", {})
    L = [
        "# Raman agent benchmark",
        "",
        f"- Agent: **{m.get('agent', '?')}**",
        f"- Run: `{m.get('run_id', '?')}`",
        f"- Generated: {s['generated_at']}",
    ]
    cfg = m.get("config") or {}
    if cfg:
        L.append(f"- Instrument: {cfg.get('laser_nm', '?')} nm, centre "
                 f"{cfg.get('center_cm1', '?')} cm-1, {cfg.get('pixels', '?')} px")
    L += [
        "",
        "## Result",
        "",
        f"**Solved {t['solved']} / {t['gradable']} gradable tasks "
        f"({t['solve_rate_pct']:.1f}%)**",
        "",
        "A task counts as solved only when every one of its checks passes. There is no "
        "partial credit.",
        "",
        "| | tasks |",
        "|---|---|",
        f"| Solved | {t['solved']} |",
        f"| Failed | {t['failed']} |",
        f"| Not gradable - instrument limits (`blocked`) | {t['blocked']} |",
        f"| Not gradable - run failed (`error`) | {t['errors']} |",
        f"| **Total defined** | **{t['tasks_total']}** |",
        "",
        "`blocked` and `error` are excluded from the solve rate: they record cases where "
        "the harness or the instrument, not the agent, prevented an answer from being "
        "graded.",
        "",
        "## By capability axis",
        "",
        "| Axis | Solved | Gradable | Rate | Excluded |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis, v in s["by_axis"].items():
        L.append(f"| {axis} | {v['solved']} | {v['gradable']} | "
                 f"{v['solve_rate_pct']:.1f}% | {v['blocked'] + v['errors']} |")

    L += ["", "## Per task", "",
          "| Task | Axis | Result | Checks | First failure |",
          "|---|---|---|---:|---|"]
    ICON = {PASS: "pass", FAIL: "**fail**", BLOCKED: "_blocked_", ERROR: "_error_"}
    for r in s["by_task"]:
        reason = (r.get("reason") or "").replace("|", "\\|").replace("\n", " ")
        L.append(f"| {r['task']} | {r['axis']} | {ICON.get(r['result'], r['result'])} | "
                 f"{r['checks_passed']}/{r['checks_total']} | {reason[:150]} |")

    if s.get("problems"):
        NOTE = {"blocked": "Not gradable (instrument limits)",
                "errors": "Not gradable (run failed)",
                "not_run": "Defined but not run"}
        L += ["", "## Excluded from the solve rate", ""]
        for k, v in s["problems"].items():
            L.append(f"- **{NOTE.get(k, k)}** ({len(v)}): {', '.join(v)}")

    L += ["", "---", "",
          "Each task has a companion `<TASK>.json` in this folder with the full prompt, "
          "the agent's answer, every check with its expected and observed value, and the "
          "tool calls it made.", ""]

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "SUMMARY.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def print_summary(s: dict, stream=None) -> None:
    """콘솔 요약 — 여기는 돌리는 사람이 그 자리에서 읽는 화면이라 한국어로 둔다.
    파일로 나가는 것(<문항>.json / summary.json / SUMMARY.md)은 전부 영어다."""
    import sys
    w = stream or sys.stdout
    t = s["total"]
    print("=" * 72, file=w)
    print(f"해결  {t['solved']} / {t['gradable']}문항  ({t['solve_rate_pct']:.1f}%)"
          f"      실패 {t['failed']}", file=w)
    if t["blocked"] or t["errors"]:
        print(f"      채점 제외 {t['blocked']} · 실행 실패 {t['errors']}"
              f"   (해결률 분모에서 빠짐)", file=w)
    print("-" * 72, file=w)
    for axis, v in s["by_axis"].items():
        bar = "█" * int(v["solve_rate_pct"] / 5)
        extra = ""
        if v["blocked"] or v["errors"]:
            extra = f"  (제외 {v['blocked'] + v['errors']})"
        print(f"  {axis:20s} {v['solved']:3d}/{v['gradable']:3d}  "
              f"{v['solve_rate_pct']:5.1f}% {bar}{extra}", file=w)
    if s["problems"]:
        LABEL = {"blocked": "채점 제외(장비 설정 미달)",
                 "errors": "실행 실패(인프라·무응답)", "not_run": "미실행"}
        print("-" * 72, file=w)
        for k, v in s["problems"].items():
            print(f"  [{LABEL.get(k, k)}] {len(v)}개: {', '.join(v[:14])}"
                  f"{' ...' if len(v) > 14 else ''}", file=w)
    print("=" * 72, file=w)
