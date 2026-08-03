# -*- coding: utf-8 -*-
"""문항을 전부 돌리고 성적을 취합한다.

    python -m backend.server                       # 터미널 1 — 장비를 쥔 프로세스
    python benchmark/run_all.py --agent AILA                 # 터미널 2
    python benchmark/run_all.py --agent CoALA --tasks T001,T062
    python benchmark/run_all.py --check                      # 실행 없이 문항 파일만 점검

[한 문항이 도는 순서]
    reset()          앞 문항이 무엇을 바꿔 놨든 전 장비를 기본값으로
    setup(b)         이 문항이 요구하는 상태를 만든다(문항 파일에 있으면)
    state()          채점용 시작 상태
    run(task)        에이전트 실행
    state()          채점용 종료 상태
    evaluate(b, run) 문항 파일이 판정 목록을 돌려준다
    teardown()·reset()  락·패치를 풀고 다시 기본값으로
결과는 results/<run_id>/<문항>.평가.json, 취합은 같은 폴더의 성적.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent          # benchmark/
PROJ = HERE.parent                              # 프로젝트 루트(backend 를 import 하려고)
for _p in (str(HERE), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bench import Bench, chk                      # noqa: E402
from bench import report as R                     # noqa: E402
from bench import spectra as SPX                  # noqa: E402
from bench.tools import TOOL_NAMES                # noqa: E402

TASKS_DIR = HERE / "tasks"
RESULTS = PROJ / "results"


# ══════════════════════════════════════════════════════════════════════════════
# 문항 파일 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_tasks(only=None) -> list:
    """tasks/*.py 를 읽어 [(모듈, Task), ...] 로. 계약을 안 지킨 파일은 즉시 알린다."""
    out, bad = [], []
    for p in sorted(TASKS_DIR.glob("[TN]*.py")):
        if only and p.stem not in only:
            continue
        try:
            mod = importlib.import_module(f"tasks.{p.stem}")
        except Exception as e:
            bad.append(f"{p.stem}: import 실패 — {type(e).__name__}: {e}")
            continue
        task = getattr(mod, "TASK", None)
        if task is None:
            bad.append(f"{p.stem}: TASK 가 없습니다")
        elif not callable(getattr(mod, "evaluate", None)):
            bad.append(f"{p.stem}: evaluate(b, run) 이 없습니다")
        elif task.id != p.stem:
            bad.append(f"{p.stem}: TASK.id({task.id}) 가 파일 이름과 다릅니다")
        else:
            out.append((mod, task))
    if bad:
        print("[fatal] 문항 파일이 계약을 지키지 않습니다:", file=sys.stderr)
        for b in bad:
            print(f"    {b}", file=sys.stderr)
        raise SystemExit(2)
    return out


def check_only(pairs) -> int:
    """실행 없이 점검 — 배점 합, 판정 항목 유무, 도구 이름 오탈자."""
    import inspect
    total = sum(t.score for _, t in pairs)
    print(f"문항 {len(pairs)}개 / 총 배점 {total:g}점")

    unknown, empty = [], []
    for mod, task in pairs:
        src = inspect.getsource(mod)
        for name in set(__import__("re").findall(r'"([a-z_]{4,})"', src)):
            if name.endswith("_") or name in TOOL_NAMES:
                continue
        # 판정 항목이 하나도 안 나오는 문항은 만점이 되어 버린다.
        if "chk." not in src:
            empty.append(task.id)
        for m in __import__("re").finditer(r'chk\.(?:called|arg|arg_set|arg_not|order)\('
                                           r'run,\s*"([a-z_]+)"', src):
            if m.group(1) not in TOOL_NAMES:
                unknown.append(f"{task.id}: 존재하지 않는 도구 {m.group(1)!r}")
    todo = [t.id for m, t in pairs if "[남은 일]" in inspect.getsource(m)]

    for label, items in (("판정 항목 없음", empty), ("도구 이름 오류", unknown),
                         ("재계산 이식 대기", todo)):
        if items:
            print(f"  [{label}] {len(items)}개: {', '.join(map(str, items[:20]))}"
                  f"{' ...' if len(items) > 20 else ''}")
    return 1 if (empty or unknown) else 0


# ══════════════════════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════════════════════
def run_one(b: Bench, mod, task, run_id: str, axis=None) -> tuple:
    """(평가 payload, 치명적 사유 or None)"""
    infra: list[str] = []

    rs = b.reset()
    if rs.get("critical"):
        return None, "리셋 실패(치명적): " + "; ".join(rs["critical"])
    if rs.get("failed"):
        infra.append("리셋 경고: " + "; ".join(rs["failed"][:3]))

    if task.inputs:
        up = b.upload(task.inputs)
        if not up.get("ok", True):
            infra.append(f"입력 파일 업로드 실패: {up.get('error')}")

    setup = getattr(mod, "setup", None)
    if setup and task.mode == "live":
        # 가정형은 '장비를 건드리지 말고 답만 하라'는 문항이라 사전 세팅을 걸면 전제가 무너진다.
        try:
            setup(b)
        except Exception as e:
            infra.append(f"사전 세팅 실패: {type(e).__name__}: {e}")

    # 시작 상태는 **사전 세팅을 마친 뒤**의 상태다. 그래야 '문항이 시작한 자리에서
    # 무엇이 달라졌는가'가 에이전트의 몫이 된다.
    before = b.state()
    if not before:
        infra.append("시작 상태를 읽지 못했습니다 — 상태 판정이 전부 실패로 남습니다")
    run = b.run(task, run_id)
    run.state_before = before
    run.state_after = b.state()
    return _finish(b, mod, task, run, infra), None


def _finish(b, mod, task, run, infra) -> dict:
    run.errors = list(run.errors) + infra
    try:
        checks = mod.evaluate(b, run)
    except Exception:
        checks = [chk.fail("채점 예외", traceback.format_exc(limit=4))]
    payload = R.score_task(task, checks)
    payload["errors"] = run.errors
    payload["elapsed_s"] = round(run.elapsed_s, 2)
    payload["session_id"] = run.session_id
    payload["tool_calls"] = [c.get("name") for c in run.calls]
    return payload


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="AILA", choices=["AILA", "CoALA"])
    ap.add_argument("--tasks", default="", help="쉼표로 구분한 문항 번호")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--check", action="store_true", help="실행 없이 문항 파일만 점검")
    args = ap.parse_args()

    only = {t.strip() for t in args.tasks.split(",") if t.strip()} or None
    pairs = load_tasks(only)
    all_tasks = [t for _, t in pairs]
    if args.check:
        return check_only(pairs)

    b = Bench(args.server, agent=args.agent)
    h = b.health()
    if not h["ok"]:
        print(f"[fatal] 서버에 붙지 못했습니다({args.server}): {h['detail']}\n"
              f"        먼저 `python -m backend.server` 를 띄우세요 — 장비를 쥔 프로세스가 "
              f"실행을 맡습니다.", file=sys.stderr)
        return 2

    # 파수축 점검. 설정이 문항 구간을 못 덮으면 정답을 내도 0점이 되므로, 실행 전에 알린다.
    pf = b.preflight()
    axis = pf.get("axis") if pf.get("ok") else None
    if axis:
        print(f"[preflight] 파수축 {min(axis):.1f} ~ {max(axis):.1f} cm-1 "
              f"({len(axis)} px) / {pf.get('config', {}).get('config_path', '?')}")
    else:
        print(f"[preflight] 파수축을 읽지 못했습니다: {pf.get('error')}")
    blocked = {t.id: [w[0] for w in t.windows if not SPX.covers(axis, w[1], w[2], w[3])]
               for t in all_tasks if t.windows}
    blocked = {k: v for k, v in blocked.items() if v}
    if blocked:
        print(f"[preflight] 이 설정으로는 채점 구간이 모자란 문항 {len(blocked)}개: "
              f"{', '.join(sorted(blocked))}")
        print("[preflight] 그 문항의 해당 판정은 0점이 아니라 '채점 제외'로 처리합니다.")

    run_id = args.run_id or f"{date.today().isoformat()}_{args.agent}"
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    b.run_id = run_id

    print(f"에이전트 {args.agent} / 문항 {len(pairs)}개 / 출력 {out}\n")
    results = []
    for i, (mod, task) in enumerate(pairs, 1):
        tag = "?" if task.mode == "hypothetical" else " "
        print(f"  [{i:3d}/{len(pairs)}]{tag}{task.id} ...", end="", flush=True)
        payload, fatal = run_one(b, mod, task, run_id)
        if fatal:
            print(f"\n[fatal] {task.id}: {fatal}\n        장비가 이상한 상태로 남았을 수 "
                  f"있습니다. 확인 후 다시 시작하세요.", file=sys.stderr)
            break
        for name in blocked.get(task.id, []):
            payload["checks"].append(
                chk.blocked(name, "장비 파수축이 이 구간을 덮지 않습니다").as_dict())
        b.teardown()
        b.reset()
        R.write_eval(out, task.id, payload)
        results.append(payload)
        print(f" {payload['score']:.1f}/{payload['score_max']:g}점  "
              f"({payload['passed']}/{payload['checks_total']} 판정)"
              f"{'  !' + payload['errors'][0][:36] if payload['errors'] else ''}")

    summary = R.summarize(results, all_tasks,
                          meta={"agent": args.agent, "run_id": run_id,
                                "server": args.server,
                                "config": pf.get("config", {}),
                                "memory_scope": pf.get("memory_scope", "?")})
    (out / "성적.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print()
    R.print_summary(summary)
    print(f"→ {out / '성적.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
