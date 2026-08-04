# -*- coding: utf-8 -*-
"""문항을 전부 돌리고 성적을 취합한다.

    python -m backend.server                       # 터미널 1 — 장비를 쥔 프로세스
    python benchmark/run_all.py --agent AILA                 # 터미널 2
    python benchmark/run_all.py --agent CoALA --tasks T001,T062
    python benchmark/run_all.py --check                      # 실행 없이 문항 파일만 점검
    python benchmark/run_all.py --agent CoALA --task-timeout 600   # 문항당 10분에서 컷
    python benchmark/run_all.py --agent AILA  --task-timeout 0     # 상한 없음(예전과 동일)

문항당 상한은 기본 900 초다. 넘기면 그 문항을 끊고 **그때까지 한 일로 그대로 채점**한다
(판정 항목은 안 늘어난다 — 상한 없이 돈 예전 결과와 checks_total 이 같아야 비교가 된다).
끊었는데도 에이전트가 안 멈추면 실행 전체를 세운다 — --task-timeout 주석 참고.

[한 문항이 도는 순서]
    reset()          앞 문항이 무엇을 바꿔 놨든 전 장비를 기본값으로
    setup(b)         이 문항이 요구하는 상태를 만든다(문항 파일에 있으면)
    state()          채점용 시작 상태
    run(task)        에이전트 실행
    state()          채점용 종료 상태
    evaluate(b, run) 문항 파일이 판정 목록을 돌려준다
    teardown()·reset()  락·패치를 풀고 다시 기본값으로
결과는 results/<run_id>/<문항>.json, 취합은 같은 폴더의 summary.json.
결과 파일은 전부 영어다 — 표·스크립트·외부 협업자를 거치기 때문.
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
from bench.tools import TOOL_NAMES, TOOL_PARAMS  # noqa: E402

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
        for m in __import__("re").finditer(r'chk\.(?:called|arg|arg_pair|arg_set|arg_not|order)\('
                                           r'run,\s*"([a-z_]+)"', src):
            if m.group(1) not in TOOL_NAMES:
                unknown.append(f"{task.id}: 존재하지 않는 도구 {m.group(1)!r}")
        # setup() 이 장비에 직접 거는 명령은 인자 이름까지 맞춰 본다.
        # 여기가 틀리면 실행 시 TypeError 로 잡히긴 하지만, 그때는 '사전 세팅 실패'로
        # 기록될 뿐이라 그 문항이 **전제 없이 돌아 낮은 점수를 받는다**. 실행 전에 잡는다.
        for m in __import__("re").finditer(r'b\.hw\(\s*"([a-z_]+)"((?:\s*,\s*\w+\s*=[^)]*)?)\)',
                                           src):
            tool, argtext = m.group(1), m.group(2)
            if tool not in TOOL_NAMES:
                unknown.append(f"{task.id}: setup 이 없는 도구를 부릅니다 {tool!r}")
                continue
            for key in __import__("re").findall(r'(\w+)\s*=', argtext):
                if key not in TOOL_PARAMS.get(tool, set()):
                    unknown.append(f"{task.id}: setup {tool}({key}=…) — 실제 인자는 "
                                   f"{sorted(TOOL_PARAMS.get(tool, []))}")
    # 입력 파일이 없으면 그 문항은 '데이터 없이' 돌아 낮은 점수를 받는다. 에이전트가
    # 아니라 준비가 틀린 것이므로 실행 전에 잡는다.
    missing_inputs = [f"{t.id}: {n}" for _, t in pairs for n in t.inputs
                      if not (HERE / "inputs" / n).is_file()]
    todo = [t.id for m, t in pairs if "[남은 일]" in inspect.getsource(m)]
    undeclared, unused = _check_answer_keys(pairs)
    stale_gt = _check_stale_gt(pairs)

    for label, items in (("판정 항목 없음", empty), ("도구 이름 오류", unknown),
                         ("입력 파일 없음", missing_inputs),
                         ("답 키 미선언", undeclared),
                         ("선언했으나 안 읽는 키", unused),
                         ("안 쓰이는 GT 파일", stale_gt),
                         ("재계산 이식 대기", todo)):
        if items:
            print(f"  [{label}] {len(items)}개: {', '.join(map(str, items[:20]))}"
                  f"{' ...' if len(items) > 20 else ''}")
    return 1 if (empty or unknown or missing_inputs or undeclared) else 0


def _check_stale_gt(pairs) -> list:
    """gt/<문항>.json 은 채점 경로가 읽지 않는다 — 갈라진 정답의 온상이다.

    실효 GT 는 두 곳뿐이다: tasks/*.py 안의 인라인 값과 gt/arrays/*.csv.
    gt/<문항>.json 144 개는 예전 구조(정답이 다섯 파일에 흩어져 있던 시절)의 잔재로
    generate/ 스크립트만 참조한다. 남겨 두는 것 자체는 괜찮지만, 문항 파일의 인라인
    값과 조용히 어긋나면 어느 쪽이 정답인지 알 수 없게 된다. 대응 문항이 사라진 것만
    이라도 눈에 띄게 한다(gt/T091.json 은 이미 그렇게 고아가 됐다).
    """
    gt_dir = HERE / "gt"
    if not gt_dir.is_dir():
        return []
    have = {t.id for _, t in pairs}
    # --tasks 로 일부만 돌릴 때는 비교가 무의미하다.
    if len(have) < len(list(TASKS_DIR.glob("[TN]*.py"))):
        return []
    return [f"{p.stem}: 대응 문항 없음" for p in sorted(gt_dir.glob("[TN]*.json"))
            if p.stem not in have]


def _answer_keys_read(src: str) -> set:
    """evaluate() 가 답 JSON 에서 실제로 꺼내는 키 이름들.

    A.seq/value/flag/grid 는 후보 키를 여러 개 받는다. 그중 **첫 번째**가 정본이다
    (bench.answer._first 가 앞에서부터 찾는다). 나머지는 예전 답 모양을 받아 주는
    관대함이라 선언 대상이 아니다.
    """
    import re
    body = src[src.find("def evaluate"):]
    keys = set(re.findall(r'run\.answer\.get\(\s*["\']([^"\']+)', body))
    keys |= set(re.findall(r'chk\.reported(?:_label)?\(\s*run,\s*["\']([^"\']+)', body))
    keys |= set(re.findall(r'chk\.has_answer_key\(\s*run,\s*["\']([^"\']+)', body))
    for m in re.finditer(r'A\.(?:seq|value|flag|grid)\(\s*run,\s*["\']([^"\']+)', body):
        keys.add(m.group(1))
    # plan_order 는 answer["plan"] 을 읽는다(bench.client.Run.plan).
    if "chk.plan_order(" in body:
        keys.add("plan")
    return keys


def _check_answer_keys(pairs) -> tuple:
    """선언(Task.answer_keys)과 채점기가 읽는 키가 맞는지.

    어긋나면 그 문항은 **조용히 0 점**이 된다 — 에이전트는 선언된 이름으로 답하고
    채점기는 다른 이름을 찾으니, 정답을 내고도 '보고 없음'으로 떨어진다. 실제로
    그 사고로 T044·T126 이 만점짜리 답을 내고 0 점을 받았다. 실행 전에 잡는다.
    """
    import inspect
    undeclared, unused = [], []
    for mod, task in pairs:
        read = _answer_keys_read(inspect.getsource(mod))
        declared = {k for k, _ in task.answer_keys}
        for k in sorted(read - declared):
            undeclared.append(f"{task.id}: 채점기는 {k!r} 를 읽는데 answer_keys 에 없음")
        for k in sorted(declared - read):
            unused.append(f"{task.id}: {k!r} 를 선언했는데 채점기가 안 읽음")
    return undeclared, unused


# ══════════════════════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════════════════════
def run_one(b: Bench, mod, task, run_id: str, axis=None, timeout_s=None) -> tuple:
    """(평가 payload, 치명적 사유 or None)

    [치명과 권고를 가르는 기준 — 2026-08-03]
    fatal 은 '이 문항은 채점이 성립하지 않는다'는 것만 담는다: 입력 파일이 안 올라갔다,
    사전 세팅이 안 걸렸다(전제 없이 돌았다), 시작 상태를 못 읽었다. 이것들은 결과가
    result="error" 가 되어 해결률 분모에서 빠진다.
    warn 은 남겨 둘 값어치는 있지만 채점을 좌우하지 않는 것이다. 리셋이 비치명적으로
    일부 실패한 경우가 그렇다 — 문항은 정상 전제로 돌았고 답도 받았다. 예전에는 이
    경고 한 줄이 fatal 과 같은 목록에 들어가, 모든 판정을 통과한 실행이 error 로
    빠지면서 해결률 분모가 장비 잡음에 흔들렸다.
    """
    fatal: list[str] = []
    warn: list[str] = []

    rs = b.reset()
    if rs.get("critical"):
        return None, "리셋 실패(치명적): " + "; ".join(rs["critical"])
    if rs.get("failed"):
        # 비치명적 리셋 실패. 문항 전제는 살아 있으므로 채점은 그대로 한다.
        warn.append("reset warning: " + "; ".join(rs["failed"][:3]))

    if task.inputs:
        up = b.upload(task.inputs)
        if not up.get("ok", True):
            fatal.append(f"input upload failed: {up.get('error')}")

    setup = getattr(mod, "setup", None)
    if setup and task.mode == "live":
        # 가정형은 '장비를 건드리지 말고 답만 하라'는 문항이라 사전 세팅을 걸면 전제가 무너진다.
        b.setup_errors = []
        try:
            setup(b)
        except Exception as e:
            fatal.append(f"setup failed: {type(e).__name__}: {e}")
        # 예외가 안 나도 도구가 ok=false 를 돌려줄 수 있다. 그러면 문항이 요구한 전제
        # 없이 돌게 되므로, 그 실행으로 에이전트를 평가하면 안 된다.
        fatal += [f"setup did not take effect - {e}" for e in b.setup_errors]

    # 시작 상태는 **사전 세팅을 마친 뒤**의 상태다. 그래야 '문항이 시작한 자리에서
    # 무엇이 달라졌는가'가 에이전트의 몫이 된다.
    before = b.state()
    if not before:
        fatal.append("could not read the starting state - every state check will fail")
    run = b.run(task, run_id, timeout_s=timeout_s)
    # 시간 상한에 걸린 것은 warn 이지 fatal 이 아니다. 상한을 넘겼다는 것 자체가
    # 에이전트가 못 끝냈다는 뜻이므로, 그때까지 한 일로 그대로 채점한다 —
    # fatal 로 올리면 result=error 가 되어 해결률 분모에서 빠지고, 영원히 도는
    # 에이전트가 오히려 분모에서 사라져 성적이 좋아 보인다.
    if run.timed_out:
        warn.append(f"cut at the {timeout_s:.0f}s task time limit "
                    f"(ran {run.elapsed_s:.0f}s, {len(run.calls)} tool calls)")
    run.state_before = before
    run.state_after = b.state()
    return _finish(b, mod, task, run, fatal, warn), None


def _finish(b, mod, task, run, fatal, warn) -> dict:
    run.errors = list(run.errors) + fatal
    run.warnings = list(getattr(run, "warnings", [])) + warn
    try:
        checks = mod.evaluate(b, run)
    except Exception:
        checks = [chk.fail("grading raised", traceback.format_exc(limit=4))]
    payload = R.score_task(task, checks, run)
    # 판정 목록은 건드리지 않고 키만 더 얹는다. 상한 없이 돈 예전 결과 파일에는 이
    # 키들이 없으므로 읽는 쪽은 .get("timed_out", False) 로 봐야 한다.
    payload["timed_out"] = bool(getattr(run, "timed_out", False))
    payload["abandoned"] = bool(getattr(run, "abandoned", False))
    return payload


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="AILA", choices=["AILA", "CoALA"])
    ap.add_argument("--tasks", default="", help="쉼표로 구분한 문항 번호")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--check", action="store_true", help="실행 없이 문항 파일만 점검")
    # 문항당 벽시계 상한.
    #
    # [왜 필요한가 — 2026-08-04]
    # CoALA 가 N07 에서 같은 좌표를 30 회 재측정하며 72 분을 먹고도 안 끝났다.
    # 에이전트 안의 가드(_MAX_CYCLES=150)는 사이클당 2.4 분이면 6 시간이라 사실상
    # 없는 것과 같았고, 실행 전체를 죽이는 수밖에 없었다.
    #
    # [기본값을 900 으로 정한 근거]
    # 2026-08-04 AILA 143 문항의 실측: 중앙값 56 초, 평균 92 초, 최대 574 초(T031).
    # 10 분(600 초)은 최장 문항 위로 26 초밖에 안 남겨서, T031 류가 조금만 느려지면
    # 잘린다 — 그러면 그 문항의 결과가 바뀌어 상한 없이 돈 예전 실행과 비교가
    # 깨진다. 900 초는 최장 문항의 1.6 배라 정상 문항은 걸리지 않으면서, N07 이
    # 72 분을 먹은 것 같은 폭주는 확실히 자른다.
    # 0 이면 무제한(= 상한이 없던 예전 실행과 동일).
    ap.add_argument("--task-timeout", type=float, default=900.0, metavar="SEC",
                    help="문항당 상한(초). 0 이면 무제한. 기본 900(15분)")
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

    # 어느 에이전트 모듈이 도는지. 벤치 사본(single_agent_*_bench)이 아니면 출력 규약이
    # 안 걸려 답의 키가 어긋나고, 그 결과가 '에이전트가 못 맞혔다'로 기록된다.
    # 143 문항을 다 돌린 뒤에 알면 늦으므로 여기서 끊는다.
    if pf.get("agent_module"):
        print(f"[preflight] 에이전트 모듈 {pf['agent_module']}")
        if not pf.get("agent_is_bench"):
            print(f"[fatal] 벤치 사본이 아니라 운영 모듈이 잡혔습니다"
                  f"({pf['agent_module']}).\n"
                  f"        backend/agents/single_agent_{args.agent}_bench.py 가 있는지, "
                  f"import 에러가 없는지 확인하세요.", file=sys.stderr)
            return 2

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

    cap = max(0.0, args.task_timeout)
    print(f"에이전트 {args.agent} / 문항 {len(pairs)}개 / 출력 {out}"
          + (f" / 문항당 상한 {cap:.0f}s" if cap else " / 문항당 상한 없음") + "\n")
    results = []
    for i, (mod, task) in enumerate(pairs, 1):
        tag = "?" if task.mode == "hypothetical" else " "
        print(f"  [{i:3d}/{len(pairs)}]{tag}{task.id} ...", end="", flush=True)
        payload, fatal = run_one(b, mod, task, run_id, timeout_s=cap or None)
        if fatal:
            print(f"\n[fatal] {task.id}: {fatal}\n        장비가 이상한 상태로 남았을 수 "
                  f"있습니다. 확인 후 다시 시작하세요.", file=sys.stderr)
            break
        for name in blocked.get(task.id, []):
            payload["checks"].append(
                chk.blocked(name, "the instrument axis does not cover this window").as_dict())
        b.teardown()
        b.reset()
        R.write_eval(out, task.id, payload)
        results.append(payload)
        MARK = {"pass": "O 맞음", "fail": "X 틀림",
                "blocked": "- 채점제외", "error": "! 실행실패"}
        print(f" {MARK.get(payload['result'], payload['result']):9s}"
              f" ({payload['checks_passed']}/{payload['checks_total']} 판정)"
              f"{'  [시간초과 컷]' if payload.get('timed_out') else ''}"
              f"{'  ' + payload['reason'][:44] if payload['reason'] else ''}")

        # 중단을 요청했는데 안 멈춘 에이전트. 이 문항의 결과는 위에 이미 남겼고,
        # 여기서 실행을 세운다 — 장비를 쥔 채 도는 에이전트 위에서 남은 문항을
        # 돌리면 그 결과가 전부 못 쓰게 되고, 그 사이 시료에는 계속 빔이 들어간다.
        # (teardown·reset 은 위에서 이미 한 번 시도했다 — reset 이 레이저를 끈다.)
        if payload.get("abandoned"):
            print(f"\n[fatal] {task.id}: 시간 상한으로 중단을 요청했으나 에이전트가 멈추지 "
                  f"않았습니다.\n"
                  f"        장비를 계속 쥐고 있을 수 있어 여기서 세웁니다 — 남은 "
                  f"{len(pairs) - i}개 문항은 돌리지 않습니다.\n"
                  f"        레이저 상태를 직접 확인하고, 서버(python -m backend.server)를 "
                  f"내렸다 올린 뒤 --tasks 로 이어서 돌리세요.", file=sys.stderr)
            break

    summary = R.summarize(results, all_tasks,
                          meta={"agent": args.agent, "run_id": run_id,
                                "server": args.server,
                                "config": pf.get("config", {}),
                                "memory_scope": pf.get("memory_scope", "?")})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    md = R.write_markdown(out, summary)
    print()
    R.print_summary(summary)
    print(f"→ {out / 'summary.json'}")
    print(f"→ {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
