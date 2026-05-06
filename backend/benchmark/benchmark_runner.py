"""
라만 분광기 에이전트 벤치마크 러너.

사용법:
    # 서버를 먼저 실행해야 함:  python -m backend.server
    python backend/benchmark/benchmark_runner.py
    python backend/benchmark/benchmark_runner.py --category A
    python backend/benchmark/benchmark_runner.py --tasks Q001,Q003,Q025
    python backend/benchmark/benchmark_runner.py --skip-laser --delay 3
    python backend/benchmark/benchmark_runner.py --tasks Q001 --output results/
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import requests

# benchmark 패키지 내부에서 직접 import
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from verifiers import run_verifiers, VerifyResult
from report_generator import save_json, generate_html

TASKS_FILE = _HERE / "benchmark_tasks.json"
SERVER_BASE = "http://localhost:8000"
LASER_IDS = {"Q005", "Q006", "Q007", "Q008", "Q020", "Q026", "Q027", "Q028",
             "Q029", "Q030", "Q031", "Q032", "Q033", "Q034", "Q036", "Q037",
             "Q038", "Q039", "Q040"}


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _get(path: str, timeout: int = 5) -> dict:
    try:
        r = requests.get(f"{SERVER_BASE}{path}", timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, body: dict, timeout: int = 300) -> dict:
    try:
        r = requests.post(f"{SERVER_BASE}{path}", json=body, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def check_server() -> bool:
    resp = _get("/api/health", timeout=5)
    if "error" in resp:
        print(f"[오류] 서버 연결 실패: {resp['error']}")
        print("      먼저 서버를 실행하세요: python -m backend.server")
        return False
    print(f"[OK] 서버 연결됨: {SERVER_BASE}")
    return True


def capture_state() -> dict:
    return _get("/api/hardware/state", timeout=10)


def run_task(task: dict, delay: float) -> dict:
    task_id = task["id"]
    desc = task["description"]

    print(f"\n{'='*60}")
    print(f"[{task_id}] {desc}")
    print(f"{'='*60}")

    if task.get("skip"):
        print(f"  SKIP: {task.get('skip_reason', '')}")
        return {
            "task": task,
            "skipped": True,
            "skip_reason": task.get("skip_reason", ""),
            "llm_response": None,
            "tool_trace": [],
            "pre_state": None,
            "post_state": None,
            "auto_results": [],
            "elapsed_sec": 0.0,
        }

    print("  pre-state 수집 중...")
    pre_state = capture_state()

    print("  LLM 에이전트 호출 중...")
    t0 = time.time()
    resp = _post("/api/hardware-command", {"command": desc}, timeout=600)
    elapsed = round(time.time() - t0, 2)

    if "error" in resp:
        print(f"  [오류] {resp['error']}")
        return {
            "task": task,
            "skipped": False,
            "error": resp["error"],
            "llm_response": None,
            "tool_trace": [],
            "pre_state": pre_state,
            "post_state": None,
            "auto_results": [{"passed": False, "verifier_type": "server_error",
                               "detail": resp["error"], "is_human_only": False}],
            "elapsed_sec": elapsed,
        }

    llm_response = resp.get("message", "")
    tool_trace = resp.get("tool_trace", [])

    print(f"  완료 ({elapsed}s) — tool 호출: {len(tool_trace)}회")
    for t in tool_trace:
        print(f"    step {t['step']}: {t['tool']}({_fmt_args(t['args'])}) → {_short(t['result'])}")

    print("  post-state 수집 중...")
    post_state = capture_state()

    print("  검증 중...")
    auto_results_raw = run_verifiers(task.get("verifiers", []), tool_trace, pre_state, post_state)
    auto_results = [
        {
            "passed": v.passed,
            "verifier_type": v.verifier_type,
            "detail": v.detail,
            "is_human_only": v.is_human_only,
        }
        for v in auto_results_raw
    ]

    for v in auto_results:
        icon = "✓" if v["passed"] else ("?" if v["is_human_only"] else "✗")
        print(f"    {icon} [{v['verifier_type']}] {v['detail']}")

    if delay > 0:
        print(f"  다음 태스크까지 {delay}s 대기...")
        time.sleep(delay)

    return {
        "task": task,
        "skipped": False,
        "llm_response": llm_response,
        "tool_trace": tool_trace,
        "pre_state": pre_state,
        "post_state": post_state,
        "auto_results": auto_results,
        "elapsed_sec": elapsed,
    }


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])


def _short(result: dict) -> str:
    s = json.dumps(result, ensure_ascii=False)
    return s[:80] + "…" if len(s) > 80 else s


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="라만 분광기 벤치마크 러너")
    parser.add_argument("--tasks", help="실행할 태스크 ID (쉼표 구분, 예: Q001,Q003)")
    parser.add_argument("--category", choices=["A", "B"], help="카테고리 필터 (A 또는 B)")
    parser.add_argument("--skip-laser", action="store_true", help="레이저 관련 태스크 스킵")
    parser.add_argument("--delay", type=float, default=2.0, help="태스크 간 대기 시간 (초, 기본 2)")
    parser.add_argument("--output", default="backend/benchmark/results", help="출력 디렉터리")
    parser.add_argument("--server", default="http://localhost:8000", help="서버 URL")
    args = parser.parse_args()

    global SERVER_BASE
    SERVER_BASE = args.server

    if not check_server():
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = json.loads(TASKS_FILE.read_text(encoding="utf-8"))

    # 필터 적용
    if args.tasks:
        ids = {t.strip().upper() for t in args.tasks.split(",")}
        tasks = [t for t in tasks if t["id"] in ids]
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]
    if args.skip_laser:
        tasks = [
            {**t, "skip": True, "skip_reason": "--skip-laser 옵션으로 스킵"}
            if t["id"] in LASER_IDS else t
            for t in tasks
        ]

    print(f"\n총 {len(tasks)}개 태스크 (필터 적용 후)")

    results = []
    for task in tasks:
        result = run_task(task, delay=args.delay)
        results.append(result)

    # 최종 집계
    run = [r for r in results if not r.get("skipped")]
    auto_pass = sum(1 for r in run if all(
        v["passed"] or v["is_human_only"]
        for v in r.get("auto_results", [])
    ))
    auto_fail = len(run) - auto_pass

    print(f"\n{'='*60}")
    print(f"벤치마크 완료")
    print(f"  실행: {len(run)} / {len(results)} (스킵: {len(results)-len(run)})")
    print(f"  자동PASS: {auto_pass}  자동FAIL: {auto_fail}")
    print(f"{'='*60}")

    json_path = save_json(results, output_dir)
    html_path = generate_html(results, output_dir)

    print(f"\n보고서:")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    print("\n브라우저에서 HTML 파일을 열어 인간 채점을 진행하세요.")


if __name__ == "__main__":
    main()
