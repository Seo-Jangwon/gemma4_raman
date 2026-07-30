# -*- coding: utf-8 -*-
"""
라만 에이전트 벤치마크 러너 — AILA / CoALA 를 동일 질문으로 돌리고, 각 실행의
상세 트레이스를 모은다. 에이전트 코드는 건드리지 않는다(공개 HTTP API만 사용).

[핵심 원리]
  · 질문마다 '새 session_id' 를 주면 run_experiment 가 무상태로 새 세션을 연다
    (히스토리 누적/컨텍스트 폭주 없음, 문항 간 공정성 유지).
  · 에이전트가 매 턴 DetailLog/<AGENT>_<시각>_<sid>.json 에 전부 기록하므로,
    그 파일을 sid 로 찾아 읽으면 툴 호출/인자/결과/최종답/planning/dose/시간을 얻는다.
  · 툴트레이스만으론 못 보는 하드웨어 상태 검증(stage_position 등)을 위해 실행 전후
    /api/hardware/state 스냅샷을 뜬다.

[실행 방법 — 사전 준비]
  1) 장비 PC에서 서버를 띄우고 하드웨어를 연결한다(카메라/스테이지/레이저/CCD):
        python -m backend.server
     그리고 프론트나 API로 각 장비 connect 를 완료해 둔다(레이저 측정 문항이 있으므로
     실제 레이저가 발사됨 — dose 회로차단기가 폭주만 막는다).

     [평가 모드 환경변수 — 반드시 '서버를 띄울 때' 준다]
     이 러너는 HTTP 클라이언트라 이미 떠 있는 서버의 에이전트 설정을 바꿀 수 없다.
     아래는 서버 프로세스의 환경변수로만 적용된다.

        ※ RAMAN_SAFETY_PROMPT 는 폐지됐다(2026-07-29). 자율 실행이 '기본값'이 되어
          더 이상 아무 코드도 이 변수를 읽지 않는다 — 줘도 조용히 무시된다.
          되묻기/사람승인 동작으로 되돌릴 때만 RAMAN_AUTONOMOUS=0 을 준다.

        RAMAN_AUTONOMOUS=0        (선택) 자율 모드를 끄고 '시료 미상 시 되묻기' + 그리드
                                  사람-승인 게이트를 되살린다. 벤치마크에서는 주지 말 것 —
                                  되묻고 턴을 끝내면 그 문항은 수행 실패로 채점된다.
        RAMAN_LLM_TIMEOUT_S=600   (선택) LLM HTTP 호출 상한(초, 기본 600). Ollama 가 요청을
                                  유실하면 예전엔 서버가 '영원히' 멈췄다(실측 17분+ 정지).
                                  이제 이 시간에 명확한 에러로 끊긴다.
        RAMAN_MEMORY_SCOPE=session CoALA 장기기억(episodic+semantic)을 세션별로 격리
                                  — 벤치는 문항마다 새 session_id 를 주므로, 문항이 넘어갈
                                    때마다 메모리가 빈 상태에서 시작한다. 켜지 않으면 1번
                                    문항은 경험 0건, 200번 문항은 199개 문항의 경험으로
                                    푸는 셈이라 문항 순서가 결과를 바꾼다(재현 불가).
                                    도구/프롬프트는 그대로라 CoALA 아키텍처는 온전하고,
                                    각 문항이 무엇을 기록했는지는
                                    backend/agents/coala_memory/sessions/<sid>/ 에 남아
                                    채점 근거로 쓸 수 있다.
        RAMAN_EPISODIC_MEMORY=0   (선택) episodic 액션 자체를 액션 공간에서 제거하는
                                  ablation. 위 MEMORY_SCOPE 로 오염이 잡히므로 평소엔
                                  불필요하고, 'episodic 유무' 비교를 따로 뽑을 때만 쓴다.

     권장(벤치 기본) — AILA 만 돌릴 때는 환경변수가 아예 필요 없다:
        python -m backend.server
     CoALA 를 포함해 돌릴 때만 메모리 격리를 준다:
        $env:RAMAN_MEMORY_SCOPE="session"; python -m backend.server
     서버 기동 로그의 [info] 줄로 실제 적용 여부를 확인하고 벤치를 시작할 것. 기대 출력:
        [info] AILA autonomous mode (default). Set RAMAN_AUTONOMOUS=0 for interactive/...
        [info] CoALA autonomous mode (default). ...
  2) 입력 태스크 준비:
        python -m backend.benchmark.make_task_spectra   # 문항ID별 스펙트럼 생성 + tasks_raw 프롬프트에 파일명 주입
        python -m backend.benchmark.build_tasks          # tasks_raw.json + tasks_enriched.json -> tasks.json
     주의: 정답 원본은 tasks_raw.json / tasks_enriched.json 다(직접 편집). xlsx 재추출
     스크립트(xlsx_to_tasks)는 raw 편집을 덮어써 위험하므로 제거했다.
  3) 러너 실행:
        python -m backend.benchmark.run_bench                 # 전체
        python -m backend.benchmark.run_bench --agents AILA   # 한 에이전트만
        python -m backend.benchmark.run_bench --ids T021,T057 # 특정 문항
        python -m backend.benchmark.run_bench --from T037     # T037 부터 끝까지(이어서 돌리기)
        python -m backend.benchmark.run_bench --from T037 --to T060   # 구간
        python -m backend.benchmark.run_bench --skip-done     # 이미 돌린 (문항,에이전트) 제외
        python -m backend.benchmark.run_bench --categories 5  # '5.'로 시작하는 카테고리
        python -m backend.benchmark.run_bench --dry-run       # 전송 없이 프롬프트만 출력
        python -m backend.benchmark.run_bench --server http://localhost:8000

     ※ 이 러너는 기본적으로 '항상 tasks.json 을 처음부터' 돌린다. 중간에 끊겼을 때
       그냥 다시 실행하면 앞 문항에 레이저를 다시 쏜다 — 이어서 돌릴 때는 반드시
       --from 이나 --skip-done 을 줄 것. --from/--to 는 tasks.json 의 '순서' 기준이고
       (id 문자열 비교가 아님) 양끝을 포함한다.
  결과: results/raw_runs.jsonl (실행마다 append) + results/runs_<시각>.json.
  이후 grade.py 로 자동채점, report.py 로 수동채점용 HTML 을 만든다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_DETAIL_LOG_DIR = _PROJECT_ROOT / "DetailLog"     # detail_log._LOG_DIR 와 동일 규칙
_DEFAULT_TASKS = _HERE / "tasks.json"
_RESULTS_DIR = _HERE / "results"

AGENTS = ("AILA", "CoALA")


# ── HTTP ─────────────────────────────────────────────────────────────────────

def _get(base: str, path: str, timeout: int = 10) -> dict:
    try:
        r = requests.get(f"{base}{path}", timeout=timeout)
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def _post(base: str, path: str, body: dict, timeout: int = 900) -> dict:
    try:
        r = requests.post(f"{base}{path}", json=body, timeout=timeout)
        try:
            return r.json()
        except Exception:
            return {"_error": f"non-JSON response (status {r.status_code})", "_text": r.text[:500]}
    except Exception as e:
        return {"_error": str(e)}


def check_server(base: str) -> bool:
    resp = _get(base, "/api/health", timeout=5)
    if "_error" in resp:
        print(f"[오류] 서버 연결 실패: {resp['_error']}")
        print(f"      먼저 장비 PC에서 서버를 실행하세요: python -m backend.server  (base={base})")
        return False
    print(f"[OK] 서버 연결됨: {base}")
    return True


# ── DetailLog 조회 ────────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """detail_log._sanitize 와 동일 — sid 로 파일명을 역추적하기 위함."""
    return re.sub(r"[^0-9A-Za-z_-]", "-", str(text))[:64] or "nosession"


def find_detail_log_turn(agent: str, session_id: str, retries: int = 6) -> dict | None:
    """이 (agent, sid) 세션의 DetailLog 파일에서 첫(유일) 턴을 읽어 반환.

    /api/experiment/run 은 동기라 응답 시점엔 이미 기록되지만, 파일 flush 지연을 대비해
    잠깐 재시도한다. 못 찾으면 None.
    """
    sid_s = _sanitize(session_id)
    pattern = f"{agent}_*_{sid_s}.json"
    for _ in range(retries):
        matches = sorted(_DETAIL_LOG_DIR.glob(pattern))
        if matches:
            try:
                doc = json.loads(matches[-1].read_text(encoding="utf-8"))
                turns = doc.get("turns") or []
                if turns:
                    turn = dict(turns[-1])
                    turn["_file"] = matches[-1].name
                    return turn
            except Exception:
                pass
        time.sleep(0.4)
    return None


# ── 실행 ─────────────────────────────────────────────────────────────────────

def run_one(base: str, item: dict, agent: str, stamp: str, timeout: int) -> dict:
    """한 문항을 한 에이전트로 실행하고 트레이스+상태를 모은 레코드를 반환."""
    run_id = item.get("run_id", item["id"])
    # sid 는 ascii-only(영숫자/_/-) 로 만들어 sanitize 후에도 그대로여야 파일 역추적이 된다.
    session_id = f"bench_{run_id}_{agent}_{stamp}"
    prompt = item["prompt"]

    pre_state = _get(base, "/api/hardware/state", timeout=15)
    t0 = time.time()
    resp = _post(base, "/api/experiment/run",
                 {"message": prompt, "session_id": session_id, "agent": agent},
                 timeout=timeout)
    elapsed = round(time.time() - t0, 2)
    post_state = _get(base, "/api/hardware/state", timeout=15)

    turn = find_detail_log_turn(agent, session_id)

    rec = {
        "run_id": run_id,
        "id": item["id"],
        "variant": item.get("variant", "none"),
        "agent": agent,
        "session_id": session_id,
        "category": item.get("category", ""),
        "capability": item.get("capability", ""),
        "task_kind": item.get("task_kind", ""),
        "is_safety_ambiguous": item.get("is_safety_ambiguous", False),
        "auto_gradable": item.get("auto_gradable", False),
        "prompt": prompt,
        "grading_criteria": item.get("grading_criteria", ""),
        "expected_tools": item.get("expected_tools", []),
        "verifiers": item.get("verifiers", []),
        "manual_note": item.get("manual_note", ""),
        "elapsed_sec": elapsed,
        "http_error": resp.get("_error"),
        "final_report": resp.get("final_report", ""),
        "pre_state": pre_state,
        "post_state": post_state,
        # DetailLog 발췌 (없으면 None)
        "detail_log_file": turn.get("_file") if turn else None,
        "answer": (turn or {}).get("Ans", ""),
        "tool_calls": (turn or {}).get("Tool calls", []),
        "planning": (turn or {}).get("planning evaluation process", []),
        "tool_call_order": (turn or {}).get("tool_call_order", []),
        "response_type": (turn or {}).get("response_type"),   # done | chat | error
        "dose_mj": (turn or {}).get("dose_mj"),
        "detail_error": (turn or {}).get("error"),
    }
    return rec


def _short(s: str, n: int = 70) -> str:
    s = (s or "").replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def main():
    ap = argparse.ArgumentParser(description="라만 에이전트 벤치마크 러너 (AILA/CoALA)")
    ap.add_argument("--tasks", default=str(_DEFAULT_TASKS), help="tasks.json 경로")
    ap.add_argument("--server", default="http://localhost:8000", help="서버 URL")
    ap.add_argument("--agents", default="AILA,CoALA", help="실행 에이전트(쉼표, 기본 둘 다)")
    ap.add_argument("--ids", default=None, help="특정 문항 id(쉼표, 예: T021,T057)")
    ap.add_argument("--from", dest="from_id", default=None,
                    help="이 id 부터(포함) — tasks.json 순서 기준. 예: --from T037")
    ap.add_argument("--to", dest="to_id", default=None,
                    help="이 id 까지(포함). 예: --to T128")
    ap.add_argument("--skip-done", action="store_true",
                    help="results/raw_runs.jsonl 에 이미 (id, agent) 기록이 있는 문항은 건너뛴다")
    ap.add_argument("--categories", default=None, help="카테고리 접두(쉼표, 예: 5,2)")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만")
    ap.add_argument("--delay", type=float, default=1.0, help="실행 간 대기(초)")
    ap.add_argument("--timeout", type=int, default=900, help="한 실행 최대 대기(초)")
    ap.add_argument("--dry-run", action="store_true", help="전송 없이 프롬프트만 출력")
    ap.add_argument("--out", default=str(_RESULTS_DIR), help="결과 디렉터리")
    args = ap.parse_args()

    tasks: list[dict] = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    for a in agents:
        if a not in AGENTS:
            print(f"[오류] 알 수 없는 에이전트: {a} (허용: {AGENTS})"); sys.exit(1)

    if args.ids:
        want = {x.strip().upper() for x in args.ids.split(",")}
        unknown = want - {t["id"].upper() for t in tasks}
        if unknown:
            print(f"[경고] tasks.json 에 없는 id 무시: {sorted(unknown)}")
        tasks = [t for t in tasks if t["id"].upper() in want]

    # --from/--to 는 'tasks.json 의 순서' 기준 구간이다(id 문자열 비교가 아니다).
    # 파일 순서가 곧 실행 순서이므로, 중단된 지점부터 이어서 돌릴 때 이게 제일 안전하다.
    if args.from_id or args.to_id:
        order = [t["id"].upper() for t in tasks]

        def _pos(val, default):
            if not val:
                return default
            v = val.strip().upper()
            if v not in order:
                print(f"[오류] --from/--to 의 id 를 찾을 수 없음: {val}"); sys.exit(1)
            return order.index(v)

        i0 = _pos(args.from_id, 0)
        i1 = _pos(args.to_id, len(order) - 1)
        if i0 > i1:
            print(f"[오류] --from({args.from_id}) 이 --to({args.to_id}) 보다 뒤에 있음"); sys.exit(1)
        tasks = tasks[i0:i1 + 1]

    if args.categories:
        prefixes = tuple(p.strip() for p in args.categories.split(","))
        tasks = [t for t in tasks if str(t.get("category", "")).startswith(prefixes)]
    if args.limit:
        tasks = tasks[: args.limit]

    # --skip-done: 이미 돌린 (id, agent) 조합을 제외한다. run_bench 는 원래 '항상
    # tasks.json 을 처음부터' 돌리므로, 중간에 끊긴 뒤 다시 실행하면 앞 문항을 전부
    # 다시 측정한다(실제로 T001~T034 가 두 번 발사됐다). 레이저를 쏘는 벤치라
    # 재실행 비용이 공짜가 아니므로 이 가드를 둔다.
    skipped_done: list[str] = []
    if args.skip_done:
        done: set[tuple[str, str]] = set()
        jl = Path(args.out) / "raw_runs.jsonl"
        if jl.exists():
            for line in jl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # 크래시로 잘린 마지막 줄
                # 오류로 끝난 실행은 '완료'로 보지 않는다 — 다시 돌릴 값이 있다.
                if r.get("http_error") or r.get("response_type") == "error":
                    continue
                done.add((str(r.get("run_id") or r.get("id")), str(r.get("agent"))))
        keep = []
        for t in tasks:
            rid = t.get("run_id", t["id"])
            if all((rid, a) in done for a in agents):
                skipped_done.append(t["id"])
            else:
                keep.append(t)
        tasks = keep

    print(f"\n대상: {len(tasks)}개 문항 x {len(agents)}개 에이전트 = {len(tasks)*len(agents)}회 실행")
    if skipped_done:
        print(f"  (--skip-done: 이미 완료된 {len(skipped_done)}개 제외 — "
              f"{skipped_done[0]}…{skipped_done[-1]})")
    if tasks:
        print(f"  범위: {tasks[0]['id']} … {tasks[-1]['id']}")
    else:
        print("  실행할 문항이 없다. 필터를 확인할 것.")
        return

    if args.dry_run:
        for t in tasks:
            print(f"\n[{t['id']} / {t.get('variant','none')}] ({t.get('category','')})")
            print(f"  PROMPT: {_short(t['prompt'], 160)}")
        print("\n(--dry-run: 전송하지 않음)")
        return

    if not check_server(args.server):
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = out_dir / "raw_runs.jsonl"        # append 로 크래시 안전
    results: list[dict] = []

    for ti, item in enumerate(tasks, 1):
        for agent in agents:
            print(f"\n[{ti}/{len(tasks)}] {item['id']}/{item.get('variant','none')} · {agent}")
            print(f"    {_short(item['prompt'], 90)}")
            rec = run_one(args.server, item, agent, stamp, args.timeout)
            results.append(rec)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_tools = len(rec["tool_calls"])
            flag = f"⚠️{rec['http_error']}" if rec["http_error"] else \
                   ("no-detaillog" if rec["detail_log_file"] is None else f"{rec['response_type']}·{n_tools}tools")
            print(f"    → {rec['elapsed_sec']}s · {flag} · tools={rec['tool_call_order']}")
            if args.delay > 0:
                time.sleep(args.delay)

    final = out_dir / f"runs_{stamp}.json"
    final.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[완료] {len(results)}개 실행 → {final}")
    print(f"       (append 로그: {jsonl_path})")
    print(f"\n다음: python -m backend.benchmark.grade --runs {final}")


if __name__ == "__main__":
    main()
