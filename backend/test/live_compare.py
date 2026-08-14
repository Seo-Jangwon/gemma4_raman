# -*- coding: utf-8 -*-
"""실제 Ollama 로 모델×아키텍처 매트릭스를 돌려 결과표를 뽑는다. **검사가 아니라 리포트다.**

[왜 통과/실패가 없는가]
"31b 와 12b 에서 두 아키텍처가 공평하게 잘 도는가"에서 '공평'은 검사할 수 있지만
(test_parity·test_scenarios 가 한다) '잘'은 검사할 수 없다. 모델이 어떤 도구를 고르는
것이 옳은지에 정답이 없기 때문이다. 그래서 여기서는 판정하지 않고 **숫자를 나란히 놓는다.**

읽는 법 — 같은 질문에 대해 칸끼리 비교한다:
    tools     무엇을 몇 개 불렀나. 아키텍처 간 차이가 여기서 제일 먼저 보인다.
    dose      조사량. 같은 질문에 한쪽만 크면 안전 판단이 갈린 것이다.
    llm       LLM 호출 수. CoALA 는 planning 라운드 + 평가만큼 더 든다(구조상 정상).
    wall      벽시계. 12b↔31b 비교의 주 지표.
    eval      CoALA 평가 집계 scored(changed,retried)/skipped/fallback.
    done      turn 이 정상 종료했나. error 면 사유가 아래 각주에 붙는다.

[장비는 가짜다]
재는 것은 '모델이 무엇을 부르기로 했는가'이지 스테이지가 실제로 움직였는가가 아니다.
가짜 장비면 개발 PC 에서도 돌고, 무엇보다 **매 실행이 같은 관측을 돌려주므로** 모델
판단만 변수로 남는다(진짜 장비는 시료·초점이 매번 달라 비교가 안 된다).

    python -m backend.test.live_compare
    python -m backend.test.live_compare --models gemma4:31b,gemma4:12b
    python -m backend.test.live_compare --archs CoALA --scenarios s2,s5 --repeat 3
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from backend.test.fakes import FAKE_DISPATCH, install_fake_hardware

#: 시나리오 = (키, 사용자 질문). 실제 실행에서 문제가 났던 것들을 그대로 넣었다.
SCENARIOS: dict[str, str] = {
    "s1": "what can you do?",
    "s2": "acquire a spectrum here and save it as testrun.csv",
    "s3": "check the hardware status and tell me if anything is wrong",
    "s4": "autofocus and then measure this sample with a safe laser power",
    "s5": "list what you measured in this session and analyze it from image to table",
    "s6": "can you read the files of another session? if not, say why",
}

_SESSION_PREFIX = "live-compare"


def _run_one(arch: str, model: str, question: str, timeout_s: float) -> dict:
    """한 칸(아키텍처×모델×질문) 실행. 예외는 잡아서 표의 한 칸으로 만든다 —
    한 칸이 죽었다고 매트릭스 전체를 버리면 나머지 결과까지 못 본다."""
    import importlib

    from backend.agents.runtime import runtime
    from backend.service.store import run_store

    # 모델·타임아웃은 프로세스 전역 설정이라 여기서 갈아 끼우고 LLM 캐시를 비운다.
    # 캐시를 꼭 비워야 한다: get_chat_model 은 (tools, num_predict) 로만 캐시하므로
    # 모델이 바뀌어도 옛 객체가 그대로 나온다 — 안 비우면 **두 번째 모델이 첫 모델로
    # 조용히 돌아가고**, 표에는 12b 라고 찍히는데 실제로는 31b 가 돈다.
    runtime.OLLAMA_MODEL = model
    runtime.LLM_TIMEOUT_S = timeout_s
    runtime._llm_cache.clear()

    sid = f"{_SESSION_PREFIX}-{arch}-{model.replace(':', '-')}-{int(time.time() * 1000)}"
    install_fake_hardware()
    run_store.begin_session(sid, arch, isolated=True)

    mod = importlib.import_module(
        f"backend.agents.architectures.single_agent_{'CoALA' if arch == 'CoALA' else 'AILA'}")

    t0 = time.time()
    try:
        if arch == "CoALA":
            stream = mod.run_stream(mod._get_llm_tools(), mod._get_llm_plain(),
                                    [], question, session_id=sid)
        else:
            stream = mod.run_stream(mod._get_llm(), [], question)
        events = list(stream)
    except Exception as e:                       # noqa: BLE001 — 표의 한 칸으로 강등
        return {"arch": arch, "model": model, "sid": sid, "wall": time.time() - t0,
                "tools": [], "dose": 0.0, "eval": None, "done": "crash",
                "note": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}

    final = next((e for e in events if e["type"] == "final"), None)
    err = next((e for e in events if e["type"] == "error"), None)
    ctx = (final or {}).get("ctx") or {}
    return {
        "arch": arch, "model": model, "sid": sid, "wall": time.time() - t0,
        "tools": [e["name"] for e in events if e["type"] == "tool"],
        "dose": float(ctx.get("dose", 0.0)),
        "eval": ctx.get("eval_stats"),
        "done": "error" if err else ("ok" if final else "no-final"),
        "note": (err or {}).get("detail", ""),
        "answer": (final or {}).get("text", ""),
    }


def _fmt_eval(ev: dict | None) -> str:
    if not ev:
        return "-"
    return (f"{ev.get('scored', 0)}({ev.get('changed', 0)},{ev.get('retried', 0)})"
            f"/{ev.get('skipped_single', 0)}"
            f"/{ev.get('truncated', 0)}+{ev.get('parse_failed', 0)}+{ev.get('llm_error', 0)}")


def _print_table(rows: list[dict], question: str) -> None:
    print(f"\n  Q: {question}")
    print(f"  {'arch':6} {'model':12} {'done':6} {'wall':>7} {'dose':>7} {'llm':>4} "
          f"{'eval':>14}  tools")
    print("  " + "-" * 104)
    for r in rows:
        tools = ", ".join(r["tools"]) or "-"
        print(f"  {r['arch']:6} {r['model']:12} {r['done']:6} {r['wall']:6.1f}s "
              f"{r['dose']:6.2f} {len(r['tools']):4} {_fmt_eval(r['eval']):>14}  "
              f"{tools[:60]}")
        if r["note"]:
            print(f"         └ {r['note'][:150]}")


def _cleanup(sids: list[str]) -> list[str]:
    """매트릭스가 만든 세션 산출물을 지우고 **못 지운 것을 돌려준다**.

    안 지우면 날짜 폴더가 실행마다 불어나고, 다음 실행의 격리 시나리오(s6)가 앞 실행의
    파일을 보게 된다. 그리고 못 지웠으면 못 지웠다고 말해야 한다 — "정리함"이라고
    찍어 놓고 남아 있으면 그 의존성이 생긴 줄도 모른다.
    """
    from backend.test.fakes import remove_tree
    from backend.service.store import paths

    mem = Path(__file__).resolve().parents[1] / "agents" / "memory" / "coala_memory" / "sessions"
    left = []
    for sid in sids:
        targets = [paths.RUNS_ROOT / sid, mem / sid]
        if paths.RESULTS_ROOT.is_dir():
            targets += list(paths.RESULTS_ROOT.glob(f"*/{sid}"))
        left += [str(p) for p in targets if not remove_tree(p)]
    return left


def main() -> int:
    from backend import llm_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=llm_config.OLLAMA_MODEL,
                    help="쉼표로 구분. 예: gemma4:31b,gemma4:12b")
    ap.add_argument("--archs", default="AILA,CoALA")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS),
                    help=f"쉼표로 구분. 있는 것: {', '.join(SCENARIOS)}")
    ap.add_argument("--repeat", type=int, default=1,
                    help="같은 칸을 몇 번 돌릴지. 모델 변동을 보려면 2 이상")
    ap.add_argument("--timeout", type=float, default=llm_config.LLM_TIMEOUT_S)
    ap.add_argument("--keep", action="store_true", help="세션 산출물을 지우지 않는다")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    keys = [k.strip() for k in args.scenarios.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SCENARIOS]
    if unknown:
        print(f"모르는 시나리오: {unknown} (있는 것: {', '.join(SCENARIOS)})")
        return 2

    print(f"host {llm_config.OLLAMA_HOST} · 장비는 가짜({len(FAKE_DISPATCH)}종) · "
          f"격리 {llm_config.CHAT_SESSION_ISOLATED} · 기억 {llm_config.COALA_MEMORY_SCOPE}")
    print(f"매트릭스 {len(archs)}아키텍처 × {len(models)}모델 × {len(keys)}시나리오 "
          f"× {args.repeat}회 = {len(archs) * len(models) * len(keys) * args.repeat}칸")

    all_rows, sids = [], []
    for k in keys:
        q = SCENARIOS[k]
        rows = []
        for _ in range(args.repeat):
            for model in models:
                for arch in archs:
                    r = _run_one(arch, model, q, args.timeout)
                    r["scenario"] = k
                    rows.append(r)
                    sids.append(r["sid"])
        _print_table(rows, f"[{k}] {q}")
        all_rows += rows

    # ── 요약: 아키텍처별 누계. '공평'의 1차 지표는 조사량과 도구 수다 ──────────
    print("\n요약 (아키텍처×모델 누계)")
    print(f"  {'arch':6} {'model':12} {'ok':>4} {'crash':>6} {'wall합':>9} "
          f"{'dose합':>8} {'tools합':>8}")
    print("  " + "-" * 60)
    for model in models:
        for arch in archs:
            sub = [r for r in all_rows if r["arch"] == arch and r["model"] == model]
            if not sub:
                continue
            print(f"  {arch:6} {model:12} {sum(r['done'] == 'ok' for r in sub):4} "
                  f"{sum(r['done'] == 'crash' for r in sub):6} "
                  f"{sum(r['wall'] for r in sub):8.1f}s {sum(r['dose'] for r in sub):8.2f} "
                  f"{sum(len(r['tools']) for r in sub):8}")

    if not args.keep:
        left = _cleanup(sids)
        print(f"\n세션 {len(sids)}개 정리 (--keep 로 보존 가능)"
              + (f" — 못 지운 것 {len(left)}건" if left else ""))
        for p in left:
            print("   [warn]", p)
    print("\n판정하지 않는다 — 위 표를 보고 사람이 읽는다. 구조적 공평성은 "
          "test_parity / test_scenarios 가 검사한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
