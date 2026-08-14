# -*- coding: utf-8 -*-
"""행동 공평성 — 같은 대본을 두 루프에 먹여 반응을 대조한다. (Ollama·장비 불필요)

test_parity.py 는 '두 아키텍처가 같은 조건을 받았는가'를 본다. 여기서는 그 조건을 받고
**실제로 같은 일을 하는가**를 본다. 진짜 LLM 은 같은 질문에도 매번 다르게 답하므로 그
비교가 성립하지 않는다. 대본대로만 답하는 LLM 을 쓰면 모델의 변덕이 사라져 아키텍처
차이만 남는다(fakes.FakeLLM).

각 시나리오는 두 아키텍처에서 **무엇이 같아야 하고 무엇이 달라도 되는가**를 함께 적는다.
같아야 할 것: 실행한 도구와 순서, 조사량, 실패 처리, 세션 경계, 빈 응답 처리.
달라도 될 것: CoALA 의 평가 단계 개입, 후보가 여럿일 때의 선택.

    python -m backend.test.test_scenarios
"""
from __future__ import annotations

from pathlib import Path

from backend.test.fakes import drive, remove_tree

ARCHS = ("AILA", "CoALA")

#: 시나리오가 만든 산출물이 떨어지는 자리. 검사 끝에 통째로 지운다.
_SESSION = "test-scenarios"


def _both(script: list, question: str = "measure this sample",
          eval_script: list | None = None, **kw) -> dict:
    """같은 대본을 두 아키텍처에 먹이고 결과를 {arch: result} 로."""
    return {a: drive(a, list(script), question, list(eval_script or []),
                     session_id=_SESSION, **kw) for a in ARCHS}


# ══════════════════════════════════════════════════════════════════════════════
# 시나리오
# ══════════════════════════════════════════════════════════════════════════════

def s1_chat_only() -> list[str]:
    """도구 없이 대화만. 둘 다 도구 0개로 끝나고 같은 답을 낸다.

    '무엇을 할 수 있나요' 같은 질문에 한쪽만 장비를 건드리면 그건 아키텍처 차이가 아니라
    사고다 — 조사량이 붙고, 로그 성격이 done/chat 으로 갈린다.
    """
    r = _both(["I can operate the spectrometer end to end."], question="what can you do?")
    bad = []
    for a in ARCHS:
        if r[a]["tools"]:
            bad.append(f"{a}: 대화 질문에 도구를 실행했다 {r[a]['tools']}")
        if r[a]["dose"] != 0.0:
            bad.append(f"{a}: 대화 턴인데 조사량이 붙었다 {r[a]['dose']}")
    if r["AILA"]["final"] != r["CoALA"]["final"]:
        bad.append("같은 대본인데 최종 답변이 다르다")
    return bad


def s2_single_measurement() -> list[str]:
    """정렬 → 측정 → 보고. 둘 다 같은 도구를 같은 순서로 실행하고 조사량도 같다.

    조사량은 파워·노출·발수로만 정해지므로(safety_limits), 여기서 갈라지면 두 에이전트가
    같은 측정을 하고도 다른 안전 예산을 쓴다는 뜻이다.
    """
    script = [
        [("start_camera_stream", {})],
        [("analyze_microscope_image", {"question": "what is here?"})],
        [("acquire_spectrum", {"power": 20, "exposure": 2})],
        "Measured. Max 4411 ADU.",
    ]
    r = _both(script)
    bad = []
    want = ["start_camera_stream", "analyze_microscope_image", "acquire_spectrum"]
    for a in ARCHS:
        if r[a]["tools"] != want:
            bad.append(f"{a}: 도구 순서가 {r[a]['tools']} (기대 {want})")
    if r["AILA"]["dose"] != r["CoALA"]["dose"]:
        bad.append(f"조사량이 다르다 AILA={r['AILA']['dose']} CoALA={r['CoALA']['dose']}")
    if r["AILA"]["dose"] <= 0:
        bad.append("측정했는데 조사량이 0 — 회로차단기가 셀 수 없는 상태다")
    return bad


def s3_tool_failure() -> list[str]:
    """도구가 실패해도 루프가 죽지 않고 다음 수를 둔다.

    가짜 장비의 run_autofocus 는 항상 실패한다. 실패를 관측으로 받아 계속 갈 수 있어야
    하고, 실패가 예외로 새어 턴을 죽이면 안 된다 — 회복 능력은 두 아키텍처 모두의 전제다.
    """
    script = [
        [("run_autofocus", {})],
        [("acquire_spectrum", {"power": 5, "exposure": 1})],
        "Autofocus failed; measured at the current focus instead.",
    ]
    r = _both(script)
    bad = []
    for a in ARCHS:
        if r[a]["error"]:
            bad.append(f"{a}: 도구 실패가 턴을 죽였다 — {r[a]['error']}")
        if r[a]["tools"] != ["run_autofocus", "acquire_spectrum"]:
            bad.append(f"{a}: 실패 뒤 진행이 끊겼다 {r[a]['tools']}")
        failed = [e for e in r[a]["events"]
                  if e["type"] == "tool" and not e["result"].get("ok", True)]
        if not failed:
            bad.append(f"{a}: 실패한 도구 결과가 이벤트에 안 실렸다(모델이 못 본다)")
    return bad


def s4_empty_reply() -> list[str]:
    """모델이 도구도 텍스트도 안 내면 둘 다 같은 안내 문구로 턴을 닫는다.

    빈 응답은 num_ctx 초과 때 실제로 나온다. 한쪽만 빈 문자열을 사용자에게 그대로
    돌려주면 '아무 일도 안 일어난 것처럼' 보인다.
    """
    from backend.agents.runtime import runtime
    r = _both([""])
    bad = [f"{a}: 빈 응답을 안내 문구로 안 바꿨다 ({r[a]['final']!r})"
           for a in ARCHS if r[a]["final"] != runtime.EMPTY_REPLY]
    if r["AILA"]["final"] != r["CoALA"]["final"]:
        bad.append("빈 응답 처리 문구가 두 아키텍처에서 다르다")
    return bad


def s5_session_isolation() -> list[str]:
    """둘 다 남의 세션 파일을 못 연다.

    격리는 paths._accept 한 곳이 판정하지만, 그 판정이 걸리려면 두 루프가 각자
    begin_session 을 지나야 한다. 한쪽이 세션을 안 열면 그 아키텍처만 남의 데이터를 본다.
    """
    from backend.service.store import run_store, paths

    victim = "test-scenarios-other"
    d = paths.RESULTS_ROOT / "1999-01-01" / victim
    d.mkdir(parents=True, exist_ok=True)
    (d / "secret.csv").write_text("x\n1\n", encoding="utf-8")
    fid = f"results:1999-01-01/{victim}/secret.csv"
    try:
        script = [[("open_file", {"file_id": fid})], "done."]
        r = _both(script)
        bad = []
        for a in ARCHS:
            if run_store.isolated_label() is None:
                bad.append(f"{a}: 턴이 끝나고 격리 라벨이 없다(세션을 안 열었다)")
            opened = [e for e in r[a]["events"] if e["type"] == "tool"]
            if not opened:
                bad.append(f"{a}: open_file 이 아예 실행되지 않았다")
            elif opened[0]["result"].get("ok"):
                bad.append(f"{a}: **남의 세션 파일을 읽었다** — 격리가 안 걸렸다")
        return bad
    finally:
        remove_tree(paths.RESULTS_ROOT / "1999-01-01")


def s6_multi_candidate() -> list[str]:
    """후보를 여럿 냈을 때 — **여기서만 두 아키텍처가 갈라져야 한다.**

    ReAct 의 정의는 'emit 된 도구를 전부, 그 순서대로'이고 CoALA 의 정의는 '평가해서
    하나만'이다. 그것이 이 실험의 독립변수 자체이므로, 여기서 같아지면 오히려 실패다 —
    두 아키텍처를 구현했는데 실제로는 한 아키텍처가 두 벌 있는 셈이 된다.
    """
    script = [[("acquire_spectrum", {"power": 1, "exposure": 1}),
               ("acquire_spectrum", {"power": 50, "exposure": 1})],
              "done."]
    r = _both(script, eval_script=['{"scores": [0.9, 0.1], "reason": "start low"}'])
    bad = []
    if len(r["AILA"]["tools"]) != 2:
        bad.append(f"AILA 가 emit 된 도구를 전부 실행하지 않았다 {r['AILA']['tools']} "
                   f"— ReAct 의 정의가 깨졌다")
    if len(r["CoALA"]["tools"]) != 1:
        bad.append(f"CoALA 가 사이클당 하나만 실행하지 않았다 {r['CoALA']['tools']} "
                   f"— 비가역 조작(레이저)의 전제가 깨졌다")
    ev = r["CoALA"]["eval_stats"] or {}
    if ev.get("scored") != 1:
        bad.append(f"CoALA 평가 단계가 개입하지 않았다 {ev}")
    # 평가가 고른 것이 실행됐는가. 점수 [0.9, 0.1] 이면 0번(power=1)이다.
    chosen = [e for e in r["CoALA"]["events"] if e["type"] == "tool"]
    if chosen and chosen[0]["args"].get("power") != 1:
        bad.append("CoALA 가 평가 결과와 다른 후보를 실행했다")
    # 다중 후보에서 AILA 는 조사량이 더 크다 — 그건 정상이고, 오히려 같으면 이상하다.
    if r["AILA"]["dose"] <= r["CoALA"]["dose"]:
        bad.append("두 개를 쏜 AILA 의 조사량이 하나를 쏜 CoALA 보다 크지 않다")
    return bad


def s7_thinking_symmetry() -> list[str]:
    """생성된 thinking 을 히스토리에 남기는 방식이 두 아키텍처에서 같은가.

    [왜 보는가 — 2026-08-13]
    langchain-ollama 는 AIMessage.additional_kwargs["reasoning_content"] 가 있으면 그것을
    다음 요청의 thinking 필드로 **되돌려 보낸다.** 예전에는 AILA 만 모델이 준 AIMessage 를
    원본 그대로 히스토리에 넣고 CoALA 는 재구성해서 넣었다(후보 여럿 중 하나만 실행하므로
    원본을 그대로 쓰면 짝 없는 tool_call 이 남는다). 그래서 **AILA 만 자기 이전 추론을
    다시 보는** 상태였다. 그러면 성능 차이가 아키텍처 때문인지 그것 때문인지 구분이 안 된다.

    어느 쪽으로 맞추든 상관없다 — 같기만 하면 된다. 그래서 '몇 개에 실렸는가'를 대조한다.
    """
    from langchain_core.messages import AIMessage
    from langchain_ollama import ChatOllama

    conv = ChatOllama(model="_parity", base_url="http://127.0.0.1:1") \
        ._convert_messages_to_ollama_messages

    script = [[("acquire_spectrum", {"power": 1, "exposure": 1})], "done."]
    r = _both(script, carry_thinking=True)

    carried = {}
    for a in ARCHS:
        ai = [m for m in r[a]["messages"] if isinstance(m, AIMessage)]
        out = conv(ai) if ai else []
        carried[a] = sum(1 for o in out if (o.get("thinking") or ""))

    if carried["AILA"] != carried["CoALA"]:
        return [f"thinking 재전송이 비대칭이다 — AILA {carried['AILA']}개 / "
                f"CoALA {carried['CoALA']}개의 assistant 메시지가 이전 추론을 지고 간다. "
                f"한쪽만 자기 추론을 다시 보면 아키텍처 비교가 무너진다."]
    return []


# ══════════════════════════════════════════════════════════════════════════════

_SCENARIOS = [
    ("S1 대화만 (도구 0)", s1_chat_only),
    ("S2 단일 측정 (도구·조사량 동일)", s2_single_measurement),
    ("S3 도구 실패 후 진행", s3_tool_failure),
    ("S4 빈 응답 처리", s4_empty_reply),
    ("S5 세션 격리", s5_session_isolation),
    ("S6 다중 후보 (여기서만 갈라져야 함)", s6_multi_candidate),
    ("S7 thinking 재전송 대칭", s7_thinking_symmetry),
]


def main() -> int:
    from backend.service.store import paths

    problems = []
    try:
        for tag, fn in _SCENARIOS:
            found = fn()
            print(f"  {'FAIL' if found else 'ok  '}  {tag}"
                  + (f"  ({len(found)}건)" if found else ""))
            problems += [f"[{tag}] {p}" for p in found]
    finally:
        # 시나리오가 만든 세션 산출물은 남기지 않는다 — 다음 실행이 앞 실행의 파일을
        # 보게 되면 검사가 실행 순서에 의존한다(격리 검사가 그 순간 무의미해진다).
        # 남으면 남았다고 말한다. 조용히 넘어가면 그 의존성이 생긴 줄도 모른다.
        targets = [paths.RUNS_ROOT / _SESSION,
                   Path(__file__).resolve().parents[1] / "agents" / "memory"
                   / "coala_memory" / "sessions" / _SESSION]
        if paths.RESULTS_ROOT.is_dir():
            targets += list(paths.RESULTS_ROOT.glob("*/" + _SESSION))
        left = [str(p) for p in targets if not remove_tree(p)]
        if left:
            print(f"  [warn] 정리 못 한 산출물 {len(left)}건 — 다음 실행 전에 지울 것:")
            for p in left:
                print("         ", p)

    if problems:
        print(f"\n실패 {len(problems)}건:")
        for p in problems:
            print("   ·", p)
        return 1
    print(f"\n통과: 시나리오 {len(_SCENARIOS)}개 · 두 아키텍처가 같은 대본에 같이 반응하고, "
          f"다중 후보에서만 설계대로 갈라진다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
