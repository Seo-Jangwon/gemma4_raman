"""
ExperimentOrchestrator — LangGraph 그래프 실행 entry point.

두 가지 실행 방식을 제공한다:
  - run_experiment()     : 동기 1회 실행 (전체 결과를 한 번에 반환). 벤치마크/레거시용.
  - stream_experiment()  : 제너레이터. clarification(되묻기) + 진행상황 스트리밍용.
                           프론트엔드의 실시간 UI(SSE)가 이걸 쓴다.

서버 시작 시 build_graph()를 1회 compile해 _graph에 저장한다.
"""

from __future__ import annotations

from typing import Iterator, Optional

from backend.agents import clarify, experience
from backend.agents.graph import build_graph
from backend.agents.state import ClarifiedIntent, ExperimentState, initial_state
from backend.agents.translator import translate

_graph = None

# LangGraph 기본 recursion_limit은 25 (노드 방문 횟수 기준).
# Hub-and-spoke 구조는 매 step마다 Planner를 왕복하고(2배), 여기에
# C3 품질 게이트(step당 +2), 실패 재시도(+2~4), 재계획(+수 회)이 더해지면
# 정상 실험도 25회를 쉽게 넘는다 → 기본값이면 실험 막바지에
# GraphRecursionError로 크래시. 6-step 계획 + 재시도 여유를 계산해 150으로 설정.
# (무한 루프 안전망 역할은 유지 — retry_map/replan_count가 1차 방어이고
#  이 한도는 로직 버그에 대비한 최후의 회로 차단기다)
_RECURSION_LIMIT = 150

# clarification 최대 라운드. 이 횟수만큼 되물어도 정보가 안 채워지면
# 게이트를 무시하고 가진 정보로 진행한다(무한 되묻기 방지).
# 근거: hw_manager 적응형 획득이 5% 저출력 프로브부터 시작하므로 미지 시료라도
# 광손상 위험이 최소화된다 → "끝내 모르면 조심스럽게 진행"이 안전한 최종 방어선.
_MAX_CLARIFY_ROUNDS = 2

# ── 세션 저장소 (clarification 대화 유지용) ────────────────────────────────────
# {session_id: {"accumulated": str, "rounds": int}}
# clarification은 여러 턴에 걸쳐 정보를 모은다. 사용자가 앞서 한 말과 방금 준 답을
# 합쳐서 다시 번역해야 하므로, 누적 메시지를 세션별로 서버에 들고 있는다.
# 로컬 단일 사용자 도구라 in-memory dict로 충분하다(프로세스 종료 시 초기화).
# ※ 경험 저장소(experience_store.json)와 혼동 금지 — 그건 실험 "노하우"의 영속
#    기억이고, 이건 진행 중인 한 요청의 "대화 상태"라 휘발성이어도 된다.
_SESSIONS: dict[str, dict] = {}


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ══════════════════════════════════════════════════════════════════════════════
# 동기 실행 (레거시/벤치마크)
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(user_message: str, session_id: str = "") -> dict:
    """
    멀티에이전트 실험 파이프라인을 동기로 1회 실행.
    (clarification 없이 곧바로 실행 — 벤치마크는 완전한 명령을 주기 때문)
    반환값: ExperimentState 최종 상태 dict
    """
    state = initial_state(user_message=user_message, session_id=session_id)
    graph = _get_graph()
    result: ExperimentState = graph.invoke(state, config={"recursion_limit": _RECURSION_LIMIT})

    # ── 경험 기록 (H-EPM 하이브리드 기억) ────────────────────────────────────
    # 기록 지점을 여기(invoke 직후) 단일 지점으로 둔 이유:
    # 성공/실패/중단(abort) 등 "모든" 종료 경로가 이 지점을 지난다.
    # report_generator 안에서 기록하면 abort된 실험의 실패 지식이 누락된다.
    # 기록 실패가 실험 결과 반환을 막으면 안 되므로 best-effort로 감싼다.
    try:
        experience.record_experiment(dict(result))
    except Exception:
        pass

    return dict(result)


# ══════════════════════════════════════════════════════════════════════════════
# 스트리밍 실행 (clarification + 진행상황) — 프론트엔드 SSE용
# ══════════════════════════════════════════════════════════════════════════════

def _summarize_node(node: str, delta: dict, state: dict) -> str:
    """한 노드가 만든 상태 변화(delta)를 사람이 읽는 한 줄 진행 메시지로 요약한다.

    delta에는 "그 노드가 반환한 필드"만 담긴다. 노드별로 의미 있는 필드를 골라
    사용자가 '지금 무슨 일이 일어나는지' 파악할 수 있게 한다.
    (프론트에서 그대로 채팅 진행 로그로 뿌린다.)
    """
    delta = delta or {}

    if node == "planner":
        # 재계획인지, 다음 어디로 가는지, 계획이 몇 단계인지
        if delta.get("abort_reason"):
            return f"⛔ 계획 중단: {delta['abort_reason']}"
        plan = delta.get("plan")
        nxt = delta.get("next_node")
        if plan is not None and nxt == "":
            return f"🧭 계획 수립: 총 {len(plan)}단계"
        if nxt == "report_generator":
            return "📝 결과 보고서 생성 준비"
        if nxt:
            return f"🧭 다음 단계 → {nxt}"
        return "🧭 계획 검토 중"

    if node == "roi_detector":
        roi = delta.get("next_roi") or {}
        mode = roi.get("mode", "")
        if roi.get("x") is not None:
            return f"🔍 타겟 위치 확정: ({roi.get('x')}, {roi.get('y')}) [{mode}]"
        return "🔍 타겟 위치 탐색 중"

    if node == "hw_manager":
        acq = delta.get("acquisition_params") or {}
        bg = delta.get("background_reference") or {}
        if acq.get("tuned"):
            hist = acq.get("history") or []
            n = len(hist)
            last = hist[-1] if hist else {}
            return (f"⚙️ 적응형 측정 완료: {acq.get('power_pct')}% / "
                    f"{acq.get('exposure_s')}s (조사 {n}회, max={last.get('max_adu')})")
        if bg.get("summary"):
            return f"⚙️ 기판 배경 측정 완료 (max={bg.get('max_intensity')})"
        if delta.get("abort_reason"):
            return f"⛔ 측정 중단: {delta['abort_reason']}"
        return "⚙️ 하드웨어 동작 수행"

    if node == "critic":
        log = delta.get("critic_log") or []
        if log:
            e = log[-1]
            icon = {"APPROVE": "✅", "WARNING": "⚠️", "ABORT": "⛔"}.get(e.get("verdict"), "•")
            return f"{icon} 품질 점검 {e.get('checkpoint')}: {e.get('verdict')} — {e.get('reason','')[:60]}"
        return "🔎 품질 점검"

    if node == "spectrum_specialist":
        return "📊 스펙트럼 분석 완료"
    if node == "domain_specialist":
        return "🧬 도메인 해석 완료"
    if node == "rag_searcher":
        return "📚 참고문헌 검색 완료"
    if node == "debate":
        return "🗣️ 교차검증(debate) 완료"
    if node == "report_generator":
        return "📝 최종 보고서 작성 완료"
    if node == "translator":
        return "🈯 요청 해석 완료"
    return f"• {node}"


def stream_experiment(user_message: str, session_id: str = "") -> Iterator[dict]:
    """
    멀티에이전트 파이프라인을 이벤트 제너레이터로 실행한다.

    yield하는 이벤트(dict) 종류 — 모두 "type"과 "session_id"를 포함:
      {"type": "intent",        "intent": {...}}                해석된 의도
      {"type": "chat",          "reply": str}                   실험 요청이 아닌 메시지 응답(이번 턴 종료)
      {"type": "clarification", "question": str, "missing": [...]}  되묻기(이번 턴 종료)
      {"type": "node",          "node": str, "message": str}    진행상황 한 줄
      {"type": "done",          "final_report": str, "state": {...}}  완료
      {"type": "error",         "detail": str}                  오류

    [흐름]
    1. 세션에 이번 메시지를 누적한다(이전 턴의 되묻기 답변을 합침).
    2. translate()로 intent 생성 → "intent" 이벤트.
    3. intent.is_experiment_request가 False면 — 잡담/메타 질문이라 실험 파이프라인이
       필요 없다 — "chat" 이벤트로 direct_reply를 바로 돌려주고 종료한다
       (clarify 게이트를 타지 않음 — 실험도 아닌데 "시료가 뭔가요?"라고 되묻는 것을 방지).
    4. clarify.check_intent()로 필수 정보 확인.
       - 부족 & 라운드 여유 있음 → "clarification" 이벤트 후 종료(그래프 실행 안 함).
       - 충족 or 라운드 소진 → 세션 정리 후 그래프 스트리밍 진행.
    5. graph.stream(...)의 매 노드 갱신마다 "node" 이벤트.
    6. 종료 시 경험 기록 후 "done" 이벤트.
    """
    # ── session_id 확정 (빈 값이면 새로 발급해 클라이언트에 알려줌) ──
    import uuid
    sid = session_id or str(uuid.uuid4())

    def ev(d: dict) -> dict:
        d["session_id"] = sid
        return d

    try:
        # ── 1. 메시지 누적 ──
        sess = _SESSIONS.get(sid) or {"accumulated": "", "rounds": 0}
        if sess["accumulated"]:
            # 이전 턴에서 되물었고, 이번 메시지는 그 답변 → 문맥에 덧붙인다.
            sess["accumulated"] = f"{sess['accumulated']}\n[추가 정보] {user_message}"
        else:
            sess["accumulated"] = user_message

        # ── 2. 번역 ──
        intent: ClarifiedIntent = translate(sess["accumulated"])
        yield ev({"type": "intent", "intent": dict(intent)})

        # ── 3. 실험 요청이 아니면(잡담/메타 질문) 여기서 바로 응답하고 종료 ──
        # clarify 게이트를 타지 않는다 — "너 뭐 할 수 있어?" 같은 메시지에
        # "시료가 뭔가요?"로 되묻는 것은 안전 목적(sample_type 확인)에 안 맞는다.
        if not intent.get("is_experiment_request", True):
            _SESSIONS.pop(sid, None)
            yield ev({"type": "chat",
                      "reply": intent.get("direct_reply") or "라만 실험 측정을 도와드릴 수 있어요. 무엇을 측정하고 싶으신가요?"})
            return

        # ── 4. clarification 게이트 ──
        check = clarify.check_intent(intent)
        if not check["ok"] and sess["rounds"] < _MAX_CLARIFY_ROUNDS:
            sess["rounds"] += 1
            _SESSIONS[sid] = sess            # 다음 턴을 위해 대화 상태 저장
            yield ev({"type": "clarification",
                      "question": check["question"],
                      "missing": check["missing"]})
            return                            # 이번 턴은 여기서 종료(그래프 실행 안 함)

        # 진행 확정 → 세션의 대화 상태는 정리(다음 요청은 새 실험으로 시작)
        _SESSIONS.pop(sid, None)

        # ── 5. 그래프 스트리밍 실행 ──
        # intent를 사전 주입해 그래프 안 translator는 통과시킨다(중복 LLM 호출 방지).
        state = initial_state(user_message=sess["accumulated"],
                              session_id=sid, intent=intent)
        graph = _get_graph()

        # stream_mode=["updates","values"]:
        #   - "updates": 방금 실행된 노드명 + 그 노드의 반환 delta → 진행 메시지용
        #   - "values" : 매 스텝 후의 "전체" 상태 스냅샷 → 마지막 것이 최종 상태(경험 기록용)
        # 두 모드를 함께 켜면 (mode, payload) 튜플로 번갈아 들어온다.
        final_state: dict = dict(state)
        for mode, payload in graph.stream(
            state,
            config={"recursion_limit": _RECURSION_LIMIT},
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                final_state = payload
            elif mode == "updates":
                # updates payload: {node_name: delta_dict} (보통 노드 1개)
                for node_name, delta in (payload or {}).items():
                    yield ev({"type": "node",
                              "node": node_name,
                              "message": _summarize_node(node_name, delta, final_state)})

        # ── 6. 경험 기록 (성공/실패/중단 모든 종료 경로가 여기를 지남) ──
        try:
            experience.record_experiment(dict(final_state))
        except Exception:
            pass

        yield ev({"type": "done",
                  "final_report": final_state.get("final_report") or "",
                  "abort_reason": final_state.get("abort_reason"),
                  "state": _public_state(final_state)})

    except Exception as e:
        # 예외가 나도 스트림은 error 이벤트로 정상 종료시킨다(프론트가 매달리지 않게).
        yield ev({"type": "error", "detail": str(e)})


def _public_state(state: dict) -> dict:
    """프론트로 돌려줄 최종 상태의 요약(무겁거나 내부용인 필드는 제외)."""
    return {
        "intent": state.get("intent"),
        "plan": state.get("plan"),
        "spectrum_analysis": state.get("spectrum_analysis"),
        "domain_interpretation": state.get("domain_interpretation"),
        "acquisition_params": state.get("acquisition_params"),
        "background_reference": state.get("background_reference"),
        "next_roi": state.get("next_roi"),
        "abort_reason": state.get("abort_reason"),
    }
