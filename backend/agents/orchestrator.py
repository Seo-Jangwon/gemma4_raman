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

from backend.agents import experience
from backend.agents.graph import build_graph
from backend.agents.state import ClarifiedIntent, ExperimentState, initial_state
from backend.agents.translator import check_intent, translate

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

# ── 세션 저장소 (clarification 대화 + 대화 이력 유지용) ─────────────────────────
# {session_id: {"accumulated": str, "rounds": int, "history": [{"role","text"}, ...]}}
#   - accumulated/rounds : "지금 채우고 있는 한 실험"의 clarification 버퍼.
#     실험이 확정되거나 잡담으로 판명되면 즉시 초기화된다(다음 요청은 새 실험으로 시작).
#   - history            : 세션이 살아있는 동안(=같은 브라우저 탭) 계속 쌓이는 대화 기록.
#     예전엔 accumulated/rounds와 함께 세션 전체를 pop()해 지웠는데, 그러면 실험 하나가
#     끝나거나 잡담 한 번에 응답하는 순간 "이전에 뭐라고 했는지"까지 통째로 사라졌다
#     (예: 실험 완료 후 "내가 방금 뭐라고 했지?"에 답을 못함). 이제 이 필드만 별도로
#     보존해 translate()에 대화 맥락으로 전달한다.
# 로컬 단일 사용자 도구라 in-memory dict로 충분하다(프로세스 종료 시 초기화).
# ※ 경험 저장소(experience_store.json)와 혼동 금지 — 그건 실험 "노하우"의 영속
#    기억이고, 이건 대화 세션 상태라 휘발성이어도 된다.
_SESSIONS: dict[str, dict] = {}

# history에 쌓아두는 최대 메시지 수(사용자+어시스턴트 합산). 무한정 쌓이면 매 translate()
# 호출마다 LLM에 보내는 토큰이 계속 불어나므로 최근 대화만 유지한다.
_HISTORY_MAX_TURNS = 20


def _remember(sess: dict, role: str, text: str) -> None:
    """세션의 대화 이력에 한 턴을 추가한다(길이 제한 적용)."""
    if not text:
        return
    hist = sess.setdefault("history", [])
    hist.append({"role": role, "text": text})
    if len(hist) > _HISTORY_MAX_TURNS:
        del hist[: len(hist) - _HISTORY_MAX_TURNS]


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

    intent를 여기서 사전 번역해 주입한다 — 그래프에는 더 이상 translator 노드가
    없다(진입점이 planner). 번역은 항상 그래프 밖에서 일어난다.
    """
    intent = translate(user_message)
    state = initial_state(user_message=user_message, session_id=session_id,
                          intent=intent)
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
        if delta.get("final_report"):
            # 보고서 생성은 planner에 내장돼 있다 (구 report_generator 흡수)
            return "📝 최종 보고서 작성 완료 → 일관성 검증(C5)"
        plan = delta.get("plan")
        nxt = delta.get("next_node")
        if plan is not None and nxt == "critic" and delta.get("critic_checkpoint") == "C1":
            return f"🧭 계획 수립: 총 {len(plan)}단계 → 안전 검증(C1)"
        if nxt:
            return f"🧭 다음 단계 → {nxt}"
        return "🧭 계획 검토 중"

    if node == "hw_manager":
        # 위치 탐색(locate_target) / 적응형 측정 / 배경 측정을 delta 내용으로 구분
        roi = delta.get("next_roi") or {}
        acq = delta.get("acquisition_params") or {}
        bg = delta.get("background_reference") or {}
        if delta.get("abort_reason"):
            return f"⛔ 측정 중단: {delta['abort_reason']}"
        if acq.get("tuned") is not None and acq:
            hist = acq.get("history") or []
            n = len(hist)
            last = hist[-1] if hist else {}
            return (f"⚙️ 적응형 측정 완료: {acq.get('power_pct')}% / "
                    f"{acq.get('exposure_s')}s (조사 {n}회, max={last.get('max_adu')})")
        if bg.get("summary"):
            return f"⚙️ 기판 배경 측정 완료 (max={bg.get('max_intensity')})"
        if roi.get("x") is not None:
            return f"🔍 타겟 위치 확정: ({roi.get('x')}, {roi.get('y')}) [{roi.get('mode', '')}]"
        return "⚙️ 하드웨어 동작 수행"

    if node == "critic":
        log = delta.get("critic_log") or []
        if log:
            e = log[-1]
            icon = {"APPROVE": "✅", "WARNING": "⚠️", "ABORT": "⛔"}.get(e.get("verdict"), "•")
            return f"{icon} 품질 점검 {e.get('checkpoint')}: {e.get('verdict')} — {e.get('reason','')[:60]}"
        return "🔎 품질 점검"

    if node == "analyst":
        return "📊 스펙트럼 분석 + 도메인 해석 완료"
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
    1. 세션의 accumulated 버퍼에 이번 메시지를 누적한다(이전 턴의 되묻기 답변을 합침).
       세션의 history(지난 턴 전체 기록)는 translate()에 대화 맥락으로 함께 전달한다.
    2. translate(accumulated, history)로 intent 생성 → "intent" 이벤트.
    3. intent.is_experiment_request가 False면 — 잡담/메타 질문이라 실험 파이프라인이
       필요 없다 — "chat" 이벤트로 direct_reply를 바로 돌려주고 종료한다
       (clarify 게이트를 타지 않음 — 실험도 아닌데 "시료가 뭔가요?"라고 되묻는 것을 방지).
    4. clarify.check_intent()로 필수 정보 확인.
       - 부족 & 라운드 여유 있음 → "clarification" 이벤트 후 종료(그래프 실행 안 함).
       - 충족 or 라운드 소진 → accumulated/rounds만 초기화 후 그래프 스트리밍 진행
         (history는 유지 — 다음 대화도 이번 실험 맥락을 참조할 수 있게).
    5. graph.stream(...)의 매 노드 갱신마다 "node" 이벤트.
    6. 종료 시 경험 기록 + 결과를 history에 남긴 후 "done" 이벤트.

    모든 분기(chat/clarification/done)에서 어시스턴트의 응답을 _remember()로 history에
    적립한다 — "방금 뭐라고 답했어?" 같은 다음 턴 메타 질문도 근거를 갖고 답할 수 있도록.
    """
    # ── session_id 확정 (빈 값이면 새로 발급해 클라이언트에 알려줌) ──
    import uuid
    sid = session_id or str(uuid.uuid4())

    def ev(d: dict) -> dict:
        d["session_id"] = sid
        return d

    try:
        # ── 1. 메시지 누적 ──
        sess = _SESSIONS.get(sid) or {"accumulated": "", "rounds": 0, "history": []}
        # translate()에 넘길 "지난 턴들" — 이번 메시지를 history에 추가하기 전 스냅샷.
        # (이번 메시지 자체는 아래 accumulated/user_msg로 별도 전달되므로 중복을 피한다)
        history_for_context = list(sess.get("history", []))

        if sess["accumulated"]:
            # 이전 턴에서 되물었고, 이번 메시지는 그 답변 → 문맥에 덧붙인다.
            sess["accumulated"] = f"{sess['accumulated']}\n[추가 정보] {user_message}"
        else:
            sess["accumulated"] = user_message

        # ── 2. 번역 (대화 이력을 함께 전달) ──
        intent: ClarifiedIntent = translate(sess["accumulated"], history=history_for_context)
        yield ev({"type": "intent", "intent": dict(intent)})

        _remember(sess, "user", user_message)

        # ── 3. 실험 요청이 아니면(잡담/메타 질문) 여기서 바로 응답하고 종료 ──
        # clarify 게이트를 타지 않는다 — "너 뭐 할 수 있어?" 같은 메시지에
        # "시료가 뭔가요?"로 되묻는 것은 안전 목적(sample_type 확인)에 안 맞는다.
        if not intent.get("is_experiment_request", True):
            reply = intent.get("direct_reply") or "라만 실험 측정을 도와드릴 수 있어요. 무엇을 측정하고 싶으신가요?"
            _remember(sess, "assistant", reply)
            # 실험 누적 버퍼만 초기화한다(다음 요청은 새 실험으로 시작) —
            # 대화 이력(history)은 세션이 살아있는 한 계속 보존한다.
            sess["accumulated"] = ""
            sess["rounds"] = 0
            _SESSIONS[sid] = sess
            yield ev({"type": "chat", "reply": reply})
            return

        # ── 4. clarification 게이트 (translator.check_intent — 결정적 규칙) ──
        check = check_intent(intent)
        if not check["ok"] and sess["rounds"] < _MAX_CLARIFY_ROUNDS:
            sess["rounds"] += 1
            _remember(sess, "assistant", check["question"])
            _SESSIONS[sid] = sess            # 다음 턴을 위해 대화 상태 저장
            yield ev({"type": "clarification",
                      "question": check["question"],
                      "missing": check["missing"]})
            return                            # 이번 턴은 여기서 종료(그래프 실행 안 함)

        # 진행 확정 → 실험 누적 버퍼만 정리(대화 이력은 유지). 그래프에 넘길 최종
        # 사용자 메시지는 버퍼를 지우기 전에 따로 챙겨둔다.
        experiment_message = sess["accumulated"]
        sess["accumulated"] = ""
        sess["rounds"] = 0
        _SESSIONS[sid] = sess

        # ── 5. 그래프 스트리밍 실행 ──
        # intent를 사전 주입해 그래프 안 translator는 통과시킨다(중복 LLM 호출 방지).
        state = initial_state(user_message=experiment_message,
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

        report = final_state.get("final_report") or ""
        abort_reason = final_state.get("abort_reason")
        # 다음 턴에 "방금 실험 어떻게 됐어?" 같은 질문에 답할 수 있도록 결과도 이력에 남긴다.
        _remember(sess, "assistant", report or (f"[중단] {abort_reason}" if abort_reason else ""))
        _SESSIONS[sid] = sess

        yield ev({"type": "done",
                  "final_report": report,
                  "abort_reason": abort_reason,
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
