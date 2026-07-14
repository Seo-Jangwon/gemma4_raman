"""
PlannerNode — Hub 에이전트. 모든 스포크를 조율하고 실행 순서를 결정.
report_generator_node도 여기에 함께 정의.

LLM: Claude claude-opus-4-7 (파일 상단 _llm 교체로 변경 가능)

[전체 실행 흐름]
1. 첫 진입: intent 확인 → RAG 검색 → (경험 저장소 참고) LLM plan 생성 → C1 검증
2. C1 통과: plan을 step 순서대로 실행. 각 step 완료/실패 후 항상 Planner로 복귀.
3. 스펙트럼 획득 step(acquire_*)이 done 되면 → C3 품질 게이트 →
   WARNING이면 Critic의 suggestion(보정 계수)으로 파라미터를 고쳐 재실행 (한도 내).
4. step 실패 시 → step의 on_fail 정책 적용:
     retry  — 같은 step 재실행 (일시적 하드웨어 오류 대응, step당 최대 2회)
     replan — 완료된 step은 보존하고 남은 계획만 LLM으로 재작성 (최대 3회)
     skip   — step을 skipped로 표시하고 다음으로 (보조 분석 실패 등)
     abort  — 즉시 실험 중단
5. 모든 step 완료 → C4(해석 검증) → report_generator → C5(보고서 검증) → 종료.
   종료 시 이번 실험의 성공 조건을 experience store에 기록 → 다음 실험에 재사용.

[설계 결정 요약]
- 실패 판단/정책 실행은 전부 결정적 코드: LLM은 "새 계획 내용"을 만들 때만 개입.
  무한 루프 방지 근거(retry_map, replan_count)가 LLM 손에 있으면 안 되기 때문.
- C3 게이트는 Planner가 관리: hw_manager가 스스로 idx를 전진시키면 품질 게이트를
  끼워 넣을 수 없으므로, 획득 step의 전진 권한을 Planner가 가진다.
- 기존 버그 수정: report_generator가 next_node="critic"(C5)을 설정해도 그래프
  edge가 무조건 Planner로 복귀시켜 planner가 이를 덮어썼고, C5로 가는 경로가
  없어 보고서가 무한 재생성됐다(recursion limit 크래시). 이제 Planner가
  "final_report 있음 + C5 기록 없음"을 감지해 직접 C5로 라우팅한다.
"""

from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents import experience
from backend.agents.state import ExperimentState, PlanStep

# ── LLM 설정 (교체 포인트) ────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-opus-4-8", temperature=0)

# ── 한도 상수 ─────────────────────────────────────────────────────────────────
_MAX_STEP_RETRIES = 2   # step당 재시도 한도 — 같은 실패를 3번 이상 반복하지 않는다
_MAX_REPLANS = 3        # 실험당 재계획 한도 — LLM이 계획을 무한히 다시 짜는 것 방지

# step agent별 기본 실패 정책 — 계획 LLM이 on_fail을 빠뜨려도 안전한 기본값.
# 근거:
#  - hw_manager: 하드웨어 오류는 일시적(통신 지연 등)인 경우가 많다 → retry
#  - roi_detector: 탐색 실패는 반복해도 같은 결과 → 계획 자체를 바꿔야 함 → replan
#  - specialist/debate/rag: 분석·검색 실패는 실험 데이터 자체를 해치지 않는다 → skip
_DEFAULT_ON_FAIL = {
    "hw_manager": "retry",
    "roi_detector": "replan",
    "spectrum_specialist": "skip",
    "domain_specialist": "skip",
    "debate": "skip",
    "rag_searcher": "skip",
}

_PLAN_SYSTEM = """\
당신은 라만 분광 실험을 계획하는 오케스트레이터입니다.
다음 에이전트들을 사용할 수 있습니다:

- roi_detector: 측정 위치 결정.
  * 사용자가 좌표를 준 경우: params 없이 배치 (자동으로 manual 모드)
  * 타겟 위치를 모르는 경우: params에 "mode": "visual_search"와
    "target_description"(타겟 외형 설명)을 지정 — 현미경 카메라로 타겟을 찾아 이동한다.
- hw_manager: 하드웨어 제어. params의 "task" 키로 동작을 지정:
  * "acquire_target": 적응형 스펙트럼 획득. 낮은 출력 프로브 측정 후 포화/신호부족을
    자동 보정해 최적 파워/노출을 찾는다. 물질별 최적 조건을 모르면 반드시 이것을 사용.
    power_pct/exposure_s는 시작값일 뿐이며 자동 조정된다. 지정하지 않으면 저출력에서 시작.
  * "acquire_background": 기판 배경 참조 측정. 타겟 측정과 동일 조건으로 기판 위치를
    측정한다. 기판 신호와 타겟 신호의 구분이 필요하면 acquire_target 바로 다음에 배치.
  * task 생략: 자유 형식 하드웨어 작업 (action에 서술)
- spectrum_specialist: 스펙트럼 물리 분석 (배경 참조가 있으면 자동으로 대조 분석)
- domain_specialist: 도메인 전문 해석. params에 "persona" 키로 전문가 직접 지정 가능
  (biologist|materials_engineer|electrochemist|pharma_chemist|polymer_scientist|food_scientist|forensic_chemist)
- debate: spectrum_specialist와 domain_specialist 결론이 불확실하거나 교차 검증이
  필요할 때 사용. 반드시 spectrum_specialist → domain_specialist 이후에 배치

각 step에 "on_fail" 필드로 실패 정책을 지정하세요:
  "retry"(일시적 오류 재시도) | "replan"(계획 재수립) | "skip"(건너뜀) | "abort"(중단)

표준 프로토콜 (타겟 위치 불명 + 기판 배경 구분 필요 시):
[
  {"step_id": "1", "agent": "roi_detector", "action": "타겟 시각 탐색",
   "params": {"mode": "visual_search", "target_description": "..."}, "on_fail": "replan"},
  {"step_id": "2", "agent": "hw_manager", "action": "타겟 적응형 스펙트럼 획득",
   "params": {"task": "acquire_target"}, "on_fail": "retry"},
  {"step_id": "3", "agent": "hw_manager", "action": "기판 배경 참조 측정",
   "params": {"task": "acquire_background"}, "on_fail": "skip"},
  {"step_id": "4", "agent": "spectrum_specialist", "action": "타겟-배경 대조 스펙트럼 분석",
   "params": {}, "on_fail": "skip"},
  {"step_id": "5", "agent": "domain_specialist", "action": "도메인 관점 해석",
   "params": {}, "on_fail": "skip"}
]

과거 측정 경험이 제공되면 그 조건(파워/노출)을 acquire_target의 시작값으로 활용하세요.
JSON 배열만 출력하세요. 코드블록 없이."""

_REPLAN_SYSTEM = """\
당신은 라만 분광 실험의 실행 중 실패를 복구하는 오케스트레이터입니다.
이미 완료된 step은 다시 포함하지 마세요 (다시 실행하면 시편에 불필요한 레이저가 조사됩니다).
실패 원인을 분석하고, 같은 방식의 실패를 반복하지 않는 '남은 계획'을 새로 작성하세요.
예: 시각 탐색 실패 → 스테이지 중심 좌표에서 manual 측정으로 대체,
    특정 파라미터 획득 실패 → 다른 시작 파라미터의 적응형 획득으로 대체.
출력 형식은 계획과 동일한 JSON 배열 (agent/action/params/on_fail 포함).
JSON 배열만 출력하세요. 코드블록 없이."""

_REPORT_SYSTEM = """\
당신은 라만 분광 실험 보고서를 작성하는 전문가입니다.
실험 결과를 바탕으로 구조화된 보고서를 한국어로 작성하세요.
보고서에는 다음을 포함하세요:
1. 실험 목적
2. 측정 조건 (레이저 파워, 노출시간 — 적응형 튜닝 결과 명시)
3. 측정 결과 요약 (타겟 vs 기판 배경 비교 포함)
4. 스펙트럼 분석 결과
5. 도메인 해석 (전문가 토론 결과가 있으면 포함)
6. 실행 중 발생한 문제와 대응 (재시도/재계획 이력)
7. 결론 및 권고사항"""


# ── plan 파싱 ─────────────────────────────────────────────────────────────────

def _parse_json_array(text: str) -> list | None:
    try:
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        arr = json.loads(cleaned)
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


def _to_steps(steps_raw: list, start_id: int = 1) -> list[PlanStep]:
    """LLM 출력 배열 → PlanStep 리스트. 누락 필드는 안전한 기본값으로 채운다."""
    steps: list[PlanStep] = []
    for s in steps_raw:
        if not isinstance(s, dict):
            continue
        agent = s.get("agent", "hw_manager")
        steps.append(PlanStep(
            step_id=str(s.get("step_id", start_id + len(steps))),
            agent=agent,
            action=s.get("action", ""),
            params=s.get("params", {}) or {},
            status="pending",
            result=None,
            # on_fail 누락 시 agent별 기본 정책 — LLM 출력의 불완전함을 코드가 보정
            on_fail=s.get("on_fail") or _DEFAULT_ON_FAIL.get(agent, "replan"),
        ))
    return steps


def _parse_plan(text: str) -> list[PlanStep]:
    arr = _parse_json_array(text)
    steps = _to_steps(arr) if arr else []
    return steps if steps else _default_plan()


def _default_plan() -> list[PlanStep]:
    """
    LLM 계획 생성이 실패했을 때의 fallback.
    표준 프로토콜 전체(탐색→적응형 획득→배경 측정→분석→해석)를 담는다 —
    "LLM이 죽어도 실험의 뼈대는 항상 안전한 기본 절차를 따른다".
    """
    return [
        PlanStep(step_id="1", agent="roi_detector", action="측정 위치 결정",
                 params={}, status="pending", result=None, on_fail="replan"),
        PlanStep(step_id="2", agent="hw_manager", action="타겟 적응형 스펙트럼 획득",
                 params={"task": "acquire_target"}, status="pending", result=None,
                 on_fail="retry"),
        PlanStep(step_id="3", agent="hw_manager", action="기판 배경 참조 측정",
                 params={"task": "acquire_background"}, status="pending", result=None,
                 on_fail="skip"),
        PlanStep(step_id="4", agent="spectrum_specialist", action="타겟-배경 대조 분석",
                 params={}, status="pending", result=None, on_fail="skip"),
        PlanStep(step_id="5", agent="domain_specialist", action="도메인 해석",
                 params={}, status="pending", result=None, on_fail="skip"),
    ]


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _needs_c3(step: dict) -> bool:
    """스펙트럼 품질 게이트(C3)가 필요한 step인지 — 레이저로 스펙트럼을 획득한
    step만 해당한다. task 기반 판정이 1차, 결과에 스펙트럼 통계가 있으면 2차."""
    task = (step.get("params", {}) or {}).get("task", "")
    if task.startswith("acquire"):
        return True
    result = step.get("result") or {}
    return "max_intensity" in result


def _advance(plan: list, idx: int, critic_log: list) -> dict:
    """
    idx의 다음 step으로 전진. 남은 step이 없으면 C4 → report 순서로 마무리.
    모든 '전진' 경로가 이 함수 하나를 통과해야 종료 시퀀스(C4→report→C5)가
    어디서 전진하든 동일하게 보장된다.
    """
    next_idx = idx + 1
    if next_idx < len(plan):
        nxt = plan[next_idx]
        updated = list(plan)
        updated[next_idx] = {**nxt, "status": "running"}
        return {"plan": updated, "current_step_idx": next_idx, "next_node": nxt["agent"]}

    c4_done = any(e["checkpoint"] == "C4" for e in critic_log)
    if not c4_done:
        return {"plan": plan, "current_step_idx": next_idx,
                "critic_checkpoint": "C4", "next_node": "critic"}
    return {"plan": plan, "current_step_idx": next_idx, "next_node": "report_generator"}


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _summarize_execution(state: ExperimentState, upto_idx: int) -> str:
    """재계획 LLM 프롬프트용 — 완료된 step과 실패 이력을 압축 서술."""
    plan = state.get("plan", [])
    lines = []
    for i, s in enumerate(plan[:upto_idx + 1]):
        r = s.get("result") or {}
        summary = r.get("note") or r.get("error") or str(r)[:120]
        lines.append(f"- step {s['step_id']} [{s['agent']}] {s['action']} → {s['status']}: {summary}")
    fails = state.get("failure_log", [])[-5:]  # 최근 5건이면 원인 파악에 충분
    if fails:
        lines.append("실패 이력:")
        for f in fails:
            lines.append(f"  * [{f.get('agent')}] {f.get('action')}: {f.get('error')}")
    return "\n".join(lines)


def _llm_replan(state: ExperimentState, plan: list, idx: int) -> list[PlanStep] | None:
    """실패 지점부터의 '남은 계획'을 LLM으로 재작성. 실패 시 None."""
    intent = state.get("intent") or {}
    failed_step = plan[idx]

    # ── H-EPM 절차 기억 주입 ─────────────────────────────────────────────────
    # "마지막으로 성공한 step 다음에 과거 실험에서 무엇이 왔었나"를 advisory로
    # 제안한다 (논문 Fig.2의 다음-툴 제안을 계획 수준으로 번안).
    # 재계획 시점이 이 제안의 가치가 가장 큰 곳 — LLM이 실패 복구 경로를
    # 백지에서 상상하는 대신, 실제로 성공했던 전이를 참고하게 된다.
    suggestion_text = ""
    done_steps = [s for s in plan[:idx] if s.get("status") == "done"]
    if done_steps:
        last_key = experience.action_key(done_steps[-1])
        ctx = experience.build_context(state)
        suggestions = experience.suggest_next_steps(last_key, ctx)
        if suggestions:
            sug_lines = ", ".join(
                f"{s['next']} ({'유사 상황에서' if s['mode'] == 'episodic' else '자주'} 성공, {s['n']}회)"
                for s in suggestions)
            suggestion_text = (
                f"\n과거 성공 실험에서 [{last_key}] 다음에 이어진 step (참고용 제안):\n"
                f"  {sug_lines}\n")

    prompt = (
        f"실험 목적: {intent.get('primary_objective', '')}\n"
        f"샘플 종류: {intent.get('sample_type', 'unknown')}\n"
        f"기판: {intent.get('substrate', '') or '(불명)'}\n"
        f"제약 조건: {intent.get('constraints', {})}\n\n"
        f"지금까지의 실행 상황:\n{_summarize_execution(state, idx)}\n"
        f"{suggestion_text}\n"
        f"실패한 step: [{failed_step['agent']}] {failed_step['action']} "
        f"— 오류: {(failed_step.get('result') or {}).get('error', '?')}\n\n"
        "실패한 step을 대체할 방법을 포함해, 남은 실험 계획을 JSON 배열로 작성하세요."
    )
    try:
        resp = _llm.invoke([
            SystemMessage(content=_REPLAN_SYSTEM),
            HumanMessage(content=prompt),
        ])
        arr = _parse_json_array(resp.content)
        if not arr:
            return None
        # step_id 충돌 방지: 기존 마지막 id 다음 번호부터 재부여
        return _to_steps(arr, start_id=idx + 1) or None
    except Exception:
        return None


def _handle_failure(state: ExperimentState, plan: list, idx: int) -> dict:
    """
    failed step에 대한 정책 실행. 모든 분기는 결정적이며 한도가 있다:
      retry  → retry_map[step_id] < _MAX_STEP_RETRIES 이내에서만
      replan → replan_count < _MAX_REPLANS 이내에서만
      한도 초과 시 상위 정책으로 승격(retry→replan→abort) — 조용한 무한 루프 없음.
    """
    step = plan[idx]
    step_id = step.get("step_id", "?")
    policy = step.get("on_fail") or _DEFAULT_ON_FAIL.get(step.get("agent", ""), "replan")

    retry_map = dict(state.get("retry_map", {}))
    retries = retry_map.get(step_id, 0)

    # ── retry ────────────────────────────────────────────────────────────────
    if policy == "retry":
        if retries < _MAX_STEP_RETRIES:
            retry_map[step_id] = retries + 1
            updated = list(plan)
            # c3 플래그 초기화: 재실행 결과는 새로 품질 검사를 받아야 한다
            updated[idx] = {**step, "status": "running", "result": None,
                            "c3_checked": False, "c3_resolved": False}
            return {"plan": updated, "retry_map": retry_map,
                    "next_node": step["agent"]}
        # 재시도 소진 → 재계획으로 승격 (같은 방식 반복은 무의미)
        policy = "replan"

    # ── skip ─────────────────────────────────────────────────────────────────
    if policy == "skip":
        updated = list(plan)
        updated[idx] = {**step, "status": "skipped"}
        return _advance(updated, idx, state.get("critic_log") or [])

    # ── abort ────────────────────────────────────────────────────────────────
    if policy == "abort":
        return {
            "abort_reason": f"step {step_id} 실패 (정책 abort): "
                            f"{(step.get('result') or {}).get('error', '?')}",
            "next_node": "__end__",
        }

    # ── replan (기본) ────────────────────────────────────────────────────────
    replan_count = state.get("replan_count", 0)
    if replan_count >= _MAX_REPLANS:
        return {
            "abort_reason": f"재계획 한도({_MAX_REPLANS}회) 초과 — "
                            f"step {step_id} 실패: {(step.get('result') or {}).get('error', '?')}",
            "next_node": "__end__",
        }

    new_steps = _llm_replan(state, plan, idx)
    if new_steps is None:
        # 재계획 LLM 자체가 실패 — 마지막 안전망으로 해당 step만 skip.
        # (분석 없는 실험이 중단된 실험보다 낫다고 보기 어려운 경우도 있으나,
        #  여기 도달했다는 것은 LLM 장애 상황이므로 진행 가능한 것을 살린다)
        updated = list(plan)
        updated[idx] = {**step, "status": "skipped"}
        out = _advance(updated, idx, state.get("critic_log") or [])
        out["replan_count"] = replan_count + 1
        return out

    # 완료된 step(plan[:idx])은 보존 — 이미 시편에 조사한 측정을 다시 하지 않는다.
    merged = list(plan[:idx]) + new_steps
    return {
        "plan": merged,
        "current_step_idx": idx,
        "replan_count": replan_count + 1,
        # 새 계획도 C1 안전 검증을 다시 통과해야 실행된다
        "critic_checkpoint": "C1",
        "next_node": "critic",
    }


def _handle_c3_result(state: ExperimentState, plan: list, idx: int) -> dict:
    """
    C3 게이트 판정 처리. Critic이 suggestion(보정 계수)을 줬으면 결정적으로
    파라미터를 고쳐 재측정하고, 위치 문제(reposition)면 재계획으로 넘긴다.
    """
    step = plan[idx]
    step_id = step.get("step_id", "?")
    critic_log = state.get("critic_log") or []
    c3_entries = [e for e in critic_log if e["checkpoint"] == "C3"]
    last_c3 = c3_entries[-1] if c3_entries else None

    updated = list(plan)

    # APPROVE 또는 C3 기록 없음 → 품질 통과, 전진
    if last_c3 is None or last_c3["verdict"] == "APPROVE":
        updated[idx] = {**step, "c3_resolved": True}
        return _advance(updated, idx, critic_log)

    sug = last_c3.get("suggestion") or {}
    retry_map = dict(state.get("retry_map", {}))
    retries = retry_map.get(step_id, 0)

    # ── 위치 문제 (배경 우세): 파라미터로 해결 불가 → 재계획 경로로 ──────────
    if sug.get("reposition"):
        # on_fail을 "replan"으로 강제 덮어쓰는 이유: step의 기본 정책(retry)을
        # 그대로 따르면 "같은 위치에서" 재측정하게 되는데, 배경 우세는 측정
        # 위치/초점의 문제라 재측정으로는 절대 해결되지 않는다 (dose만 낭비).
        # 재계획 LLM이 타겟 재탐색(roi_detector) 등을 포함한 대안을 만들게 한다.
        updated[idx] = {**step, "status": "failed", "on_fail": "replan",
                        "result": {"error": f"C3: {last_c3['reason']}"}}
        return _handle_failure(state, updated, idx)

    # ── 파라미터 문제 (포화/신호부족): 보정 계수 적용 후 재측정 ───────────────
    acq = state.get("acquisition_params") or {}

    # 적응형 획득이 이미 한계(파워/노출 상한)에 도달해 미수렴한 경우,
    # 시작값을 키워 재시도해도 같은 한계에 부딪힌다 → dose만 낭비. 수용하고 전진.
    if (step.get("params", {}) or {}).get("task") == "acquire_target" \
            and acq and not acq.get("tuned", True) \
            and sug.get("issue") == "weak_signal":
        updated[idx] = {**step, "c3_resolved": True}
        return _advance(updated, idx, critic_log)

    if retries < _MAX_STEP_RETRIES and (sug.get("power_scale") or sug.get("exposure_scale")):
        params = dict(step.get("params", {}) or {})
        base_power = float(params.get("power_pct") or acq.get("power_pct") or 5.0)
        base_exposure = float(params.get("exposure_s") or acq.get("exposure_s") or 0.2)
        params["power_pct"] = round(
            _clamp(base_power * float(sug.get("power_scale", 1.0)), 0.004, 100.0), 3)
        params["exposure_s"] = round(
            _clamp(base_exposure * float(sug.get("exposure_scale", 1.0)), 0.05, 10.0), 3)

        retry_map[step_id] = retries + 1
        updated[idx] = {**step, "status": "running", "params": params, "result": None,
                        "c3_checked": False, "c3_resolved": False}
        return {"plan": updated, "retry_map": retry_map, "next_node": step["agent"]}

    # 재시도 소진 — 경고는 critic_log에 남아 보고서가 인지한다. 수용하고 전진.
    updated[idx] = {**step, "c3_resolved": True}
    return _advance(updated, idx, critic_log)


# ══════════════════════════════════════════════════════════════════════════════
# Planner 노드
# ══════════════════════════════════════════════════════════════════════════════

def planner_node(state: ExperimentState) -> dict:
    # ── 0. 중단 확인 ─────────────────────────────────────────────────────────
    if state.get("abort_reason"):
        return {"next_node": "__end__"}

    critic_log = state.get("critic_log") or []
    final_report = state.get("final_report")

    # ── 1. 종료 시퀀스: 보고서 → C5 → 끝 ─────────────────────────────────────
    c5_done = any(e["checkpoint"] == "C5" for e in critic_log)
    if final_report and c5_done:
        return {"next_node": "__end__"}
    if final_report and not c5_done:
        # [버그 수정] 기존에는 이 라우팅이 없어 report_generator가 무한 재호출됐다.
        # (그래프 edge가 report→planner로 고정이라 report가 설정한 next_node="critic"이
        #  planner 반환값으로 덮여 사라졌음 — 모듈 docstring 참고)
        return {"critic_checkpoint": "C5", "next_node": "critic"}

    plan = state.get("plan", [])
    idx = state.get("current_step_idx", 0)

    # ── 2. plan 없음 → RAG → LLM 계획 생성 → C1 ──────────────────────────────
    if not plan:
        if not state.get("rag_results"):
            return {"next_node": "rag_searcher"}

        intent = state.get("intent") or {}
        sample_type = intent.get("sample_type", "unknown")
        prompt = (
            f"실험 목적: {intent.get('primary_objective', '라만 스펙트럼 측정')}\n"
            f"샘플 종류: {sample_type}\n"
            f"타겟 설명: {intent.get('target_description', '') or '(없음)'}\n"
            f"제약 조건: {intent.get('constraints', {})}\n"
            f"기판: {intent.get('substrate', '') or '(불명)'}\n"
            f"RAG 참고 정보: {state.get('rag_results', [])[:2]}\n"
            # 경험 저장소 (H-EPM 하이브리드 기억): 시료/기판/영역 컨텍스트가
            # 유사한 과거 에피소드(성공 조건 + 실패 경고)와 자주 성공한 절차를
            # 계획 프롬프트에 advisory로 주입 — "노하우가 축적되는" 지점.
            # (실험 종료 시 orchestrator가 record_experiment로 기록)
            f"과거 측정 경험:\n{experience.recall_summary(experience.build_context(state))}\n\n"
            "위 정보를 바탕으로 실험 계획을 JSON 배열로 작성하세요."
        )
        try:
            response = _llm.invoke([
                SystemMessage(content=_PLAN_SYSTEM),
                HumanMessage(content=prompt),
            ])
            new_plan = _parse_plan(response.content)
        except Exception:
            # 계획 LLM 장애 → 표준 프로토콜 fallback (실험을 멈추지 않는다)
            new_plan = _default_plan()

        return {
            "plan": new_plan,
            "current_step_idx": 0,
            "critic_checkpoint": "C1",
            "next_node": "critic",
        }

    # ── 3. C1 거부 처리 ──────────────────────────────────────────────────────
    c1_entries = [e for e in critic_log if e["checkpoint"] == "C1"]
    if c1_entries and c1_entries[-1]["verdict"] == "ABORT":
        executed = any(s["status"] in ("done", "failed", "skipped") for s in plan[:idx + 1]) and idx > 0
        if executed:
            # 실행 도중 재계획한 plan이 C1에서 거부됨 — 이미 시편에 조사가 이뤄진
            # 상태에서 또 계획을 뒤집는 것은 위험 신호 → 안전하게 중단.
            return {"abort_reason": "재계획된 plan이 C1 검증 실패 — 실험 중단",
                    "next_node": "__end__"}
        replan = state.get("replan_count", 0)
        if replan < _MAX_REPLANS:
            # 아무것도 실행 전이므로 처음부터 다시: plan/RAG 리셋 후 재생성
            return {
                "plan": [],
                "rag_results": [],
                "replan_count": replan + 1,
                "next_node": "rag_searcher",
            }
        return {"abort_reason": f"C1 plan sanity 실패 — 재계획 한도({_MAX_REPLANS}회) 초과",
                "next_node": "__end__"}

    # ── 4. step 순차 실행 ────────────────────────────────────────────────────
    if idx < len(plan):
        step = plan[idx]
        status = step["status"]

        # 4a. 미실행 step → 해당 에이전트로 라우팅
        if status in ("pending", "running"):
            updated = list(plan)
            updated[idx] = {**step, "status": "running"}
            return {"plan": updated, "next_node": step["agent"]}

        # 4b. 실패 step → 정책 기반 복구 (retry/replan/skip/abort)
        if status == "failed":
            return _handle_failure(state, plan, idx)

        # 4c. skipped → 그냥 전진
        if status == "skipped":
            return _advance(plan, idx, critic_log)

        # 4d. done → 획득 step이면 C3 품질 게이트를 먼저 통과시킨다
        if status == "done":
            if _needs_c3(step) and not step.get("c3_checked"):
                updated = list(plan)
                updated[idx] = {**step, "c3_checked": True}
                return {"plan": updated, "critic_checkpoint": "C3", "next_node": "critic"}

            if step.get("c3_checked") and not step.get("c3_resolved"):
                return _handle_c3_result(state, plan, idx)

            # C3 불필요 또는 통과 → 다음 step
            return _advance(plan, idx, critic_log)

    # ── 5. 모든 step 소진 → C4 → report ─────────────────────────────────────
    c4_done = any(e["checkpoint"] == "C4" for e in critic_log)
    if not c4_done:
        return {"critic_checkpoint": "C4", "next_node": "critic"}
    return {"next_node": "report_generator"}


# ══════════════════════════════════════════════════════════════════════════════
# Report Generator (Planner와 같은 파일)
# ══════════════════════════════════════════════════════════════════════════════

def report_generator_node(state: ExperimentState) -> dict:
    intent = state.get("intent") or {}
    observations = state.get("observations", [])
    spectrum_analysis = state.get("spectrum_analysis", "분석 없음")
    domain_interpretation = state.get("domain_interpretation", "해석 없음")
    debate_result = state.get("debate_result", "")
    critic_log = state.get("critic_log", [])
    failure_log = state.get("failure_log", [])
    acq = state.get("acquisition_params") or {}
    bg = state.get("background_reference") or {}

    obs_summary = ""
    for i, obs in enumerate(observations, 1):
        obs_summary += f"측정 {i}: {obs.get('tool', '')} → {str(obs.get('result', ''))[:200]}\n"

    critic_summary = "\n".join(
        f"[{e['checkpoint']}] {e['verdict']}: {e['reason']}" for e in critic_log
    )
    failure_summary = "\n".join(
        f"[{f.get('agent')}] {f.get('action')}: {f.get('error')}" for f in failure_log
    )

    prompt = (
        f"## 실험 요청\n{intent.get('primary_objective', state.get('user_message', ''))}\n\n"
        f"## 샘플\n{intent.get('sample_type', 'unknown')}\n\n"
        f"## 확정 측정 조건 (적응형 튜닝)\n"
        f"레이저 {acq.get('power_pct', '?')}%, 노출 {acq.get('exposure_s', '?')}s, "
        f"수렴 여부: {acq.get('tuned', '?')}, 튜닝 이력: {acq.get('history', [])}\n\n"
        f"## 기판 배경 참조\n"
        f"{'측정됨 — 위치 ' + str(bg.get('position')) + ', max ' + str(bg.get('max_intensity')) + ' ADU' if bg else '미측정'}\n\n"
        f"## 측정 데이터\n{obs_summary or '없음'}\n\n"
        f"## 스펙트럼 분석\n{spectrum_analysis}\n\n"
        f"## 도메인 해석\n{domain_interpretation}\n\n"
        f"## 전문가 토론\n{debate_result or '없음'}\n\n"
        f"## Critic 로그\n{critic_summary or '없음'}\n\n"
        f"## 실행 중 실패/복구 이력\n{failure_summary or '없음'}\n\n"
        "위 정보를 바탕으로 실험 보고서를 작성하세요."
    )

    try:
        response = _llm.invoke([
            SystemMessage(content=_REPORT_SYSTEM),
            HumanMessage(content=prompt),
        ])
        report = response.content
    except Exception as e:
        # 보고서 LLM 장애 시에도 실험 데이터는 살린다 — 원시 요약으로 대체
        report = f"[보고서 생성 실패: {e}]\n\n{prompt}"

    # 노하우 기록은 여기서 하지 않는다 — orchestrator.run_experiment가 그래프
    # invoke 직후 experience.record_experiment()로 일괄 기록한다.
    # 이유: abort로 중단된 실험은 이 노드에 도달하지 못하므로, 여기서 기록하면
    # "실패 지식"(어떤 조건에서 실패했나)이 영영 축적되지 않는다.

    return {
        "final_report": report,
        # next_node는 그래프 edge상 의미 없지만(planner로 고정 복귀),
        # planner가 "final_report 있음 + C5 없음"을 보고 C5로 보낸다.
    }
