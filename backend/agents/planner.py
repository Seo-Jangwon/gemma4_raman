"""
PlannerNode — Hub 에이전트. 모든 스포크를 조율하고 실행 순서를 결정.
report_generator_node도 여기에 함께 정의.

LLM: Claude claude-opus-4-7 (파일 상단 _llm 교체로 변경 가능)

MVP 실행 흐름:
1. 첫 진입 (plan이 비어있음):
   a. intent 확인 → RAG 검색 요청 (next_node = "rag_searcher")
2. RAG 완료 후:
   b. LLM으로 plan 생성
   c. C1 체크 (next_node = "critic", critic_checkpoint = "C1")
3. C1 통과 후:
   d. plan 순서대로 에이전트 실행 (current_step_idx 증가)
4. 모든 step 완료:
   e. next_node = "report_generator"
5. abort_reason 있음:
   f. next_node = END (graph router가 처리)
"""

from __future__ import annotations

import json
import re
import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.state import ExperimentState, PlanStep

# ── LLM 설정 (교체 포인트) ────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-opus-4-7", temperature=0)

_PLAN_SYSTEM = """\
당신은 라만 분광 실험을 계획하는 오케스트레이터입니다.
다음 에이전트들을 사용할 수 있습니다:
- hw_manager: 하드웨어 제어 (스테이지 이동, 레이저 설정, 스펙트럼 획득)
- spectrum_specialist: 스펙트럼 물리 분석
- domain_specialist: 도메인 전문 해석. params에 "persona" 키로 전문가 직접 지정 가능
  (biologist|materials_engineer|electrochemist|pharma_chemist|polymer_scientist|food_scientist|forensic_chemist)
- debate: spectrum_specialist와 domain_specialist 결론이 불확실하거나 교차 검증이 필요할 때 사용
  반드시 spectrum_specialist → domain_specialist 이후에 배치
- roi_detector: 다음 측정 위치 결정

실험 계획을 JSON 배열로 출력하세요:
[
  {"step_id": "1", "agent": "hw_manager", "action": "레이저 ON 후 스펙트럼 획득",
   "params": {"power_pct": 30, "exposure_s": 1.0}},
  {"step_id": "2", "agent": "spectrum_specialist", "action": "획득된 스펙트럼 분석", "params": {}},
  {"step_id": "3", "agent": "domain_specialist", "action": "도메인 관점 해석",
   "params": {"persona": "materials_engineer"}},
  {"step_id": "4", "agent": "debate", "action": "해석 교차 검증", "params": {}}
]

JSON 배열만 출력하세요. 코드블록 없이."""

_REPORT_SYSTEM = """\
당신은 라만 분광 실험 보고서를 작성하는 전문가입니다.
실험 결과를 바탕으로 구조화된 보고서를 한국어로 작성하세요.
보고서에는 다음을 포함하세요:
1. 실험 목적
2. 측정 결과 요약
3. 스펙트럼 분석 결과
4. 도메인 해석
5. 결론 및 권고사항"""


# ── plan 파싱 ─────────────────────────────────────────────────────────────────

def _parse_plan(text: str, intent_summary: str) -> list[PlanStep]:
    try:
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        steps_raw = json.loads(text)
        steps = []
        for s in steps_raw:
            steps.append(PlanStep(
                step_id=str(s.get("step_id", len(steps) + 1)),
                agent=s.get("agent", "hw_manager"),
                action=s.get("action", ""),
                params=s.get("params", {}),
                status="pending",
                result=None,
            ))
        return steps if steps else _default_plan()
    except Exception:
        return _default_plan()


def _default_plan() -> list[PlanStep]:
    return [
        PlanStep(step_id="1", agent="hw_manager",
                 action="스펙트럼 획득", params={}, status="pending", result=None),
        PlanStep(step_id="2", agent="spectrum_specialist",
                 action="스펙트럼 분석", params={}, status="pending", result=None),
        PlanStep(step_id="3", agent="domain_specialist",
                 action="도메인 해석", params={}, status="pending", result=None),
    ]


# ── Planner 내부 단계 플래그 ──────────────────────────────────────────────────
# Planner는 상태가 없으므로 state 내 특수 필드로 진행 단계를 추적한다.
# "_planner_phase" 키 사용 (ExperimentState에 없으나 dict이므로 가능)

def planner_node(state: ExperimentState) -> dict:
    # abort 확인
    if state.get("abort_reason"):
        return {"next_node": "__end__"}

    critic_log = state.get("critic_log") or []

    # C5 완료 후 복귀 → 최종 종료
    if any(e["checkpoint"] == "C5" for e in critic_log) and state.get("final_report"):
        return {"next_node": "__end__"}

    plan = state.get("plan", [])
    idx  = state.get("current_step_idx", 0)

    # ── 단계 A: plan이 없음 → RAG 먼저 ──────────────────────────────────────
    if not plan:
        # RAG를 아직 안 했으면 먼저 요청
        rag_done = bool(state.get("rag_results"))
        if not rag_done:
            return {"next_node": "rag_searcher"}

        # RAG 완료 → LLM으로 plan 생성
        intent = state.get("intent") or {}
        prompt = (
            f"실험 목적: {intent.get('primary_objective', '라만 스펙트럼 측정')}\n"
            f"샘플 종류: {intent.get('sample_type', 'unknown')}\n"
            f"제약 조건: {intent.get('constraints', {})}\n"
            f"RAG 참고 정보: {state.get('rag_results', [])[:2]}\n\n"
            "위 정보를 바탕으로 실험 계획을 JSON 배열로 작성하세요."
        )
        response = _llm.invoke([
            SystemMessage(content=_PLAN_SYSTEM),
            HumanMessage(content=prompt),
        ])
        new_plan = _parse_plan(response.content, prompt)

        # C1 체크 요청
        return {
            "plan": new_plan,
            "current_step_idx": 0,
            "critic_checkpoint": "C1",
            "next_node": "critic",
        }

    # ── 단계 B: C1 체크 직후 (plan 있고 idx==0이고 critic_log에 C1 있음) ─────
    c1_entries = [e for e in critic_log if e["checkpoint"] == "C1"]
    if c1_entries and c1_entries[-1]["verdict"] == "ABORT":
        replan = state.get("replan_count", 0)
        if replan < 2:
            return {
                "plan": [],
                "rag_results": [],
                "replan_count": replan + 1,
                "next_node": "rag_searcher",
            }
        return {"abort_reason": "C1 plan sanity 실패 — 재계획 한도(2회) 초과", "next_node": "__end__"}

    # ── 단계 C: plan step 순서 실행 ──────────────────────────────────────────
    # 현재 step 완료 여부 확인
    if idx < len(plan):
        current_step = plan[idx]
        # 아직 실행 안 한 step이면 해당 에이전트로 라우팅
        if current_step["status"] in ("pending", "running"):
            updated_plan = list(plan)
            updated_plan[idx] = {**current_step, "status": "running"}
            return {"plan": updated_plan, "next_node": current_step["agent"]}

        # step이 완료/실패 → 다음 step을 즉시 결정해 next_node 누수 방지
        next_idx = idx + 1
        if next_idx < len(plan):
            next_step = plan[next_idx]
            updated_plan = list(plan)
            updated_plan[next_idx] = {**next_step, "status": "running"}
            return {
                "plan": updated_plan,
                "current_step_idx": next_idx,
                "next_node": next_step["agent"],
            }
        # 모든 step 완료 → C4 체크 후 report_generator
        c4_done = any(e["checkpoint"] == "C4" for e in critic_log)
        if not c4_done:
            return {"critic_checkpoint": "C4", "next_node": "critic"}
        return {"next_node": "report_generator"}

    # ── 단계 D: idx >= len(plan), 모든 step 완료 ────────────────────────────
    c4_done = any(e["checkpoint"] == "C4" for e in critic_log)
    if not c4_done:
        return {"critic_checkpoint": "C4", "next_node": "critic"}
    return {"next_node": "report_generator"}


# ── Report Generator (Planner와 같은 파일) ────────────────────────────────────

def report_generator_node(state: ExperimentState) -> dict:
    intent = state.get("intent") or {}
    observations = state.get("observations", [])
    spectrum_analysis = state.get("spectrum_analysis", "분석 없음")
    domain_interpretation = state.get("domain_interpretation", "해석 없음")
    critic_log = state.get("critic_log", [])
    plan = state.get("plan", [])

    obs_summary = ""
    for i, obs in enumerate(observations, 1):
        obs_summary += f"측정 {i}: {obs.get('tool', '')} → {str(obs.get('result', ''))[:200]}\n"

    critic_summary = "\n".join(
        f"[{e['checkpoint']}] {e['verdict']}: {e['reason']}" for e in critic_log
    )

    prompt = (
        f"## 실험 요청\n{intent.get('primary_objective', state.get('user_message', ''))}\n\n"
        f"## 샘플\n{intent.get('sample_type', 'unknown')}\n\n"
        f"## 측정 데이터\n{obs_summary or '없음'}\n\n"
        f"## 스펙트럼 분석\n{spectrum_analysis}\n\n"
        f"## 도메인 해석\n{domain_interpretation}\n\n"
        f"## Critic 로그\n{critic_summary or '없음'}\n\n"
        "위 정보를 바탕으로 실험 보고서를 작성하세요."
    )

    response = _llm.invoke([
        SystemMessage(content=_REPORT_SYSTEM),
        HumanMessage(content=prompt),
    ])

    return {
        "final_report": response.content,
        "critic_checkpoint": "C5",
        "next_node": "critic",
    }
