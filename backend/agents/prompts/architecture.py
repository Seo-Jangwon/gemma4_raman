# -*- coding: utf-8 -*-
"""
아키텍처마다 다른 것 — ReAct(AILA)와 CoALA의 프로필.

여기 있는 것은 두 종류뿐이다:
  ① 전용 섹션   : 상대 쪽에는 존재할 이유가 없는 블록
                  (CoALA 의 [Memory structure], [Decision cycle])
  ② 호칭 슬롯   : 공통 블록(prompts/common.py)의 같은 지시를 그 아키텍처의 어휘로
                  부르는 값. "도구를 호출한다" ↔ "planning action 을 쓴다",
                  "같은 턴에" ↔ "같은 턴의 다음 사이클에" 같은 것.

★ ②에 '지시 내용'을 넣지 말 것. 도메인 규칙(안전·절차·복구)이 여기로 들어오는 순간
  두 에이전트가 다른 규칙을 받게 되고, 비교 실험의 독립변수가 오케스트레이션 하나가
  아니게 된다. 그런 문장은 전부 common.py 로 간다.
"""

from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════
# ReAct (AILA) — stateless baseline
# ══════════════════════════════════════════════════════════════════════════════

REACT_PREAMBLE = """\
You are a single AI agent that directly controls a Raman spectrometer from start to finish.
You plan, measure, adjust parameters, interpret spectra, and write the report entirely by your
own judgment. There is no separate validator or assistant to check you - your judgment is the
final judgment."""

REACT = {
    "preamble": REACT_PREAMBLE,
    "extra_sections": [],                # ReAct 는 전용 섹션이 없다 — 그게 baseline 의 정의다
    "slots": {
        # [Autonomy]
        "evidence_source":  "the knowledge base",
        "execution_timing": "in the same turn",
        "autonomy_extra":   "",
        # [Attached data files]
        "inspect_lead":      "call list_uploaded_files, then call inspect_file on each relevant file",
        "run_analysis_lead": "Then call run_analysis with file_ids to compute",
        # [Measurement procedure]
        "procedure_mode":  "proceed step by step, using your own judgment",
        "gather_tools":    "infer it from the request, the knowledge base, and your observation tools "
                           "(analyze_microscope_image)",
        "protocol_lookup": "use search_knowledge_base",
        "execution_note":  "",
        "record_step":     "",
        "report_no":       "6",
        # [Hardware failure recovery]
        "diagnose_lead":    "call get_hardware_status",
        "recovery_lead":    "Try recovery ONCE for the failed component",
        "recovery_record":  "",
        "recovery_budget":
            "Never call the same recovery tool more than twice in a turn. If it did not work twice, it will not work\n"
            "a third time, and burning steps on it costs you the rest of the task.",
        # [Safety rules]
        "verify_with": "a tool",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# CoALA — 명시적 작업기억 + 장기기억 + 의사결정 사이클
# ══════════════════════════════════════════════════════════════════════════════

COALA_PREAMBLE = """\
You are a single AI agent that controls a Raman spectrometer, operating under the CoALA (Cognitive
Architectures for Language Agents) architecture. You have explicit working memory and long-term
memory, and you act according to the decision cycle below (planning -> execution)."""

# episodic 메모리가 켜져 있을 때와 꺼져 있을 때의 [Memory structure].
# 예전에는 완성된 프롬프트에서 이 줄을 문자열 replace 로 지웠는데, 문장이 조금만 바뀌어도
# 치환이 조용히 실패해 "없는 도구를 지시하는 프롬프트"가 남았다. 이제는 아예 안 넣는다.
COALA_MEMORY_SECTION = """\
[Memory structure]
- Working memory: the current goal, retrieved knowledge, and recent observations are provided in the prompt.
- Semantic memory: read with search_knowledge_base (curated protocols) and recall_insights (insights I left),
  and write with record_insight (reusable rules).{episodic_line}"""

COALA_EPISODIC_LINE = ("\n- Episodic memory: read past experiments with recall_experiences and "
                       "write with record_experience (know-how).")

COALA_DECISION_CYCLE = """\
[Decision cycle - distinguish planning from execution]
Your actions are of two kinds, and their nature is completely different.

  · Planning actions (information gathering): {planning_tools}.
    - These do not turn on the laser; they 'gather evidence'. Call them several times in a row as needed
      to fill working memory sufficiently. They make nothing irreversible, so use them freely, but do not
      repeat the same lookup.

  · Execution actions (commit): hardware tools (stage move, laser, acquire_spectrum, camera, etc.) and
    recording tools ({recording_tools}).
    - These actually change the world or long-term memory. In particular, acquire_spectrum irradiates the
      sample with the laser, and photodamage is irreversible.
    - Always execute 'only one at a time'. When execution is needed, choose the single most valuable
      execution action in the current situation. After seeing its result (observation), decide the next
      execution in the following cycle.

  · Principle: finish the necessary planning actions (information gathering) 'before' choosing an execution
    action. The laser should only be fired after enough evidence is gathered.

  · finish: if there are no more tools to call, write the final report in English without tools and this turn ends."""


def coala(episodic: bool = True) -> dict:
    """CoALA 프로필. episodic=False 면 에피소딕 메모리를 '언급조차 하지 않는' 프롬프트가 된다.

    RAMAN_EPISODIC_MEMORY=0 ablation 에서 쓴다 — 도구를 바인딩에서 빼는 것만으로는
    부족하고(모델이 없는 도구를 호출하려 사이클을 태운다) 지시문도 함께 빠져야 한다.
    """
    planning_tools = ("search_knowledge_base, recall_experiences, recall_insights,\n"
                      "    list_uploaded_files, inspect_file") if episodic else \
                     ("search_knowledge_base, recall_insights,\n"
                      "    list_uploaded_files, inspect_file")
    recording_tools = "record_experience, record_insight" if episodic else "record_insight"

    # 측정 절차의 근거 수집 도구와 프로토콜 조회 문구도 episodic 유무를 따라간다.
    gather = ("gather evidence with planning actions (recall_experiences, recall_insights, "
              "search_knowledge_base, analyze_microscope_image)") if episodic else \
             ("gather evidence with planning actions (recall_insights, search_knowledge_base, "
              "analyze_microscope_image)")
    protocol = ("query past know-how with recall_experiences, accumulated rules with recall_insights, "
                "and protocols with search_knowledge_base") if episodic else \
               ("query accumulated rules with recall_insights and protocols with search_knowledge_base")
    record_step = ("6. When the measurement is done, before writing the report, record this experience with\n"
                   "   record_experience (and, if possible, generalized knowledge with record_insight) into\n"
                   "   long-term memory - this is how know-how accumulates for the next experiment.\n") if episodic else \
                  ("6. When the measurement is done, before writing the report, record generalized knowledge\n"
                   "   with record_insight into long-term memory - this is how know-how accumulates for the\n"
                   "   next experiment.\n")

    memory_section = COALA_MEMORY_SECTION.format(
        episodic_line=COALA_EPISODIC_LINE if episodic else "")
    decision_cycle = COALA_DECISION_CYCLE.format(
        planning_tools=planning_tools, recording_tools=recording_tools)

    return {
        "preamble": COALA_PREAMBLE,
        "extra_sections": [memory_section, decision_cycle],
        "slots": {
            # [Autonomy]
            "evidence_source":  "your memories",
            "execution_timing": "in a following cycle of the same turn",
            "autonomy_extra":
                "\n- Insufficient evidence is a reason to run more planning actions, not a reason to ask the user.",
            # [Attached data files]
            "inspect_lead": ("gather evidence with the planning actions list_uploaded_files "
                             "and then inspect_file on each relevant file"),
            "run_analysis_lead": "Then run_analysis with file_ids (an execution action - one per cycle) to compute",
            # [Measurement procedure]
            "procedure_mode":  "proceed on your own through the cycles",
            "gather_tools":    gather,
            "protocol_lookup": protocol,
            "execution_note":  " These are execution actions - one at a time.",
            "record_step":     record_step,
            "report_no":       "7",
            # [Hardware failure recovery]
            "diagnose_lead":   "get_hardware_status is a planning action (no side effects) - call it first",
            "recovery_lead":   "Then ONE recovery execution action",
            "recovery_record": ("\n   Record what happened with record_experience so the next run does not repeat\n"
                                "   the same dead end.") if episodic else "",
            "recovery_budget":
                "Never spend more than two cycles on the same recovery - that is evidence it cannot be fixed from here.",
            # [Safety rules]
            "verify_with": "a planning action",
        },
    }
