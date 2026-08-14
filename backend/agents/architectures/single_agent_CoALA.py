# -*- coding: utf-8 -*-
"""
CoALA — 라만 분광기를 LLM 하나로 제어하되, Sumers et al. 2024 "Cognitive Architectures for
Language Agents" 의 **의사결정 사이클**로 구조화한 단일 에이전트.

이 파일에는 그 사이클 하나만 있다. 프롬프트는 backend.agents.prompts, 배관은
backend.agents.runtime, 장기기억은 backend.agents.long_term_memory 에 있다.

[single_agent_AILA 와의 관계]
AILA 는 stateless ReAct baseline 이다: 모델이 한 번에 emit 한 tool_call 을 **전부** 실행하고,
장기기억이 없고, 오케스트레이션 로직이 사실상 없다. 이 파일은 그 baseline 과 비교되는 두
번째 에이전트이고, 비교 실험의 독립변수는 **오직 오케스트레이션 아키텍처**다. 따라서 서로를
import 하지 않되, 어긋나면 비교가 무너지는 것(프롬프트의 도메인 지시·LLM 설정·조사량 상한·
관측 축약·배관)은 사본을 두지 않고 둘 다 공통 상위 모듈을 본다.

[CoALA 매핑 — 논문 §4]
  · Working memory  : 아래 WorkingMemory — LLM 호출 간 지속되는 자료구조
                      (goal / retrieved / observations / messages).
  · Semantic memory : 읽기 search_knowledge_base·recall_insights, 쓰기 record_insight.
  · Episodic memory : 읽기 recall_experiences, 쓰기 record_experience.
  · Procedural      : 이 파일의 코드 + TOOL_DISPATCH + LLM 가중치(설계자가 초기화).
  · Action space    : external=grounding(하드웨어), internal=retrieval/reasoning/learning.
  · Decision cycle  : planning(propose·evaluate·select) → execution → observe (§4.6).

[planning stage 분리 — 논문 Figure 4B / §4.6 정합]
retrieval 은 planning 의 '수단'이지 실행 대상이 아니다. 그래서 사이클 내부에 planning
내부 루프를 둔다:
  · 모델이 retrieval 을 제안 → 즉시 실행해 working memory 에 정보를 쌓고, 사이클을 닫지
    않고 다시 propose (정보 수집 반복).
  · 모델이 grounding/learning 을 제안 → planning 종료. 그것들'만' evaluate·select·execute
    대상이 된다.
  · retrieval 과 grounding 이 한 응답에 섞이면 retrieval 만 먼저 실행하고 grounding 은 버려
    재제안하게 한다 — 정보를 다 모은 '뒤'에 실행을 결정하도록.
    (라만 비가역성: 레이저는 planning 이 끝난 뒤의 판단으로만 발사된다.)

[공개 API — 서버가 의존하는 계약 (AILA 와 동일)]
  ALL_TOOLS / stream_experiment() / run_experiment() / run_stream()
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.agents.utils import reason_log
from backend.agents.memory import long_term_memory as ltm
from backend.agents.runtime import runtime
from backend.service.store import run_store
from backend.agents.prompts import AUTONOMOUS, build_system_prompt
from backend.tools.non_hw_tools.file_tools import FILE_RETRIEVAL
from backend.tools.tools import BASE_TOOLS, KB_TOOL_COALA

# /api/agents/health 용 재수출 (AILA 와 동일한 계약).
from backend.llm_config import OLLAMA_HOST, OLLAMA_MODEL   # noqa: F401

ARCH = "CoALA"
SYSTEM_PROMPT = build_system_prompt("CoALA")

# ── 사이클/계획 예산 ──────────────────────────────────────────────────────────
MAX_CYCLES = 150            # 최대 의사결정 사이클 수 (= commit 행동 실행 횟수)
MAX_PLANNING_STEPS = 6      # 한 사이클 내 planning(정보수집) 라운드 상한
SOFT_PLAN_LIMIT = 4         # 이 라운드부터 "이제 실행/보고서로" 진행 문구를 강화
MAX_AGENT_STEPS = 150       # 턴 전체 propose() 호출 총량 가드(무한 루프 방지)

# ── evaluate 예산 ────────────────────────────────────────────────────────────
# 평가 프롬프트에 실을 근거의 상한. 이 값들이 곧 evaluate 호출 1회의 입력 크기다.
#
# [왜 상한이 필요한가]
# 근거를 붙이는 목적은 평가자가 추측 대신 사실로 판단하게 하는 것이지, 평가자에게 propose
# 와 같은 컨텍스트를 통째로 주는 것이 아니다. 도구 설명 하나가 4,035자(run_analysis)까지
# 있고 관측 결과에는 스펙트럼 통계가 통째로 들어 있어서, 자르지 않으면 evaluate 가 propose
# 만큼 비싸진다. 아래 상한이면 근거를 다 채워도 입력이 ~2,600자(≈700 토큰)를 넘지 않는다.
#
# [VRAM 과는 무관하다]
# Ollama 의 KV 캐시는 num_ctx(=100,000)로 미리 잡히므로, 실제 프롬프트가 239 토큰이든
# 1,200 토큰이든 점유 VRAM 은 같다. 여기서 아끼는 것은 생성 시간과 네트워크다.
MAX_EVAL_CANDIDATES = 4     # 후보가 더 많아도 앞의 4개만 채점한다(나머지는 폐기)
EVAL_DESC_CHARS = 400       # 후보 도구 설명을 이 길이에서 자른다 — 앞머리에 계약이 있다
EVAL_OBS_CHARS = 500        # 직전 관측 1건의 최대 길이
EVAL_OBS_COUNT = 2          # 몇 건까지 실을 것인가
# 평가 응답 출력 토큰 상한. **thinking 토큰이 여기 포함된다** — 답이 JSON 한 줄이라고
# 상한을 그 크기로 잡으면 안 된다. 512 로 잡았다가 2026-08-13 실행에서 평가 2회가 전부
# `done=length` 로 잘렸다(thinking 1,971자·2,176자를 쓰고 `{"scores": [1.0,` 에서 절단).
# 근거를 붙인 뒤 thinking 이 실측 500~550 토큰이므로 그 3배로 잡는다. 근거 없던 시절의
# 폭주값(2,584 토큰)은 여전히 상한에 걸린다.
EVAL_NUM_PREDICT = 1536


# ══════════════════════════════════════════════════════════════════════════════
# 액션 공간 — internal action 도 tool 로 노출한다 (논문 §4.1: 모든 액션이 action space)
# ══════════════════════════════════════════════════════════════════════════════

# 도구 스키마는 backend.tools.tools 한 곳에서 조립된다. BASE_TOOLS 는 AILA 와 '같은 리스트
# 객체'라 두 에이전트의 하드웨어·파일 분석 능력이 구조적으로 어긋날 수 없다.
#
# KB 검색만 스키마가 따로다(KB_TOOL_COALA): 구현은 AILA 와 완전히 같은 함수지만, 설명문에
# "planning · does not end the cycle" 이 붙어야 한다. 그게 없으면 모델이 조회 한 번에
# 사이클이 닫히는 줄 알고 정보 수집을 아낀다.
ALL_TOOLS = BASE_TOOLS + [KB_TOOL_COALA] + ltm.SCHEMAS

# 어느 도구가 어느 CoALA 액션 범주인지 — planning/execution 분리의 판정 기준이다.
#   retrieval : planning 도구. 사이클을 닫지 않고 working memory 에 정보를 쌓는다.
#   learning  : execution(commit) 액션. grounding 과 함께 propose→evaluate→select 대상.
#   그 외(하드웨어) : grounding = execution(commit) 액션.
# 첨부 파일 조회(list_uploaded_files/inspect_file)도 부수효과 없는 정보 수집이라 retrieval 에
# 합친다. 안 그러면 파일을 한 번 들여다볼 때마다 사이클이 하나씩 닫혀, 구조만 확인하다 예산을
# 태운다. 반면 run_analysis 는 결과물(그림)을 만드는 실행 액션이라 FILE_RETRIEVAL 에 없고
# commit 으로 남는다.
RETRIEVAL_ACTIONS = {"search_knowledge_base"} | ltm.RETRIEVAL_TOOL_NAMES | FILE_RETRIEVAL
LEARNING_ACTIONS = ltm.LEARNING_TOOL_NAMES


def _get_llm_tools():
    """ALL_TOOLS 를 바인딩한 Runnable — propose/execute 용 (실패 시 None)."""
    return runtime.get_chat_model(ALL_TOOLS)


def _get_llm_plain():
    """도구 없이 텍스트만 내는 Runnable — evaluate 용 (실패 시 None).

    evaluate 는 도구를 '호출'하는 게 아니라 후보를 '점수화'하는 순수 추론이라 tool 바인딩이
    없는 편이 JSON 출력을 방해받지 않아 안정적이다. 모델·호스트는 동일하다.

    출력만 EVAL_NUM_PREDICT 로 묶는다 — 답이 점수 JSON 한 줄로 정해져 있어서다. 상한에
    걸려 잘리면 파싱이 실패하고 첫 후보로 폴백하는데, 그 폴백은 EvalStats 에 남는다.
    """
    return runtime.get_chat_model(None, num_predict=EVAL_NUM_PREDICT)


# ══════════════════════════════════════════════════════════════════════════════
# Working memory (논문 §4.1) — LLM 호출 간 지속되는 자료구조
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkingMemory:
    """현재 의사결정 사이클의 활성 상태. 매 propose 프롬프트에 직렬화되어 주입된다.

    messages 는 tool_call ↔ ToolMessage 쌍을 담는 LangChain 메시지 로그로, 모델이 관측을
    '보는' 실제 컨텍스트다. 나머지 필드는 사람이 읽는 요약이자 planning 프롬프트의 상단
    컨텍스트가 된다.
    """
    goal: str = ""
    retrieved: list = field(default_factory=list)      # 조회한 semantic/episodic 지식 요약
    observations: list = field(default_factory=list)   # 최근 grounding 관측 요약
    messages: list = field(default_factory=list)       # LangChain 메시지 로그

    def render(self) -> str:
        """작업기억을 planning 프롬프트에 넣을 텍스트 블록으로 직렬화한다."""
        lines = ["[Working memory]",
                 f"- Current goal: {self.goal or '(no clear measurement goal yet)'}"]
        if self.retrieved:
            lines.append("- Retrieved knowledge (memory):")
            lines += [f"    · {s}" for s in self.retrieved[-6:]]
        else:
            lines.append("- Retrieved knowledge (memory): none yet (query with a planning action if needed)")
        if self.observations:
            lines.append("- Recent observations:")
            lines += [f"    · {s}" for s in self.observations[-6:]]
        return "\n".join(lines)

    def absorb(self, name: str, result: dict, action: str) -> None:
        """실행 결과를 요약(retrieved/observations)에 반영한다."""
        if not isinstance(result, dict):
            return
        if action == "retrieval":
            self._absorb_retrieval(name, result)
        elif not result.get("ok", True):
            self.observations.append(f"⚠️ {name} failed: {result.get('error', '')}")
        elif name == "acquire_spectrum":
            self.observations.append(f"Spectrum acquired (max {result.get('max_intensity', 0)} ADU)")
        elif name == "run_grid_scan":
            self.observations.append(
                f"Grid scan: {result.get('n_measured', 0)}/{result.get('n_points', 0)} points measured")
        elif action == "learning":
            self.observations.append(
                f"Memory recorded: {name} → {result.get('sample') or result.get('topic', '')}")
        else:
            self.observations.append(f"{name} executed")

    def _absorb_retrieval(self, name: str, result: dict) -> None:
        hits = result.get("results", [])
        if not hits:
            self.retrieved.append(f"[{name}] {result.get('note', 'no match')}")
            return
        # 세 retrieval 도구가 형태가 다른 항목을 돌려주므로 폴백 체인으로 뽑는다.
        #   search_knowledge_base → {"title", "recommended_params", ...}   (평평)
        #   recall_insights       → {"topic", "insight"}                   (평평)
        #   recall_experiences    → {"sample", "params_used", ...}         (projection 후)
        # 예전에는 최상위에서 sample/params 를 찾았는데 에피소드는 이들이 중첩돼 있어 늘
        # "?" 로 찍혔다 — render() 가 매 propose 맨 위에 싣는 요약이 공란이 되어, 모델이
        # raw JSON 덤프에만 의존했다.
        for h in hits[:3]:
            title = h.get("title") or h.get("sample") or h.get("topic") or "?"
            if h.get("substrate"):
                title += f" on {h['substrate']}"
            params = h.get("recommended_params") or h.get("params_used") or h.get("params")
            extra = f" params {params}" if params else ""
            # 성공/실패와 조건 불일치는 요약 단계에서부터 눈에 띄어야 한다 — 그래야 모델이
            # raw 덤프를 안 읽어도 '따라할 것/피할 것'을 구분한다. 마커는 ASCII 로 — 이
            # 문자열은 프롬프트 본문에 들어가고 콘솔에도 찍힐 수 있다(cp949 인코딩 에러).
            if h.get("is_success") is True:
                extra += " [OK]"
            elif h.get("is_success") is False:
                extra += " [FAILED-avoid]"
            if h.get("condition_warning"):
                extra += " [different-substrate]"
            self.retrieved.append(f"[{name}] {title}{extra}")


# ══════════════════════════════════════════════════════════════════════════════
# 사이클 요소: 후보 분류 / propose / evaluate+select
# ══════════════════════════════════════════════════════════════════════════════

def _label(tc: dict) -> str:
    """tool_call 후보를 사람이 읽는 한 줄로."""
    args = tc.get("args") or {}
    return f"{tc.get('name', '?')}({', '.join(f'{k}={v}' for k, v in list(args.items())[:4])})"


def _action_of(name: str) -> str:
    if name in RETRIEVAL_ACTIONS:
        return "retrieval"
    return "learning" if name in LEARNING_ACTIONS else "grounding"


def _partition(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """제안된 tool_call 들을 (planning 행동, commit 행동)으로 나눈다.

    이 분리가 이 파일의 핵심이다(논문 §4.6): retrieval 은 planning 의 '수단'이지 실행
    대상이 아니다. commit(grounding+learning)이 하나 실행되면 사이클이 닫힌다.
    """
    planning = [tc for tc in candidates if tc.get("name", "") in RETRIEVAL_ACTIONS]
    commit = [tc for tc in candidates if tc.get("name", "") not in RETRIEVAL_ACTIONS]
    return planning, commit


def _plan_progress_note(round_no: int, retrieval_count: int, repeated: bool) -> str:
    """planning 진행 상황을 모델에게 알리는 문구 — 매 propose 마다 새로 계산된다.

    [왜 필요한가]
    모델이 정보 수집만 반복하다 계획 예산을 소진하면 측정 한 번 못 하고 턴이 끝난다.
    하드 넛지 한 번 대신 "지금 이 사이클에서 몇 번째 계획 중인지"를 매번 알려줘, 모델이
    스스로 '이제 실행/보고서로 넘어갈 때'를 판단하게 한다. 영구 히스토리가 아니라 매 호출
    새로 만드는 SystemMessage 에만 실려 세션에 남지 않는다(다음 턴 오염 없음).
    """
    left = MAX_PLANNING_STEPS - round_no
    note = (f"[Planning progress] This is planning (information gathering) round {round_no} in this cycle "
            f"({retrieval_count} consecutive lookups, {left} planning rounds left).")
    if repeated:
        note += (" You are repeating the same lookup as just before - do not repeat the same lookup; "
                 "with the evidence gathered so far, choose one execution action (grounding/learning) "
                 "or write the report.")
    elif round_no >= SOFT_PLAN_LIMIT:
        note += " Information looks sufficient. Now choose one execution action or write the final report."
    if left <= 0:
        note += " This is the last planning round - next you must execute or write the report."
    return note


def _propose(llm_tools, wm: WorkingMemory, plan_note: str = "",
             rlog=None, stage: str = "CoALA propose", step: int = 0) -> AIMessage:
    """Propose — 작업기억(+계획 진행 문구)을 주입해 다음 행동 후보(tool_calls)를 생성한다.

    반환된 AIMessage.tool_calls 가 후보 목록이다(0개면 finish = 최종 답변).
    시스템 프롬프트·작업기억·진행 문구는 세션 히스토리에 남기지 않고 매 호출마다 새로
    붙인다(중복·오염 방지).
    """
    content = SYSTEM_PROMPT + "\n\n" + wm.render()
    # 세션 요약도 매 호출마다 새로 만든다 — 작업기억에 넣으면 낡은 산출물 목록이 누적돼
    # 모델이 지난 상태를 현재로 오인한다.
    session_note = run_store.summary_for_prompt()
    if session_note:
        content += f"\n\n[This session]\n{session_note}\n"
    if plan_note:
        content += "\n\n" + plan_note
    rlog = rlog or reason_log.NULL
    return rlog.invoke([SystemMessage(content=content)] + wm.messages,
                       llm=llm_tools, stage=stage, step=step)


#: 도구 이름 → 모델에게 준 설명. 평가자에게 붙일 근거의 출처이고, propose 가 보는 것과
#: 같은 문장이다(다른 문장을 쓰면 두 단계가 서로 다른 계약을 보게 된다).
_TOOL_DESC: dict[str, str] = {
    t["function"]["name"]: t["function"].get("description", "")
    for t in ALL_TOOLS if isinstance(t, dict) and "function" in t
}


def _eval_stats() -> dict:
    """evaluate 결과 집계용 카운터. 턴 시작 시 ctx 에 하나 만든다.

    [왜 세는가]
    evaluate 의 실패·생략 경로가 셋인데 **전부 첫 후보로 폴백**한다(후보 1개 / 파싱 실패 /
    LLM 예외). 셋 다 사이클을 멈추지 않으므로, 세지 않으면 평가가 실제로 돌았는지 아무도
    모른 채 결과만 남는다. 그 상태에서 "CoALA 가 AILA 보다 낫다/못하다"를 말하면 근거가
    없다 — 평가 단계가 한 번도 개입하지 않은 실행일 수도 있기 때문이다.

    changed 는 '평가가 선택을 바꾼 횟수'다. 폴백은 언제나 0번 후보를 고르므로, 0번이
    아닌 것을 골랐을 때만 evaluate 가 결과에 실제로 기여했다고 말할 수 있다.

    [세는 방식 — 결과 4종은 배타적, retried 는 별개]
      scored / skipped_single / truncated / parse_failed / llm_error 는 평가 1회의 결과라
      서로 겹치지 않는다. 다 더하면 평가 시도 횟수가 된다.
      retried 는 결과가 아니라 사건이다 — 잘려서 think 를 끄고 다시 물은 횟수. 재시도가
      성공하면 결과는 scored 이고 retried 만 올라간다. 그래서 'scored 인데 retried 가
      같이 오른' 상태가 곧 '상한이 빠듯하지만 결론은 건졌다'는 뜻이 된다.
    """
    return {"scored": 0, "skipped_single": 0, "parse_failed": 0, "truncated": 0,
            "llm_error": 0, "changed": 0, "retried": 0}


def _clip(text: str, limit: int) -> str:
    """길이 상한. 잘렸다는 사실을 남겨 '원래 이만큼'인지 '잘린 것'인지 구분되게 한다."""
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _eval_evidence(wm: WorkingMemory, candidates: list[dict]) -> str:
    """평가자에게 줄 근거 — 후보 도구의 계약 + 직전 관측.

    [왜 필요한가 — 2026-08-13 실행 로그]
    평가 프롬프트는 wm.render() 만 받아 239 토큰이었다(같은 시점 propose 는 18,091 토큰).
    그래서 평가자는 도구가 무엇을 하는지도, 방금 무슨 일이 있었는지도 모른 채 점수를 매겼고,
    실제로 이런 추론을 남겼다:

        "Wait, can analyze_microscope_image work without start_camera_stream?
         In many API setups, taking a snapshot implies the camera is active…"

    답은 도구 설명에 그대로 적혀 있다("Streaming must be ON before analyze_microscope_image
    … can get a frame"). 평가자에게 안 보였을 뿐이다. 이 프로젝트의 도구 계약 대신 일반적인
    에이전트 관례로 판단한 셈이고, 그런 점수는 채점이 아니라 추측이다.

    [무엇을 싣는가]
    · 후보에 등장하는 도구의 설명 앞머리(EVAL_DESC_CHARS) — 선행조건·안전 규칙이 거기 있다.
      같은 도구가 여러 후보에 나오면 한 번만 싣는다(인자만 다른 후보들이 흔하다).
    · 직전 관측 EVAL_OBS_COUNT 건 — wm.observations 는 한 줄 요약이라 '방금 무엇이
      실패했는지'가 남지 않는다. 실제 ToolMessage 본문 꼬리를 잘라 싣는다.
    """
    parts = []

    seen: list[str] = []
    lines = []
    for c in candidates:
        name = c.get("name", "")
        if name in seen or name not in _TOOL_DESC:
            continue
        seen.append(name)
        lines.append(f"- {name}: {_clip(_TOOL_DESC[name], EVAL_DESC_CHARS)}")
    if lines:
        parts.append("[What these tools actually do]\n" + "\n".join(lines))

    # 메시지 로그 꼬리에서 도구 결과만 최신순으로 모은 뒤, 읽기 좋게 시간순으로 되돌린다.
    obs = []
    for m in reversed(wm.messages):
        if isinstance(m, ToolMessage):
            obs.append(_clip(m.content, EVAL_OBS_CHARS))
            if len(obs) >= EVAL_OBS_COUNT:
                break
    if obs:
        parts.append("[Most recent observations (newest last)]\n"
                     + "\n".join(f"- {o}" for o in reversed(obs)))

    return "\n\n".join(parts)


def _evaluate_and_select(llm_plain, wm: WorkingMemory, candidates: list[dict],
                         dose: float, rlog=None, stats: dict | None = None) -> tuple[dict, dict]:
    """Evaluate + Select — 실행(commit) 후보가 여럿이면 점수화 후 argmax, 하나면 그대로.

    여기 들어오는 candidates 는 grounding/learning 뿐이다 — retrieval 은 planning 단계에서
    이미 처리되어 이 지점에 오지 않는다.

    후보 1개면 LLM 호출 없이 통과시킨다(논문 §4.6: 단순 상황은 평가 생략). 여럿이면 plain
    LLM 으로 유용성·안전(dose/광손상)·근거성을 0~1 점수화한다. 실패하면 첫 후보로 폴백해
    사이클이 멈추지 않게 하고, 폴백했다는 사실은 stats 에 남긴다.

    Parameters
    ----------
    stats : _eval_stats() 카운터. 없으면 집계만 생략하고 동작은 같다.

    Returns
    -------
    (선택된 tool_call, meta)
        meta = {"scores": [...], "reason": str, "mode": str}
        mode ∈ {"skipped_single", "scored", "truncated", "parse_failed", "llm_error"} —
        어느 경로로 골랐는지다. 폴백 넷이 전부 같은 후보를 고르므로, 이 필드가 없으면
        로그만 보고는 평가가 돈 것과 안 돈 것을 구분할 수 없다.

        truncated 를 parse_failed 와 가르는 이유: 원인과 처방이 다르다. 잘림은
        EVAL_NUM_PREDICT 를 올리면 되고, 파싱 실패는 모델이 JSON 을 안 냈다는 뜻이라
        프롬프트 문제다. 뭉쳐 두면 '점수 JSON 파싱 실패' 한 줄만 보고 상한이 원인인 줄
        모른다 — 실제로 2026-08-13 실행에서 그렇게 놓쳤다.
    """
    rlog = rlog or reason_log.NULL

    def _bump(key: str) -> None:
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1

    if len(candidates) == 1:
        _bump("skipped_single")
        rlog.phase("evaluate", "1 candidate - evaluation skipped (paper 4.6: simple case)",
                   _label(candidates[0]))
        return candidates[0], {"scores": [1.0], "mode": "skipped_single",
                               "reason": "single execution candidate - evaluation skipped"}

    # 후보가 지나치게 많으면 앞에서 자른다 — 평가 입력이 후보 수에 비례해 커지는 유일한
    # 자리다. 잘린 후보는 다음 사이클에 다시 제안될 수 있으므로 정보가 사라지지 않는다.
    dropped = candidates[MAX_EVAL_CANDIDATES:]
    candidates = candidates[:MAX_EVAL_CANDIDATES]
    if dropped:
        rlog.phase("evaluate", f"{len(dropped)} candidate(s) beyond the scoring cap "
                               f"({MAX_EVAL_CANDIDATES}) dropped",
                   " / ".join(_label(c) for c in dropped))

    listing = "\n".join(f"{i}. {_label(c)}" for i, c in enumerate(candidates))
    evidence = _eval_evidence(wm, candidates)
    prompt = (
        "You are the 'execution-action evaluator' of a Raman experiment agent. Looking at the working memory "
        "and execution candidates below, score how valuable each candidate is to execute right now from 0.0 to 1.0. "
        "Consider usefulness (progress toward the goal), safety (photodamage/dose), and groundedness "
        "(consistency with working memory/memory).\n\n"
        f"{wm.render()}\n\n"
        + (evidence + "\n\n" if evidence else "")
        + f"Current cumulative dose {dose:.1f}/{runtime.MAX_DOSE_MJ_PER_TURN} mJ. For laser irradiation "
        "(acquire_spectrum), consider this budget together with photodamage (irreversible).\n\n"
        f"[Execution candidates]\n{listing}\n\n"
        "Exactly one candidate will be executed and the rest are discarded, so judge them as alternatives, "
        "not as steps of a plan. The tool descriptions above are the actual contract of this instrument - "
        "when they state a precondition or a limit, treat it as fact rather than guessing from convention.\n\n"
        "Output only the following JSON (no explanation): "
        '{"scores": [number, ...], "reason": "one sentence on why the highest candidate was chosen"}'
    )
    rlog.phase("evaluate",
               f"Scoring {len(candidates)} candidates (cumulative dose {dose:.1f} mJ considered, "
               f"evidence {len(evidence)} chars)",
               listing)

    def _cut_off(resp) -> bool:
        """출력 상한에 걸려 잘렸는가. 잘리면 JSON 이 안 닫혀 파싱도 실패하는데, 그때
        '파싱 실패'로만 적으면 상한이 원인이라는 것이 로그에 안 남는다."""
        return (getattr(resp, "response_metadata", None) or {}).get("done_reason") == "length"

    scores, reason, mode = None, "", "scored"
    try:
        msg = [HumanMessage(content=prompt)]
        resp = rlog.invoke(msg, llm=llm_plain,
                           stage="CoALA evaluate (evaluator call without tools)")

        # ── 잘렸으면 think 를 끄고 한 번만 다시 묻는다 ────────────────────────
        # think 토큰도 출력 상한을 먹는다. 답이 점수 JSON 한 줄인데 thinking 이 예산을
        # 다 써서 결론만 잘리는 일이 실제로 있었다(2026-08-13: `{"scores": [1.0,` 에서
        # 절단, 2회). 그 상태로 폴백하면 평가가 통째로 없던 일이 된다.
        #
        # 상한을 올려 재시도하지 않고 think 를 끄는 이유: 숙고는 1차에서 이미 끝났고
        # 로그에도 남아 있다. 잘린 것은 결론 한 줄뿐이므로, think 없이 다시 물으면
        # 같은 근거를 보고 결론만 받아 낼 수 있다 — 그리고 1~2초면 끝난다(상한을 올려
        # 재시도하면 thinking 을 처음부터 다시 돌려 8~9초가 더 든다).
        #
        # 2차 점수는 숙고를 안 거쳤으니 1차보다 못하다. 하지만 폴백(무조건 0번 후보)
        # 에는 판단이 아예 없으므로, 못한 판단이라도 있는 쪽이 낫다.
        if _cut_off(resp):
            _bump("retried")     # 재시도했다는 사실은 성공하든 아니든 남긴다 —
                                 # 성공했더라도 '상한이 빠듯하다'는 신호다.
            rlog.phase("evaluate",
                       f"evaluator reply hit the output cap ({EVAL_NUM_PREDICT} tokens) - "
                       f"thinking ate the budget. Retrying once with thinking OFF.")
            resp = rlog.invoke(msg, llm=llm_plain, reasoning=False,
                               stage="CoALA evaluate (retry, thinking OFF)")
            if _cut_off(resp):
                # 2차까지 잘렸으면 think 가 아니라 상한 자체가 모자라다 — 처방을 적는다.
                mode = "truncated"
                rlog.phase("evaluate",
                           f"the retry was cut off too -> falling back to the first candidate. "
                           f"Raise EVAL_NUM_PREDICT (now {EVAL_NUM_PREDICT}); even without "
                           f"thinking the reply does not fit.")

        m = re.search(r"\{.*\}", runtime.text_of(resp), re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            scores = [float(x) for x in data.get("scores", [])][:len(candidates)]
            reason = str(data.get("reason", ""))
    except Exception as e:
        # LLM 자체가 죽은 것과 응답이 이상한 것을 가른다 — 전자는 평가가 아예 없었다는
        # 뜻이라, 같은 폴백이어도 실험 해석이 달라진다.
        scores, mode = None, "llm_error"
        rlog.phase("evaluate", f"evaluator LLM call failed ({type(e).__name__}) "
                               f"-> falling back to the first candidate")

    if not scores or len(scores) != len(candidates):
        if mode == "scored":       # 잘림도 LLM 오류도 아닌데 점수가 안 나왔다 = 형식 문제
            mode = "parse_failed"
            rlog.phase("evaluate", "Score JSON parse failed -> falling back to the first candidate")
        _bump(mode)
        return candidates[0], {"scores": [], "mode": mode,
                               "reason": f"evaluation {mode} - first candidate chosen"}

    best = max(range(len(candidates)), key=lambda i: scores[i])
    _bump("scored")
    if best != 0:
        # 폴백은 언제나 0번을 고른다. 0번이 아닌 것을 골랐을 때만 평가가 결과를 바꿨다.
        _bump("changed")
    rlog.phase("evaluate", "Scores " + ", ".join(
        f"{_label(c)}={s:.2f}" for c, s in zip(candidates, scores)), reason)
    return candidates[best], {"scores": scores, "reason": reason, "mode": "scored"}


# ══════════════════════════════════════════════════════════════════════════════
# Planning stage — retrieval 을 반복 호출해 근거를 쌓고, commit 후보가 나오면 종료
# ══════════════════════════════════════════════════════════════════════════════

def _planning_stage(llm_tools, ctx: dict, wm: WorkingMemory, propose_budget: list,
                    outcome: dict, rlog=None, cycle: int = 0) -> Iterator[dict]:
    """한 사이클의 planning 단계(논문 §4.6 / Figure 4B).

    reasoning(모델의 내재 CoT)·retrieval 로 근거를 쌓다가, 실행(grounding/learning) 후보가
    제안되면 planning 을 끝낸다. retrieval 호출은 사이클을 닫지 않고 working memory 만
    갱신하며, 필요한 만큼 반복(interleave)된다.

    결과는 `outcome` dict 에 기록한다:
      outcome["kind"] ∈ {"commit", "finish", "stuck", "error"}
      outcome["commit"]      = [tool_call, ...]   (kind=="commit" 일 때)
      outcome["commit_text"] = str                (commit 제안에 딸린 모델 텍스트)
      outcome["final_text"]  = str                (kind=="finish" 일 때 최종 보고서)

    propose_budget 은 [남은_propose_예산] 을 공유하는 가변 리스트다(턴 전체 총량 가드).
    """
    prior_sigs: list = []      # 이번 사이클의 조회 시그니처 이력 — 반복 조회 감지용
    rlog = rlog or reason_log.NULL

    for step in range(MAX_PLANNING_STEPS):
        if propose_budget[0] <= 0:
            rlog.phase("plan", f"cycle {cycle} · propose budget exhausted -> stuck")
            outcome["kind"] = "stuck"
            return

        # 직전 조회가 그 이전에도 나왔던 것이면 반복으로 본다.
        repeated = bool(prior_sigs) and prior_sigs[-1] in prior_sigs[:-1]
        plan_note = _plan_progress_note(step + 1, len(prior_sigs), repeated)
        rlog.phase("plan", f"cycle {cycle} · planning round {step + 1}/{MAX_PLANNING_STEPS}"
                           + (" · repeated identical retrieval detected" if repeated else ""),
                   plan_note)
        try:
            ai_msg = _propose(llm_tools, wm, plan_note, rlog=rlog,
                              stage=f"CoALA propose (cycle {cycle}, plan {step + 1})",
                              step=MAX_AGENT_STEPS - propose_budget[0] + 1)
        except Exception as e:
            outcome["kind"] = "error"
            outcome["detail"] = runtime.llm_error_detail(e, "LLM call (propose)")
            return
        propose_budget[0] -= 1

        candidates = list(ai_msg.tool_calls or [])

        # ── 후보 없음 = finish. 모델이 최종 보고서를 낸 것 → 이번 턴 종료 ──────────
        if not candidates:
            rlog.phase("plan", f"cycle {cycle} · no tool candidate -> final report, turn ends")
            wm.messages.append(ai_msg)
            outcome["kind"] = "finish"
            outcome["final_text"] = runtime.text_of(ai_msg).strip() or runtime.EMPTY_REPLY
            return

        planning_actions, commit_actions = _partition(candidates)

        # ── retrieval 이 없고 commit 후보만 있으면: planning 종료 ────────────────
        if not planning_actions:
            rlog.phase("plan", f"cycle {cycle} · retrieval done · {len(commit_actions)} execution "
                               f"candidate(s) -> evaluate/select",
                       " / ".join(_label(c) for c in commit_actions))
            outcome["kind"] = "commit"
            outcome["commit"] = commit_actions
            outcome["commit_text"] = runtime.text_of(ai_msg)
            return

        # ── retrieval 이 있으면: 그것만 실행하고 계속 계획 ──────────────────────
        # 이 응답에 섞여 나온 commit_actions 는 '버린다' — 근거를 더 모은 뒤 다음 propose 에서
        # 재제안하게 해서, 실행은 항상 최신 작업기억으로 결정되게 한다.
        wm.messages.append(AIMessage(content=runtime.text_of(ai_msg), tool_calls=planning_actions))
        prior_sigs.append(tuple(sorted(
            f"{c.get('name')}:{json.dumps(c.get('args') or {}, sort_keys=True, ensure_ascii=False)}"
            for c in planning_actions)))

        rlog.phase("retrieval",
                   f"cycle {cycle} · running {len(planning_actions)} retrieval action(s) "
                   f"(does not close the cycle)",
                   " / ".join(_label(c) for c in planning_actions))
        if commit_actions:
            rlog.phase("retrieval",
                       f"cycle {cycle} · dropped {len(commit_actions)} execution candidate(s) mixed "
                       f"into the same reply (re-proposed on the next propose)",
                       " / ".join(_label(c) for c in commit_actions))

        yield {"type": "phase", "phase": "plan",
               "message": "Information gathering (retrieval): "
                          + " / ".join(_label(c) for c in planning_actions),
               "candidates": [_label(c) for c in planning_actions]}

        for tc in planning_actions:
            yield from _execute_and_observe(ctx, wm, tc, rlog)
        # 같은 사이클 안에서 다시 propose (planning 반복)

    rlog.phase("plan", f"cycle {cycle} · used all {MAX_PLANNING_STEPS} planning rounds without "
                       f"deciding to execute or finish -> stuck")
    outcome["kind"] = "stuck"


def _execute_and_observe(ctx: dict, wm: WorkingMemory, tc: dict, rlog) -> Iterator[dict]:
    """tool_call 하나를 실행하고 관측을 작업기억·메시지 로그에 반영한다."""
    name = tc["name"]
    ex = runtime.execute_tool(ctx, name, dict(tc.get("args") or {}), tc.get("id") or "",
                              extra_handlers=ltm.HANDLERS)
    action = _action_of(name)
    rlog.observed(name, ex["result"], ex["elapsed_ms"], action)
    yield {"type": "tool", "name": name, "args": ex["args"], "result": ex["result"], "action": action}

    wm.messages.append(ToolMessage(
        content=json.dumps(ex["result"], ensure_ascii=False, default=str),
        tool_call_id=ex["tool_call_id"]))
    wm.absorb(name, ex["result"], action)
    if ex["img_b64"]:
        rlog.phase("observe", f"{name} · injected 1 image into the model "
                              f"(base64 {len(ex['img_b64'])} chars, not written to this log)")
        wm.messages.append(runtime.image_message(ex["img_b64"], ex["question"]))


# ══════════════════════════════════════════════════════════════════════════════
# 의사결정 사이클: [planning] → evaluate → select → execute → observe
# ══════════════════════════════════════════════════════════════════════════════

def run_stream(llm_tools, llm_plain, history: list, user_message: str,
               session_id: str = "") -> Iterator[dict]:
    """CoALA 의사결정 사이클 루프(논문 §4.6 / Figure 4B).

    각 사이클:
      1) planning stage — reasoning·retrieval 로 근거를 쌓는다(retrieval 은 사이클을 닫지
         않고 working memory 만 채운다). 실행 후보(grounding/learning)가 나오면 종료.
      2) evaluate + select — 실행 후보가 여럿이면 점수화 후 하나 선택.
      3) execute + observe — 선택된 '하나'만 실행하고 관측을 남긴다.
    그리고 다음 사이클로. 선택된 후보만 담은 AIMessage 를 히스토리에 남겨
    tool_call ↔ ToolMessage 짝을 항상 유효하게 유지한다.

    yield 이벤트:
      {"type": "phase", "phase": str, "message": str}   사이클 단계 진행
      {"type": "tool",  "name": str, "args": dict, "result": dict, "action": str}
      {"type": "error", "detail": str}
      {"type": "final", "text": str, "ctx": dict, "messages": list}
    """
    rlog = reason_log.open_turn(ARCH, user_message, session_id=session_id)
    try:
        if llm_tools is None or llm_plain is None:
            rlog.failed(runtime.LLM_NOT_CONNECTED)
            yield {"type": "error", "detail": runtime.LLM_NOT_CONNECTED}
            return

        ctx = {"dispatch": runtime.get_tool_dispatch(), "dose": 0.0, "session_id": session_id,
               "tool_call_order": [], "learned": False, "goal": user_message.strip(),
               # evaluate 가 실제로 돌았는지의 집계. ctx 에 두는 이유는 턴이 어디서 끝나든
               # (finish/stuck/error/사이클 소진) 같은 자리에 남아 로그·벤치가 읽을 수
               # 있어야 하기 때문이다.
               "eval_stats": _eval_stats()}
        wm = WorkingMemory(goal=user_message.strip(),
                           messages=list(history) + [HumanMessage(content=user_message)])
        propose_budget = [MAX_AGENT_STEPS]   # 턴 전체 propose 총량 가드(가변 공유)

        for n in range(MAX_CYCLES):
            cycle = n + 1
            rlog.phase("cycle", f"────── cycle {cycle} start "
                                f"(propose budget {propose_budget[0]}/{MAX_AGENT_STEPS}, "
                                f"cumulative dose {ctx['dose']:.1f} mJ) ──────")

            # ── 1) PLANNING ──────────────────────────────────────────────────
            outcome: dict = {}
            yield from _planning_stage(llm_tools, ctx, wm, propose_budget, outcome,
                                       rlog=rlog, cycle=cycle)
            kind = outcome.get("kind")

            if kind == "error":
                rlog.failed(outcome.get("detail", "planning failed"), ctx)
                yield {"type": "error", "detail": outcome.get("detail", "planning failed")}
                return
            if kind == "finish":
                rlog.final(outcome["final_text"], ctx)
                yield {"type": "final", "text": outcome["final_text"],
                       "ctx": ctx, "messages": wm.messages}
                return
            if kind != "commit":
                # stuck — 계획만 반복하고 실행/종료를 못 정함. 안전하게 턴을 닫는다.
                stuck = ("Reached the planning-stage budget, ending this turn. "
                         "Please check the progress and request again.")
                rlog.final(stuck, ctx)
                yield {"type": "final", "text": stuck, "ctx": ctx, "messages": wm.messages}
                return

            commit_candidates = outcome["commit"]
            labels = [_label(c) for c in commit_candidates]

            # ── 2) EVALUATE + SELECT ─────────────────────────────────────────
            if len(commit_candidates) > 1:
                yield {"type": "phase", "phase": "evaluate",
                       "message": f"Evaluating {len(commit_candidates)} execution candidates…",
                       "candidates": labels}
            try:
                chosen, meta = _evaluate_and_select(llm_plain, wm, commit_candidates,
                                                    ctx["dose"], rlog=rlog,
                                                    stats=ctx["eval_stats"])
            except Exception as e:
                detail = runtime.llm_error_detail(e, "LLM call (evaluate)")
                rlog.failed(detail, ctx)
                yield {"type": "error", "detail": detail}
                return

            rlog.phase("select", f"cycle {cycle} · selected -> {_label(chosen)}",
                       (meta.get("reason") or "")
                       + (f"\n({len(labels)} candidates: " + " / ".join(labels) + ")"
                          if len(labels) > 1 else ""))
            # select 이벤트에 propose→evaluate→select 전 과정을 실어 벤치마크 로그가
            # 후보/점수/이유/선택을 다 담게 한다.
            yield {"type": "phase", "phase": "select",
                   "message": f"Selected → {_label(chosen)}"
                              + (f"  ({meta['reason']})" if meta.get("reason") else ""),
                   "candidates": labels, "scores": meta.get("scores") or None,
                   "reason": meta.get("reason") or None, "chosen": _label(chosen)}

            # ── 3) EXECUTE + OBSERVE — 선택된 하나만 (나머지는 버린다) ─────────
            wm.messages.append(AIMessage(content=outcome.get("commit_text", ""), tool_calls=[chosen]))
            rlog.executing(chosen["name"], dict(chosen.get("args") or {}))
            yield from _execute_and_observe(ctx, wm, chosen, rlog)
            # 다음 사이클로 (관측을 반영해 다시 planning)

        capped = (f"Stopped after reaching the maximum number of cycles ({MAX_CYCLES}). "
                  "Please check the progress and request again.")
        rlog.final(capped, ctx)
        yield {"type": "final", "text": capped, "ctx": ctx, "messages": wm.messages}
    finally:
        # 벤치 러너가 중단하면 서버가 gen.close() 를 부른다(GeneratorExit).
        rlog.close()


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API — 껍데기는 runtime 이 갖고 있고, 여기서는 사이클만 꽂아 준다
# ══════════════════════════════════════════════════════════════════════════════

_SESSIONS: runtime.SessionStore = {}


def stream_experiment(user_message: str, session_id: str = "") -> Iterator[dict]:
    """이 에이전트를 프론트엔드 SSE 이벤트 제너레이터로 실행한다."""
    return runtime.stream_turn(
        ARCH, _SESSIONS,
        lambda history, sid: run_stream(_get_llm_tools(), _get_llm_plain(),
                                        history, user_message, session_id=sid),
        user_message, session_id,
        interactive_grid_gate=not AUTONOMOUS)   # 자율 정책은 AILA 와 동일해야 한다


def run_experiment(user_message: str, session_id: str = "") -> dict:
    """동기 1회 실행 — 벤치마크/레거시용 (세션 히스토리 없이 매번 새로 시작)."""
    return runtime.run_turn_once(
        ARCH,
        lambda history, sid: run_stream(_get_llm_tools(), _get_llm_plain(),
                                        history, user_message, session_id=sid),
        user_message, session_id)


# ══════════════════════════════════════════════════════════════════════════════
# 자체 점검:  python -m backend.agents.architectures.single_agent_CoALA
#
# 가상 에이전트로 사이클 전체를 돈다 — Ollama 도 장비도 없이.
#   가짜 LLM      대본대로 tool_calls / 점수 JSON 을 돌려준다
#   가짜 디스패치 하드웨어 도구 이름을 받아 그럴듯한 결과 dict 를 돌려준다
# 검사 대상은 '판단의 질'이 아니라 **배선**이다: 후보가 여럿일 때 평가가 실제로 도는가,
# 평가 프롬프트에 근거가 실리는가, 폴백 셋이 구분되어 집계되는가, 사이클당 하나만
# 실행되는가. 판단의 질은 실기 로그로만 볼 수 있다.
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    class _FakeLLM:
        """대본을 순서대로 돌려주는 LLM 대역.

        propose 용은 tool_calls 를, evaluate 용은 텍스트를 낸다. 받은 프롬프트를 전부
        모아 두어(seen) 무엇이 실렸는지 사후에 검사한다 — 이 대역의 진짜 목적이다.
        """

        def __init__(self, script: list):
            self.script, self.seen, self.calls = list(script), [], 0
            self.reasoning_seen = []          # 호출마다 think 를 켰는지 — 재시도 검사용

        def invoke(self, messages, **kw):
            self.calls += 1
            self.reasoning_seen.append(kw.get("reasoning"))
            self.seen.append("\n".join(getattr(m, "content", "") or "" for m in messages))
            item = self.script.pop(0) if self.script else ""
            if isinstance(item, Exception):
                raise item
            if isinstance(item, list):                     # tool_calls 대본
                return AIMessage(content="", tool_calls=[
                    {"name": n, "args": a, "id": f"call_{self.calls}_{i}"}
                    for i, (n, a) in enumerate(item)])
            if isinstance(item, dict):                     # 잘린 응답 대본
                return AIMessage(content=item["text"],
                                 response_metadata={"done_reason": item["done_reason"]})
            return AIMessage(content=str(item))            # 텍스트(최종 보고서 / 점수 JSON)

    # 하드웨어 도구 이름만 쓴다 — FILE_DISPATCH·RUNTIME_DISPATCH 는 가로채기가 먼저라
    # 가짜 디스패치까지 오지 않는다(runtime._dispatch 의 순서).
    _fake_dispatch = {
        "start_camera_stream": lambda a: {"ok": True, "already_streaming": False},
        "acquire_spectrum": lambda a: {"ok": True, "max_intensity": 1294.0,
                                       "laser_power_pct": a.get("power"), "length": 1024},
        "move_stage": lambda a: {"ok": True, "position": {"x": a.get("x"), "y": a.get("y")}},
    }
    runtime.get_tool_dispatch = lambda: _fake_dispatch        # noqa: E731
    run_store.begin_session("selfcheck-coala", ARCH, isolated=True)

    def _drive(propose_script, eval_script):
        """가상 에이전트 한 턴. (이벤트들, ctx, 평가LLM) 을 돌려준다."""
        plan_llm, eval_llm = _FakeLLM(propose_script), _FakeLLM(eval_script)
        events = list(run_stream(plan_llm, eval_llm, [], "measure this sample",
                                 session_id="selfcheck-coala"))
        ctx = next(e["ctx"] for e in events if e["type"] == "final")
        return events, ctx, eval_llm

    # ── 1) 후보 2개 → 평가가 실제로 돌고, 점수가 선택을 바꾼다 ────────────────
    # 점수 [0.2, 0.9] 는 1번을 가리킨다. 폴백은 언제나 0번이므로, 1번이 실행됐다면
    # 그것만으로 '평가가 결과에 개입했다'가 증명된다.
    events, ctx, ev_llm = _drive(
        [[("acquire_spectrum", {"power": 50}), ("acquire_spectrum", {"power": 1})],
         "done."],
        ['{"scores": [0.2, 0.9], "reason": "start low to avoid photodamage"}'])
    st = ctx["eval_stats"]
    assert st["scored"] == 1 and st["changed"] == 1, st
    assert ctx["tool_call_order"] == ["acquire_spectrum"], ctx["tool_call_order"]
    tool_ev = [e for e in events if e["type"] == "tool"]
    assert tool_ev[0]["args"]["power"] == 1, "평가가 고른 후보가 실행되지 않았다"

    # 평가 프롬프트에 근거가 실렸는가 — 이것이 a/b 수정의 핵심이다.
    prompt = ev_llm.seen[0]
    assert "[What these tools actually do]" in prompt
    assert "irradiates" in prompt or "laser" in prompt.lower(), "도구 설명이 안 실렸다"
    assert "alternatives" in prompt, "후보를 대안으로 보라는 지시가 빠졌다"
    # 근거는 상한 안에 있어야 한다(입력이 후보 수에 비례해 터지지 않게).
    assert len(prompt) < 6000, f"평가 프롬프트가 너무 크다: {len(prompt)}자"

    # ── 2) 후보 1개 → LLM 호출 없이 통과(논문 §4.6) ──────────────────────────
    events, ctx, ev_llm = _drive([[("start_camera_stream", {})], "done."], [])
    assert ctx["eval_stats"] == {**_eval_stats(), "skipped_single": 1}, ctx["eval_stats"]
    assert ev_llm.calls == 0, "후보 1개인데 평가 LLM 을 불렀다"

    # ── 3) 점수 JSON 이 깨지면 첫 후보로 폴백하고, 그 사실이 남는다 ──────────
    events, ctx, _ = _drive(
        [[("acquire_spectrum", {"power": 50}), ("acquire_spectrum", {"power": 1})], "done."],
        ["I think the second one is better, honestly."])
    assert ctx["eval_stats"]["parse_failed"] == 1, ctx["eval_stats"]
    assert [e for e in events if e["type"] == "tool"][0]["args"]["power"] == 50

    # ── 3-b) 잘리면 think 를 끄고 한 번 더 물어 점수를 건진다 ────────────────
    # 1차는 상한에 걸려 잘리고, 2차(think OFF)가 성공하는 대본. 폴백으로 끝나면 안 되고
    # 2차 점수대로 골라야 한다 — 여기서는 [0.2, 0.9] 라 1번(power=1)이 실행돼야 한다.
    events, ctx, ev_llm = _drive(
        [[("acquire_spectrum", {"power": 50}), ("acquire_spectrum", {"power": 1})], "done."],
        [{"text": '```json\n{"scores": [1.0,', "done_reason": "length"},
         '{"scores": [0.2, 0.9], "reason": "retry succeeded"}'])
    assert ctx["eval_stats"] == {**_eval_stats(), "scored": 1, "changed": 1, "retried": 1}, \
        ctx["eval_stats"]
    assert ev_llm.calls == 2, "재시도가 일어나지 않았다"
    assert ev_llm.reasoning_seen[1] is False, "재시도인데 thinking 을 안 껐다"
    assert ev_llm.seen[0] == ev_llm.seen[1], "재시도는 같은 프롬프트여야 한다"
    assert [e for e in events if e["type"] == "tool"][0]["args"]["power"] == 1

    # ── 3-c) 재시도까지 잘리면 폴백하고, 그때만 truncated 로 센다 ────────────
    # 원인과 처방이 다르다: 잘림은 EVAL_NUM_PREDICT 를 올리면 되고, 파싱 실패는 모델이
    # JSON 을 안 낸 것이라 프롬프트 문제다. 실기(2026-08-13)에서 뭉쳐 세다가 상한이
    # 원인인 것을 못 봤다. thinking 토큰도 상한에 포함된다는 점이 원인이었다.
    cut = {"text": '```json\n{"scores": [1.0,', "done_reason": "length"}
    events, ctx, ev_llm = _drive(
        [[("acquire_spectrum", {"power": 50}), ("acquire_spectrum", {"power": 1})], "done."],
        [cut, cut])
    assert ctx["eval_stats"] == {**_eval_stats(), "truncated": 1, "retried": 1}, ctx["eval_stats"]
    assert [e for e in events if e["type"] == "tool"][0]["args"]["power"] == 50

    # ── 4) 평가 LLM 이 죽어도 턴은 살고, 파싱 실패와 구분된다 ────────────────
    events, ctx, _ = _drive(
        [[("acquire_spectrum", {"power": 50}), ("acquire_spectrum", {"power": 1})], "done."],
        [RuntimeError("ollama down")])
    assert ctx["eval_stats"] == {**_eval_stats(), "llm_error": 1}, ctx["eval_stats"]
    assert next(e for e in events if e["type"] == "final")["text"] == "done."

    # ── 5) 후보가 상한을 넘으면 앞에서 자른다(평가 입력이 터지지 않게) ────────
    many = [("move_stage", {"x": i, "y": 0}) for i in range(7)]
    events, ctx, ev_llm = _drive(
        [many, "done."], ['{"scores": [0.1, 0.2, 0.3, 0.4], "reason": "last one"}'])
    assert ctx["eval_stats"]["scored"] == 1, ctx["eval_stats"]
    assert ev_llm.seen[0].count("move_stage(x=") == MAX_EVAL_CANDIDATES, "상한이 안 걸렸다"
    assert [e for e in events if e["type"] == "tool"][0]["args"]["x"] == 3

    # ── 6) 사이클당 실행은 언제나 하나 — 나머지 후보는 버려진다 ──────────────
    # (레이저 비가역성의 근거가 되는 불변식이라 여기서 못 박는다.)
    events, ctx, _ = _drive(
        [[("acquire_spectrum", {"power": 1}), ("acquire_spectrum", {"power": 2})],
         [("acquire_spectrum", {"power": 3})],
         "done."],
        ['{"scores": [0.9, 0.1], "reason": "a"}'])
    assert ctx["tool_call_order"] == ["acquire_spectrum", "acquire_spectrum"], ctx["tool_call_order"]

    # ── 7) 프롬프트가 '대안을 내라'고 말하는가 — d1 이 살아 있는지 ────────────
    assert "ALTERNATIVES" in SYSTEM_PROMPT, "CoALA 프롬프트에 대안 제안 지시가 없다"
    assert "Do NOT propose a sequence" in SYSTEM_PROMPT

    print(f"통과: 평가 개입(선택 변경) · 후보1 생략 · 잘림→think OFF 재시도로 점수 회수 · "
          f"폴백 3종 구분(truncated/parse/llm) · 후보 상한 {MAX_EVAL_CANDIDATES} · "
          f"사이클당 1회 실행 · 근거 주입 ({len(prompt)}자 프롬프트, "
          f"출력 상한 {EVAL_NUM_PREDICT} 토큰)")
