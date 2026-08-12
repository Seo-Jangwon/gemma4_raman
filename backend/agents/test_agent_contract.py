# -*- coding: utf-8 -*-
"""
에이전트 계약 검사 — 프롬프트/공용코드 분리 때 만든 안전망.

두 가지를 본다:
  ① 공개 API   두 에이전트 모듈이 서버·벤치 러너가 부르는 심볼을 전부 갖고 있는가.
                하나라도 빠지면 /api/agents/health 나 /api/experiment/* 가 런타임에
                AttributeError 로 죽는다 — 그건 실행해 봐야 알 수 있어서 늦다.
  ② 프롬프트    두 에이전트의 도메인 지시가 다시 갈라지지 않았는가. 공통 블록에 있어야
                할 문장이 한쪽에만 있으면 비교 실험의 독립변수가 오케스트레이션 하나가
                아니게 된다(이 리팩터링의 이유 자체).

①은 AST 로 본다 — 에이전트 모듈은 장비 SDK·삭제된 모듈을 import 하므로 개발 PC 에서
import 가 안 된다. ②는 prompts 패키지만 import 하면 되고 그건 순수 문자열이라 어디서든 돈다.

    python backend/agents/test_agent_contract.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_AGENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_AGENTS.parents[1]))

# 서버(web_controller/controllers/agent.py)가 에이전트 모듈에서 직접 읽는 이름.
# 여기를 줄이려면 그쪽 호출부부터 고쳐야 한다.
REQUIRED_COMMON = {
    "ALL_TOOLS",            # /api/agents/health 가 len() 호출
    "OLLAMA_MODEL",         # /api/agents/health 가 보고
    "OLLAMA_HOST",
    "stream_experiment",    # /api/experiment/stream
    "run_experiment",       # /api/experiment/run
    "run_stream",           # 두 진입점(stream/run_experiment)의 공통 심장부
    "SYSTEM_PROMPT",
}
REQUIRED_PER_AGENT = {
    "single_agent_AILA.py":  {"_get_llm"},
    "single_agent_CoALA.py": {"_get_llm_tools", "_get_llm_plain"},
}


def _top_level_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {a.asname or a.name.split(".")[0] for a in node.names}
    return names


def check_public_api() -> list[str]:
    problems = []
    for fname, extra in REQUIRED_PER_AGENT.items():
        have = _top_level_names(_AGENTS / fname)
        missing = (REQUIRED_COMMON | extra) - have
        for m in sorted(missing):
            problems.append(f"{fname}: 공개 심볼 '{m}' 없음")
    return problems


# 아키텍처와 무관한 도메인 지시 — 두 프롬프트에 **모두** 있어야 한다.
# 예전에 실제로 한쪽에만 있던 것들이라 그대로 회귀 검사가 된다.
SHARED_SENTENCES = [
    "Instrument-status questions",                     # 예전엔 AILA 에만
    "whether the connected ones actually respond",     # 예전엔 AILA 에만
    "A missing camera does not block a stage move",    # 예전엔 AILA 에만
    "Record what happened",                            # 예전엔 CoALA 에서 빠짐
    "not by itself a reason to turn on the laser",     # 예전엔 CoALA 에만
    "answer immediately in English",                   # 인사말 처리
    "Stage coordinate units: mm",
    "you cannot see your own plots",
]
# 반대로, 상대 쪽으로 새면 안 되는 아키텍처 전용 문구.
COALA_ONLY = ["[Decision cycle", "[Memory structure]", "planning action"]


def check_prompt_parity() -> list[str]:
    from backend.agents.prompts import build_system_prompt

    # 공백을 정규화해서 본다 — 줄바꿈 위치는 프롬프트 의미에 아무 영향이 없으므로,
    # 문장이 어디서 접혔는지 때문에 검사가 실패해선 안 된다.
    def flat(s: str) -> str:
        return " ".join(s.split())

    react = flat(build_system_prompt("ReAct"))
    coala = flat(build_system_prompt("CoALA"))

    problems = []
    for s in SHARED_SENTENCES:
        if s not in react:
            problems.append(f"ReAct 프롬프트에 공통 문장 없음: {s!r}")
        if s not in coala:
            problems.append(f"CoALA 프롬프트에 공통 문장 없음: {s!r}")
    for s in COALA_ONLY:
        if s in react:
            problems.append(f"CoALA 전용 문구가 ReAct 로 샘: {s!r}")
        if s not in coala:
            problems.append(f"CoALA 프롬프트에 전용 문구 없음: {s!r}")
    if "{" in react or "{" in coala:
        problems.append("미치환 슬롯이 프롬프트에 남음")

    # 토글이 실제로 먹는지 (예전에는 문자열 치환이라 조용히 무력화될 수 있었다)
    if "recall_experiences" in flat(build_system_prompt("CoALA", episodic=False)):
        problems.append("RAMAN_EPISODIC_MEMORY=0 인데 episodic 지시문이 남음")
    interactive = flat(build_system_prompt("ReAct", autonomous=False))
    if "[Autonomy" in interactive or "ask the user first" not in interactive:
        problems.append("RAMAN_AUTONOMOUS=0 인데 대화 모드로 안 바뀜")
    return problems


def main() -> int:
    problems = check_public_api() + check_prompt_parity()
    for p in problems:
        print(f"  [실패] {p}")
    if problems:
        print(f"\n실패 {len(problems)}건")
        return 1
    print(f"통과: 공개 API {len(REQUIRED_COMMON)}종 + 공통 문장 {len(SHARED_SENTENCES)}개 "
          f"+ 전용 문구 {len(COALA_ONLY)}개 + 토글 2종")
    return 0


if __name__ == "__main__":
    sys.exit(main())
