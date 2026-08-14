# -*- coding: utf-8 -*-
"""구조적 공평성 — 두 아키텍처가 **같은 조건**을 받고 있는가. (Ollama·장비 불필요)

이 실험의 독립변수는 오케스트레이션 하나여야 한다. 그런데 두 에이전트는 서로를 import
하지 않으므로(그 자체가 설계 원칙이다) 공유해야 할 것이 조용히 갈라지기 쉽다 — 실제로
예전에 프롬프트의 도메인 지시가 한쪽에만 있었고, 모델명이 일곱 군데에 복사돼 있었다.
그런 갈라짐은 에러를 내지 않는다. 그냥 실험 결과를 못 믿게 만들 뿐이다.

여기서 보는 것:
  ① 공개 API      서버가 두 모듈에서 같은 이름을 읽는가
  ② 프롬프트      아키텍처와 무관한 도메인 지시가 양쪽에 다 있는가
  ③ 도구          같은 도구를 보는가. 차이가 있다면 그것이 '설명된 차이'인가
  ④ 단일 출처     모델·조사량·관측축약·배관을 같은 객체에서 읽는가
  ⑤ 독립성        서로를 import 하지 않는가

행동(같은 대본에 같이 반응하는가)은 test_scenarios.py 가 본다.

    python -m backend.test.test_parity
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_ARCH_DIR = Path(__file__).resolve().parents[1] / "agents" / "architectures"
_AILA_PY = _ARCH_DIR / "single_agent_AILA.py"
_COALA_PY = _ARCH_DIR / "single_agent_CoALA.py"

#: 서버(web_controller/controllers/agent.py)와 러너가 에이전트 모듈에서 직접 읽는 이름.
#: 하나라도 빠지면 /api/agents/health 나 /api/experiment/* 가 런타임 AttributeError 로
#: 죽는다 — 실행해 봐야 아는 종류라 늦다. 여기를 줄이려면 그쪽 호출부부터 고쳐야 한다.
REQUIRED_COMMON = {
    "ALL_TOOLS", "OLLAMA_MODEL", "OLLAMA_HOST",
    "stream_experiment", "run_experiment", "run_stream", "SYSTEM_PROMPT",
}
REQUIRED_PER_AGENT = {
    _AILA_PY: {"_get_llm"},
    _COALA_PY: {"_get_llm_tools", "_get_llm_plain"},
}

#: 아키텍처와 무관한 도메인 지시 — 두 프롬프트에 **모두** 있어야 한다.
#: 전부 예전에 실제로 한쪽에만 있던 문장이라, 목록 자체가 회귀 검사다.
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

#: CoALA 에만 있어도 되는 도구 — 장기기억은 CoALA 아키텍처의 정의 자체다(논문 §4.5).
#: 이 집합 밖의 차이가 생기면 '설명 안 되는 능력 차이'이므로 실패시킨다.
COALA_ONLY_TOOLS = {"recall_experiences", "recall_insights",
                    "record_experience", "record_insight"}


def _norm(text: str) -> str:
    """공백을 하나로 눌러 준다. 줄바꿈 위치가 바뀐 것은 갈라짐이 아니다 —
    실제로 프롬프트를 다시 감싸면서 'A missing camera does not block a stage\\n move'
    가 되어, 문자열 그대로 비교하던 옛 검사가 헛 실패했다."""
    return re.sub(r"\s+", " ", text)


def _top_level_names(path: Path) -> set[str]:
    """모듈의 최상위 이름들. **AST 로 읽는다 — import 하지 않는다.**

    에이전트 모듈은 장비 SDK 를 끌고 들어올 수 있어 개발 PC 에서 import 가 실패할 수
    있다. 그러면 이 검사가 '갈라짐'이 아니라 '환경'을 이유로 못 도는데, 안 도는 검사는
    없는 검사다.
    """
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


def _imported_modules(path: Path) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


# ══════════════════════════════════════════════════════════════════════════════
# ① 공개 API
# ══════════════════════════════════════════════════════════════════════════════

def check_public_api() -> list[str]:
    bad = []
    for path, extra in REQUIRED_PER_AGENT.items():
        missing = (REQUIRED_COMMON | extra) - _top_level_names(path)
        bad += [f"{path.name}: 공개 심볼 '{m}' 없음" for m in sorted(missing)]
    return bad


# ══════════════════════════════════════════════════════════════════════════════
# ② 프롬프트
# ══════════════════════════════════════════════════════════════════════════════

def check_prompt_parity() -> list[str]:
    from backend.agents.prompts import build_system_prompt
    aila, coala = _norm(build_system_prompt("ReAct")), _norm(build_system_prompt("CoALA"))
    bad = []
    for s in SHARED_SENTENCES:
        n = _norm(s)
        where = [k for k, p in (("AILA", aila), ("CoALA", coala)) if n not in p]
        if where:
            bad.append(f"공유 도메인 지시가 {'/'.join(where)} 에 없음: {s!r}")
    # 아키텍처 전용 문구가 반대편에 새지 않았는가 — 방향이 반대인 같은 병이다.
    # CoALA 쪽은 planning/execution 분리, ReAct 쪽은 '검증자 없는 단독 판단'이 표지다.
    for tag, needle, must_be_in, must_not in (
            ("CoALA 사이클", "Planning actions (information gathering)", coala, aila),
            ("CoALA 기억", "long-term memory", coala, aila),
            ("ReAct", "There is no separate validator or assistant to check you", aila, coala)):
        if _norm(needle) in must_not:
            bad.append(f"{tag} 전용 지시가 반대편 프롬프트에 샜음: {needle!r}")
        if _norm(needle) not in must_be_in:
            bad.append(f"{tag} 전용 지시가 자기 프롬프트에서 사라짐: {needle!r}")
    return bad


# ══════════════════════════════════════════════════════════════════════════════
# ③ 도구
# ══════════════════════════════════════════════════════════════════════════════

def check_tools() -> list[str]:
    from backend.agents.architectures import single_agent_AILA as A
    from backend.agents.architectures import single_agent_CoALA as C
    from backend.tools.tools import BASE_TOOLS

    bad = []
    a_names = {t["function"]["name"] for t in A.ALL_TOOLS}
    c_names = {t["function"]["name"] for t in C.ALL_TOOLS}

    # 하드웨어·파일 도구는 '같은 리스트 객체'를 공유해야 한다. 사본이면 한쪽만 늘어난다.
    base = {t["function"]["name"] for t in BASE_TOOLS}
    for tag, names in (("AILA", a_names), ("CoALA", c_names)):
        if not base <= names:
            bad.append(f"{tag}: BASE_TOOLS 중 빠진 것 {sorted(base - names)}")

    if a_names - c_names:
        bad.append(f"AILA 에만 있는 도구 {sorted(a_names - c_names)} — 설명되지 않는 능력 차이")
    unexplained = (c_names - a_names) - COALA_ONLY_TOOLS
    if unexplained:
        bad.append(f"CoALA 에만 있는데 장기기억이 아닌 도구 {sorted(unexplained)}")

    # KB 검색은 구현이 같고 설명만 다르다(CoALA 는 '사이클을 닫지 않는다'가 붙어야 한다).
    # 이름이나 인자까지 갈라지면 두 에이전트가 다른 도구를 쓰는 셈이 된다.
    ka = next(t for t in A.ALL_TOOLS if t["function"]["name"] == "search_knowledge_base")
    kc = next(t for t in C.ALL_TOOLS if t["function"]["name"] == "search_knowledge_base")
    if ka["function"]["parameters"] != kc["function"]["parameters"]:
        bad.append("search_knowledge_base 의 인자 스키마가 두 에이전트에서 다름")
    if ka["function"]["description"] == kc["function"]["description"]:
        bad.append("search_knowledge_base 설명이 같음 — CoALA 용 문구(planning 표시)가 사라졌다")
    return bad


# ══════════════════════════════════════════════════════════════════════════════
# ④ 단일 출처
# ══════════════════════════════════════════════════════════════════════════════

def check_single_sources() -> list[str]:
    """두 모듈이 같은 **객체**를 보고 있는가.

    값이 같은지가 아니라 출처가 하나인지를 본다. 값 비교로는 '지금은 우연히 같은' 사본을
    못 잡는다 — 모델명이 일곱 군데 복사돼 있던 시절에도 값은 한동안 같았다.
    """
    from backend.agents.architectures import single_agent_AILA as A
    from backend.agents.architectures import single_agent_CoALA as C
    from backend import llm_config
    from backend.agents.runtime import runtime

    bad = []
    for name in ("OLLAMA_MODEL", "OLLAMA_HOST"):
        vals = {getattr(A, name), getattr(C, name), getattr(llm_config, name)}
        if len(vals) != 1:
            bad.append(f"{name} 이 갈라짐: {vals}")

    # 배관은 공용 모듈 하나여야 한다. 두 파일이 각자 복사해 두면 조사량 회로차단기나
    # 관측 축약이 한쪽에만 적용된다 — 그건 실험이 아니라 두 개의 다른 실험이다.
    for mod, tag in ((A, "AILA"), (C, "CoALA")):
        if mod.runtime is not runtime:
            bad.append(f"{tag} 가 다른 runtime 을 보고 있음")

    # 조사량 상한은 runtime 이 safety_limits 에서 읽는 값 하나뿐이어야 한다.
    from backend.service.safety import safety_limits
    if runtime.MAX_DOSE_MJ_PER_TURN != safety_limits.MAX_DOSE_MJ_PER_TURN:
        bad.append("조사량 상한이 runtime 과 safety_limits 에서 다름")
    return bad


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 독립성
# ══════════════════════════════════════════════════════════════════════════════

def check_independence() -> list[str]:
    """두 에이전트가 서로를 import 하지 않는가.

    한쪽이 다른 쪽을 끌어다 쓰면 '두 아키텍처'가 아니라 '한 아키텍처와 그 변형'이 된다.
    공유해야 할 것은 서로가 아니라 공통 상위 모듈(prompts/runtime/tools)이다.
    """
    bad = []
    for path, forbidden in ((_AILA_PY, "single_agent_CoALA"), (_COALA_PY, "single_agent_AILA")):
        if any(forbidden in m for m in _imported_modules(path)):
            bad.append(f"{path.name} 가 {forbidden} 를 import 함")
    return bad


# ══════════════════════════════════════════════════════════════════════════════

_CHECKS = [
    ("공개 API", check_public_api),
    ("프롬프트 공유 지시", check_prompt_parity),
    ("도구 집합", check_tools),
    ("단일 출처", check_single_sources),
    ("아키텍처 독립성", check_independence),
]


def main() -> int:
    problems = []
    for tag, fn in _CHECKS:
        found = fn()
        print(f"  {'FAIL' if found else 'ok  '}  {tag}"
              + (f"  ({len(found)}건)" if found else ""))
        problems += [f"[{tag}] {p}" for p in found]

    if problems:
        print(f"\n실패 {len(problems)}건 — 두 아키텍처의 조건이 갈라졌다:")
        for p in problems:
            print("   ·", p)
        return 1
    print(f"\n통과: {len(_CHECKS)}개 항목 · 공유 지시 {len(SHARED_SENTENCES)}개 · "
          f"두 아키텍처는 오케스트레이션 말고 같은 조건이다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
