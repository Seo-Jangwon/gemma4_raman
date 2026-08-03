# -*- coding: utf-8 -*-
"""Task — 문항의 정의부.

문항 파일 맨 위에 이것 하나만 보면 '무엇을 묻는 문항인지'가 다 나오게 한다.
채점 로직은 같은 파일 아래쪽 evaluate() 에 있다.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Task:
    """한 문항.

    id       "T062" / "N01"
    score    배점(총점 가중치). xlsx '배점' 열과 같아야 한다
    axis     역량축. 축별 집계에 쓴다
    prompt   에이전트에게 그대로 보내는 문장
    criteria 이 문항을 무엇으로 맞다고 보는지 — 한 줄 규칙. 결과 파일에 그대로 실린다.
             영어로 쓴다: 결과 JSON 은 사람이 아니라 도구·외부 협업자도 읽는다.
    inputs   업로드해야 하는 입력 파일 이름들(benchmark 입력 폴더 기준)

    mode     "live"        장비를 실제로 조작하게 하고 그 결과를 채점한다(기본)
             "hypothetical" 장비를 건드리지 말고 '어떻게 하겠는가'만 답하게 한다.
                            위험한 상태를 안전하게 재현할 수 없는 문항에 쓴다.

    windows  이 문항의 채점이 성립하려면 파수축이 덮어야 하는 구간
             [(이름, 하한, 상한, 최소 점 수), ...]
             축은 장비 Config.ini 가 정하므로, 설정이 어긋나면 정답을 내도 0점이 된다.
             그 경우를 '오답'이 아니라 '채점 불가'로 빼기 위해 문항이 스스로 밝힌다.

    needs    사람이 미리 해 두면 좋은 것(선택). 안 해도 채점은 성립해야 한다.

    answer_keys
             에이전트가 답 JSON 블록에 담아야 할 키와 그 모양. [(키, 설명), ...]
             client 가 프롬프트 끝의 출력 규약에 그대로 박아 보낸다.

             [왜 필요한가 — 2026-08-03]
             출력 규약은 "Use the exact names the task asked for" 라고 하는데, 정작
             문항이 이름을 대지 않았다. 그래서 에이전트가 지은 이름과 evaluate() 가
             찾는 이름이 갈렸다. 실제로 T044 는 정답 [1001, 1602, 1031] 을 내고도
             키가 highest_intensity_peaks 라 0 점, T126 은 물질 5 개를 다 맞히고도
             파일명을 키로 써서 0 점이었다. 채점기가 못 읽은 것을 못 맞힌 것으로
             기록하는 셈이라, 이름을 문항이 먼저 밝힌다.

             모양까지 적는다. T043 은 키가 맞았는데도
                 [{"position": 620.0, "intensity": 180.6}, ...]
             로 내서 0/7 이었다 — 채점기가 기다린 건 [620.0, 793.0, ...] 였다.
             그래서 설명에 "list of numbers" 같은 모양을 반드시 넣는다.

             evaluate() 가 읽는 키와 여기 선언이 어긋나면 그 문항은 조용히 0 점이
             된다. run_all.py --check 가 둘을 대조해 실행 전에 잡는다.
    """
    id: str
    score: float
    axis: str
    prompt: str
    criteria: str = ""
    inputs: list[str] = field(default_factory=list)
    mode: str = "live"
    windows: list[tuple] = field(default_factory=list)
    needs: str = ""
    answer_keys: list[tuple] = field(default_factory=list)

    # 그리드 승인 인터록을 이 문항에서 강제할지.
    #
    # 기본 False. 켜면 preview→턴 종료→사람 승인 경로가 필요한데 벤치에는 사람이 없어
    # 격자 스캔이 전부 거부되고 실행이 끊긴다. 그런데 통째로 꺼 두면 '같은 턴 실행이
    # 막히는가'(N01)를 물을 수 없고, 인터록을 우회해 시료에 빔을 쏜 실행이 안전 문항에서
    # 만점을 받는다. 그래서 '거부가 정답'인 문항만 켠다 — 그 문항은 승인 턴이 필요 없다.
    enforce_grid_gate: bool = False

    def __post_init__(self):
        if self.mode not in ("live", "hypothetical"):
            raise ValueError(f"{self.id}: mode 는 live 또는 hypothetical 이어야 합니다: {self.mode}")
        if not self.prompt.strip():
            raise ValueError(f"{self.id}: prompt 가 비어 있습니다")
        if self.score <= 0:
            raise ValueError(f"{self.id}: 배점이 0 이하입니다")
        # 결과 파일은 '그것만 보고 무엇을 틀렸는지 알 수 있어야' 한다. 정답 기준이 비면
        # 판정 이름만 남아 근거 없는 감점처럼 보인다.
        if not self.criteria.strip():
            raise ValueError(f"{self.id}: criteria 가 비어 있습니다")
        # 선언이 (키, 설명) 두 칸이 아니면 렌더링에서 조용히 깨진다. 여기서 잡는다.
        for i, item in enumerate(self.answer_keys):
            if not (isinstance(item, (tuple, list)) and len(item) == 2
                    and all(isinstance(s, str) and s.strip() for s in item)):
                raise ValueError(f"{self.id}: answer_keys[{i}] 는 (키, 설명) 두 문자열이어야 "
                                 f"합니다: {item!r}")

    def answer_contract(self) -> str:
        """답 JSON 블록의 키·모양을 에이전트에게 알리는 문장. 없으면 빈 문자열."""
        if not self.answer_keys:
            return ""
        lines = "\n".join(f'  "{k}": {desc}' for k, desc in self.answer_keys)
        # 두 군데를 조심해서 쓴다.
        #   '중첩 금지' 라고 쓰면 안 된다 — T113·T121 처럼 객체 목록을 요구하는 문항이
        #   있어 규약과 키 설명이 정면으로 어긋난다. 모양은 키 설명이 정한다.
        #   '이 키만' 이라고 쓰면 안 된다 — 여분 키는 아무 판정도 보지 않으므로 해가
        #   없는데, 금지해 두면 설명을 덧붙인 답이 규약 위반이 된다.
        return ("\nThe JSON block must include these keys, spelled exactly like this, "
                "with the shape described for each:\n" + lines + "\n")
