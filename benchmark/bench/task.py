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
