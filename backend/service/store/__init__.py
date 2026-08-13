# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

from pathlib import Path

# 현재 파일(store/__init__.py) 기준으로 3단계 상위 폴더가 프로젝트 루트(gemma4_raman)입니다.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# 프로젝트 루트 아래의 'data' 폴더를 DATA_ROOT로 지정합니다.
DATA_ROOT = _PROJECT_ROOT / "data"

"""순수 계산·규약 모듈. 장비 SDK 도 Config.ini 도 건드리지 않는다.

여기 있는 것들의 공통점은 **여러 계층이 같은 답을 내야 해서** 사본을 둘 수 없다는 것이다.
그래서 위치가 util 이다 — 하드웨어가 없는 PC 에서도, 분석용 별도 프로세스에서도 import 된다.

    safety_limits   조사량 상한과 계산식   (에이전트 층 · 툴 층이 공유)
    spectro_math    IPBSA 배경 제거        (도구 · 분석 샌드박스가 공유)
    tool_slim       도구 결과 요약 규칙    (AILA · CoALA 가 공유)
    detail_log      턴 단위 JSON 로그      (AILA · CoALA 가 공유)
"""
