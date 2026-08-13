# -*- coding: utf-8 -*-
"""장비 도구 계층 — 에이전트가 부르는 '장비를 만지는' 함수들.

    hw_core.py       장비 핸들 4개·직렬화 락·세션 상태·공용 검사 (도구가 아니다)
    stage_tools.py   스테이지 이동·속도
    laser_tools.py   레이저 발사·파워(ND 필터)·가이드빔
    ccd_tools.py     CCD 설정 조회·변경
    camera_tools.py  카메라 스트리밍·캡처·픽셀→좌표 이동·오토포커스
    acquire_tools.py 스펙트럼 수집·측정점 기록·격자 매핑 (레이저가 실제로 나가는 것들)
    system_tools.py  연결 상태 진단·재연결

    optics_map.py    픽셀 ↔ 스테이지 좌표 변환 (단일 출처)
    vision.py        프레임 전처리·가이드빔 스팟 검출
    USE_*.py         장비별 저수준 드라이버 래퍼

스키마와 디스패치 표는 여기가 아니라 backend/tools/tools.py 한 곳에서 조립된다.
이 폴더의 모듈은 함수만 제공하고, '모델에게 무엇을 보여줄지'는 모른다.

[hw_core 를 거쳐 핸들을 읽어야 하는 이유]
_stage/_laser/_ccd/_camera 는 init_hardware() 가 global 로 재바인딩한다. 다른 모듈이
`from hw_core import _stage` 로 받으면 그 시점의 None 을 영원히 붙들어, 장비를 연결해도
모든 도구가 "not connected" 를 돌려준다. 그래서 도구 모듈은 `hw_core as _hw` 로 모듈을
받아 `_hw._stage` 로 읽는다 — hw_core 의 자체 점검이 이 계약을 실제로 실행해 확인한다.
"""
