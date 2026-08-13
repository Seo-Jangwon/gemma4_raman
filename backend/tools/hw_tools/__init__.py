# -*- coding: utf-8 -*-
"""장비 계층. Config.ini 와 벤더 SDK(Andor·TUCam)를 읽으므로, 장비 PC 가 아니면
이 아래 대부분이 import 자체에 실패한다 — 그게 정상이고, 호출부는 그 경우를 다룬다.

    config.py           Config.ini 파싱 (좌표 원점·카메라 해상도 등)
    hardware_manager.py 장비 4종의 연결 수명주기. get_manager() 싱글턴.
    hw_tools/           도구 계층 — 에이전트가 실제로 부르는 함수들
    SDKs/               벤더 제공 코드 (우리가 경로를 정하지 않는다)
"""
