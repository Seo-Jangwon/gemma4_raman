# -*- coding: utf-8 -*-
"""벤더 제공 SDK 바인딩. 우리가 경로도 스타일도 정하지 않는다 — 손대지 말 것.

    andor_codes/   Andor CCD (ATMCD DLL) + RamanCalibrator
    TuCam/         TUCSEN 카메라 (TUCam.dll)

둘 다 import 시점에 DLL 을 로드하므로 장비 PC 밖에서는 실패한다. 그래서 호출부는
전부 함수 안에서 지연 import 한다(hardware_manager._init_camera 주석 참고).
"""
