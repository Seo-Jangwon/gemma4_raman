# -*- coding: utf-8 -*-
"""TUCSEN 카메라 SDK 바인딩(ctypes).

[DLL 을 어디서 찾는가]
TUCam.py 는 **자기 파일 위치 기준**으로 DLL 을 연다:

    <이 폴더>/lib/x64/TUCam.dll

lib/ 는 .gitignore 대상이라 저장소에 없다. 장비 PC 에서는 이 폴더 아래에 있어야 한다.
(예전에는 backend/TuCam/ 에 있었다 — 사본을 이쪽으로 일원화하면서 lib/ 도 따라와야 한다.)
"""
