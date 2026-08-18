# -*- coding: utf-8 -*-
"""가상 장비 계층 — hao/ 의 실물 드라이버 자리에 꽂는 대역 4종.

`RAMAN_VIRTUAL_HW=1` 이면 hardware_manager 의 _init_* 네 곳과 controllers/hardware.py 두
곳이 실물 클래스 대신 여기 것을 만든다. 그 여섯 곳 말고는 아무도 이 패키지를 모른다.

    scene.py           시료 이미지 한 장과 그 좌표계 (카메라·CCD 가 공유)
    virtual_stage.py   VirtualStage   ← TangoController
    virtual_laser.py   VirtualLaser   ← LaserController
    virtual_camera.py  VirtualCamera  ← StreamingTUCam
    virtual_ccd.py     VirtualCCD     ← AndorCCD

[무엇을 위한 것인가]
장비 없는 PC 에서 **도구 계층을 실제로 실행**하기 위한 것이다. backend/test/fakes.py 는
runtime.get_tool_dispatch 를 통째로 갈아 끼워 도구 함수 자체가 안 돈다 — 락·범위 검증·
조사량 가드·저장·이벤트가 전부 건너뛰어진다. 이쪽은 한 층 아래(장비 핸들)를 바꾸므로
그 로직이 전부 실행된다. 목적이 달라 둘은 공존한다.

[한계 — 알고 쓸 것]
  · Z 축이 없다. 시료가 이미지 한 장(2D)이라 존재하지 않는다.
    run_autofocus 는 VirtualStage.has_z=False 를 보고 스스로 거절한다.
    run_grid_scan 은 autofocus='none' 으로만 돈다.
  · 스펙트럼 내용이 아직 없다. virtual_ccd._synthesize() 가 평탄한 신호를 만든다.
    색 → 피크는 그 함수 하나에 넣으면 된다(다음 단계).
  · reconnect_hardware 는 돌지만 실제로 해제할 자원이 없다. 성공으로 끝난다.
  · set_camera_auto_exposure 는 실물과 달리 소프트웨어 보정이다(목표 평균 밝기).

[상속하지 않는 이유]
실물 클래스를 상속하면 벤더 SDK(Tango DLL·TUCam·pyAndorSDK2)를 import 하게 되어 목적
자체가 무너진다. hw_core.init_hardware() 는 타입 검사를 하지 않으므로 이름만 맞으면 된다.
계약은 '어떤 메서드·속성을 갖는가'이고, 그 목록은 각 파일 머리말에 적혀 있다.
"""
from backend.tools.hw_tools.hao_vertual.scene import VirtualScene, get_scene
from backend.tools.hw_tools.hao_vertual.virtual_camera import VirtualCamera
from backend.tools.hw_tools.hao_vertual.virtual_ccd import VirtualCCD
from backend.tools.hw_tools.hao_vertual.virtual_laser import VirtualLaser
from backend.tools.hw_tools.hao_vertual.virtual_stage import VirtualStage

__all__ = ["VirtualScene", "get_scene", "VirtualStage", "VirtualLaser",
           "VirtualCamera", "VirtualCCD"]
