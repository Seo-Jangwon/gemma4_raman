# -*- coding: utf-8 -*-
"""LLM 이 부를 수 있는 것들의 선언부 — 스키마 + 디스패치.

service/ 와의 차이는 '무엇을 하는가'가 아니라 '누가 부르는가'다.
service/ 는 우리 코드가 부르는 함수고, 여기 있는 것은 **모델이 이름으로 골라 부르는** 것이다.
그래서 여기 있는 파일은 전부 OpenAI function 스키마와 이름→핸들러 dict 를 짝으로 갖는다.

    file_tools.py   list_uploaded_files · inspect_file · run_analysis · list_session_artifacts

[장비 도구가 여기 없는 이유]
raman_tools.TOOL_DISPATCH(측정·스테이지·레이저)와 raman_tool_schemas.RAMAN_TOOLS 도
성격은 같지만 hw_tools/ 에 남는다 — Config.ini 와 벤더 SDK 에 묶여 있어서, 옮기면
장비 없는 PC 에서 이 폴더 전체가 import 되지 않는다. 여기 있는 것들은 하드웨어와 무관해서
장비가 죽어 있어도 그대로 돈다(그게 runtime.call_tool 이 하드웨어 가드보다 먼저 보는 이유).
"""
