# -*- coding: utf-8 -*-
"""LLM 이 부를 수 있는 것들의 선언부 — 스키마 + 디스패치.

service/ 와의 차이는 '무엇을 하는가'가 아니라 '누가 부르는가'다.
service/ 는 우리 코드가 부르는 함수고, 여기 있는 것은 **모델이 이름으로 골라 부르는** 것이다.

    schema.py       인자 계약 — 함수 시그니처 → OpenAI function 스키마
    result.py       결과 계약 — {"ok": ...} 응답 형식(ok/fail/normalize)
    tools.py        ALL_TOOLS + TOOL_DISPATCH. **스키마와 디스패치의 단일 진입점**
    file_tools.py   list_uploaded_files · open_file · list_session_artifacts (+ run_analysis 가로채기)
    data_tools.py   service 계층 어댑터 — 목록·병합·요약·묶음·분석·웹검색·KB
    bg_tools.py     배경 제거(IPBSA)·스펙트럼 CSV 읽기

앞의 둘은 도구가 아니라 **도구의 계약**이라 아무것도 import 하지 않는다. 덕분에 장비
계층(hw_tools)이든 서비스 계층이든 어디서 불러도 순환이 생기지 않는다.

[장비 도구가 여기 없는 이유]
스테이지·레이저·CCD·측정 도구도 성격은 같지만 hw_tools/ 에 남는다 — Config.ini 와 벤더
SDK 에 묶여 있어서, 옮기면 장비 없는 PC 에서 이 폴더 전체가 import 되지 않는다.

[그래서 tools.py 와 file_tools.py 의 무게가 다르다]
tools.py 는 장비 모듈을 import 하므로 Config.ini 가 없으면 통째로 죽는다. 반면
file_tools.py·data_tools.py 는 하드웨어를 import 하지 않아 그 상황에서도 살아 있고,
runtime._dispatch 가 그 둘을 '하드웨어 가드보다 먼저' 본다. 파일 분석·KB 조회·순수 계산이
장비 유무에 묶이지 않는 것은 그 순서 덕분이다 — 두 모듈에 hw_tools import 를 늘리지 말 것.
"""
