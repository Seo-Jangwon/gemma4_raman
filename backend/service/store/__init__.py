# -*- coding: utf-8 -*-
"""디스크에 무엇을 어디에 남기는가 — 저장소 3종.

    run_store        세션 산출물 인덱스   data/runs/<세션>/
    spectrum_store   측정 결과·플롯       data/results/<날짜>/<세션>/
    upload_store     사용자 첨부 파일     data/uploads/<날짜>/

셋 다 세션 라벨로 묶인다(발급처는 run_store 하나). 그래서 "이번 문항에서 나온 것"을
셋 다에서 같은 이름으로 찾을 수 있다.

[DATA_ROOT 가 여기 있는 이유]
예전에는 세 파일이 각자 __file__ 로 프로젝트 루트를 되짚었다(`parents[2]`,
`parent.parent`). 파일 위치가 한 단계라도 바뀌면 세 곳이 각각 다른 곳을 가리키는데,
에러 없이 '엉뚱한 폴더에 저장'만 되는 종류라 발견이 늦다. 되짚는 계산은 여기 한 번만 한다.
"""
from __future__ import annotations

from pathlib import Path

# backend/service/store/__init__.py → parents[3] = 프로젝트 루트(gemma4_raman)
DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
