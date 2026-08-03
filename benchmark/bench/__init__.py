# -*- coding: utf-8 -*-
"""라만 에이전트 벤치마크 — 문항 하나 = 파일 하나.

[구조]  전부 benchmark/ 아래에 있다
    tasks/      T001.py ... N15.py — 문항 143개. 한 문항이 정의·세팅·채점을 다 갖는다
    run_all.py  전부 실행하고 성적 취합
    bench/      그 문항 파일들이 쓰는 공용 부품
      task.py     Task  — 문항의 정의(문제·배점·입력파일)
      client.py   Bench — 장비를 쥔 서버에 시키는 창구 / Run — 실행 기록
      check.py    chk   — 판정 원자. 각 판정은 Check 하나를 돌려준다
      spectra.py        저장 스펙트럼 읽기 + 규약 재계산(SNR·피크·코사인)
      report.py         평가.json 쓰기 / 전체 취합
      tools.py          도구 이름·인자 표(오탈자 검사용)
    inputs/ gt/ generate/   문항이 쓰는 데이터 자산

[문항 파일이 지켜야 하는 계약]
    TASK               Task 인스턴스 (필수)
    setup(b)           측정 전에 만들어야 하는 상태. 없으면 생략 가능
    evaluate(b, run)   -> list[Check]. 이 목록이 그대로 점수가 된다 (필수)

[왜 이렇게 바꿨나 — 2026-08-03]
예전 구조는 한 문항의 정답이 다섯 파일에 흩어져 있었다: xlsx 의 산문, gt/manifest.json 의
값, grader/specs.py 의 규격, grader/posthoc.py 의 재계산 함수, grader/grade.py 의 분기.
T062 하나를 이해하려면 그 다섯 곳을 다 열어야 했고, 고칠 때도 다섯 곳이 어긋날 수 있었다.
지금은 tasks/T062.py 한 파일만 읽으면 된다.
"""
from .check import Check, chk
from .client import Bench
from .task import Task

__all__ = ["Task", "Bench", "Check", "chk"]
