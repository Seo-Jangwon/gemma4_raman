# -*- coding: utf-8 -*-
"""도메인 서비스 계층 — 웹 컨트롤러와 에이전트가 함께 쓰는 '일하는 코드'.

    store/       저장소 3종 (run_store · spectrum_store · upload_store)
    knowledge/   지식베이스 (search · ingest)
    search/      외부 웹 검색
    analysis_sandbox  생성 코드 실행용 격리 프로세스 (run_analysis 도구의 실체)

여기서는 위(web_controller · agents)를 import 하지 않는다. 화살표는 항상 아래로만
간다 — controller → service → util. 설정(llm_config)과 순수 계산(util)만이 더 아래다.
"""
