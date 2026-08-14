# -*- coding: utf-8 -*-
"""AILA ↔ CoALA 공평성 검사.

이 실험의 주장은 "오케스트레이션 아키텍처가 성능을 가른다"이다. 그 주장이 성립하려면
**오케스트레이션 말고는 아무것도 달라선 안 된다** — 같은 모델, 같은 도구, 같은 프롬프트
도메인 지시, 같은 조사량 상한, 같은 관측 축약, 같은 컨텍스트 내용. 하나라도 갈라지면
성능 차이가 아키텍처 때문인지 그것 때문인지 영영 구분할 수 없다.

여기 있는 것은 그 '하나라도 갈라졌는지'를 잡는 그물이다.

    fakes.py            가짜 LLM·가짜 장비. 아래 두 검사가 공유한다.
    test_parity.py      LLM 없이 — 두 아키텍처가 구조적으로 같은 조건인가
    test_scenarios.py   가짜 LLM 으로 — 같은 대본을 두 루프에 먹여 행동을 대조
    live_compare.py     실제 Ollama 로 — 모델×아키텍처 매트릭스 실행 후 결과표

앞의 둘은 통과/실패가 있는 검사다. 마지막은 검사가 아니라 **리포트**다 — 실제 모델의
판단 품질에는 정답이 없으므로 숫자를 뽑아 사람이 보게 한다.

    python -m backend.test.test_parity                  (Ollama 불필요, 장비 불필요)
    python -m backend.test.test_scenarios               (Ollama 불필요, 장비 불필요)
    python -m backend.test.live_compare --models gemma4:31b,gemma4:12b

[왜 장비 없이 도는가]
세 파일 모두 하드웨어 디스패치를 가짜로 갈아 끼운다(fakes.FAKE_DISPATCH). 개발 PC 에서
돌아야 회귀를 실험 전에 잡을 수 있고, 장비 PC 에서만 도는 검사는 결국 아무도 안 돌린다.
live_compare 도 마찬가지다 — 거기서 재는 것은 '모델이 무엇을 부르기로 했는가'이지
장비가 실제로 무엇을 했는가가 아니다.
"""
