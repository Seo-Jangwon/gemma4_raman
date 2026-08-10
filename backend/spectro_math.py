# -*- coding: utf-8 -*-
"""스펙트럼 수치 규약 — 도구와 샌드박스가 **같은 구현을 공유하는** 곳.

[왜 별도 모듈인가 — 2026-08-10]
IPBSA 는 두 경로로 쓰인다.
    도구      apply_background_subtraction   방금 측정한 것(과 업로드 파일)에 건다
    코드      run_analysis 안의 ipbsa()      코드로 굴리는 중간 배열에 건다
같은 이름의 연산이 두 곳에서 각자 구현되면, 어느 쪽으로 풀었느냐에 따라 답이 달라진다.
그건 에이전트가 통제할 수 없는 차이라 성적에 실어서는 안 된다. 그래서 구현은 여기
하나만 두고 양쪽이 부른다.

[의존성을 numpy 로만 묶는 이유]
run_analysis 는 하드웨어를 못 만지는 별도 프로세스에서 돈다. 이 모듈이 raman_tools 를
끌어오면 그 프로세스가 장비 모듈(DLL·COM 포트)을 import 하게 되므로, 반대로 raman_tools
가 이 모듈을 부르는 방향으로 둔다.
"""
from __future__ import annotations

import numpy as np


def poly_baseline(y, order: int = 5, x=None):
    """다항식 배경을 **한 번** 피팅해 돌려준다.

    ipbsa() 가 쓰는 재료를 그대로 노출한 것이다. 반복 규칙을 직접 정하고 싶은 쪽은
    이것을 조립하면 되고(마스킹·가중치·수렴 조건을 자기 방식으로), 기본 조합이면
    ipbsa() 를 부르면 된다.

    x 를 생략하면 0~1 로 정규화한 축을 쓴다 — 도구가 쓰는 규약과 같다. 다항식은
    선형 재매개변수화에 대해 같은 곡선을 내므로 실제 파수축을 넣어도 수학적으로는
    같지만, 수치 조건이 달라 끝자리가 갈릴 수 있다.
    """
    y = np.asarray(y, dtype=np.float64)
    xx = np.linspace(0.0, 1.0, len(y)) if x is None else np.asarray(x, dtype=np.float64)
    return np.polyval(np.polyfit(xx, y, deg=int(order)), xx)


def ipbsa_detail(y, order: int = 5, max_iterations: int = 100,
                 threshold: float = 0.001, x=None):
    """IPBSA(반복 다항식 배경 제거). (보정, 배경, 반복횟수, 수렴여부).

    한 번 피팅한 배경보다 큰 점을 배경 쪽으로 끌어내리며(working = min(y, bg)) 다시
    피팅하기를 반복한다. 봉우리는 배경 위로 솟아 있으므로 회를 거듭할수록 배경이
    봉우리를 빼고 아래쪽 포락선에 붙는다.

    보정값을 0 으로 자르는 것(clip)은 도구가 예전부터 하던 동작이라 그대로 둔다 —
    두 경로가 같은 답을 내는 것이 이 모듈의 존재 이유다.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    xx = np.linspace(0.0, 1.0, n) if x is None else np.asarray(x, dtype=np.float64)
    working = y.copy()
    prev_bg = np.zeros(n, dtype=np.float64)
    bg = prev_bg
    converged = False
    # 0 이하가 들어오면 아래 루프가 한 번도 안 돌아 bg 가 미정의가 된다. 도구는 인자
    # 검증으로 막지만 샌드박스 헬퍼에는 검증이 없으므로 여기서 하한을 세운다.
    iters = max(1, int(max_iterations))
    i = 0
    for i in range(iters):
        bg = np.polyval(np.polyfit(xx, working, deg=int(order)), xx)
        working = np.minimum(y, bg)
        denom = np.linalg.norm(prev_bg)
        if denom > 0 and np.linalg.norm(bg - prev_bg) / denom < threshold:
            converged = True
            break
        prev_bg = bg.copy()
    return np.clip(y - bg, 0.0, None), bg, i + 1, converged


def ipbsa(y, order: int = 5, max_iterations: int = 100,
          threshold: float = 0.001, x=None):
    """IPBSA 배경 제거 — (보정된 세기, 배경) 두 배열.

    run_analysis 안에서 쓰는 얼굴이다. 반복 횟수·수렴 여부까지 필요하면
    ipbsa_detail() 을 쓴다.
    """
    corrected, bg, _, _ = ipbsa_detail(y, order=order, max_iterations=max_iterations,
                                       threshold=threshold, x=x)
    return corrected, bg
