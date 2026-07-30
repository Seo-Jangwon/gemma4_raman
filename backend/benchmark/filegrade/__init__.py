# -*- coding: utf-8 -*-
"""파일처리(analysis_file) 48문항 전용 채점 패키지.

[왜 별도 폴더인가]
backend/benchmark/ 는 러너(run_bench)·태스크 빌드(build_tasks)·합성(make_task_spectra)·
채점 콘솔(review) 이 한 폴더에 섞여 있다. 여기 담긴 것은 그중 '파일처리 문항의
정답 기준을 다시 정의하고 채점하는' 한 가지 관심사뿐이라 따로 뺐다.

    task_class.py     A/B 부류 분류표 (정답 기준 단일 소스)
    shape_match.py    부류 B 모양새 판정 엔진
    matching_truth.py T111~T128 라이브러리 매칭 GT 엔진
    checks.py         diagnostics.CHECKS 에 등록할 누락 문항 진단
    report.py         채점 리포트(HTML/JSON) 생성
    grade_files.py    CLI 진입점
    results/          채점 산출물 (benchmark/results/ 와 분리)

[임포트 규칙]
상위 benchmark/ 의 모듈들(verifiers, diagnostics, spectra_panel …)은 서로를 플랫하게
임포트한다(`import spectra_panel`). 이 패키지가 어디서 실행되든 그게 되도록
아래에서 상위 폴더를 sys.path 에 넣어 둔다.
"""
from __future__ import annotations

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BENCH_DIR.parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
