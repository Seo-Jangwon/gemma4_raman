# -*- coding: utf-8 -*-
"""T073 — 데이터 처리 (3점)

[문제]
  After the same preprocessing as T072, cluster the spectra of T073.csv into 2 groups with
  KMeans(n_clusters=2, n_init=10, random_state=0) and show each coordinate's group as a
  map.

[정답 기준]
  GT=좌표별 클러스터 레이블. 알고리즘과 시드를 못박아 재현성을 확보했다(원문은 시드가 없어 GT 불가였다). 확인=레이블 번호의 순열을 동일 취급(0↔1 교환
  허용)한 뒤 할당 일치율 100%. 맵 1장.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T073",
    score=3,
    axis="data processing",
    mode="live",
    inputs=['T073.csv'],
    criteria="EXACT(labels, 100% after relabeling)",
    prompt=(
        "After the same preprocessing as T072, cluster the spectra of T073.csv into 2 groups "
        "with KMeans(n_clusters=2, n_init=10, random_state=0) and show each coordinate's "
        "group as a map. "
    ),
    answer_keys=[
        ("labels",
         "list of numbers - the cluster label (0 or 1) of each coordinate, in the "
         "row order of the input file"),
    ],
)


# 정답 — 좌표 25 개의 군집 번호를, T073.csv 에 좌표가 처음 나오는 순서로.
#
# [왜 인라인 값인가 — 2026-08-09]
# 예전에는 _kmeans_labels() 가 채점할 때마다 T073.csv 를 다시 읽어 KMeans 를 돌렸는데,
# 그 코드가 파일 포맷을 잘못 읽어 **정답이 아닌 값**을 정답으로 삼고 있었다:
#
#   T073.csv 는 long format 이다 — 한 행이 (x, y, raman_shift, intensity) 한 점이고,
#   좌표 25 개 × 시프트 1,801 점 = 45,025 행이다. 그런데 옛 코드는 자기 docstring 이
#   "각 행 스펙트럼" 이라고 적은 대로 **한 행 = 한 좌표의 스펙트럼(wide format)** 을
#   전제하고 rows[1:] 을 그대로 샘플로 넘겼다. 그래서 좌표 25 개가 아니라 45,025 개
#   점을 [y, 시프트, 세기] 3 차원 벡터로 군집화하고 레이블 45,025 개를 정답이라 했다.
#   문항이 요구하는 IPBSA 베이스라인(T072 와 같은 전처리)도 아예 건너뛰었다.
#   결과: 25 개를 맞게 낸 답안이 "25 reported (expected 45025 items)" 로 오답 처리됐다.
#
# 그래서 재계산을 걷어내고 값을 고정한다. 이 값의 출처는 데이터를 만든
# generate/make_dataset.py 의 파이프라인 그대로다 — 좌표별 스펙트럼 →
# synth.ipbsa(order=5) → l2 → 평균중심화 → KMeans(n_clusters=2, n_init=10,
# random_state=0). 그 파이프라인을 T073.csv 에 다시 적용해 (25, 1801) 행렬로 재현했고,
# generate 가 남긴 gt/T073.json 의 assignments[].cluster 와 25/25 일치를 확인했다.
# 인라인으로 두는 이유는 run_all._check_stale_gt 주석 참고 — 채점 경로가 읽는 실효 GT 는
# tasks/*.py 의 인라인 값과 gt/arrays/*.csv 두 곳뿐이고 gt/<문항>.json 은 읽지 않는다.
#
# 평균중심화를 전처리에 포함하든 말든 레이블은 같다(k-means 는 평행이동 불변) — 문항의
# "same preprocessing as T072" 가 어디까지인지 모호해도 채점 결과는 갈리지 않는다.
GT_CLUSTERS = [1, 1, 1, 1, 1,
               1, 1, 1, 1, 0,
               1, 1, 1, 0, 0,
               1, 1, 0, 0, 0,
               1, 0, 0, 0, 0]


def evaluate(b, run):
    """이 목록이 그대로 T073 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    want = GT_CLUSTERS
    got = run.answer.get("labels")
    if not isinstance(got, list):
        return [chk.fail("cluster labels", "answer.labels is missing", weight=2.0)]
    if len(got) != len(want):
        # 좌표 수와 어긋난 것이므로 무엇을 세야 했는지까지 알려준다 — 옛 채점기가
        # 45,025(행 수)를 요구하는 바람에 정답이 오답으로 처리된 적이 있다.
        return [chk.fail("cluster labels",
                         f"{len(got)} reported (expected {len(want)} - one label per "
                         f"coordinate, in the order the coordinates first appear in "
                         f"T073.csv)", weight=2.0)]
    try:
        g = [int(v) for v in got]
    except (TypeError, ValueError):
        return [chk.fail("cluster labels", "answer.labels holds non-numeric values",
                         weight=2.0)]
    # 레이블 번호 자체는 임의다. 0↔1 을 바꾼 것도 같은 군집이므로 둘 중 잘 맞는 쪽으로 본다.
    same = sum(1 for a, c in zip(g, want) if a == c)
    hit = max(same, len(want) - same)
    return [
        chk.ok("cluster labels (relabeling allowed)", hit == len(want),
               f"{hit}/{len(want)} matched", weight=2.0),
    ]
