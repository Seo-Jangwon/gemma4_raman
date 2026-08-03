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
)


def evaluate(b, run):
    """이 목록이 그대로 T073 의 점수가 된다."""
    before, after = run.state_before, run.state_after

    K = 2
    src = _input(b, "T073.csv")
    if src is None:
        return [chk.fail("cluster labels", "could not read the input T073.csv", weight=2.0)]
    got = run.answer.get("labels")
    if not isinstance(got, list):
        return [chk.fail("cluster labels", "answer.labels is missing", weight=2.0)]

    want = _kmeans_labels(src, K)
    if want is None:
        return [chk.fail("cluster labels", "recomputation failed", weight=2.0)]
    if len(got) != len(want):
        return [chk.fail("cluster labels",
                         f"{len(got)} reported (expected {len(want)} items)", weight=2.0)]
    # 레이블 번호 자체는 임의다. 0↔1 을 바꾼 것도 같은 군집이므로 둘 중 잘 맞는 쪽으로 본다.
    g = [int(v) for v in got]
    same = sum(1 for a, c in zip(g, want) if a == c)
    flip = sum(1 for a, c in zip(g, want) if a != c)
    hit = max(same, flip)
    return [
        chk.ok("cluster labels (relabeling allowed)", hit == len(want),
               f"{hit}/{len(want)} matched", weight=2.0),
    ]


def _kmeans_labels(src, k):
    """규약: 각 행 스펙트럼을 L2 정규화한 뒤 k-means(고정 시드)."""
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return None
    import csv as _csv
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / "T073.csv"
    rows = list(_csv.reader(p.read_text(encoding="utf-8-sig").splitlines()))
    body = [r for r in rows[1:] if len(r) > 2]
    try:
        X = np.array([[float(v) for v in r[1:]] for r in body])
    except ValueError:
        return None
    X = np.array([sp.l2(v) for v in X])
    return list(KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X))


def _input(b, name):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "inputs" / name
    return sp.read_xy(p) if p.exists() else None
