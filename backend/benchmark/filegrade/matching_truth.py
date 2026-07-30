# -*- coding: utf-8 -*-
"""T111~T128 라이브러리 매칭 문항의 정답(GT) 엔진.

[왜 필요한가]
매칭 블록 17문항의 verifier 는 `tool_called` + `answer_contains("polystyrene")` 뿐이다.
그런데 실제 채점기준은 이런 것들을 요구한다.

  T111  top-3 의 식별정보가 정답 순위와 정확히 일치, 점수 내림차순, 각 점수 1e-4 이내
  T119  8개가 각각 한 번씩, 전체 순위가 정답과 일치, 점수 내림차순
  T128  질의별 precision@3 의 산술평균이 레퍼런스와 1e-6 이내

`answer_contains` 는 답변 어딘가에 "polystyrene" 이 있으면 통과하므로, 순위가 뒤집혀도
점수가 오름차순이어도 통과한다. tasks.json 의 human_only note 에도 "similarity
scores/ranking are not checkable by the verifier vocabulary" 라고 적혀 있다.
이 모듈이 그 빈칸을 메운다.

[유사도 지표를 무엇으로 고정할 것인가]
프롬프트가 유사도 정의를 지정하지 않는다. 그래서 세 가지로 전부 계산해 봤다.

    문항   cos_raw            pearson            baseline+L2 후 cosine
    T111   PS 0.9983 …        PS 0.9974 …        PS 0.9972 …
    T119   PS 0.9982 …        PS 0.9973 …        PS 0.9971 …
    T122   CAL 0.9962 / ARA 0.9290   CAL 0.9942 / ARA 0.8916   CAL 0.9939 / ARA 0.8869

**물질·ID 순위는 세 지표에서 모두 동일했다.** 따라서 순위와 식별은 지표에 무관하게
채점할 수 있다(부류 A). 반면 **점수값은 지표마다 다르다**. 그래서 "점수가 레퍼런스와
1e-4 이내"라는 기준은 값 비교로 쓰지 않고, `reproduce_score()` 로
'에이전트가 선언한 지표로 재계산했을 때 그 값이 나오는가'를 본다.
순위가 지표에 따라 갈리는 문항이 있으면 `metric_invariant()` 가 False 를 돌려주고
리포트에 '지표 의존 — 판정 보류'로 표시된다.

[동점]
reference_library.csv 의 PS_01 과 PS_02 는 **비트 단위로 동일하다**(측정 확인).
reference_library_8.csv 는 PS_01=PS_02=PS_03 3중 동점, PET_01=PET_02 2중 동점.
동일한 참조끼리는 어떤 순서로 답해도 옳으므로, 순위 비교는 동점군 내부를 무시한다.
이건 관용이 아니라 문항 설계상 순서가 정의되지 않기 때문이다.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import BENCH_DIR, PROJECT_ROOT

_UPLOADS = PROJECT_ROOT / "data" / "uploads"
METRICS = ("cos_raw", "pearson", "cos_baseline_l2")


# ── 입력 ─────────────────────────────────────────────────────────────────────

def _rows(path: Path) -> list[dict]:
    lines = [l for l in path.read_text(encoding="utf-8-sig").splitlines()
             if not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def find_input(name: str) -> Path | None:
    """가장 최근 날짜 폴더의 파일을 쓴다 — diagnostics.find_input 과 같은 규칙."""
    hits = sorted(_UPLOADS.glob(f"*/{name}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def load_query(name: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    p = find_input(name)
    if p is None:
        return None, None
    r = _rows(p)
    if not r:
        return None, None
    return (np.array([float(a["raman_shift_cm-1"]) for a in r]),
            np.array([float(a["intensity"]) for a in r]))


@dataclass
class Library:
    ids: list[str]
    material: dict[str, str]
    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    ties: list[list[str]]          # 비트 동일한 참조들의 묶음

    def tie_of(self, sid: str) -> list[str]:
        for g in self.ties:
            if sid in g:
                return g
        return [sid]


def load_library(name: str = "reference_library.csv") -> Library | None:
    p = find_input(name)
    if p is None:
        return None
    acc: dict[str, dict] = {}
    for r in _rows(p):
        d = acc.setdefault(r["spectrum_id"], {"m": r["material"], "x": [], "y": []})
        d["x"].append(float(r["raman_shift_cm-1"]))
        d["y"].append(float(r["intensity"]))
    ids = list(acc)
    X = {k: np.array(v["x"]) for k, v in acc.items()}
    Y = {k: np.array(v["y"]) for k, v in acc.items()}
    M = {k: v["m"] for k, v in acc.items()}

    ties: list[list[str]] = []
    seen: set[str] = set()
    for a in ids:
        if a in seen:
            continue
        g = [a]
        for b in ids:
            if b != a and b not in seen and Y[a].shape == Y[b].shape and np.array_equal(Y[a], Y[b]):
                g.append(b)
        seen.update(g)
        ties.append(g)
    return Library(ids, M, X, Y, ties)


def load_peak_library(name: str = "peak_library.csv") -> dict[str, list[float]]:
    p = find_input(name)
    if p is None:
        return {}
    out: dict[str, list[float]] = {}
    for r in _rows(p):
        out.setdefault(r["material"], []).append(float(r["peak_cm-1"]))
    return out


# ── 유사도 ───────────────────────────────────────────────────────────────────

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _bl_l2(x: np.ndarray, y: np.ndarray, order: int = 5) -> np.ndarray:
    v = y - np.polyval(np.polyfit(x, y, order), x)
    n = np.linalg.norm(v)
    return v / n if n else v


def similarity(x, q, xr, r, metric: str = "cos_raw") -> float:
    """질의와 참조의 유사도. 축이 다르면 참조를 질의 축으로 보간한다(T126 요구사항)."""
    q = np.asarray(q, float)
    r = np.asarray(r, float)
    if len(xr) != len(x) or not np.allclose(xr, x):
        r = np.interp(x, xr, r)
    if metric == "cos_raw":
        return _cos(q, r)
    if metric == "pearson":
        return _cos(q - q.mean(), r - r.mean())
    if metric == "cos_baseline_l2":
        return _cos(_bl_l2(x, q), _bl_l2(x, r))
    raise ValueError(metric)


@dataclass
class Ranked:
    metric: str
    rows: list[tuple[str, str, float]]        # (spectrum_id, material, score) 내림차순

    @property
    def ids(self) -> list[str]:
        return [r[0] for r in self.rows]

    @property
    def materials(self) -> list[str]:
        return [r[1] for r in self.rows]

    @property
    def scores(self) -> list[float]:
        return [r[2] for r in self.rows]

    def top(self, k: int) -> list[tuple[str, str, float]]:
        return self.rows[:k]


def rank(x, q, lib: Library, metric: str = "cos_raw") -> Ranked:
    """참조 전체를 유사도 내림차순으로. 동점은 spectrum_id 사전순으로 안정 정렬한다."""
    sc = [(sid, lib.material[sid], similarity(x, q, lib.x[sid], lib.y[sid], metric))
          for sid in lib.ids]
    sc.sort(key=lambda t: (-t[2], t[0]))
    return Ranked(metric, sc)


# 문항이 실제로 요구하는 순위 깊이. 이 깊이까지만 지표 무관하면 엄격 채점이 가능하다.
REQUIRED_DEPTH = {
    "T111": 3,    # top-3
    "T112": 1, "T113": 1, "T114": 1, "T115": 1, "T116": 1, "T117": 1,
    "T118": 1, "T120": 1, "T121": 1, "T122": 2,   # T122 는 2위(aragonite)까지 봐야 한다
    "T123": 2,    # 1위 + 2위 후보
    "T124": 3, "T125": 1, "T126": 1, "T127": 1, "T128": 3,
    "T119": 8,    # 전체 순위 — 이 깊이에서 지표 무관성이 깨진다(문항 결함으로 보고)
}


def metric_invariant(x, q, lib: Library, depth: int | None = None
                     ) -> tuple[bool, dict[str, list[str]]]:
    """세 지표의 순위가 상위 `depth` 까지 같은가.

    전체 순위를 요구하면 거의 항상 깨진다 — 유사도가 낮은 꼬리(서로 안 닮은 참조들)의
    순서는 지표 정의에 민감하기 때문이다. 예를 들어 T111 은 3위까지는 세 지표가 모두
    PS_01/PS_02/PMMA_01 로 같지만, 5위 아래로는 PET·CAL·ARA·SI 가 뒤섞인다.
    그래서 '문항이 실제로 묻는 깊이'까지만 따진다. 그 깊이에서 깨지면 그 문항은
    지표를 못박지 않는 한 순위를 채점할 수 없다는 뜻이고, 리포트에 그렇게 적는다.

    동점군은 하나로 접어 비교한다(동일 참조끼리는 순서가 정의되지 않는다).
    """
    seq: dict[str, list[str]] = {}
    canon: dict[str, list] = {}
    for m in METRICS:
        r = rank(x, q, lib, m)
        ids = r.ids[:depth] if depth else r.ids
        seq[m] = [f"{s}({mat})" for s, mat, _ in r.rows[:depth]] if depth else \
                 [f"{s}({mat})" for s, mat, _ in r.rows]
        # 깊이는 '참조 개수'로 자르고 그 다음에 동점군을 접는다. 순서를 반대로 하면
        # 동점군 하나가 항목 하나로 세어져 의도보다 훨씬 깊이 비교하게 된다.
        canon[m] = _canon(ids, lib)
    first = canon[METRICS[0]]
    return all(canon[m] == first for m in METRICS), seq


def reproduce_score(x, q, lib: Library, sid: str, reported: float,
                    tol: float = 1e-4) -> tuple[bool, str | None, float]:
    """보고된 점수가 세 지표 중 하나로 재현되는가.

    점수값은 지표 의존이므로 '레퍼런스 값과 같은가'를 물으면 안 된다. 대신
    '어떤 정당한 지표로 계산하면 이 값이 나오는가'를 묻는다. 재현되면 그 지표 이름을
    함께 돌려줘서, 에이전트가 실제로 무엇을 계산했는지 리포트에 남긴다.
    """
    best_m, best_d = None, float("inf")
    for m in METRICS:
        v = similarity(x, q, lib.x[sid], lib.y[sid], m)
        d = abs(v - reported)
        if d < best_d:
            best_m, best_d = m, d
    return (best_d <= tol), best_m, best_d


# ── 순위 비교 (동점군 무시) ──────────────────────────────────────────────────

def _canon(seq: list[str], lib: Library) -> list[frozenset]:
    """동점군을 하나의 집합으로 접어, 동점 내부 순서 차이를 없앤 표준형."""
    out: list[frozenset] = []
    for s in seq:
        g = frozenset(lib.tie_of(s))
        if not out or out[-1] != g:
            out.append(g)
    return out


def ranking_matches(reported_ids: list[str], truth_ids: list[str], lib: Library) -> bool:
    """동점군 내부 순서를 무시하고 두 순위가 같은가."""
    return _canon(reported_ids, lib) == _canon(truth_ids, lib)


def descending(scores: list[float], tol: float = 1e-9) -> bool:
    return all(a >= b - tol for a, b in zip(scores, scores[1:]))


# ── 에이전트 답변 파싱 ───────────────────────────────────────────────────────

_ID_RE = re.compile(r"\b(PS|PET|PMMA|CAL|ARA|SI)_\d{2}\b")
_MAT_ALIASES = {
    "polystyrene": "polystyrene", "ps": "polystyrene",
    "pet": "PET", "polyethylene terephthalate": "PET",
    "pmma": "PMMA", "poly(methyl methacrylate)": "PMMA",
    "calcite": "calcite", "aragonite": "aragonite",
    "silicon": "silicon", "si": "silicon",
}


def parse_reported(answer: str) -> list[dict]:
    """답변의 마크다운 표에서 (rank, id, material, score) 를 뽑는다.

    에이전트들이 결과를 표로 내는 게 관측된 공통 패턴이라 표를 우선 읽는다. 표가 없으면
    빈 목록을 돌려주고, 호출부가 '자동 추출 불가 → 사람 확인'으로 넘긴다. 억지로
    추출해서 틀린 판정을 내리는 것보다 낫다.
    """
    if not answer:
        return []
    out: list[dict] = []
    for line in answer.splitlines():
        if line.count("|") < 3:
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if not cells or set("".join(cells)) <= set("-: "):
            continue
        sid = next((_ID_RE.search(c).group(0) for c in cells if _ID_RE.search(c)), None)
        mat = None
        for c in cells:
            key = c.lower().strip("*_ ")
            if key in _MAT_ALIASES:
                mat = _MAT_ALIASES[key]
                break
        score = None
        for c in reversed(cells):
            m = re.fullmatch(r"[*\s]*(-?\d*\.\d+|-?\d+)[*\s]*", c)
            if m:
                v = float(m.group(1))
                if -1.0001 <= v <= 1.0001 and ("." in m.group(1)):
                    score = v
                    break
        if sid or (mat and score is not None):
            out.append({"id": sid, "material": mat, "score": score})
    return out


def declared_metric(answer: str, code: str = "") -> str | None:
    """에이전트가 어떤 유사도를 썼다고 선언했는지. 점수 재현 판정의 근거로 쓴다."""
    t = f"{answer}\n{code}".lower()
    if "pearson" in t or "corrcoef" in t or "correlation coefficient" in t:
        return "pearson"
    if "cosine" in t or "cosine_similarity" in t:
        return "cos_raw"
    return None


# ── 문항별 GT ────────────────────────────────────────────────────────────────

def truth_for(task_id: str) -> dict:
    """문항의 정답을 실제 데이터로 재계산한다. 리포트가 그대로 싣는 '정답 정의'다."""
    tid = str(task_id)
    lib_name = "reference_library_8.csv" if tid == "T119" else "reference_library.csv"
    lib = load_library(lib_name)
    if lib is None:
        return {"error": f"{lib_name} 를 찾지 못했다"}

    if tid in ("T124", "T128"):
        order = ["polystyrene", "PET", "PMMA", "calcite", "silicon"]
        per, prec = [], []
        for k, truth in enumerate(order, start=1):
            x, q = load_query(f"{tid}_{k}.csv")
            if q is None:
                continue
            r = rank(x, q, lib, "cos_raw")
            top3 = r.top(3)
            same = sum(1 for _, mat, _ in top3 if mat == truth)
            per.append({"query": f"{tid}_{k}.csv", "truth": truth,
                        "top1": top3[0][1] if top3 else None,
                        "top3": [m for _, m, _ in top3], "relevant_in_top3": same,
                        "precision_at_3": same / 3})
            prec.append(same / 3)
        res = {"library": lib_name, "per_query": per, "truth_order": order}
        if tid == "T128":
            res["macro_precision_at_3"] = float(np.mean(prec)) if prec else None
            res["ceiling_note"] = ("물질당 참조가 2개뿐이라 top-3 의 최대 정밀도는 "
                                   "2/3 = 0.6667 이다 — 1.0 을 보고했다면 계산이 틀린 것이다.")
        return res

    if tid == "T118":
        pl = load_peak_library()
        x, q = load_query("T118.csv")
        if q is None or not pl:
            return {"error": "T118.csv / peak_library.csv 를 찾지 못했다"}
        from .shape_match import peaks_of
        got = peaks_of(x, q, 0.05)
        best, scores = None, {}
        for mat, peaks in pl.items():
            hit = sum(1 for p in peaks if got.size and np.min(np.abs(got - p)) <= 3.0)
            scores[mat] = hit / len(peaks)
            if best is None or scores[mat] > scores[best]:
                best = mat
        return {"library": "peak_library.csv", "detected_peaks": [round(float(v), 1) for v in got],
                "match_ratio": {k: round(v, 4) for k, v in scores.items()}, "truth": best}

    if tid == "T121":
        pl = load_peak_library()
        x, q = load_query("T121.csv")
        if q is None:
            return {"error": "T121.csv 를 찾지 못했다"}
        i = int(np.argmax(q))
        return {"library": "peak_library.csv", "peak_used_cm-1": round(float(x[i]), 2),
                "truth": "silicon",
                "note": "silicon 은 peak_library 에서 520 cm⁻¹ 단일 피크로 정의된다."}

    x, q = load_query(f"{tid}.csv")
    if q is None:
        return {"error": f"{tid}.csv 를 찾지 못했다"}

    metric = "cos_baseline_l2" if tid in ("T113", "T126") else "cos_raw"
    r = rank(x, q, lib, metric)
    depth = REQUIRED_DEPTH.get(tid)
    inv, seq = metric_invariant(x, q, lib, depth)
    res: dict = {
        "library": lib_name,
        "metric_used": metric,
        "required_depth": depth,
        "metric_invariant": inv,
        "order_by_metric": seq,
        "ranking": [{"rank": i + 1, "id": s, "material": m, "score": round(v, 6)}
                    for i, (s, m, v) in enumerate(r.rows)],
        "tie_groups": [g for g in lib.ties if len(g) > 1],
        "top1_material": r.rows[0][1],
        "top1_id": r.rows[0][0],
        "max_score": r.rows[0][2],
    }

    if tid == "T111":
        res["top3"] = res["ranking"][:3]
    elif tid == "T112":
        res["threshold"] = 0.85
        res["decision"] = "동일 물질로 볼 수 있다" if r.rows[0][2] >= 0.85 else "볼 수 없다"
        res["by_metric"] = {m: round(rank(x, q, lib, m).rows[0][2], 6) for m in METRICS}
    elif tid == "T114":
        res["threshold"] = 0.75
        res["decision"] = ("신뢰할 만한 매칭 없음" if r.rows[0][2] < 0.75
                           else f"매칭 있음({r.rows[0][1]})")
        res["by_metric"] = {m: round(rank(x, q, lib, m).rows[0][2], 6) for m in METRICS}
    elif tid == "T116":
        res["mixture_note"] = "PET70 / PMMA30 혼합. 우세 성분이 정답이다."
    elif tid == "T119":
        res["expected_ids"] = r.ids
        # 5위 아래는 지표에 따라 뒤바뀐다(PET/CAL/SI 가 재배열). 채점기준은 "전체 순위가
        # 정답과 정확히 일치"를 요구하지만 프롬프트가 지표를 못박지 않아 그 요구는
        # 충족 여부를 판정할 수 없다. 지표 무관하게 물을 수 있는 것만 채점한다.
        inv4, _ = metric_invariant(x, q, lib, 4)
        res["invariant_prefix_depth"] = 4 if inv4 else 1
        res["gradable_strictly"] = ["8개가 각각 한 번씩 나왔는가",
                                    "점수가 내림차순인가",
                                    "상위 4위(PS 3개 동점 + PMMA_01)가 맞는가"]
        res["not_gradable"] = ("5~8위(PET/CAL/SI)의 순서는 유사도 지표에 따라 달라진다 — "
                               "cos_raw 는 PET>CAL>SI, cos_baseline_l2 는 SI>PET>CAL. "
                               "프롬프트가 지표를 지정하지 않으므로 이 구간은 "
                               "'에이전트가 선언한 지표로 재현되는가'로만 본다.")
    elif tid == "T122":
        cal = max((v for s, m, v in r.rows if m == "calcite"), default=None)
        ara = max((v for s, m, v in r.rows if m == "aragonite"), default=None)
        res["polymorph"] = {"calcite": round(cal, 6) if cal else None,
                            "aragonite": round(ara, 6) if ara else None}
    elif tid == "T123":
        second = next((m for _, m, _ in r.rows if m != r.rows[0][1]), None)
        pl = load_peak_library()
        p1 = set(pl.get(r.rows[0][1], []))
        p2 = set(pl.get(second, []))
        res["second_candidate"] = second
        res["distinguishing_peaks"] = sorted(p1 - p2)
        res["shared_peaks"] = sorted(p1 & p2)
    elif tid == "T125":
        res["claimed"] = "PET"
        res["verdict"] = ("불일치" if r.rows[0][1] != "PET" else "일치")
    elif tid == "T126":
        tie = [s for s, m, v in r.rows if abs(v - r.rows[0][2]) < 1e-12]
        res["tie_candidates"] = sorted(tie)
        res["tie_rule_pick"] = sorted(tie)[0] if tie else None
    elif tid == "T127":
        # 이동량 추정: 축을 -d 만큼 밀며 최대 유사도가 되는 d 를 찾는다.
        best_d, best_s = 0.0, -2.0
        for d in np.arange(-15, 15.01, 0.5):
            s = max(similarity(x, np.interp(x, x - d, q), lib.x[i], lib.y[i], "cos_raw")
                    for i in lib.ids)
            if s > best_s:
                best_d, best_s = float(d), float(s)
        res["estimated_shift_cm-1"] = best_d
        res["similarity_after_correction"] = round(best_s, 6)
        res["similarity_before"] = round(r.rows[0][2], 6)
        rc = rank(x, np.interp(x, x - best_d, q), lib, "cos_raw")
        res["material_after_correction"] = rc.rows[0][1]
    return res


# ── 자기검증 ─────────────────────────────────────────────────────────────────

def selftest(verbose: bool = True) -> bool:
    lib8 = load_library("reference_library_8.csv")
    lib = load_library("reference_library.csv")
    if lib is None or lib8 is None:
        print("selftest 생략 — 라이브러리를 찾지 못했다")
        return True
    ok = True

    if verbose:
        print("동점군 (비트 단위로 동일한 참조):")
        print(f"  reference_library.csv   : {[g for g in lib.ties if len(g) > 1]}")
        print(f"  reference_library_8.csv : {[g for g in lib8.ties if len(g) > 1]}")

    t = truth_for("T119")
    exp_mat = ["polystyrene", "polystyrene", "polystyrene", "PMMA", "PET", "PET",
               "calcite", "silicon"]
    got_mat = [r["material"] for r in t["ranking"]]
    hit = got_mat == exp_mat
    ok &= hit
    if verbose:
        print(f"\nT119 GT 전체 순위 (metric={t['metric_used']}, "
              f"요구깊이={t['required_depth']} 에서 지표무관={t['metric_invariant']}):")
        for r in t["ranking"]:
            print(f"  {r['rank']}. {r['id']:8} {r['material']:12} {r['score']:.4f}")
        print(f"  기대 물질순서 일치: {hit}")
        print(f"  지표 무관 접두 깊이: {t['invariant_prefix_depth']}위까지")
        print(f"  → {t['not_gradable']}")

    # AILA 의 T119 답변을 실제로 대조한다 — 동점군 무시하면 정답이어야 한다.
    import json
    p = BENCH_DIR / "results" / "raw_runs.jsonl"
    if p.exists():
        rec = next((json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                    if '"T119"' in l and '"AILA"' in l), None)
        if rec:
            rep = parse_reported(rec.get("answer") or "")
            rid = [r["id"] for r in rep if r["id"]]
            match = ranking_matches(rid, t["expected_ids"], lib8)
            desc = descending([r["score"] for r in rep if r["score"] is not None])
            uniq = len(set(rid)) == len(rid) == 8
            ok &= (match and desc and uniq)
            if verbose:
                print(f"\nAILA T119 파싱: {rid}")
                print(f"  8개 각 1회={uniq} · 동점군 무시 순위일치={match} · 점수 내림차순={desc}")

    if verbose:
        print("\n문항별 GT 와 지표 무관성 (요구 깊이 기준):")
        for tid in ("T111", "T112", "T113", "T114", "T115", "T116", "T117", "T118",
                    "T120", "T121", "T122", "T123", "T125", "T126", "T127", "T128"):
            r = truth_for(tid)
            key = ("macro_precision_at_3" if tid == "T128" else
                   "decision" if tid in ("T112", "T114") else
                   "estimated_shift_cm-1" if tid == "T127" else
                   "verdict" if tid == "T125" else "truth" if tid in ("T118", "T121")
                   else "top1_material")
            d = r.get("required_depth")
            iv = r.get("metric_invariant")
            tag = "" if iv is None else f"  [깊이 {d} 지표무관={'예' if iv else '아니오'}]"
            print(f"  {tid}: {key} = {r.get(key)}{tag}")
    # 요구 깊이에서 지표 무관성이 깨지는 문항은 T119 하나여야 한다 — 그 외에서 깨지면
    # 그 문항도 '지표 미지정' 결함이 있다는 뜻이므로 알려야 한다.
    broken = [t for t in REQUIRED_DEPTH
              if (truth_for(t).get("metric_invariant") is False)]
    if verbose:
        print(f"\n요구 깊이에서 지표 의존인 문항: {broken or '없음'}")
    ok &= (broken == ["T119"])
    if verbose:
        print(f"selftest: {'통과' if ok else '실패'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(0 if selftest() else 1)
