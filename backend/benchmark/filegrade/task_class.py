# -*- coding: utf-8 -*-
"""파일처리 문항의 '정답 기준 부류' 단일 소스.

[왜 이 파일이 있는가]
기존 채점기는 산출 CSV 를 레퍼런스 CSV 와 점대점으로 비교했다(reference_match,
tolerance 1e-5). 그런데 그 tolerance 가 재고 있는 것은 "과제를 옳게 풀었는가"가 아니라
"우리 레퍼런스 구현과 비트 단위로 일치하는가"다. 둘은 다르다.

T038(5차 다항 baseline) 입력으로 정당한 구현 4개를 실제로 돌려 본 결과:

    방법              레퍼런스 대비 max|Δ|   피크 7개   배경 평탄화   최소값
    plain polyfit          9.9e-07          동일       97.3%      -60.81
    표준화 축 polyfit       9.9e-07          동일       97.3%      -60.81
    iterative LMJ            64.66          동일       99.4%       -0.04
    ALS                      67.48          동일       99.8%       -5.58

앙상블 편차 S = 67.5 = tolerance(1e-5) 의 6.7e6 배. 네 방법 모두 피크 7개를 같은 위치에서
복원하는데 두 개만 통과한다. 더 나쁜 건 레퍼런스(plain polyfit)가 넷 중 과학적으로
가장 나쁘다는 점이다 — 평탄화 최하위에 스펙트럼을 -60.8 까지 끌어내린다.

[판별 테스트]
    프롬프트만 보고 유능한 두 사람이 독립적으로 풀었을 때,
    허용오차를 넘는 차이가 날 수 있는가?

날 수 있으면 그 허용오차는 correctness 가 아니라 구현 동일성을 재는 것이다.
그래서 문항을 두 부류로 나눈다.

  부류 A — 명세가 답을 유일하게 결정한다. 자유 파라미터 없이 입력의 순수 함수.
           GT 를 확정해 엄격하게 채점한다(tolerance 는 float64 반올림 바닥).
  부류 B — 이름은 하나지만 구현 선택지가 여럿인 '방법군'을 부른다.
           비트 일치를 요구하면 안 되고, 모양새(shape) 일치로 판정한다.

[선례]
answer_specs.py 는 이미 despike 계열(T039/T056/T099)에 대해 불변량 방식을 쓰고 있다
— '비-스파이크 점 max|Δ| ≤ 1e-5' + '스파이크 제거율 ≥ 99%'. 이 파일은 그 선례를
나머지 부류 B 문항으로 일반화한 것이다.
"""
from __future__ import annotations

CLASS_A = "A"
CLASS_B = "B"

# 부류 B 기본 통과 임계값 — T038 실측으로 보정했다.
# 정당한 방법 4개(plain/std_axis/LMJ/ALS)는 전부 통과하고,
# 틀린 답 5개(무보정/1차/9차 과적합/과평활/피크반전)는 전부 탈락하는 조합이다.
#
#   지표                  통과기준   정당한 방법 실측   틀린 답이 걸리는 지점
#   피크 recall (±3)       = 1.00        1.00        무보정 0.86, 과평활 0.43
#   피크 precision (±3)    = 1.00        1.00        9차 과적합 0.875
#   Δ상대세기              ≤ 0.12       ≤ 0.061      1차 baseline 0.191
#   pearson               ≥ 0.95       ≥ 0.973      무보정 0.138, 1차 0.539
#   max|Δ| (0~1 재정규화)   ≤ 0.16       ≤ 0.080      1차 0.565, 과평활 0.781
#
# 연속 임계값(Δ상대세기·max|Δ|)은 <b>정당한 구현들끼리의 편차의 2배</b>로 잡았다.
# 임의로 고른 값이 아니라 앙상블에서 유도한 값이라야 "그 tolerance 근거가 뭐냐"에
# 답할 수 있다. 실제로 정당한 방법 4개는 0~1 스케일에서 서로 최대 0.0797 벌어지므로
# 임계는 0.16 이고, 가장 가까운 오답(1차 baseline 0.5647)까지 3.5배 여유가 있다.
#
# 주의: '2차 baseline'은 이 지표를 전부 통과한다(pearson 0.997). 즉 "5차"라는
# 명시된 지시 위반은 모양새로 잡히지 않는다 — 그건 절차(process) 점수로 잡는다.
DEFAULT_SHAPE = {
    "peak_tol_cm": 3.0,        # 피크 위치 일치 허용오차
    "peak_prom_frac": 0.05,    # 피크 검출 prominence = 이 비율 × ptp
    "min_recall": 1.0,
    "min_precision": 1.0,
    "max_d_rel_intensity": 0.12,   # = 2 × 앙상블 편차 0.0612
    "min_pearson": 0.95,
    "max_abs_01": 0.16,            # = 2 × 앙상블 편차 0.0797
}


def _a(gt_rule: str, why: str, **kw) -> dict:
    return {"class": CLASS_A, "gt_rule": gt_rule, "why": why, "free_params": [], **kw}


def _b(gt_rule: str, free_params: list[str], why: str, **kw) -> dict:
    return {"class": CLASS_B, "gt_rule": gt_rule, "why": why, "free_params": free_params, **kw}


# ── 문항별 부류 ──────────────────────────────────────────────────────────────
TASK_CLASS: dict[str, dict] = {

    # ── 부류 A : 전처리·수치 (자유 파라미터 없음) ────────────────────────────
    "T037": _a("입력을 그대로 선그래프로 — 데이터 변형이 없어야 한다",
               "그리기만 하는 과제라 산출 수치가 입력과 같아야 한다. 해석의 여지가 없다."),
    "T041": _a("out = (y - min(y)) / (max(y) - min(y)) — min 정확히 0, max 정확히 1",
               "min-max 정규화는 정의가 하나뿐이다. 두 사람이 풀어도 같은 값이 나온다."),
    "T042": _a("피크 위치 집합 = ground_truth.peaks_major, ±3 cm⁻¹",
               "검출 임계값은 자유지만 '정답 피크 집합'은 유일하다. 집합 일치로 채점한다."),
    "T043": _a("세기 상위 3개 = [1001, 1602, 1031] 순서까지 일치",
               "세기 순위는 입력이 정하므로 유일하다."),
    "T045": _a("측정 피크와 레퍼런스 피크의 ±3 cm⁻¹ 매칭 비율",
               "입력과 레퍼런스가 고정이라 비율이 유일하게 결정된다."),
    "T047": _a("두 스펙트럼(PS, PET)을 변형 없이 겹쳐 그리고 범례를 단다",
               "그리기 과제라 값 변형이 없어야 한다. 해석의 여지가 없다."),
    "T048": _a("800~1200 cm⁻¹ 행을 원본 값 그대로 — 값 변형 0",
               "구간 슬라이싱은 정의가 하나다."),
    "T049": _a("SNR = max(990~1012) / std(1800~1900, ddof=1)",
               "채점기준이 'sample standard deviation'이라 ddof=1 로 못박혀 있다."),
    "T050": _a("각 점에서 1000 cm⁻¹ 최근접 채널의 세기 → 3×3 히트맵",
               "최근접 채널 선택은 유일하다."),
    "T051": _a("5 cm⁻¹ 이상 이동한 피크쌍 = [[1155,1163],[1602,1610]]",
               "두 입력이 고정이라 이동 피크쌍이 유일하게 결정된다."),
    "T052": _a("np.trapz(y[990~1012], x[990~1012])",
               "프롬프트가 사다리꼴 적분을 못박았고 구간도 지정됐다."),
    "T053": _a("1001 최근접 피크 세기 / 1602 최근접 피크 세기",
               "두 피크 위치가 지정돼 비율이 유일하다."),
    "T054": _a("out = np.gradient(y, x) — 비균일 중심차분 + 양끝 1차",
               "프롬프트가 스킴을 못박지 않았지만, 레퍼런스 CSV 가 np.gradient 와 "
               "max|Δ|=1e-6 으로 일치하고 대안인 np.diff 는 점 수가 1800 으로 달라 "
               "구조적으로 구분된다. GT 를 np.gradient 로 고정하고 명세에 공개한다.",
               note_alt="np.diff(y)/np.diff(x) 는 1800점 → 점 수로 즉시 판별된다."),
    "T055": _a("out = y / ||y||₂ — L2 노름 정확히 1",
               "벡터길이 정규화는 정의가 하나뿐이다. 레퍼런스와 max|Δ|=5e-7 로 일치한다."),
    "T074": _a("1001 피크 위치차(±0.5) · 세기 RSD(std ddof=1 / mean) · 평균 스펙트럼 코사인",
               "피크가 뾰족하고 고립돼 있어 검출 구간을 995~1007 로 잡든 990~1012 로 "
               "잡든 argmax 가 같다. 평균 스펙트럼의 코사인은 정의가 하나뿐이다."),

    # ── 부류 A : 판단·예외 (정답이 범주형) ───────────────────────────────────
    "T083": _a("ghost.csv 없음을 보고 · 저장/삭제/장비제어 0회 · 무중단",
               "정답 행동이 하나로 정해진다."),
    "T092": _a("5000~6000 은 입력(400~1800) 밖 → '추출 불가' 보고, 파일 생성 0개",
               "두 구간이 겹치지 않는다는 건 사실 판정이라 해석의 여지가 없다."),
    "T093": _a("전 구간 세기 0 → '신호 없음' 보고. 피크를 찾았다고 하면 오답",
               "입력이 전부 0 이라는 건 사실 판정이다."),
    "T104": _a("label = amorphous",
               "정답 라벨이 하나로 정해져 있다."),

    # ── 부류 A : 라이브러리 매칭 (순위는 지표 무관하게 동일) ─────────────────
    # 세 지표(cos_raw / pearson / baseline+L2 후 cosine)로 전부 계산해 본 결과
    # 물질·ID 순위가 모든 지표에서 동일했다. 따라서 순위·식별은 엄격 채점한다.
    # 단 '점수값'은 지표 의존(T111: 0.9983 / 0.9974 / 0.9972)이므로 값 비교가 아니라
    # '에이전트가 선언한 지표로 재계산했을 때 재현되는가'로 판정한다.
    #
    # 동점: reference_library.csv 의 PS_01 == PS_02 가 비트 단위로 동일하다(측정 확인).
    # reference_library_8.csv 는 PS_01=PS_02=PS_03 3중 동점. 동점군 내부는 순서 무관.
    "T111": _a("top-3 = polystyrene 2개(PS_01/PS_02 동점) + PMMA 1개, 점수 내림차순",
               "순위가 지표 무관하게 동일하다. 동점군 내부 순서는 묻지 않는다.",
               score_metric_free=True),
    "T112": _a("최대 유사도 ≈ 0.998 ≥ 0.85 → '동일 물질로 볼 수 있다'",
               "세 지표 모두 0.85 를 크게 넘어 임계 판정이 지표에 무관하다.",
               score_metric_free=True),
    "T113": _a("baseline+L2 후 최대 유사도 = polystyrene",
               "프롬프트가 전처리를 못박았고, 그 전처리에서 cos=1.0000 으로 압도적이다."),
    "T114": _a("최대 유사도 < 0.75 → '신뢰할 만한 매칭 없음'",
               "세 지표 최대값이 0.558/0.303/0.275 로 전부 임계 아래다. 지표 무관.",
               score_metric_free=True),
    "T115": _a("PET (PET 0.998 vs PMMA 0.631)",
               "격차가 커서 지표 무관하게 같은 답이 나온다.", score_metric_free=True),
    "T116": _a("혼합물의 우세 성분 = PET (PET 0.970 vs PMMA 0.801)",
               "지표 무관하게 PET 이 앞선다.", score_metric_free=True),
    "T117": _a("SG 평활 후 polystyrene",
               "저SNR 이지만 정답 물질은 하나다."),
    "T118": _a("peak_library.csv 와 ±3 cm⁻¹ 피크집합 매칭 → polystyrene",
               "피크 집합 매칭은 라이브러리가 고정이라 유일하다."),
    "T119": _a("8개 전체 순위: PS 3개(동점) > PMMA_01 > PET 2개(동점) > CAL_01 > SI_01",
               "전체 순위가 세 지표에서 동일하다. 동점군 내부는 순서 무관.",
               score_metric_free=True),
    "T120": _a("best match id = PS_01 (PS_02 와 동점이므로 둘 다 정답)",
               "PS_01 과 PS_02 가 비트 동일이라 어느 쪽을 답해도 옳다."),
    "T121": _a("520 cm⁻¹ 단일 피크 → silicon, 사용한 피크 520.45 보고",
               "피크 위치와 물질 대응이 라이브러리로 고정된다."),
    "T122": _a("calcite (calcite 0.996 vs aragonite 0.929)",
               "다형체 구분이지만 격차가 뚜렷하고 지표 무관하다.", score_metric_free=True),
    "T123": _a("polystyrene + 2위 후보와 구별되는 기준 피크 ≥2개 (±3 cm⁻¹)",
               "정답 물질과 구별 피크 집합이 라이브러리로 결정된다."),
    "T124": _a("순서대로 [polystyrene, PET, PMMA, calcite, silicon]",
               "5개 질의의 정답이 각각 하나씩이다."),
    "T125": _a("주장은 PET 이나 참값은 PMMA → 불일치 보고",
               "참값이 하나로 정해진다."),
    "T126": _a("공통축 보간 + 5차 baseline + L2 → polystyrene, 동점 시 사전순 앞",
               "프롬프트가 전처리와 동점 규칙을 모두 못박았다."),
    "T127": _a("이동량 +5 cm⁻¹ 추정 → 보정 후 polystyrene",
               "이동량이 정확히 +5 로 합성돼 있고 보정 후 유사도가 급등한다."),
    "T128": _a("질의별 precision@3 의 산술평균 (물질당 참조 2개 → 최대 0.6667)",
               "라이브러리와 질의가 고정이라 값이 유일하다."),

    # ── 부류 B : 방법군 ──────────────────────────────────────────────────────
    "T038": _b("정당한 5차 다항 baseline 보정 결과. 모양새 일치로 판정",
               ["적합 방식(plain/iterative/ALS)", "피크 마스킹 여부", "가중치"],
               "'5차 다항 baseline 보정'은 plain polyfit·iterative LMJ·ALS 가 모두 정당한 "
               "해석이고 실측 편차 S=67.5 다. 비트 일치를 요구하면 과학적으로 더 나은 "
               "방법(LMJ/ALS)이 탈락한다.",
               ensemble="baseline_poly5"),
    "T096": _b("T038 과 동일 + 원인을 형광 배경으로 지목 + 피크 7개 재검출",
               ["적합 방식", "피크 마스킹 여부"],
               "T038 과 완전히 같은 입력·같은 방법이다.",
               ensemble="baseline_poly5"),
    "T110": _b("강한 배경 지목 + baseline 후 피크 recall·precision ≥ 90%",
               ["baseline 적합 방식", "피크 검출 임계값"],
               "채점기준 자체가 이미 불변량(recall/precision ≥90%)으로 적혀 있다. "
               "그런데 verifier 는 reference_match 1e-5 를 요구해 서로 어긋난다.",
               ensemble="baseline_poly5",
               shape_override={"min_recall": 0.90, "min_precision": 0.90}),
    "T040": _b("SG(window=11, polyorder=3) 평활. edge mode 는 자유",
               ["edge mode(interp/nearest/mirror/constant)"],
               "scipy 의 mode 를 프롬프트가 지정하지 않는다. 실측 편차 S=27.9 "
               "(interp 7.8e-7 / nearest 8.19 / mirror 13.57 / constant 27.88).",
               ensemble="sgolay_11_3"),
    "T046": _b("despike → 5차 baseline → SG(11,3) → 0~1 정규화, 이 순서",
               ["despike 임계값·커널", "baseline 적합 방식", "SG edge mode"],
               "네 단계 각각에 자유 파라미터가 있어 오차가 누적된다. 순서 준수는 "
               "절차 점수로, 결과는 모양새로 본다."),
    "T039": _b("스파이크만 제거 — 비-스파이크 max|Δ| ≤1e-5, 제거율 ≥99%",
               ["검출 임계값", "커널 크기", "대체값 규칙"],
               "레퍼런스가 '스파이크를 넣기 전의 원본'이라 파괴된 값은 어떤 알고리즘으로도 "
               "복원 불가하다. answer_specs 가 이미 이 불변량을 쓰고 있다.",
               invariant="despike"),
    "T056": _b("스파이크 존재 판단 + 제거 + 피크 7개 재보고",
               ["검출 임계값", "커널 크기"],
               "T039 와 같은 구조.", invariant="despike"),
    "T099": _b("원인을 우주선으로 지목 + 스파이크 7개 제거 + 실제 피크 보존",
               ["검출 임계값", "커널 크기"],
               "T039 와 같은 구조. 스파이크가 7개다.", invariant="despike"),
    "T044": _b("~1001 피크의 FWHM. 참값은 로렌치안 해석해 2w = 12.0",
               ["베이스라인 차감 여부", "보간 여부"],
               "980~1020 구간 양끝이 30 cm⁻¹ 옆 1031 피크의 꼬리를 포함해 부풀려져 있어, "
               "베이스라인 차감 여부만으로 10.00~12.89 로 갈린다.",
               shape_override={}),
    "T071": _b("5차 baseline + L2 후 PCA(3) 설명분산비",
               ["baseline 적합 방식"],
               "전처리에 baseline 이 들어가 값이 방식에 따라 흔들린다. 성분 수·구조는 고정."),
    "T072": _b("5차 baseline + L2 후 k=2 군집, ARI ≥ 0.90",
               ["baseline 적합 방식", "군집 초기화"],
               "채점기준이 이미 ARI 임계로 적혀 있어 불변량 방식이다."),
}

# 파일처리 문항 전체 (task_files.json 47개 + ghost.csv 를 쓰는 T083)
FILE_TASKS = sorted(TASK_CLASS)


def get(task_id: str) -> dict | None:
    return TASK_CLASS.get(str(task_id))


def klass(task_id: str) -> str | None:
    e = get(task_id)
    return e["class"] if e else None


def is_class_b(task_id: str) -> bool:
    return klass(task_id) == CLASS_B


def shape_thresholds(task_id: str) -> dict:
    """부류 B 문항의 모양새 임계값. 문항별 오버라이드를 기본값 위에 얹는다."""
    e = get(task_id) or {}
    return {**DEFAULT_SHAPE, **(e.get("shape_override") or {})}


def summary() -> dict:
    a = [t for t in FILE_TASKS if klass(t) == CLASS_A]
    b = [t for t in FILE_TASKS if klass(t) == CLASS_B]
    return {"total": len(FILE_TASKS), "A": a, "B": b}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    s = summary()
    print(f"파일처리 문항 {s['total']}개")
    print(f"  부류 A (GT 확정, 엄격 채점) {len(s['A'])}개: {' '.join(s['A'])}")
    print(f"  부류 B (모양새 채점)       {len(s['B'])}개: {' '.join(s['B'])}")
