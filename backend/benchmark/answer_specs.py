# -*- coding: utf-8 -*-
"""방법이 프롬프트에 지정되지 않은 문항의 '정답 정의' 데이터.

[왜 이 파일이 있는가]
T038/T039/T044/T046/T056/T096/T099 은 프롬프트가 방법을 지정하지 않아서, 에이전트가
정당한 대안을 쓰면 정답 레퍼런스와 어긋난다. 그런 문항은 'max|Δ| 가 크다'만 봐서는
오답인지 방법차이인지 구분할 수 없다. 그래서 문항마다
  ① 정답이 정확히 어떻게 만들어졌는가(생성기 레시피 원문과 그 재현 함수)
  ② 왜 애매한가
  ③ 무엇을 만족하면 정답으로 볼 것인가
를 여기 적어 두고, review.py 가 채점 콘솔의 '정답 기준' 블록에 그대로 싣는다.
recipe_fn 은 review.py 가 '이 레시피가 레퍼런스 CSV 를 실제로 재현하는지' 검증해
그 오차를 함께 보여주는 데 쓴다 — 정답 정의가 추측이 아니라는 증거다.

[핵심 — 스파이크 계열의 정답 기준]
T039/T056/T099 의 레퍼런스는 '스파이크를 지운 스펙트럼'이 아니라 '스파이크를 넣기
전의 깨끗한 스펙트럼'이다(make_task_spectra: refs={...: (AXIS, ps)}). 스파이크는 단일
점에 +5000 을 더한 것이라 원래 값은 소실됐고, 어떤 despike 알고리즘도 정확히 복원할
수 없다. 따라서 '전 구간 일치'는 정답 조건이 될 수 없고, 올바른 조건은 두 가지다:
  · 스파이크가 아닌 점을 건드리지 않았는가(비트 단위 동일)
  · 스파이크 위치에서 +5000 초과분을 얼마나 제거했는가(제거율)
이 기준으로 보면 max|Δ| 가 2.1e-01 이라 '불일치' 로 보였던 답이 실제로는
비-스파이크 1796점 완전일치 + 스파이크 100% 제거였다.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import medfilt, savgol_filter

# 생성기 상수 (make_task_spectra.py 와 일치해야 한다)
_LW = 6.0
_SPIKE_HEIGHT = 5000.0

def poly_bl(axis, inten, order=5):
    return inten - np.polyval(np.polyfit(axis, inten, order), axis)


def despike(inten, thresh=6.0, win=5):
    med = medfilt(inten, kernel_size=win)
    resid = inten - med
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    mask = np.abs(resid) > thresh * 1.4826 * mad
    out = inten.copy()
    out[mask] = med[mask]
    return out


def sgolay(inten, w=11, p=3):
    return savgol_filter(inten, window_length=w, polyorder=p, mode="interp")


def minmax(inten):
    lo, hi = float(inten.min()), float(inten.max())
    return (inten - lo) / (hi - lo) if hi > lo else np.zeros_like(inten)


# ── 문항별 정답 명세 ─────────────────────────────────────────────────────────
# kind: exact(전 구간 일치) | despike(스파이크만 제거) | baseline(베이스라인 추정)
#       | pipeline(다단 파이프라인) | scalar(단일 수치)

SPECS: dict[str, dict] = {
    "T038": {
        "kind": "baseline",
        "ref": "T038_reference.csv",
        "recipe": "reference = input - polyval(polyfit(raman_shift, input, 5), raman_shift)",
        "recipe_note": (
            "5차 다항식을 <b>전체 1801점에 통상 최소제곱으로 한 번</b> 적합한다. "
            "피크 마스킹도, 반복(iterative) 재적합도, 가중치도 없다. "
            "적합 변수는 <b>라만 시프트 축(200~2000 cm⁻¹)</b>이며 인덱스가 아니다."),
        "recipe_fn": lambda x, y: poly_bl(x, y),
        "pass_if": [
            "<b>지정 방법을 요구한다면</b>: 전 구간 max|Δ| ≤ 1e-5. 즉 통상 최소제곱 5차 적합이어야 한다.",
            "<b>임의의 타당한 5차 베이스라인을 허용한다면</b>: 피크 위치·상대 세기가 보존되면 정답. "
            "아래 '선형관계' 기울기가 1.000 이면 피크 형상은 보존된 것이고, 차이는 베이스라인 추정값뿐이다.",
        ],
        "why_ambiguous": (
            "프롬프트는 “5차 다항식 베이스라인 보정”만 말하고 '피크를 제외하지 말 것'이나 "
            "'반복하지 말 것'을 말하지 않는다. 실제 라만 분석에서는 피크를 마스킹하고 반복 적합하는 "
            "쪽이 더 표준적이라, 에이전트가 그쪽을 고르면 레퍼런스와 어긋난다."),
    },
    "T096": {
        "kind": "baseline",
        "ref": "T096_reference.csv",
        "recipe": "reference = input - polyval(polyfit(raman_shift, input, 5), raman_shift)",
        "recipe_note": "T038 과 <b>완전히 같은 방법·같은 입력</b>이다(생성기에서 동일한 ps_fl 배열을 쓴다).",
        "recipe_fn": lambda x, y: poly_bl(x, y),
        "pass_if": [
            "원인을 <b>형광 배경(fluorescence)</b>으로 지목했는가 — 이건 answer_contains 로 자동 확인된다.",
            "보정 후 주요 피크 7개를 재검출했는가 — answer_numeric(±3 cm⁻¹) 로 자동 확인된다.",
            "스펙트럼 파일 자체는 T038 과 같은 판정 기준을 적용한다(위 두 갈래).",
        ],
        "why_ambiguous": "T038 과 동일.",
    },
    "T039": {
        "kind": "despike",
        "ref": "T039_reference.csv",
        "recipe": "reference = 스파이크를 넣기 전의 깨끗한 스펙트럼 (despike 결과가 아니다)",
        "recipe_note": (
            "입력은 <code>out[i] += 5000</code> 으로 단일 점 5곳에 +5000 을 더한 것이다. "
            "레퍼런스는 그 이전의 원본이므로, <b>파괴된 원래 값을 정확히 복원하는 것은 어떤 "
            "알고리즘으로도 불가능하다</b>. 따라서 '전 구간 일치'를 요구하면 안 된다."),
        "recipe_fn": None,
        "pass_if": [
            "<b>스파이크가 아닌 점을 건드리지 않았는가</b> — 비-스파이크 max|Δ| ≤ 1e-5.",
            "<b>스파이크 위치에서 +5000 초과분을 제거했는가</b> — 제거율 ≥ 99%.",
            "이 두 조건을 만족하면 정답이다. 남은 잔차(수 단위)는 국소 보간 오차이며 방법 무관하게 발생한다.",
        ],
        "why_ambiguous": (
            "채점기준에 '절대오차 1e-8' 이 적혀 있으나 <b>물리적으로 도달 불가</b>하다 — "
            "레퍼런스 CSV 가 세기를 %.6f 로 저장해 양자화 오차만 5e-7 이고, 스파이크 위치의 "
            "원래 값은 애초에 소실됐다."),
    },
    "T056": {
        "kind": "despike",
        "ref": "T056_reference.csv",
        "recipe": "reference = 스파이크를 넣기 전의 깨끗한 스펙트럼",
        "recipe_note": (
            "T039 와 같은 구조(+5000 단일점 스파이크)다. 다만 ground_truth 에 스파이크 위치가 "
            "기록돼 있지 않으므로, 이 리포트는 <b>입력과 레퍼런스의 차가 1000 을 넘는 점</b>을 "
            "스파이크로 역산해서 판정한다."),
        "recipe_fn": None,
        "pass_if": [
            "<b>스파이크 존재를 옳게 판단했는가</b> (있다고 답해야 정답).",
            "<b>비-스파이크 점 max|Δ| ≤ 1e-5</b>, <b>스파이크 제거율 ≥ 99%</b>.",
            "<b>despike 후 주요 피크 7개 보고</b> — answer_numeric(±3 cm⁻¹) 로 자동 확인된다.",
        ],
        "why_ambiguous": "T039 와 동일.",
    },
    "T099": {
        "kind": "despike",
        "ref": "T099_reference.csv",
        "recipe": "reference = 스파이크를 넣기 전의 깨끗한 스펙트럼 (스파이크 7개)",
        "recipe_note": "T039 와 같은 구조. 스파이크가 5개가 아니라 7개다.",
        "recipe_fn": None,
        "pass_if": [
            "원인을 <b>우주선(cosmic ray)</b>으로 지목했는가 — answer_contains 로 자동 확인.",
            "<b>비-스파이크 점 max|Δ| ≤ 1e-5</b>, <b>스파이크 제거율 ≥ 99%</b>.",
            "<b>보호 대상 실제 피크(7개)가 훼손되지 않았는가</b> — 비-스파이크 일치가 이를 함의한다.",
        ],
        "why_ambiguous": "T039 와 동일. 채점기준의 '면적오차 5% 이내'는 비-스파이크 일치로 자동 충족된다.",
    },
    "T046": {
        "kind": "pipeline",
        "ref": "T046_reference.csv",
        "recipe": "reference = minmax( savgol( poly_bl( despike(input) ) ) )",
        "recipe_note": (
            "네 단계를 <b>이 순서로</b> 적용한다:<br>"
            "① despike: 5점 중앙값 필터와의 잔차가 <code>6.0 × 1.4826 × MAD</code> 를 넘는 점을 "
            "그 중앙값으로 대체<br>"
            "② baseline: 5차 다항식 통상 최소제곱(전체 점, 피크 마스킹 없음)<br>"
            "③ smoothing: Savitzky-Golay window=11, polyorder=3, <b>mode='interp'</b>(양끝을 "
            "다항 보간으로 처리, 패딩 아님)<br>"
            "④ normalize: (x−min)/(max−min) → 0~1"),
        "recipe_fn": lambda x, y: minmax(sgolay(poly_bl(x, despike(y)))),
        "pass_if": [
            "네 단계를 <b>지정 순서대로</b> 적용했는가(순서가 바뀌면 결과가 달라진다).",
            "<b>0~1 정규화 스케일에서 max|Δ| ≤ 0.01(=1%)</b> 이면 형상은 정답으로 볼 수 있다. "
            "① 의 스파이크 대체값이 조금 달라도 ②③④ 를 타고 전 구간에 작은 차이로 퍼진다.",
            "완전 일치(≤1e-5)를 요구하려면 ① 의 임계값·커널까지 같아야 하는데 프롬프트가 그걸 지정하지 않는다.",
        ],
        "why_ambiguous": (
            "프롬프트는 “spike removal”만 말하고 검출 임계값·커널 크기·대체값 규칙을 지정하지 않는다. "
            "SG 의 edge 처리 모드도 지정하지 않는다."),
    },
    "T044": {
        "kind": "scalar",
        "ref": None,
        "expected": 12.0,
        "rel_tol": 0.05,
        "recipe": "ground_truth = round(2 × _LW, 2) = round(2 × 6.0, 2) = 12.0",
        "recipe_note": (
            "이 값은 측정값이 아니라 <b>합성에 쓴 로렌치안의 해석적 FWHM</b> 이다. "
            "피크 모형은 <code>L(x) = w²/((x−c)² + w²)</code>, w=6.0 이고 이 함수의 FWHM 은 "
            "정확히 2w = 12.0 이다. 즉 '참값'이 12.0 이다.<br><br>"
            "직접 4가지 방법으로 재봤을 때:<br>"
            "· 베이스라인 미차감 + 보간 없음 → <b>12.00</b> (참값과 일치)<br>"
            "· 베이스라인 미차감 + 보간 → 12.89<br>"
            "· 구간 양끝 베이스라인 차감 + 보간 없음 → 10.00<br>"
            "· 구간 양끝 베이스라인 차감 + 보간 → <b>10.62</b> (에이전트)<br><br>"
            "베이스라인을 차감하면 <b>더 좁게</b> 나오는데, 이유는 980~1020 구간의 양끝값이 "
            "순수 바닥(40)이 아니라 <b>30 cm⁻¹ 옆 1031 피크의 꼬리</b>를 포함해 190 근처로 "
            "부풀려지기 때문이다. 그 부풀려진 값을 베이스라인으로 빼면 반치 높이가 올라가 "
            "폭이 좁아진다. 즉 에이전트의 방법은 인접 피크 오염으로 <b>계통 편향</b>을 얻었다."),
        "recipe_fn": None,
        "pass_if": [
            "채점기준은 <b>레퍼런스의 5% 이내</b> = 11.4 ~ 12.6 이다.",
            "에이전트의 10.62 는 참값 12.0 대비 <b>−11.5%</b> 로 기준을 벗어난다 → 기준대로면 오답.",
            "다만 프롬프트가 베이스라인 처리를 지정하지 않았다는 점은 감안할 여지가 있다. "
            "베이스라인 차감 자체는 통상 옳은 절차이고, 여기서 문제가 된 건 인접 피크가 구간에 "
            "걸쳐 있다는 이 문항 특유의 사정이다.",
        ],
        "why_ambiguous": "프롬프트가 “980-1020 구간에서 FWHM 계산”만 말하고 베이스라인 처리를 지정하지 않는다.",
    },
}
