# -*- coding: utf-8 -*-
"""T105 — 진단 복구 (3점)

[문제]
  T105.csv holds 10 consecutive measurements of the same spot (column frame_index 0-9).
  Determine whether the Raman signal decreases and the background increases over time,
  quantify both trends by linear regression slope, and name the cause. For each frame take
  the background as the median of that frame, and the signal as the frame maximum minus
  that median.

[정답 기준]
  [TS-A] 실제 광표백은 시료를 손상시키므로 재현하지 않는다. 추세를 주입한 시계열 파일로 대체했다. GT(원인)=광표백(photobleaching) —
  키워드 채점. GT(수치)=신호 기울기<0, 배경 기울기>0 (각 기울기 값은 상대오차 10%). 확인=두 기울기의 부호가 GT와 일치할 것(부호가 핵심,
  크기는 부차).

  [신호 정의가 없어 기울기가 2배로 갈렸다 — 2026-08-06]
  GT 는 make_dataset 이 프레임마다 signal = max - median, background = median 으로
  뽑아 회귀한 값인데 프롬프트에는 그 정의가 없었다. 입력으로 재 보면:
      배경  median 24.86(GT) / 1800-1900 평균 25.07 / 최소 25.40 / 전체평균 23.97
            → 어떤 합리적 정의를 써도 ±10% 안이라 안전하다
      신호  max - median -49.29(GT) / 1001밴드 - 1800-1900평균 -49.50
            **원시 피크 최대값 -24.43** ← 정확히 절반. 확정 오답이었다
  주입값이 진폭 -50/frame 과 배경 +25/frame 이라 배경을 안 빼면 둘이 상쇄된다. 즉 이
  문항은 "배경을 뺐는가"만으로 정오가 갈리는데 그걸 묻지 않고 있었다. 정의를 프롬프트에
  넣어 GT 와 일치시킨다(GT 값·데이터는 그대로).

  [프롬프트가 요구한 것을 안 재고 있었다 — 2026-08-06]
  프롬프트가 "name the cause" 를 요구하고 criteria 가 KEYWORD(photobleaching) 를
  약속하고 GT json 에 cause_keywords 까지 있는데 evaluate 에는 그 판정이 없었다 —
  "원인은 온도 드리프트입니다" 라고 답해도 만점이었다. 그리고 정답 기준이 "부호가 핵심"
  이라고 적었는데 정작 부호 판정 없이 크기 판정만 있었다. 둘 다 세운다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401

TASK = Task(
    id="T105",
    score=3,
    axis="diagnostics",
    mode="live",
    inputs=['T105.csv'],
    criteria="KEYWORD(photobleaching) + NUM(slope 10%) + REL(sign)",
    prompt=(
        "T105.csv holds 10 consecutive measurements of the same spot (column frame_index "
        "0-9). Determine whether the Raman signal decreases and the background increases "
        "over time, quantify both trends by linear regression slope against the frame "
        "index, and name the cause. For each frame take the background as the median of "
        "that frame, and the signal as the frame maximum minus that median. "
    ),
    answer_keys=[
        ("signal_slope", "number - slope of the Raman signal over the frames"),
        ("background_slope", "number - slope of the background over the frames"),
        ("cause", "string - the physical cause, in one or two words"),
    ],
)

GT_SIGNAL_SLOPE = -49.28848874219068
GT_BACKGROUND_SLOPE = 24.855960678628154


def evaluate(b, run):
    """이 목록이 그대로 T105 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.reported(run, "signal_slope", GT_SIGNAL_SLOPE, rel=0.1, name="signal slope"),
        chk.reported(run, "background_slope", GT_BACKGROUND_SLOPE, rel=0.1,
                     name="background slope"),
        # 부호가 이 문항의 핵심이다 — 정답 기준이 그렇게 적어 놓고 재지 않고 있었다.
        # 크기 판정과 따로 세워야 '방향은 맞는데 정의가 달랐다'가 구분된다.
        chk.relation("signal decreases", run.answer.get("signal_slope"), "<", 0),
        chk.relation("background increases", run.answer.get("background_slope"), ">", 0),
        _cause_named(run),
    ]


CAUSE_KEYWORDS = ['photobleach', 'bleach', '광표백']


def _cause_named(run):
    """원인을 광표백이라고 답했는가 — answer 우선, 없으면 본문.

    표기가 여러 가지(photobleaching / photo-bleaching / 광표백)라 레이블 완전일치는
    문체를 재게 된다. 뜻이 같으면 통과시키되, answer_keys 로 선언한 'cause' 를 실제로
    읽는다(선언만 하고 안 읽으면 run_all --check 가 잡는다).
    """
    said = f"{run.answer.get('cause') or ''} {run.text or ''}".lower()
    hit = [k for k in CAUSE_KEYWORDS if k in said]
    return chk.ok("cause named", bool(hit),
                  f"found {hit}" if hit else f"not found {CAUSE_KEYWORDS}", kind="KEYWORD")
