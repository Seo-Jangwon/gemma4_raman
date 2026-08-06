# -*- coding: utf-8 -*-
"""T107 — 진단 복구 (3점)

[문제]
  Suppose a spectrum shows a strong broad component unrelated to the sample and you suspect
  room light entering the spectrometer. State the tools you would call, in order, to
  quantify the external-light contribution, and state the subtraction you would perform.
  decision must be one of: normal_minus_dark, dark_minus_normal. Do not operate the
  instrument for this question — answer only.

[정답 기준]
  GT(plan)=[acquire_spectrum, acquire_spectrum] — 셔터를 닫고 한 번, 열고 한 번.
  여분 단계는 용서한다(chk.plan_order). GT(차감 단계)=두 스펙트럼을 빼겠다는 말이
  계획이나 본문에 있을 것 — 도구명을 어느 것으로 대든 상관없다.
  GT(decision)=normal_minus_dark. 차감 방향이 뒤집히면 오답 — 그게 이 문항의 핵심이다.

[실제 측정을 채점하지 않는 이유 — 2026-08-03]
  예전 채점기는 mode="hypothetical" 인데도 acquire_spectrum(shutter=…) 실호출 2 건과
  저장 파일 2 개를 요구하고 그 배열로 차감 결과를 검증했다. 프롬프트는 "장비를 만지지
  말고 부를 도구를 순서대로 말하라"고 한다. 계획을 정확히 낸 실행이 '측정을 안 했다'는
  이유로 0 점이었다. 명세가 요구한 것(계획과 차감 방향)만 본다.

  [메타데이터가 판정과 다른 말을 하고 있었다 — 2026-08-06]
  criteria 가 "PROC(shutter args EXACT) + ARRAY(post-hoc GT, rtol 1e-6) + NUM(ratio 5%)"
  였다 — 2026-08-03 개정 때 evaluate 만 바꾸고 이 문자열을 안 고친 잔재로, 실제 판정과
  아무 관계가 없는데 결과 파일에 그대로 실려 읽는 사람을 오해시켰다. 판정에 맞춰 다시 썼다.
  그리고 위 [문제] 에는 ```json 블록 형식 지시와 rationale 키가 있었는데 TASK.prompt 와
  answer_keys 에는 없었다. 출력 규약은 harness(client.output_contract)가 모든 문항에
  똑같이 붙이므로, 문항이 따로 적을 것이 아니다 — [문제] 에서 지웠다.

  [계획 GT 가 특정 도구명 하나에 걸려 있었다 — 2026-08-06]
  plan GT 가 [acquire_spectrum, acquire_spectrum, **run_analysis**] 였다. 프롬프트가
  요구한 것은 "부를 도구를 순서대로 말하고, 수행할 차감을 말하라" 이고 차감 방향은
  decision 키로 따로 채점된다. 세 번째 단계로 run_analysis 를 대는 것은 '이 프로젝트에서
  수치 연산은 코드 실행 도구로 한다' 는 구현 지식이지 이 문항이 재려는 진단 능력이 아니라,
  두 번 찍고 "그다음 두 배열을 빼겠다" 고 정확히 서술한 답이 떨어졌다. 차감 단계는 도구명을
  가리지 않고 본다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1        # noqa: F401
from bench import spectra as sp                          # noqa: F401
import numpy as np                                       # noqa: F401

TASK = Task(
    id="T107",
    score=3,
    axis="diagnostics",
    mode="hypothetical",
    criteria="PROC(0 acquisitions) + PLAN(2 acquisitions in order) + EXACT(subtraction direction)",
    needs=(
        "Optional: turning the room light on lets real stray light in. Grading works "
        "without it (the fraction is simply near 0)."
    ),
    prompt=(
        "Suppose a spectrum shows a strong broad component unrelated to the sample and you "
        "suspect room light entering the spectrometer. State the tools you would call, in order, "
        "to quantify the external-light contribution, and state the subtraction you would "
        "perform. decision must be one of: normal_minus_dark, dark_minus_normal. "
        "Do not operate the instrument for this question — answer only. "
    ),
    answer_keys=[
        ("plan", "list of tool-name strings, in the order you would call them"),
        ("decision", 'string - either "normal_minus_dark" or "dark_minus_normal"'),
    ],
)


def evaluate(b, run):
    """이 목록이 그대로 T107 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        # 답만 하라고 했다 — 장비를 만지면 지시 불이행.
        chk.called(run, "acquire_spectrum", times=0),
        chk.ok("answer present", len((run.text or "").split()) >= 20,
               f"{len((run.text or '').split())} words", kind="PLAN"),
        # 셔터 닫고 한 번, 열고 한 번. 확인 단계를 끼워 넣는 것은 용서한다.
        chk.plan_order(run, ['acquire_spectrum', 'acquire_spectrum']),
        # 그리고 두 스펙트럼을 뺀다. 어느 도구로 빼겠다고 하든 상관없다 — 프롬프트가
        # 요구한 것은 '수행할 차감을 말하라' 이지 도구명을 맞히라가 아니다.
        chk.ok("subtraction step stated",
               any(t in [str(s).lower() for s in (run.answer.get("plan") or [])]
                   for t in ("run_analysis", "combine_spectra",
                             "apply_background_subtraction"))
               or any(k in (run.text or "").lower()
                      for k in ("subtract", "difference", "minus")),
               "an explicit subtraction of the two spectra was stated", kind="PLAN"),
        # 방향이 뒤집히면 배경이 아니라 신호를 빼는 것이다 — 이 문항의 핵심.
        chk.reported_label(run, "decision", "normal_minus_dark",
                           ['normal_minus_dark', 'dark_minus_normal'], name="decision"),
    ]
