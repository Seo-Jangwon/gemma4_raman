# -*- coding: utf-8 -*-
"""T063 — 시각 판단 (3점)

[문제]
  Look at the microscope camera image, find the single bright circular target, move the
  stage onto it, and acquire one spectrum there. Report the pixel coordinates of the target
  centre in the image you were shown.

[정답 기준]
  GT=T063.png 의 표적 중심 (690, 300) — 합성 장면이라 정답 픽셀이 정확히 확정된다. 확인=(1)
  analyze_microscope_image 를 실제로 호출, (2) 보고한 픽셀이 ±30 px, (3) move_to_pixel 을 그 좌표로 호출
  (한 호출 안에서 x·y 둘 다), (4) 그 자리에서 측정 1회.

  [가정형에서 실측형으로 — 2026-08-03]
  예전 이 문항은 mode="hypothetical" 이었고 프롬프트가 "Suppose analyze_microscope_image
  returns one bright circular target centred at pixel (512, 384)" 로 **정답 좌표를 직접 줬다**.
  그러면 재는 것은 '타깃을 찾는 능력'이 아니라 '찾겠다고 말하는 능력', 정확히는
  픽셀↔스테이지 좌표계를 아는가 하나뿐이었다. 시각 축 3문항(T037·T063·T076)이 전부 그랬으므로
  이 벤치의 시각 해결률은 시각을 재고 있지 않았다.
  필요한 재료는 이미 다 있었다: benchmark/inputs/T063.png 가 존재하고,
  bench_ops.inject_scene 이 analyze_microscope_image **하나만** 합성 장면으로 갈아끼우며
  (나머지 광학은 진짜라 오토포커스가 깨지지 않는다), 벤치 에이전트는 도구가 돌려준
  image_base64 를 별도 이미지 블록으로 모델에 실제로 보여 준다.
  그림 속 표적 중심은 직접 재서 확인했다 — 1060×800 화면에 밝은 원 1개, 중심 (690, 300),
  반지름 약 41 px. 예전 프롬프트가 말하던 (512, 384) 와는 아무 관계가 없는 값이었다.
  허용오차 ±30 px 는 표적 반지름 안쪽이라 '표적을 찍었는가'가 되고, 1060×800 에서 그 상자는
  전체 면적의 0.4% 라 찍어서 맞을 수는 없다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1, TOL_PIXEL   # noqa: F401
from bench import spectra as sp                                # noqa: F401

# 합성 장면에서 직접 측정한 표적 중심(픽셀). 그림이 바뀌지 않는 한 정답은 이 하나다.
TARGET = [690.0, 300.0]

TASK = Task(
    id="T063",
    score=3,
    axis="vision",
    mode="live",
    criteria="NUM(pixel ±30px) + PROC(move_to_pixel on that pixel) + PROC(1 measurement)",
    prompt=(
        "Look at the microscope camera image, find the single bright circular target, move "
        "the stage onto it, and acquire one spectrum there. Report the pixel coordinates of "
        "the target centre in the image you were shown. "
    ),
    answer_keys=[
        ("target_pixel",
         "list of 2 numbers - [x, y] pixel coordinates of the target centre in the "
         "microscope image. Numbers only, not objects"),
    ],
)


def setup(b):
    """측정 전에 만들어야 하는 상태."""
    b.inject_scene("T063.png")      # analyze_microscope_image 하나만 합성 장면을 본다


def evaluate(b, run):
    """이 목록이 그대로 T063 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "analyze_microscope_image", at_least=1),
        chk.point("reported target pixel", run.answer.get("target_pixel"), TARGET,
                  tol=TOL_PIXEL, weight=2.0),
        # 본 것을 실제로 실행에 옮겼는가. chk.arg 를 두 번 쓰면 x 는 A 호출에서 y 는 B
        # 호출에서 맞아도 통과하므로 짝으로 본다.
        chk.arg_pair(run, "move_to_pixel", ("pixel_x", "pixel_y"), TARGET,
                     tol=TOL_PIXEL, name="move_to_pixel aimed at the target", weight=2.0),
        chk.called(run, "acquire_spectrum", times=1),
    ]
