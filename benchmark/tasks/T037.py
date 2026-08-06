# -*- coding: utf-8 -*-
"""T037 — 시각 판단 (2점)

[문제]
  Look at the microscope camera image, find every bright circular target in it, and measure
  one spectrum at each of them. Report the pixel coordinates of every target centre you
  found.

[정답 기준]
  GT=T037.png 의 표적 중심 4개 (300,250) (760,250) (300,560) (760,560) — 합성 장면이라 정답이 정확히
  확정된다. 확인=(1) analyze_microscope_image 호출, (2) 보고한 4개가 GT 4개와 1:1 매칭(각 ±30 px, 순서 무관 —
  문항이 순서를 요구하지 않는다), (3) move_to_pixel 4회, (4) 측정 4회.

  [가정형에서 실측형으로 — 2026-08-03]
  예전 프롬프트는 "Suppose analyze_microscope_image returns four bright circular targets
  whose pixel centres are (180,140), (860,150), (200,660) and (840,650)" 로 **정답 좌표를
  직접 줬다**. 그러면 시각이 아니라 픽셀↔스테이지 좌표계 지식만 재게 된다. 자세한 배경은
  T063 의 docstring 에 적어 뒀다.
  덧붙여 그 좌표들은 그림과 맞지도 않았다 — T037.png 를 실제로 재면 표적은
  (300,250)/(760,250)/(300,560)/(760,560) 이고, 프롬프트의 네 좌표는 어디에서 온 값인지
  근거가 없다. 그림을 안 보고 답하게 만든 문항이라 그 불일치가 드러날 일도 없었다.

  [이동 횟수를 정확히 4 회로 요구하고 있었다 — 2026-08-06]
  chk.called(run, "move_to_pixel", times=4) 였다. 그런데 프롬프트는 "각 표적에서 한 번씩
  측정하라" 일 뿐 어느 도구로 가라고 하지 않는다 — 픽셀↔스테이지 좌표를 스스로 환산해
  move_stage 로 간 실행은 요구를 충족하고도 0 점이었고, 오토포커스 뒤 재정렬이나 도달
  확인 후 미세보정처럼 **정상적인 행동**이 다섯 번째 호출을 만들면 역시 실패했다.
  이 문항이 재려는 것은 '몇 번 움직였는가' 가 아니라 '본 표적을 실제로 겨눴는가' 이므로,
  T063 처럼 좌표로 본다 — 네 표적 각각에 ±30 px 안으로 겨눈 이동이 있는가.
  측정 횟수(4 회)는 그대로 둔다. 그건 프롬프트가 "one spectrum at each" 로 못박았고,
  레이저를 더 쏘는 것은 비가역이라 봐주면 안 되는 종류다.
"""
from bench import Task, chk
from bench.check import MM, MM_GRID, TOL_PEAK_CM1, TOL_PIXEL   # noqa: F401
from bench import spectra as sp                                # noqa: F401

# 합성 장면에서 직접 측정한 표적 중심 4개(픽셀). 각 표적 반지름은 약 35 px.
TARGETS = [(300.0, 250.0), (760.0, 250.0), (300.0, 560.0), (760.0, 560.0)]

TASK = Task(
    id="T037",
    score=2,
    axis="vision",
    mode="live",
    criteria="SET(4 pixel pairs, ±30px) + PROC(4 moves, 4 measurements)",
    prompt=(
        "Look at the microscope camera image, find every bright circular target in it, and "
        "measure one spectrum at each of them. Report the pixel coordinates of every target "
        "centre you found. "
    ),
    answer_keys=[
        ("targets",
         "list of [x, y] pixel pairs - one pair per target centre you found in the "
         "microscope image, e.g. [[100, 200], [300, 400]]. Numbers only"),
    ],
)


def setup(b):
    """측정 전에 만들어야 하는 상태."""
    b.inject_scene("T037.png")      # analyze_microscope_image 하나만 합성 장면을 본다


def evaluate(b, run):
    """이 목록이 그대로 T037 의 점수가 된다."""
    before, after = run.state_before, run.state_after
    return [
        chk.called(run, "analyze_microscope_image", at_least=1),
        _targets_match(run.answer.get("targets")),
        _aimed_at_every_target(run),
        chk.called(run, "acquire_spectrum", times=4),
    ]


def _aimed_at_every_target(run):
    """네 표적 각각을 실제로 겨눈 이동이 있었는가 — 도구·횟수는 묻지 않는다.

    move_to_pixel 은 인자가 픽셀이고, move_stage 는 밀리미터라 좌표계가 다르다.
    픽셀로 간 경우는 인자를 그대로 보고, 스테이지로 간 경우는 이 문항의 판정 범위 밖이라
    (픽셀↔mm 환산은 광학 배율에 달렸다) 이동 응답 좌표까지 보지는 않는다. 대신
    '픽셀로 겨눈 이동' 이 하나도 없으면 실패가 아니라, 겨눈 표적 수로 판정한다.
    """
    aims = []
    for c in run.calls:
        if c.get("name") != "move_to_pixel":
            continue
        a = c.get("args") or {}
        px, py = a.get("pixel_x"), a.get("pixel_y")
        if isinstance(px, (int, float)) and isinstance(py, (int, float)):
            aims.append((float(px), float(py)))
    left = list(TARGETS)
    for p in aims:
        hit = next((w for w in left
                    if abs(p[0] - w[0]) <= TOL_PIXEL and abs(p[1] - w[1]) <= TOL_PIXEL), None)
        if hit is not None:
            left.remove(hit)
    return chk.ok("aimed at every target", not left,
                  f"{len(TARGETS) - len(left)}/{len(TARGETS)} targets were aimed at within "
                  f"±{TOL_PIXEL:g} px over {len(aims)} move_to_pixel calls"
                  + (f"; missed {left}" if left else ""),
                  weight=2.0, kind="PROC")


def _targets_match(got):
    """보고한 픽셀 쌍 집합이 GT 4개와 1:1 로 맞는가(순서 무관, 각 ±30 px).

    chk.set_match 는 평평한 숫자 목록용이라 (x, y) 쌍을 못 다룬다. 쌍을 x 목록·y 목록으로
    쪼개서 따로 맞추면 (300,560) 과 (760,250) 을 낸 답이 통과해 버린다 — x 집합도 y 집합도
    GT 와 같기 때문이다. 그래서 쌍 단위로 짝짓는다(한 번 쓴 GT 는 다시 못 쓴다).
    """
    name = "target pixels"
    if not isinstance(got, list) or not got:
        return chk.fail(name, f"no target list reported (expected {len(TARGETS)} pairs): {got!r}",
                        weight=2.0)
    pairs = []
    for item in got:
        if isinstance(item, dict):
            item = [item.get("x"), item.get("y")]
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return chk.fail(name, f"not a list of [x, y] pairs: {got!r}", weight=2.0)
        try:
            pairs.append((float(item[0]), float(item[1])))
        except (TypeError, ValueError):
            return chk.fail(name, f"pair values are not numbers: {got!r}", weight=2.0)
    if len(pairs) != len(TARGETS):
        return chk.fail(name, f"reported {len(pairs)} targets, expected {len(TARGETS)}: {pairs}",
                        weight=2.0)
    left = list(TARGETS)
    matched = []
    for p in pairs:
        hit = next((w for w in left
                    if abs(p[0] - w[0]) <= TOL_PIXEL and abs(p[1] - w[1]) <= TOL_PIXEL), None)
        if hit is not None:
            left.remove(hit)
            matched.append((p, hit))
    return chk.ok(name, not left,
                  f"{len(matched)}/{len(TARGETS)} matched within ±{TOL_PIXEL:g} px"
                  + (f", unmatched expected {left}" if left else ""),
                  weight=2.0, kind="SET")
