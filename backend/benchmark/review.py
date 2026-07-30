# -*- coding: utf-8 -*-
"""단일 채점 콘솔 — 문항 하나를 한 화면에서 보고 정답/오답을 한 번에 찍는다.

[왜 이 파일이 있는가]
채점 산출물이 셋으로 쪼개져 있었다(트레이스+스펙트럼 리포트 / 방법 미지정 문항의 정답
근거 / 콘솔 요약). 셋을 번갈아 보면서 좌표는 트레이스 텍스트로 상상해야 했다. 그래서
전부 이 파일 하나로 합쳤다 — 채점에 필요한 것을 (문항×에이전트) 하나의 셀에 모아 넣고,
판정 버튼을 그 자리에 붙인다:

  · 로그          — 전체 툴 트레이스(인자·결과 원본 JSON 접기), CoALA planning 단계
  · 정답 기준     — 문항의 채점기준 + 수동채점 노트 + 기계검증기를 '사람 말'로 풀어쓴 목록
                    + 방법 미지정 문항이면 answer_specs 의 정답 정의·판정조건과
                      '그 레시피가 레퍼런스를 재현하는지' 실측 오차까지
  · 신호 가공     — 입력 원본 / 산출물 / 정답 레퍼런스를 같은 x축에 겹친 그림 + max|Δ|
  · 화면          — 현미경 스냅샷·그리드 프리뷰·측정 PNG·분석 그림 갤러리
  · 좌표          — 어디를 어떤 순서로 찍었는지 XY 지도 + 격자성 진단(간격·등간격·중심)
  · 판정          — 정답 / 부분 / 오답 버튼 + 메모

[무응답은 자동 오답]
모델이 응답을 못 만들면 두 에이전트 모두 "Failed to generate a response." 를 최종 답변에
넣는다(verifiers._no_answer). 이 경우 판정을 '오답'으로 미리 채워 저장하고 배지를 달아
따로 볼 필요가 없게 한다. 사람이 뒤집을 수는 있다.

[판정 저장]
브라우저 localStorage 에 `rev1.<run_id>|<agent>` 키로 저장한다. 리포트를 다시 생성해도
키가 같아서 판정이 유지된다(파일명·시각을 키에 넣지 않는 이유다). 상단 '내보내기' 로
JSON/CSV 를 받는다.

다만 저장소는 그 브라우저 안에만 있다. 그림은 data: URI 로 HTML 에 박혀 있어 파일만
옮겨도 보이지만 판정은 안 따라가고, file:// 에서는 브라우저가 저장소를 통째로 막는
경우도 있다. 그래서 '채점 포함 저장'(exportSnapshot) 이 현재 판정·메모를 HTML 안에
`window.__GRADES__` 로 구워 새 파일로 내려받는다 — 그 한 장이면 어느 PC 에서든 그림과
채점이 함께 열린다. 직렬화 직전에 하이드레이트된 data: URI 를 걷어내 같은 base64 가
두 번 들어가지 않게 한다(stage_map.HYDRATE_JS 가 data-img 색인을 남겨 두는 이유).
열 때는 빈 셀만 채우고(그 PC 의 판정이 우선), 배너의 '전부 덮어쓰기' 로만 대체한다.

[채점 경로는 이것 하나다]
raw_runs.jsonl(run_bench 가 append) → review.py → HTML. 중간 파일(runs_*.json,
graded_*.json)을 만드는 별도 단계는 없앴다. 중복 실행 제거와 채점설정 갱신을 이 안에서
하기 때문이다(_dedupe / _refresh_config).

실행:
  python -m backend.benchmark.review                                   # raw_runs.jsonl 기본
  python -m backend.benchmark.review --agents AILA,CoALA
  python -m backend.benchmark.review --runs <다른 runs_*.json 또는 jsonl>
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))          # 스크립트/모듈 양쪽 실행 지원

import spectra_panel                     # noqa: E402
import stage_map                         # noqa: E402
from grade import grade_one, summarize    # noqa: E402
from verifiers import _no_answer          # noqa: E402

try:
    import diagnostics                    # noqa: E402  (numpy/scipy/sklearn 필요)
except Exception as _e:                   # noqa: BLE001
    diagnostics = None
    print(f"      [주의] diagnostics 임포트 실패 -> 자동 진단 생략 ({_e})")

# 파일처리(CSV) 문항의 정답 기준. 부류 A(명세가 답을 유일하게 결정) / 부류 B(방법군을
# 부름 → 모양새로 판정)를 나눠 들고 있고, 절차/결과 판정을 나누는 규칙도 여기 있다.
# 없어도 콘솔은 그대로 뜬다 — 그 경우 파일처리 문항이 기존처럼 수동 판정으로 남을 뿐이다.
try:
    from filegrade import task_class as fg_class          # noqa: E402
    from filegrade.grade_files import split_verdicts as fg_split   # noqa: E402
except Exception as _e:                   # noqa: BLE001
    fg_class = None
    fg_split = None
    print(f"      [주의] filegrade 임포트 실패 -> CSV 문항 정답기준 블록 생략 ({_e})")

AGENT_ORDER = ("AILA", "CoALA")
STORE_PREFIX = "rev1."                   # localStorage 키 접두어(형식 바뀌면 rev2 로)


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


# ── 기계검증기를 '사람 말'로 ──────────────────────────────────────────────────

def _j(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def describe_verifier(v: dict) -> str:
    """검증기 dict → 정확한 통과 조건 한 줄. 채점자가 기준을 눈으로 확인하게 한다."""
    t = v.get("type")
    tol = v.get("tolerance")
    if t == "tool_called":
        return f"툴 <b>{esc(', '.join(v.get('tools') or []))}</b> 를 모두 호출했는가"
    if t == "tool_called_any":
        return f"툴 <b>{esc(' 또는 '.join(v.get('tools') or []))}</b> 중 하나를 호출했는가"
    if t == "tool_call_count":
        lo, hi = v.get("min_count"), v.get("max_count")
        rng = f"{lo}회" if lo == hi else f"{lo if lo is not None else 0}~{hi if hi is not None else '∞'}회"
        return f"<b>{esc(v.get('tool'))}</b> 호출 횟수가 <b>{esc(rng)}</b> 인가"
    if t == "tool_args":
        return (f"<b>{esc(v.get('tool'))}</b> 의 인자 <code>{esc(v.get('field'))}</code> 가 "
                f"<b>{esc(_j(v.get('expected')))}</b> 인가"
                + (f" (허용 ±{tol:g})" if tol else " (정확히 일치)"))
    if t == "tool_arg_any":
        cands = ", ".join(f"{c.get('tool')}.{c.get('field')}" for c in v.get("candidates") or [])
        return (f"{esc(cands)} 중 어느 하나가 <b>{esc(_j(v.get('expected')))}</b> 인가"
                + (f" (허용 ±{tol:g})" if tol else ""))
    if t == "tool_sequence":
        return f"툴 호출 순서가 <b>{esc(' → '.join(v.get('tools_in_order') or []))}</b> 를 포함하는가"
    if t == "tool_args_sequence":
        return (f"<b>{esc(v.get('tool'))}</b> 의 <code>{esc(v.get('field'))}</code> 가 "
                f"<b>{esc(_j(v.get('expected_sequence')))}</b> 순서로 나왔는가")
    if t == "tool_result_ok":
        return f"<b>{esc(v.get('tool'))}</b> 가 오류 없이 성공했는가"
    if t == "stage_position":
        return (f"최종 스테이지 좌표가 <b>{esc(_j(v.get('expected')))}</b> mm 인가 "
                f"(허용 ±{(tol or 0):g} mm)")
    if t == "stage_velocity":
        return f"스테이지 속도가 <b>{esc(_j(v.get('expected')))}</b> 인가 (허용 ±{(tol or 0):g})"
    if t == "ccd_exposure":
        return f"CCD 노출시간이 <b>{esc(v.get('expected'))}</b> s 인가 (허용 ±{(tol or 0):g})"
    if t == "ccd_read_mode":
        return f"CCD 읽기모드가 <b>{esc(v.get('expected'))}</b> 인가"
    if t == "ccd_temperature":
        return f"CCD 목표온도가 <b>{esc(v.get('expected'))}</b> ℃ 인가"
    if t == "ccd_temperature_reported":
        return "CCD 온도를 실제로 조회해 답변에 보고했는가"
    if t == "laser_state":
        return f"최종 레이저 상태가 <b>{'ON' if v.get('expected_on') else 'OFF'}</b> 인가"
    if t == "spectrum_valid":
        return f"<b>{esc(v.get('tool'))}</b> 가 유효한(빈 배열 아님) 스펙트럼을 돌려줬는가"
    if t == "reference_match":
        if v.get("spike_aware"):
            return (f"산출 CSV 를 <code>{esc(v.get('reference'))}</code> 와 비교해 "
                    f"<b>스파이크가 아닌 점은 오차 ≤ {v.get('tolerance', 0):g}</b> 이고 "
                    f"<b>스파이크 제거율 ≥ {v.get('min_removal_pct', 99.0):g}%</b> 인가"
                    " <span class='hint'>(레퍼런스는 스파이크를 넣기 전 원본이라 "
                    "전 구간 일치는 원리적으로 불가 — 아래 '정답의 정의' 참조)</span>")
        return (f"산출 CSV 가 <code>{esc(v.get('reference'))}</code> 와 점대점 일치하는가 "
                f"(허용오차 {v.get('tolerance', 0):g}, 초과 허용 {v.get('max_bad_points', 0)}점)")
    if t == "answer_numeric":
        exp = v.get("expected")
        cond = f"허용 ±{tol:g}" if tol is not None else f"상대오차 {100*v.get('rel_tolerance', 0):g}% 이내"
        order = " (등장 순서까지 일치)" if v.get("ordered") else ""
        return (f"답변 텍스트에 <b>{esc(_j(exp))}</b> 가 등장하는가 ({esc(cond)}){order}"
                + (f" — {esc(v.get('note'))}" if v.get("note") else ""))
    if t == "answer_contains":
        exp = v.get("expected")
        mode = v.get("mode", "any" if isinstance(exp, list) else "all")
        head = (f"답변에 <b>{esc(_j(exp))}</b> 중 "
                f"{'하나 이상' if mode == 'any' else '전부'} 가 등장하는가")
        if v.get("forbidden"):
            head += (f", 그리고 <b>{esc(_j(v.get('forbidden')))}</b> 를 "
                     f"<u>결론으로</u> 말하지 않았는가 "
                     "<span class='hint'>(대조·소거 문맥의 언급은 통과)</span>")
        return head + (f" — {esc(v.get('note'))}" if v.get("note") else "")
    if t == "human_only":
        return f"<i>사람 확인 항목</i> — {esc(v.get('note'))}"
    return f"{esc(t)} {esc(_j({k: x for k, x in v.items() if k != 'type'}))}"


# ── 좌표/그리드 문항의 판정 기준 ─────────────────────────────────────────────

_POS_TOOLS = {"run_grid_scan", "preview_grid_scan", "move_stage", "move_stage_relative",
              "move_to_pixel", "acquire_spectrum", "save_point_data", "run_autofocus"}
_GRID_RE = re.compile(r"(\d+)\s*[x×]\s*(\d+)")
_SPACING_RE = re.compile(
    r"(?:spacing|step|pitch|interval|apart|간격)[^0-9]{0,14}(\d+(?:\.\d+)?)\s*(mm|µm|um)"
    r"|(\d+(?:\.\d+)?)\s*(mm|µm|um)\s*(?:spacing|step|pitch|interval|apart|간격)",
    re.IGNORECASE)
# 'X=37.8, 37.9, 38.0 mm' 처럼 축 하나에 좌표가 여럿 나열되는 문항이 있어 콤마 목록까지 받는다
# (첫 숫자만 잡으면 기대 좌표를 절반만 적어 채점자를 오도한다).
_COORD_RE = re.compile(
    r"\b([XYZxyz])\s*=\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*)")


def positional_criteria(meta: dict) -> str:
    """스테이지 지도에서 무엇을 확인해야 정답인지. 프롬프트에서 기대값을 뽑아 같이 적는다.

    프롬프트 원문을 정규식으로 읽은 값이므로 참고용이다 — 지도·표의 실측값과 대조하는
    용도이고, 검증기(tool_args/stage_position)가 있는 항목은 이미 기계로 확인된다.
    """
    tools = set(meta.get("expected_tools") or [])
    for v in meta.get("verifiers") or []:
        tools.update(v.get("tools") or [])
        if v.get("tool"):
            tools.add(v["tool"])
    if not (tools & _POS_TOOLS):
        return ""

    prompt = str(meta.get("prompt") or "")
    exp: list[str] = []
    gm = _GRID_RE.search(prompt)
    if gm:
        r, c = int(gm.group(1)), int(gm.group(2))
        exp.append(f"격자 <b>{r}×{c} = {r*c}점</b> → 지도의 ‘쏜 점 개수’와 "
                   f"‘서로 다른 X/Y 좌표’가 이와 같아야 하고 ‘완전 격자 여부=예’ 여야 한다")
    sm_ = _SPACING_RE.search(prompt)
    if sm_:
        val = sm_.group(1) or sm_.group(3)
        unit = sm_.group(2) or sm_.group(4)
        exp.append(f"점 간격 <b>{esc(val)} {esc(unit)}</b> → 지도의 ‘X/Y 간격’이 이 값이고 "
                   f"‘등간격=예’ 여야 한다")
    coords = _COORD_RE.findall(prompt)
    if coords:
        exp.append("프롬프트에 적힌 좌표 <b>"
                   + ", ".join(f"{a.upper()}={b}" for a, b in coords)
                   + "</b> → 좌표 목록의 해당 이동/측정 지점과 일치해야 한다")
    if not exp:
        exp.append("지도의 이동 순서(회색 파선)와 ‘레이저 쏜 점’의 개수·위치가 "
                   "프롬프트의 요구와 맞는지 본다")
    return ('<div class="crit-pos"><b>좌표로 확인할 것</b> '
            '<span class="hint">(아래 「스테이지 지도」의 수치와 대조)</span><ol>'
            + "".join(f"<li>{e}</li>" for e in exp) + "</ol></div>")


def _load_specs() -> dict:
    """answer_specs.SPECS — 방법 미지정 문항의 정답 정의.
    numpy/scipy 가 없어 임포트가 실패하면 정답 정의 블록만 빠지고 콘솔은 동작한다."""
    try:
        from answer_specs import SPECS
        return SPECS
    except Exception as e:                       # noqa: BLE001
        print(f"      [주의] answer_specs 임포트 실패 → 정답정의 블록 생략 ({e})")
        return {}


def verify_recipe(tid: str, spec: dict) -> str:
    """'이 레시피가 레퍼런스 CSV 를 실제로 재현하는가' 를 지금 계산해 한 줄로 돌려준다.

    정답 정의가 추측이 아니라는 증거다 — 입력 CSV 에 레시피를 그대로 적용해 레퍼런스와
    비교한다. 실측 오차는 5e-7~1e-6 수준으로, 레퍼런스 CSV 가 세기를 %.6f 로 저장해
    생기는 양자화 한계와 같다. 레퍼런스나 입력이 없으면 ''(문장 생략).
    """
    fn, ref = spec.get("recipe_fn"), spec.get("ref")
    if fn is None or not ref:
        return ""
    try:
        import numpy as np
        hits = sorted((_HERE.parents[1] / "data" / "uploads").glob(f"*/{tid}.csv"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        rp = spectra_panel._TASK_REFS / ref
        if not hits or not rp.exists():
            return ""
        ci = spectra_panel.read_curves(hits[0], max_groups=1)
        cr = spectra_panel.read_curves(rp, max_groups=1)
        if not (ci and cr and ci[0]["y"] and cr[0]["y"]):
            return ""
        x = np.asarray(ci[0]["x"], dtype=float)
        y = np.asarray(ci[0]["y"], dtype=float)
        yr = np.asarray(cr[0]["y"], dtype=float)
        rep = np.asarray(fn(x, y), dtype=float)
        n = min(len(rep), len(yr))
        err = float(np.abs(rep[:n] - yr[:n]).max())
    except Exception as e:                       # noqa: BLE001
        return f'<div class="spec-verify bad">레시피 재현 검증 실패: {esc(e)}</div>'
    return (f'<div class="spec-verify">✓ 이 레시피로 레퍼런스 CSV 를 재현 검증했다 — '
            f'최대 오차 <code>{err:.3e}</code> (CSV 가 %.6f 로 저장되어 생기는 양자화 '
            f'오차 수준). 즉 위 정의가 정답의 실제 생성 방법이다.</div>')


# ── 로그 ─────────────────────────────────────────────────────────────────────

def _short(v, n=180) -> str:
    s = _j(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + " …"


def render_log(rec: dict) -> str:
    """전체 툴 트레이스. 한 줄 요약 + 클릭하면 인자/결과 원본 JSON."""
    calls = rec.get("tool_calls") or []
    if not calls:
        return '<div class="empty">— 툴 호출 없음 —</div>'
    rows = []
    for c in calls:
        ok = c.get("ok", True)
        args, res = c.get("args") or {}, c.get("result")
        head = (f'<span class="tstep">#{c.get("step", "?")}</span>'
                f'<span class="tname">{esc(c.get("name"))}</span>'
                f'<span class="targs">({esc(_short(args, 110))})</span> '
                f'<span class="tres">→ {"✓" if ok else "✗"} {esc(_short(res, 150))}</span>')
        rows.append(
            f'<details class="tool{"" if ok else " err"}"><summary>{head}</summary>'
            f'<pre class="tjson">args = {esc(json.dumps(args, ensure_ascii=False, indent=2))}\n\n'
            f'result = {esc(json.dumps(res, ensure_ascii=False, indent=2, default=str))}</pre>'
            f'</details>')
    return "".join(rows)


def render_model_output(rec: dict) -> str:
    """에이전트가 실제로 돌린 코드와 그 출력.

    [왜 따로 빼는가]
    이게 '모델의 산출물' 이고 채점의 핵심 근거인데, 예전에는 툴 트레이스의 접힌 결과
    JSON 안에 파묻혀 있었다(실측: stdout 이 있는 호출 120개가 전부 화면에 안 보였다).
    답변 텍스트는 요약일 뿐이고 실제 계산은 여기서 일어난다 — 예: T049 는 코드의
    `np.std(...)` 한 줄이 표본/모표준편차 판정을 가른다.
    """
    blocks = []
    n = 0
    for c in rec.get("tool_calls") or []:
        r = c.get("result") if isinstance(c.get("result"), dict) else {}
        code = (c.get("args") or {}).get("code")
        out = r.get("stdout")
        if not (code or out):
            continue
        n += 1
        fail = "" if c.get("ok", True) else ' <span class="chip cerr">실패</span>'
        parts = [f'<div class="mo-head">#{c.get("step", "?")} '
                 f'<b>{esc(c.get("name"))}</b>{fail}</div>']
        if code:
            parts.append(f'<pre class="mo-code">{esc(code)}</pre>')
        if out:
            parts.append('<div class="mo-label">출력(stdout)</div>'
                         f'<pre class="mo-out">{esc(out)}</pre>')
        if r.get("error"):
            parts.append(f'<pre class="mo-err">{esc(r.get("error"))}</pre>')
        blocks.append(f'<div class="mo">{"".join(parts)}</div>')
    if not blocks:
        return ""
    return (f'<details class="modelout" open><summary>모델이 돌린 코드와 출력 ({n})</summary>'
            f'{"".join(blocks)}</details>')


def render_diagnostics(rows: list[dict]) -> str:
    """자동 진단 표 — 기준 / 실측(기준값) / 에이전트 보고값 / 판정."""
    if not rows:
        return ""
    trs = []
    for r in rows:
        v = r.get("verdict")
        mark = {"pass": '<span class="dv ok">통과</span>',
                "fail": '<span class="dv bad">실패</span>'}.get(
                    v, '<span class="dv info">참고</span>')
        note = f'<div class="dnote">{r["note"]}</div>' if r.get("note") else ""
        trs.append(f'<tr class="d-{esc(v)}"><td class="l">{r["name"]}</td>'
                   f'<td class="l crit">{r["criterion"]}{note}</td>'
                   f'<td class="l mono">{r["measured"]}</td>'
                   f'<td class="l mono rep">{r["reported"]}</td><td>{mark}</td></tr>')
    n_ok = sum(1 for r in rows if r.get("verdict") == "pass")
    n_bad = sum(1 for r in rows if r.get("verdict") == "fail")
    return (f'<details class="diag" open><summary>자동 진단 — 통과 {n_ok} · 실패 {n_bad} · '
            f'참고 {len(rows)-n_ok-n_bad} <span class="hint">(입력 파일로 채점기준을 '
            f'다시 계산한 결과)</span></summary>'
            f'<table class="diagtbl"><thead><tr><th class="l">항목</th><th class="l">기준</th>'
            f'<th class="l">실측(기준값)</th><th class="l">에이전트 보고</th><th>판정</th></tr>'
            f'</thead><tbody>{"".join(trs)}</tbody></table></details>')


def _diag_chip(v: str | None) -> str:
    if v == "pass":
        return '<span class="chip cok">진단 ✓</span>'
    if v == "fail":
        return '<span class="chip cfail">진단 ✗</span>'
    return ""


# ── CSV(파일처리) 문항 : 정답 ↔ 대답 대조 ────────────────────────────────────

def render_gt_block(tid: str) -> str:
    """이 문항의 정답을 무엇으로 정의했고 왜 그렇게 정의했는지.

    파일처리 문항은 '레퍼런스 CSV 와 비트 단위로 같은가'로 채점하면 안 되는 것들이
    섞여 있다("5차 다항 baseline" 같은 지시는 정당한 구현이 여럿이고 그 편차가
    tolerance 의 6.7e6 배다). 그래서 문항마다 부류를 먼저 밝히고 시작한다.
    """
    if fg_class is None:
        return ""
    e = fg_class.get(tid)
    if not e:
        return ""
    is_b = e["class"] == fg_class.CLASS_B
    if is_b:
        th = fg_class.shape_thresholds(tid)
        how = (f'<div class="gt-how"><b>판정 방법 — 모양새 일치</b> '
               f'<span class="hint">(값 일치를 요구하지 않는다)</span><br>'
               f'피크 recall ≥ {th["min_recall"]:.2f} · precision ≥ {th["min_precision"]:.2f} '
               f'(±{th["peak_tol_cm"]:g} cm⁻¹) · 상대세기 Δ ≤ {th["max_d_rel_intensity"]:.2f} · '
               f'pearson ≥ {th["min_pearson"]:.2f} · 0~1 재정규화 후 max|Δ| ≤ {th["max_abs_01"]:.2f}'
               f'<br><span class="hint">연속 임계값은 정당한 구현 앙상블 편차의 2배로 '
               f'유도했다 — 임의로 고른 값이 아니다.</span></div>')
    else:
        how = ('<div class="gt-how"><b>판정 방법 — GT 엄격 비교</b> '
               '<span class="hint">(자유 파라미터가 없어 답이 유일하다)</span></div>')
    free = (f'<div class="gt-free"><b>자유 파라미터</b> · {esc(", ".join(e["free_params"]))}</div>'
            if e.get("free_params") else "")
    return (f'<div class="gtdef cls{e["class"]}">'
            f'<div class="gt-head"><span class="gtbadge b{e["class"]}">부류 {e["class"]}</span>'
            f'{"방법군을 부르는 과제" if is_b else "명세가 답을 유일하게 결정"}</div>'
            f'<div class="gt-rule"><b>정답</b> · {esc(e["gt_rule"])}</div>'
            f'{free}{how}'
            f'<div class="gt-why"><b>이 부류인 이유</b> · {esc(e["why"])}</div></div>')


_VLABEL = {"pass": ("정답", "ok"), "fail": ("오답", "bad")}


def render_compare(drows: list[dict], tid: str) -> str:
    """한눈 대조표 — 판정을 가른 항목만 '기준 / 정답 / 에이전트 답 / 판정' 네 칸으로.

    아래 '자동 진단' 표는 참고행까지 전부 담아 길다. 사람이 먼저 보고 싶은 것은
    "무엇을 봤고, 정답이 뭐였고, 얘는 뭐라 했고, 그래서 맞았나" 네 가지뿐이라
    그것만 위로 끌어올린다.
    """
    if fg_split is None or not drows:
        return ""
    sv = fg_split(drows)
    dec = [r for r in sv["decisive"] if r.get("verdict") in ("pass", "fail")]
    if not dec:
        return ""

    trs = []
    for r in dec:
        v = r.get("verdict")
        lbl, cls = _VLABEL.get(v, ("참고", "info"))
        note = f'<div class="dnote">{r["note"]}</div>' if r.get("note") else ""
        rep = r.get("reported")
        # 진단 행에는 두 관례가 섞여 있다.
        #   ① filegrade 가 만든 행 : measured = 정답(재계산), reported = 에이전트 답
        #   ② 기존 diagnostics 행  : reported 가 비어 있고 measured 자체가
        #                            '에이전트 산출물을 재어 본 값'이다(기대값은 기준 문장에).
        # 대조표는 ②를 왼쪽(정답) 칸에 두면 안 된다 — 에이전트 답을 정답인 양 보여 주게 된다.
        if rep in ("", "-", "—", None):
            gt_cell = '<span class="empty">← 기준 참조</span>'
            got_cell = r["measured"]
        else:
            gt_cell = r["measured"]
            got_cell = rep
        trs.append(f'<tr class="d-{esc(v)}"><td class="l">{r["name"]}</td>'
                   f'<td class="l crit">{r["criterion"]}</td>'
                   f'<td class="l mono gtv">{gt_cell}</td>'
                   f'<td class="l mono rep">{got_cell}{note}</td>'
                   f'<td><span class="dv {cls}">{lbl}</span></td></tr>')

    def chip(label, v):
        l, c = _VLABEL.get(v, ("해당없음", "info"))
        return f'<span class="pochip {c}">{label} {l}</span>'

    ov = sv["verdict"]
    lbl, cls = _VLABEL.get(ov, ("판정보류", "info"))
    return (f'<div class="cmp {cls}">'
            f'<div class="cmp-head"><span class="cmp-v {cls}">{lbl}</span>'
            f'{chip("절차", sv["process_verdict"])}{chip("결과", sv["outcome_verdict"])}'
            f'<span class="cmp-why">{esc(sv["why"])}</span></div>'
            f'<table class="cmptbl"><thead><tr><th class="l">무엇을 봤나</th>'
            f'<th class="l">통과 조건</th><th class="l">정답 (재계산)</th>'
            f'<th class="l">에이전트 답</th><th>판정</th></tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table></div>')


def render_planning(rec: dict) -> str:
    pl = rec.get("planning") or []
    if not pl:
        return ""
    rows = []
    for p in pl:
        extra = ""
        if p.get("scores") is not None:
            extra += f' <span class="pscore">scores={esc(p.get("scores"))}</span>'
        if p.get("chosen"):
            extra += f' <span class="pchosen">→ {esc(p.get("chosen"))}</span>'
        rows.append(f'<div class="pl"><b>{esc(p.get("phase", ""))}</b> '
                    f'{esc(p.get("message", ""))}{extra}</div>')
    return (f'<details class="planning"><summary>CoALA planning 단계 ({len(pl)})</summary>'
            f'{"".join(rows)}</details>')


def render_verifier_results(rec: dict) -> str:
    vrs = rec.get("verifier_results") or []
    if not vrs:
        return '<div class="empty">— 기계검증기 없음(전적으로 사람 판정) —</div>'
    rows = []
    for v in vrs:
        if v.get("is_human_only"):
            cls, mark = "vr human", "✎"
        elif v.get("passed"):
            cls, mark = "vr pass", "✓"
        else:
            cls, mark = "vr fail", "✗"
        rows.append(f'<div class="{cls}">{mark} <b>{esc(v.get("type"))}</b> '
                    f'{esc(v.get("detail"))}</div>')
    return "".join(rows)


# ── 정답 기준 블록 ───────────────────────────────────────────────────────────

def render_criteria(meta: dict, spec: dict | None, spec_verify: str = "") -> str:
    """이 문항을 무엇으로 판정하는지 전부. 사람이 여기만 읽고 판정할 수 있어야 한다."""
    parts = []
    # CSV 문항이면 '정답을 어떻게 정의했나'를 맨 위에 둔다. 그게 없으면 아래 채점기준의
    # 허용오차(1e-6, 1e-8 …)를 곧이곧대로 읽게 되는데, 그 값들은 방법군 문항에서는
    # correctness 가 아니라 구현 동일성을 재는 값이다.
    gt = render_gt_block(str(meta.get("id") or ""))
    if gt:
        parts.append(gt)
    parts.append(f'<div class="crit-prose"><b>문항의 채점기준</b><br>'
                 f'{esc(meta.get("grading_criteria"))}</div>')
    if meta.get("manual_note"):
        parts.append(f'<div class="crit-note"><b>수동채점 노트</b><br>'
                     f'{esc(meta.get("manual_note"))}</div>')

    vs = meta.get("verifiers") or []
    mach = [v for v in vs if v.get("type") != "human_only"]
    hum = [v for v in vs if v.get("type") == "human_only"]
    if mach:
        parts.append('<div class="crit-list"><b>기계로 확인하는 조건</b><ol>'
                     + "".join(f"<li>{describe_verifier(v)}</li>" for v in mach)
                     + "</ol></div>")
    if hum:
        parts.append('<div class="crit-list human"><b>사람이 확인할 조건</b><ol>'
                     + "".join(f"<li>{describe_verifier(v)}</li>" for v in hum)
                     + "</ol></div>")
    if not vs:
        parts.append('<div class="crit-note">이 문항에는 검증기가 없다 — '
                     '위 채점기준 문장만으로 판정한다.</div>')

    pos = positional_criteria(meta)
    if pos:
        parts.append(pos)

    if spec:
        parts.append(
            '<div class="crit-spec"><b>정답의 정확한 정의</b> '
            '<span class="hint">(레퍼런스를 만든 생성기 레시피 — 프롬프트가 방법을 '
            '지정하지 않은 문항)</span>'
            f'<pre class="recipe">{esc(spec["recipe"])}</pre>'
            f'<div class="spec-note">{spec["recipe_note"]}</div>{spec_verify}'
            f'<div class="spec-why"><b>애매한 이유</b> · {spec["why_ambiguous"]}</div>'
            '<div class="spec-pass"><b>이걸 만족하면 정답</b><ol>'
            + "".join(f"<li>{c}</li>" for c in spec["pass_if"]) + "</ol></div></div>")

    return f'<div class="criteria">{"".join(parts)}</div>'


# ── 셀(문항×에이전트) ────────────────────────────────────────────────────────

def _chips(rec: dict, noans: bool) -> str:
    out = []
    av = rec.get("auto_verdict")
    if noans:
        out.append('<span class="chip cerr">무응답 → 자동 오답</span>')
    elif av == "pass":
        out.append(f'<span class="chip cok">자동 ✓ {rec.get("n_machine_pass")}/{rec.get("n_machine")}</span>')
    elif av == "fail":
        out.append(f'<span class="chip cfail">자동 ✗ {rec.get("n_machine_pass")}/{rec.get("n_machine")}</span>')
    else:
        out.append('<span class="chip cman">기계검증 없음</span>')
    out.append(f'<span class="chip">{esc(rec.get("response_type"))}</span>')
    if rec.get("fired_laser"):
        out.append('<span class="chip cfire">레이저 발사</span>')
    if rec.get("asked_clarification"):
        out.append('<span class="chip cask">되물음</span>')
    if rec.get("run_error"):
        out.append('<span class="chip cerr">실행오류</span>')
    return " ".join(out)


def render_cell(rec: dict | None, run_id: str, agent: str, out_dir: Path,
                meta_task: dict | None = None, tf_entry: dict | None = None) -> str:
    if rec is None:
        return (f'<div class="cell none"><div class="chead"><b>{esc(agent)}</b> '
                f'<span class="empty">미실행</span></div></div>')

    answer = rec.get("answer") or rec.get("final_report") or ""
    noans = _no_answer(answer)
    gid = f"{run_id}|{agent}"
    auto = "fail" if noans else (rec.get("auto_verdict") or "")

    def safe(fn, label):
        """패널 하나가 깨져도 채점을 못 하게 되면 안 된다 — 셀 단위로 막는다."""
        try:
            return fn()
        except Exception as e:                   # noqa: BLE001
            return f'<div class="warnbox">{esc(label)} 생성 실패: {esc(f"{type(e).__name__}: {e}")}</div>'

    spectra = safe(lambda: spectra_panel.build_spectra_panel(rec, out_dir), "스펙트럼 패널")
    smap = safe(lambda: stage_map.build_stage_map(rec), "스테이지 지도")
    shots = safe(lambda: stage_map.build_images(rec, out_dir), "이미지 갤러리")
    modelout = safe(lambda: render_model_output(rec), "모델 출력")
    drows = []
    if diagnostics is not None:
        try:
            drows = diagnostics.run(rec, meta_task or rec, tf_entry or {})
        except Exception as e:                   # noqa: BLE001
            drows = [{"name": "진단 오류", "criterion": "자동 진단이 실패했다",
                      "measured": esc(f"{type(e).__name__}: {e}"), "reported": "—",
                      "verdict": "info", "note": ""}]
    diag = safe(lambda: render_diagnostics(drows), "자동 진단")
    dverdict = diagnostics.overall(drows) if (diagnostics and drows) else None

    # 파일처리(CSV) 문항은 정답 기준이 부류 A/B 로 재정의돼 있다. 그 판정을 셀의
    # 자동판정으로 쓴다 — 기계검증기(tool_called 등)는 "run_analysis 를 불렀다"만
    # 확인하므로 CSV 문항에서는 판정 근거가 되기에 약하다.
    cmp_html, fg_v = "", None
    tid = str((meta_task or rec).get("id") or "")
    if fg_split is not None and fg_class is not None and fg_class.get(tid) and drows:
        cmp_html = safe(lambda: render_compare(drows, tid), "정답 대조표")
        try:
            fg_v = fg_split(drows)["verdict"]
        except Exception:                        # noqa: BLE001
            fg_v = None
    if fg_v and not noans:
        auto = fg_v

    meta = (f'{rec.get("elapsed_sec", "?")}s · dose {rec.get("dose_mj", "?")}mJ · '
            f'툴 {len(rec.get("tool_calls") or [])}회 · {esc(rec.get("session_id") or "")}')
    ans_cls = "answer noans" if noans else "answer"

    return f'''<div class="cell" data-gid="{esc(gid)}" data-auto="{esc(auto)}"
         data-noans="{int(noans)}" data-agent="{esc(agent)}" data-diag="{esc(dverdict or "")}">
      <div class="chead"><b class="aname">{esc(agent)}</b> {_chips(rec, noans)}
        {_diag_chip(dverdict)}<span class="cmeta">{meta}</span></div>
      {cmp_html}
      <details class="{ans_cls}" open><summary>최종 답변</summary>
        <pre>{esc(answer) or "<span class='empty'>(빈 답변)</span>"}</pre></details>
      {diag}
      <details class="vres" open><summary>기계검증 결과</summary>
        {render_verifier_results(rec)}</details>
      {modelout}{spectra}{smap}{shots}
      <details class="logs"><summary>실행 로그 — 툴 {len(rec.get("tool_calls") or [])}회
        (클릭하면 인자·결과 원본)</summary>{render_log(rec)}{render_planning(rec)}</details>
      <div class="judge" data-gid="{esc(gid)}">
        <button class="jb pass" data-v="pass">정답</button>
        <button class="jb partial" data-v="partial">부분</button>
        <button class="jb fail" data-v="fail">오답</button>
        <input class="jnote" type="text" placeholder="메모 (선택)">
        <span class="jstate"></span>
      </div>
    </div>'''


def render_card(run_id: str, meta: dict, cells: dict, spec: dict | None,
                out_dir: Path, agents: list[str], spec_verify: str = "",
                tf_entry: dict | None = None) -> str:
    verds = {a: (cells.get(a) or {}).get("auto_verdict") for a in agents}
    present = [a for a in agents if cells.get(a)]
    disagree = len({verds[a] for a in present}) > 1
    noans = any(_no_answer((cells.get(a) or {}).get("answer")
                           or (cells.get(a) or {}).get("final_report") or "")
                for a in present)
    anyfail = any(verds[a] == "fail" for a in present)
    cat0 = str(meta.get("category", "")).split(".")[0].strip()

    badges = [f'<span class="badge cat">{esc(meta.get("category"))}</span>',
              f'<span class="badge">{esc(meta.get("capability"))}</span>',
              f'<span class="badge kind">{esc(meta.get("task_kind"))}</span>']
    if fg_class is not None:
        _e = fg_class.get(str(meta.get("id") or ""))
        if _e:
            badges.append(
                f'<span class="badge fgc b{_e["class"]}">부류 {_e["class"]} · '
                f'{"모양새 채점" if _e["class"] == fg_class.CLASS_B else "GT 엄격"}</span>')
    if meta.get("is_safety_ambiguous"):
        badges.append('<span class="badge amb">안전-애매</span>')
    if meta.get("variant") and meta.get("variant") != "none":
        badges.append(f'<span class="badge var">{esc(meta.get("variant"))}</span>')
    if spec:
        badges.append('<span class="badge spec">방법 미지정</span>')

    grid = "".join(render_cell(cells.get(a), run_id, a, out_dir, meta, tf_entry)
                   for a in agents)
    # 셀 렌더 결과에서 진단 실패 여부를 읽는다(셀이 판정을 계산하므로 여기서 재계산하지 않는다)
    has_dfail = int('data-diag="fail"' in grid)
    ncols = max(1, len(agents))
    return f'''<section class="card" id="card-{esc(run_id)}"
        data-cat="{esc(cat0)}" data-disagree="{int(disagree)}" data-noans="{int(noans)}"
        data-anyfail="{int(anyfail)}" data-amb="{int(bool(meta.get("is_safety_ambiguous")))}"
        data-spec="{int(bool(spec))}" data-diagfail="{has_dfail}">
      <div class="cardhead">
        <span class="tid">{esc(meta.get("id"))}</span>
        {"".join(badges)}
        <span class="cardnav"><a href="#card-{esc(run_id)}">#</a></span>
      </div>
      <div class="prompt"><b>프롬프트</b><br>{esc(meta.get("prompt"))}</div>
      {render_criteria(meta, spec, spec_verify)}
      <div class="cells" style="grid-template-columns:repeat({ncols},minmax(0,1fr))">{grid}</div>
    </section>'''


# ── HTML 셸 ──────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing:border-box; }
body { font-family:system-ui, sans-serif; margin:0; background:#eef0f4; color:#111827; font-size:13px; line-height:1.55; }
header { position:sticky; top:0; z-index:20; background:#111827; color:#fff; padding:9px 14px; box-shadow:0 1px 6px rgba(0,0,0,.25); }
header h1 { margin:0; font-size:15px; font-weight:650; }
header .sub { color:#9ca3af; font-size:11.5px; }
.bar { display:flex; gap:5px; flex-wrap:wrap; align-items:center; margin-top:7px; }
.bar button { background:#374151; color:#fff; border:1px solid #4b5563; border-radius:6px; padding:3px 9px; cursor:pointer; font-size:11.5px; }
.bar button:hover { background:#4b5563; }
.bar button.on { background:#2563eb; border-color:#60a5fa; }
.bar button.act { background:#059669; border-color:#10b981; }
.bar button.snap { background:#7c3aed; border-color:#a78bfa; }
.bar button:disabled { opacity:.55; cursor:progress; }
.baked { color:#fde68a; margin-right:8px; }
.baked button { background:#b45309 !important; border-color:#f59e0b !important; }
.bar .sp { flex:1; }
.bar .tally { font-size:11.5px; color:#e5e7eb; font-family:ui-monospace,monospace; }
.bar .tally b.p { color:#34d399; } .bar .tally b.q { color:#fbbf24; } .bar .tally b.f { color:#f87171; }
.wrap { padding:10px 14px 120px; }
.card { background:#fff; border:1px solid #d7dbe2; border-radius:9px; margin:11px 0; padding:11px 12px; }
.cardhead { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.tid { font-weight:750; font-size:15px; letter-spacing:.02em; }
.cardnav { margin-left:auto; } .cardnav a { color:#c7cbd3; text-decoration:none; font-size:14px; }
.badge { font-size:10.5px; padding:1px 7px; border-radius:9px; border:1px solid #dde1e7; background:#f7f8fa; color:#4b5563; }
.badge.cat { background:#eef2ff; border-color:#c7d2fe; color:#3730a3; }
.badge.kind { font-style:italic; }
.badge.amb { background:#fef3c7; border-color:#fcd34d; color:#92400e; }
.badge.var { background:#dcfce7; border-color:#86efac; color:#166534; }
.badge.spec { background:#fce7f3; border-color:#f9a8d4; color:#9d174d; }
.prompt { margin:8px 0; padding:7px 9px; background:#f8fafc; border-left:3px solid #2563eb; border-radius:4px; white-space:pre-wrap; }
.criteria { display:flex; flex-direction:column; gap:6px; margin-bottom:9px; }
.criteria > div { padding:7px 9px; border-radius:5px; font-size:12.5px; }
.crit-prose { background:#f5f3ff; border-left:3px solid #7c3aed; }
.crit-note { background:#fffbeb; border-left:3px solid #f59e0b; }
.crit-list { background:#f0fdf4; border-left:3px solid #10b981; }
.crit-list.human { background:#eef2ff; border-left-color:#6366f1; }
.crit-pos { background:#ecfeff; border-left:3px solid #0891b2; }
.criteria ol { margin:3px 0 0 19px; padding:0; }
.criteria li { margin:2px 0; }
.crit-spec { background:#fdf2f8; border-left:3px solid #db2777; }
.crit-spec .recipe { background:#0f172a; color:#e2e8f0; padding:7px 9px; border-radius:4px; font-family:ui-monospace,monospace; font-size:11.5px; overflow-x:auto; margin:5px 0; white-space:pre-wrap; }
.crit-spec .spec-note, .crit-spec .spec-why { margin:4px 0; }
.crit-spec .spec-why { color:#9d174d; }
.crit-spec .spec-pass { background:#fff; border:1px solid #f9a8d4; border-radius:4px; padding:5px 8px; margin-top:5px; }
.hint { color:#6b7280; font-weight:400; font-size:11px; }
code { background:#f1f3f6; padding:0 3px; border-radius:3px; font-family:ui-monospace,monospace; }
.cells { display:grid; gap:9px; }
.cell { border:1px solid #dfe3e9; border-radius:7px; padding:8px; background:#fcfdfe; }
.cell.none { opacity:.45; }
.cell.active { outline:3px solid #2563eb; outline-offset:1px; }
.cell.judged-pass { background:#f4fdf7; border-color:#86efac; }
.cell.judged-partial { background:#fffdf2; border-color:#fcd34d; }
.cell.judged-fail { background:#fef6f6; border-color:#fca5a5; }
.chead { display:flex; align-items:center; gap:5px; flex-wrap:wrap; border-bottom:1px solid #eef0f3; padding-bottom:5px; margin-bottom:5px; }
.aname { font-size:13.5px; }
.cmeta { margin-left:auto; color:#9ca3af; font-size:10.5px; font-family:ui-monospace,monospace; }
.chip { font-size:10px; padding:1px 6px; border-radius:8px; background:#eceff3; color:#374151; }
.chip.cok { background:#d1fae5; color:#065f46; } .chip.cfail { background:#fee2e2; color:#991b1b; }
.chip.cman { background:#e0e7ff; color:#3730a3; } .chip.cfire { background:#fecaca; color:#7f1d1d; }
.chip.cask { background:#fef9c3; color:#854d0e; } .chip.cerr { background:#dc2626; color:#fff; }
details { margin-top:5px; } summary { cursor:pointer; font-size:11.5px; color:#4b5563; }
details > summary:hover { color:#111827; }
.answer pre { white-space:pre-wrap; font-size:12px; background:#f8fafc; padding:7px; border-radius:4px; max-height:300px; overflow:auto; margin:4px 0 0; }
.answer.noans pre { background:#fef2f2; color:#991b1b; font-weight:600; }
.vr { font-size:11px; padding:1px 0; }
.vr.pass { color:#059669; } .vr.fail { color:#dc2626; } .vr.human { color:#7c3aed; }
.empty { color:#9ca3af; font-style:italic; font-size:11px; }
.logs .tool { font-family:ui-monospace,monospace; font-size:10.5px; border-radius:3px; padding:1px 3px; }
.logs .tool.err { background:#fef2f2; }
.logs .tool > summary { color:#374151; font-family:ui-monospace,monospace; font-size:10.5px; }
.tstep { color:#9ca3af; margin-right:3px; } .tname { color:#1d4ed8; font-weight:650; }
.targs { color:#6b7280; } .tres { color:#374151; }
.tjson { background:#0f172a; color:#e2e8f0; padding:7px; border-radius:4px; font-size:10.5px; overflow:auto; max-height:340px; white-space:pre-wrap; }
.planning .pl { font-size:11px; color:#4b5563; }
.modelout, .diag { margin-top:8px; border-top:1px dashed #e5e7eb; padding-top:6px; }
.modelout > summary, .diag > summary { font-weight:600; color:#1f2a37; }
.mo { margin:5px 0 8px; }
.mo-head { font-size:11px; color:#4b5563; font-family:ui-monospace,monospace; }
.mo-label { font-size:10px; color:#9ca3af; text-transform:uppercase; margin-top:3px; }
.mo-code { background:#0f172a; color:#e2e8f0; padding:7px 9px; border-radius:4px; font-size:11px;
           overflow:auto; max-height:300px; white-space:pre-wrap; margin:3px 0; }
.mo-out { background:#f0fdf4; border:1px solid #bbf7d0; color:#14532d; padding:6px 8px;
          border-radius:4px; font-size:11px; overflow:auto; max-height:240px; white-space:pre-wrap; margin:2px 0; }
.mo-err { background:#fef2f2; border:1px solid #fecaca; color:#991b1b; padding:6px 8px;
          border-radius:4px; font-size:11px; white-space:pre-wrap; margin:2px 0; }
.diagtbl { border-collapse:collapse; width:100%; font-size:11.5px; margin-top:4px; }
.diagtbl th, .diagtbl td { border:1px solid #e5e7eb; padding:3px 6px; vertical-align:top; text-align:center; }
.diagtbl th { background:#f3f4f6; font-weight:600; }
.diagtbl td.l, .diagtbl th.l { text-align:left; }
.diagtbl td.mono { font-family:ui-monospace,monospace; }
.diagtbl td.rep { font-weight:600; }
.diagtbl tr.d-fail { background:#fef6f6; }
.diagtbl tr.d-pass { background:#f6fefa; }
.diagtbl .crit { color:#4b5563; }
.diagtbl .dnote { color:#7c3aed; font-size:10.5px; margin-top:2px; }
.dv { font-size:10px; padding:1px 6px; border-radius:8px; white-space:nowrap; }
.dv.ok { background:#d1fae5; color:#065f46; } .dv.bad { background:#fee2e2; color:#991b1b; }
.dv.info { background:#eceff3; color:#6b7280; }
.warnbox { font-size:11px; color:#92400e; background:#fffbeb; border:1px solid #fcd34d; border-radius:4px; padding:4px 7px; margin-top:5px; }

/* ── CSV 문항: 정답 ↔ 대답 한눈 대조 ───────────────────────────────────── */
.cmp { border:2px solid #d7dbe0; border-radius:6px; padding:7px 9px; margin:7px 0 9px; background:#fcfcfd; }
.cmp.ok { border-color:#86efac; background:#f4fdf8; }
.cmp.bad { border-color:#fca5a5; background:#fff7f7; }
.cmp.info { border-color:#e5e7eb; }
.cmp-head { display:flex; align-items:center; gap:7px; flex-wrap:wrap; margin-bottom:5px; }
.cmp-v { font-size:13px; font-weight:800; padding:2px 10px; border-radius:5px; letter-spacing:.02em; }
.cmp-v.ok { background:#059669; color:#fff; } .cmp-v.bad { background:#dc2626; color:#fff; }
.cmp-v.info { background:#9ca3af; color:#fff; }
.pochip { font-size:10px; padding:1px 7px; border-radius:8px; border:1px solid currentColor; white-space:nowrap; }
.pochip.ok { color:#047857; } .pochip.bad { color:#b91c1c; } .pochip.info { color:#9ca3af; }
.cmp-why { font-size:11px; color:#4b5563; flex:1 1 100%; }
.cmptbl { border-collapse:collapse; width:100%; font-size:11.5px; }
.cmptbl th, .cmptbl td { border:1px solid #e5e7eb; padding:4px 6px; vertical-align:top; text-align:center; }
.cmptbl th { background:#eef1f5; font-weight:700; font-size:10.5px; }
.cmptbl td.l, .cmptbl th.l { text-align:left; }
.cmptbl td.mono { font-family:ui-monospace,monospace; }
.cmptbl td.gtv { background:#f8fafc; color:#0f172a; }
.cmptbl td.rep { font-weight:600; background:#fffdf5; }
.cmptbl tr.d-fail td.rep { color:#991b1b; } .cmptbl tr.d-pass td.rep { color:#065f46; }
.cmptbl .crit { color:#4b5563; } .cmptbl .dnote { color:#7c3aed; font-size:10.5px; margin-top:2px; }

/* ── 정답 정의(부류 A/B) 블록 ──────────────────────────────────────────── */
.gtdef { border-left:4px solid #94a3b8; background:#f8fafc; border-radius:0 5px 5px 0;
         padding:7px 10px; margin:0 0 9px; font-size:11.5px; line-height:1.55; }
.gtdef.clsA { border-left-color:#2563eb; background:#f5f9ff; }
.gtdef.clsB { border-left-color:#c2761a; background:#fffaf3; }
.gt-head { display:flex; align-items:center; gap:7px; font-weight:700; color:#1f2a37; margin-bottom:3px; }
.gtbadge { font-size:10px; font-weight:800; padding:1px 7px; border-radius:4px; color:#fff; }
.gtbadge.bA { background:#2563eb; } .gtbadge.bB { background:#c2761a; }
.gt-rule { color:#111827; } .gt-free { color:#92400e; } .gt-how { color:#374151; margin-top:3px; }
.gt-why { color:#6b7280; margin-top:3px; }
.badge.fgc.bA { background:#dbeafe; color:#1e40af; } .badge.fgc.bB { background:#fde9d0; color:#92400e; }
.judge { margin-top:8px; padding-top:7px; border-top:1px dashed #dfe3e9; display:flex; gap:5px; align-items:center; flex-wrap:wrap; }
.jb { border:1px solid #d1d5db; background:#fff; border-radius:6px; padding:4px 13px; cursor:pointer; font-size:12.5px; font-weight:600; color:#4b5563; }
.jb:hover { border-color:#9ca3af; }
.jb.pass.on { background:#059669; border-color:#059669; color:#fff; }
.jb.partial.on { background:#d97706; border-color:#d97706; color:#fff; }
.jb.fail.on { background:#dc2626; border-color:#dc2626; color:#fff; }
.jnote { flex:1; min-width:110px; padding:3px 6px; border:1px solid #d1d5db; border-radius:4px; font-size:12px; }
.jstate { font-size:10.5px; color:#9ca3af; font-family:ui-monospace,monospace; }
.hidden { display:none !important; }
.help { background:#fff; border:1px solid #d7dbe2; border-radius:9px; padding:10px 12px; margin:10px 0; font-size:12.5px; }
.help h2 { margin:0 0 5px; font-size:13.5px; }
.help kbd { background:#111827; color:#fff; border-radius:3px; padding:0 5px; font-size:11px; font-family:ui-monospace,monospace; }
.help ul { margin:4px 0 0 18px; padding:0; }
"""

_JS = r"""
const PRE = window.__STORE_PREFIX__;
const K = g => PRE + g;
const cells = () => Array.from(document.querySelectorAll('.cell[data-gid]'));
let active = null;

/* localStorage 를 못 쓰는 환경(브라우저 설정·엄격한 file:// 정책)에서도 채점은 되어야
   한다 — 접근이 막히면 메모리 저장소로 떨어지고, 그때만 경고를 띄운다.
   (그 경우 새로 고치면 판정이 날아가므로 내보내기를 먼저 해야 한다.) */
const STORE = (() => {
  try {
    const k = '__probe__';
    window.localStorage.setItem(k, '1'); window.localStorage.removeItem(k);
    return window.localStorage;
  } catch (e) {
    const m = new Map();
    window.addEventListener('DOMContentLoaded', () => {
      const b = document.querySelector('.bar');
      if (b) b.insertAdjacentHTML('afterbegin',
        '<span style="color:#fca5a5">⚠ 이 브라우저에서 localStorage 가 막혀 판정이 ' +
        '메모리에만 남는다 — 새로 고치기 전에 반드시 내보내기 할 것</span>');
    });
    return { getItem: k => (m.has(k) ? m.get(k) : null),
             setItem: (k, v) => m.set(k, v), removeItem: k => m.delete(k) };
  }
})();

function get(gid){ try { return JSON.parse(STORE.getItem(K(gid)) || 'null'); } catch(e){ return null; } }
function put(gid, obj){ STORE.setItem(K(gid), JSON.stringify(obj)); }

function paint(cell){
  const gid = cell.dataset.gid, s = get(gid) || {};
  cell.classList.remove('judged-pass','judged-partial','judged-fail');
  cell.querySelectorAll('.jb').forEach(b => b.classList.toggle('on', b.dataset.v === s.verdict));
  if(s.verdict) cell.classList.add('judged-' + s.verdict);
  const n = cell.querySelector('.jnote'); if(n && document.activeElement !== n) n.value = s.note || '';
  const st = cell.querySelector('.jstate');
  if(st) st.textContent = s.verdict ? (s.auto ? '자동채택' : '수동') : '미채점';
}

function setVerdict(cell, v, auto){
  const gid = cell.dataset.gid, cur = get(gid) || {};
  // 같은 버튼을 다시 누르면 판정 취소 — 잘못 누른 걸 되돌릴 방법이 있어야 한다.
  const verdict = (!auto && cur.verdict === v) ? null : v;
  put(gid, { verdict, note: (cell.querySelector('.jnote') || {}).value || cur.note || '',
             auto: !!auto, at: new Date().toISOString() });
  paint(cell); tally();
}

/* 무응답은 오답으로 미리 채운다. 이미 사람이 판정한 셀은 건드리지 않는다. */
function seedNoAnswer(){
  cells().forEach(c => {
    if(c.dataset.noans !== '1') return;
    const s = get(c.dataset.gid);
    if(s && s.verdict) return;
    put(c.dataset.gid, { verdict:'fail', note:'무응답(Failed to generate a response) → 자동 오답',
                         auto:true, at:new Date().toISOString() });
  });
}

/* 자동판정을 미채점 셀에 일괄 채택 — 이견 있는 것만 손으로 고치는 흐름 */
function acceptAuto(onlyPass){
  let n = 0;
  cells().forEach(c => {
    const s = get(c.dataset.gid); if(s && s.verdict) return;
    const a = c.dataset.auto; if(!a) return;              // 기계검증 없는 셀은 건너뛴다
    if(onlyPass && a !== 'pass') return;
    setVerdict(c, a, true); n++;
  });
  alert(n + '개 셀에 자동판정을 채택했다. (기계검증기가 없는 셀은 제외)');
}

function clearAuto(){
  if(!confirm('자동으로 채택된 판정만 지운다. 손으로 찍은 판정은 유지된다. 계속?')) return;
  cells().forEach(c => { const s = get(c.dataset.gid); if(s && s.auto) STORE.removeItem(K(c.dataset.gid)); });
  cells().forEach(paint); tally();
}

function tally(){
  const t = { pass:0, partial:0, fail:0, none:0 }, per = {};
  cells().forEach(c => {
    const s = get(c.dataset.gid), v = (s && s.verdict) || 'none';
    t[v]++;
    const a = c.dataset.agent; per[a] = per[a] || { pass:0, partial:0, fail:0, none:0 };
    per[a][v]++;
  });
  const tot = cells().length;
  let txt = `채점 ${tot - t.none}/${tot} · <b class="p">정답 ${t.pass}</b> · ` +
            `<b class="q">부분 ${t.partial}</b> · <b class="f">오답 ${t.fail}</b>`;
  const keys = Object.keys(per).sort();
  if(keys.length > 1) txt += ' ‖ ' + keys.map(a =>
      `${a} ${per[a].pass}/${per[a].pass + per[a].partial + per[a].fail + per[a].none}`).join(' · ');
  document.getElementById('tally').innerHTML = txt;
}

/* ── 필터 ── */
let F = { cat:null, flag:null, agent:null };
function applyFilters(){
  document.querySelectorAll('.card').forEach(card => {
    let show = true;
    if(F.cat && card.dataset.cat !== F.cat) show = false;
    if(F.flag){
      const cs = Array.from(card.querySelectorAll('.cell[data-gid]'));
      if(F.flag === 'todo')     show = show && cs.some(c => { const s = get(c.dataset.gid); return !(s && s.verdict); });
      if(F.flag === 'autofail') show = show && card.dataset.anyfail === '1';
      if(F.flag === 'diagfail') show = show && card.dataset.diagfail === '1';
      if(F.flag === 'noans')    show = show && card.dataset.noans === '1';
      if(F.flag === 'disagree') show = show && card.dataset.disagree === '1';
      if(F.flag === 'amb')      show = show && card.dataset.amb === '1';
      if(F.flag === 'spec')     show = show && card.dataset.spec === '1';
    }
    card.classList.toggle('hidden', !show);
  });
  if(F.agent) document.querySelectorAll('.cell[data-agent]').forEach(c =>
      c.classList.toggle('hidden', c.dataset.agent !== F.agent));
  else document.querySelectorAll('.cell[data-agent]').forEach(c => c.classList.remove('hidden'));
}
function setF(group, val, btn){
  F[group] = (F[group] === val) ? null : val;
  document.querySelectorAll('[data-fg="' + group + '"]').forEach(b => b.classList.remove('on'));
  if(F[group] !== null) btn.classList.add('on');
  applyFilters();
}

/* ── 키보드 ── */
function visibleCells(){
  return cells().filter(c => !c.classList.contains('hidden') && !c.closest('.card').classList.contains('hidden'));
}
function focusCell(cell){
  if(!cell) return;
  cells().forEach(c => c.classList.remove('active'));
  cell.classList.add('active'); active = cell;
  cell.scrollIntoView({ block:'center', behavior:'smooth' });
}
function step(d){
  const v = visibleCells(); if(!v.length) return;
  let i = active ? v.indexOf(active) : -1;
  i = (i < 0) ? 0 : Math.min(v.length - 1, Math.max(0, i + d));
  focusCell(v[i]);
}
function nextTodo(){
  const v = visibleCells();
  const from = active ? v.indexOf(active) + 1 : 0;
  for(let i = from; i < v.length; i++){ const s = get(v[i].dataset.gid); if(!(s && s.verdict)) return focusCell(v[i]); }
  for(let i = 0; i < from; i++){ const s = get(v[i].dataset.gid); if(!(s && s.verdict)) return focusCell(v[i]); }
  alert('미채점 셀이 없다.');
}
document.addEventListener('keydown', e => {
  if(e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.metaKey || e.ctrlKey) return;
  const k = e.key.toLowerCase();
  if(k === 'j'){ step(1); e.preventDefault(); }
  else if(k === 'k'){ step(-1); e.preventDefault(); }
  else if(k === 'n'){ nextTodo(); e.preventDefault(); }
  else if(active && (k === '1' || k === '2' || k === '3')){
    setVerdict(active, { '1':'pass', '2':'partial', '3':'fail' }[k], false);
    step(1); e.preventDefault();
  }
});

/* ── 내보내기 ── */
function rows(){
  return cells().map(c => {
    const s = get(c.dataset.gid) || {};
    const [run_id, agent] = c.dataset.gid.split('|');
    const card = c.closest('.card');
    return { run_id, agent, verdict: s.verdict || null, note: s.note || '',
             auto_accepted: !!s.auto, judged_at: s.at || null,
             auto_verdict: c.dataset.auto || null, no_answer: c.dataset.noans === '1',
             category: card.dataset.cat };
  });
}
function download(name, text, mime){
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type:mime }));
  a.download = name; a.click();
}
function exportJSON(){ download('review_grades.json', JSON.stringify(rows(), null, 2), 'application/json'); }

/* ── 판정 싣기 ──
   판정은 localStorage 에만 있어서 파일을 다른 PC 로 옮기면 전부 '미채점' 으로 보인다.
   게다가 file:// 에서는 브라우저가 localStorage 를 아예 막기도 해서(위 STORE 폴백)
   그 PC 에서는 저장 자체가 안 된다. 그래서 판정을 HTML 안에 __GRADES__ 로 구워
   넣을 수 있게 한다 — 그 파일은 저장소가 없어도 열자마자 채점이 다 보인다. */
function loadGrades(list, force){
  if(!Array.isArray(list)) return 0;
  const by = new Map(list.map(r => [(r.run_id || '') + '|' + (r.agent || ''), r]));
  let n = 0;
  cells().forEach(c => {
    const r = by.get(c.dataset.gid);
    if(!r || (!r.verdict && !r.note)) return;
    const cur = get(c.dataset.gid);
    if(!force && cur && cur.verdict) return;      // 이 PC 에서 이미 찍은 판정이 우선
    put(c.dataset.gid, { verdict: r.verdict || null, note: r.note || '',
                         auto: !!r.auto_accepted, at: r.judged_at || null });
    n++;
  });
  cells().forEach(paint); tally();
  return n;
}

/* 파일에 구워져 온 판정을 열 때 자동으로 싣는다(빈 셀만). 전부 덮어쓰려면 배너의 버튼. */
function applyBaked(){
  const g = window.__GRADES__;
  if(!Array.isArray(g)) return;
  const have = g.filter(r => r.verdict).length;
  if(!have) return;
  const n = loadGrades(g, false);
  const b = document.querySelector('.bar');
  if(b) b.insertAdjacentHTML('afterbegin',
    '<span class="baked">이 파일에 채점 ' + have + '건이 저장돼 있다 — ' + n + '건을 불러왔다. ' +
    '<button onclick="alert(loadGrades(window.__GRADES__, true) + \'건을 파일 내용으로 덮어썼다.\')">' +
    '전부 덮어쓰기</button></span>');
}

function importGrades(ev){
  const f = ev.target.files && ev.target.files[0];
  if(!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    let list;
    try { list = JSON.parse(rd.result); } catch(e){ alert('JSON 을 읽지 못했다: ' + e.message); return; }
    if(!Array.isArray(list)){ alert('내보내기 JSON(배열) 이 아니다.'); return; }
    const force = confirm('이 PC 의 기존 판정까지 파일 내용으로 덮어쓸까?\n' +
                          '취소를 누르면 미채점 셀만 채운다.');
    alert(loadGrades(list, force) + '건을 불러왔다.');
  };
  rd.readAsText(f);
  ev.target.value = '';
}

/* 현재 화면을 판정까지 포함해 HTML 한 장으로 다시 저장한다. */
function exportSnapshot(){
  const btn = document.getElementById('snapbtn');
  if(btn){ btn.disabled = true; btn.textContent = '저장 중…'; }
  setTimeout(() => {
    const undo = [];
    const strip = (sel, attr) => document.querySelectorAll(sel).forEach(el => {
      const v = el.getAttribute(attr);
      if(v !== null){ undo.push([el, attr, v]); el.removeAttribute(attr); }
    });
    // 화면에 들어와 이미 채워진 그림의 data: URI 를 걷어낸다. 안 걷으면 같은 base64 가
    // __IMG__ 배열과 태그 양쪽에 직렬화돼 파일이 두 배가 된다(실측 13MB → 20MB+).
    strip('img[data-img]', 'src');
    strip('a[data-img]', 'href');
    strip('a[data-img]', 'target');
    strip('[data-img]', 'data-hyd');
    // 필터·포커스는 보는 사람의 상태지 채점 결과가 아니다 — 전부 푼 상태로 저장한다.
    const hid = Array.from(document.querySelectorAll('.hidden'));
    hid.forEach(e => e.classList.remove('hidden'));
    const act = document.querySelector('.cell.active');
    if(act) act.classList.remove('active');
    // 메모는 textarea 의 value 라서 outerHTML 에 안 실린다 — 판정과 함께 __GRADES__ 로
    // 나가고 열 때 paint() 가 되돌려 준다.
    const old = document.getElementById('baked-grades');
    if(old) old.remove();
    const s = document.createElement('script');
    s.id = 'baked-grades';
    // 메모에 닫는 script 태그 문자열이 들어가면 문서가 거기서 끊긴다.
    s.textContent = 'window.__GRADES__ = ' +
                    JSON.stringify(rows()).replace(/<\//g, '<\\/') + ';';
    document.body.appendChild(s);

    let html = '';
    try { html = '<!doctype html>\n' + document.documentElement.outerHTML; }
    finally {
      s.remove();
      undo.forEach(([el, a, v]) => el.setAttribute(a, v));
      hid.forEach(e => e.classList.add('hidden'));
      if(act) act.classList.add('active');
      if(btn){ btn.disabled = false; btn.textContent = '채점 포함 저장'; }
    }
    const d = new Date(), p = n => String(n).padStart(2, '0');
    const name = `review_graded_${d.getFullYear()}${p(d.getMonth()+1)}${p(d.getDate())}`
               + `-${p(d.getHours())}${p(d.getMinutes())}.html`;
    download(name, html, 'text/html;charset=utf-8');
    const t = { pass:0, partial:0, fail:0 };
    rows().forEach(r => { if(r.verdict) t[r.verdict]++; });
    alert(`${name}\n${(html.length/1048576).toFixed(1)} MB · 판정 `
        + `${t.pass + t.partial + t.fail}건(정답 ${t.pass} · 부분 ${t.partial} · 오답 ${t.fail})을 `
        + `HTML 안에 담았다.\n이 파일만 옮기면 그림과 채점이 그대로 열린다.`);
  }, 30);
}
function exportCSV(){
  const r = rows();
  const cols = ['run_id','agent','verdict','auto_verdict','auto_accepted','no_answer','category','note'];
  const q = v => '"' + String(v === null || v === undefined ? '' : v).replace(/"/g, '""') + '"';
  // BOM 을 붙인다 — 엑셀이 UTF-8 CSV 를 cp949 로 읽어 한글 메모가 깨지는 걸 막는다.
  download('review_grades.csv', '﻿' + [cols.join(',')].concat(r.map(x => cols.map(c => q(x[c])).join(','))).join('\n'), 'text/csv');
}
function openAll(v){ document.querySelectorAll('.card:not(.hidden) details').forEach(d => d.open = v); }

document.addEventListener('click', e => {
  const b = e.target.closest('.jb');
  if(b){ setVerdict(b.closest('.cell'), b.dataset.v, false); focusCell(b.closest('.cell')); return; }
  const c = e.target.closest('.cell[data-gid]');
  if(c && !e.target.closest('a')) focusCell(c);
});
document.addEventListener('input', e => {
  if(!e.target.classList.contains('jnote')) return;
  const cell = e.target.closest('.cell'), s = get(cell.dataset.gid) || {};
  // 메모를 손으로 적으면 더는 '자동채택' 이 아니다 — '자동채택 취소' 로 메모까지 날아가면 안 된다.
  put(cell.dataset.gid, { verdict: s.verdict || null, note: e.target.value, auto: false, at: s.at || null });
  paint(cell); tally();
});
window.addEventListener('DOMContentLoaded', () => {
  // 순서 주의: 파일에 구워진 판정을 먼저 싣고 나서 무응답 자동오답을 채운다.
  // 반대로 하면 저장해 둔 사람 판정이 '무응답 → 오답' 에 덮인다.
  applyBaked(); seedNoAnswer(); cells().forEach(paint); tally(); applyFilters();
});
"""


def build_html(graded: list[dict], summary: dict, agents: list[str], specs: dict,
               out_dir: Path, stamp: str, src: str) -> str:
    groups: "OrderedDict[str, dict]" = OrderedDict()
    for rec in graded:
        rid = rec.get("run_id") or rec.get("id")
        g = groups.setdefault(rid, {"meta": None, "cells": {}})
        g["cells"][rec.get("agent")] = rec
        if g["meta"] is None:
            g["meta"] = {k: rec.get(k) for k in
                         ("id", "variant", "category", "capability", "task_kind",
                          "is_safety_ambiguous", "prompt", "grading_criteria",
                          "manual_note", "verifiers")}

    cats = sorted({str(g["meta"].get("category", "")).split(".")[0].strip()
                   for g in groups.values() if g["meta"].get("category")})
    cat_btns = "".join(f'<button data-fg="cat" onclick="setF(\'cat\',\'{esc(c)}\',this)">'
                       f'{esc(c)}류</button>' for c in cats)
    agent_btns = "".join(f'<button data-fg="agent" onclick="setF(\'agent\',\'{esc(a)}\',this)">'
                         f'{esc(a)}만</button>' for a in agents) if len(agents) > 1 else ""

    # 문항별 입력/레퍼런스 파일 목록 — 자동 진단이 입력 CSV 를 다시 읽는 데 쓴다.
    try:
        tfiles = json.loads((_HERE / "task_files.json").read_text(encoding="utf-8"))
    except Exception:                            # noqa: BLE001
        tfiles = {}
    # 레시피 재현 검증은 문항당 한 번만 계산한다(입력·레퍼런스 CSV 를 읽어 재현하므로).
    cards = "".join(
        render_card(rid, g["meta"], g["cells"], specs.get(g["meta"].get("id")), out_dir, agents,
                    verify_recipe(g["meta"].get("id"), specs[g["meta"]["id"]])
                    if g["meta"].get("id") in specs else "",
                    tfiles.get(g["meta"].get("id")) or {})
        for rid, g in groups.items())

    ba = summary.get("by_agent", {})
    sumline = " · ".join(
        f'{esc(a)}: {s.get("n")}실행 자동✓{s.get("auto_pass")} 자동✗{s.get("auto_fail")} '
        f'수동전용{s.get("manual_only")}' for a, s in ba.items())

    help_html = f'''<div class="help">
  <h2>쓰는 법</h2>
  <ul>
    <li>셀마다 <b>정답 / 부분 / 오답</b> 버튼을 누르면 즉시 브라우저에 저장된다.
        같은 버튼을 다시 누르면 판정이 취소된다.</li>
    <li>키보드: <kbd>j</kbd>/<kbd>k</kbd> 다음·이전 셀 · <kbd>n</kbd> 다음 미채점 ·
        <kbd>1</kbd> 정답 <kbd>2</kbd> 부분 <kbd>3</kbd> 오답(찍으면 자동으로 다음 셀).</li>
    <li><b>자동판정 일괄 채택</b> 을 먼저 누르면 기계검증 결과가 미채점 셀에 채워진다.
        그 뒤 <b>이견</b>(자동실패·무응답·에이전트 불일치) 필터만 돌며 손으로 고치는 게 가장 빠르다.
        기계검증기가 없는 셀은 채택 대상이 아니라 반드시 사람이 본다.</li>
    <li><b>무응답</b>(“Failed to generate a response.”)은 열 때 자동으로 오답이 찍힌다.</li>
    <li><b>CSV(파일처리) 문항</b>은 셀 맨 위에 <b>정답 ↔ 대답 대조표</b>가 뜬다 —
        <i>무엇을 봤나 · 통과 조건 · 정답(재계산) · 에이전트 답 · 판정</i> 다섯 칸이고,
        판정을 가른 항목만 추렸다. 그 위의 큰 배지가 이 셀의 자동판정이며
        <b>절차</b>(지정 차수·순서를 지켰나)와 <b>결과</b>(값·모양이 맞나)를 따로 표시한다.
        절차 ✓ / 결과 ✗ 는 “방법은 옳았고 계산만 어긋났다”는 뜻이다.</li>
    <li>문항 제목 옆 <b>부류 A / 부류 B</b> 배지와 정답 기준 블록을 먼저 읽을 것.
        <b>부류 A</b> 는 명세가 답을 유일하게 결정해 GT 와 엄격 비교한다.
        <b>부류 B</b> 는 “5차 다항 baseline” 처럼 정당한 구현이 여럿인 과제라
        <b>값 일치를 요구하지 않고 모양새로 채점</b>한다 — 실측으로 정당한 구현 4개의
        편차가 tolerance 의 6.7×10⁶ 배였고, 그중 레퍼런스가 오히려 가장 나빴다.
        그래서 <code>reference_match</code> 는 부류 B 에서 참고값으로 내려가 있다.</li>
    <li><b>자동 진단</b> 표는 채점기준의 계산을 입력 파일로 다시 해서
        <i>기준 · 실측(기준값) · 에이전트 보고값 · 판정</i>을 나란히 놓은 것이다.
        “참고”로 표시된 행은 자동 판정이 불가능한 항목(그림의 시각적 정확성 등)이니
        아래 그림·코드를 보고 사람이 정한다. <b>진단실패</b> 필터로 문제 있는 문항만 볼 수 있다.</li>
    <li><b>모델이 돌린 코드와 출력</b>은 답변 요약이 아니라 실제 계산이다 —
        수치가 미묘하게 다를 때 원인은 거기 있다(예: <code>np.std</code> 의 ddof).</li>
    <li>판정은 <code>{esc(STORE_PREFIX)}&lt;문항ID&gt;|&lt;에이전트&gt;</code> 키로 저장되므로
        이 HTML 을 다시 생성해도 유지된다. 다 끝나면 <b>JSON/CSV 내보내기</b>.</li>
    <li><b>다른 컴퓨터로 넘길 때는 <span style="color:#7c3aed">채점 포함 저장</span></b> 을 쓴다.
        그림은 원래 내장돼 있지만 <b>판정은 이 브라우저에만</b> 남아서, 파일만 복사하면
        받는 쪽에는 전부 미채점으로 보인다(<code>file://</code> 에서는 브라우저가
        저장소를 아예 막기도 한다). 이 버튼은 지금까지의 판정·메모를 HTML 안에
        <code>__GRADES__</code> 로 구워 새 파일로 내려받는다 — <b>그 파일 하나만</b> 있으면
        어느 PC 에서든 그림과 채점이 그대로 열린다.</li>
    <li>받은 쪽이 이어서 채점하면 그 브라우저에 저장되고, 다시 <b>채점 포함 저장</b> 으로
        되돌려 보낼 수 있다. 열 때 파일에 든 판정은 <b>비어 있는 셀만</b> 채우고,
        상단 배너의 <b>전부 덮어쓰기</b> 를 눌러야 이 PC 판정까지 파일 것으로 바꾼다.
        <b>채점 불러오기</b> 는 내보낸 <code>review_grades.json</code> 을 같은 방식으로 싣는다.</li>
  </ul>
</div>'''

    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>라만 벤치 채점 콘솔 — {esc(stamp)}</title>
<style>{_CSS}
{spectra_panel.CSS}
{stage_map.CSS}</style></head><body>
<header>
  <h1>라만 벤치마크 채점 콘솔
    <span class="sub">{len(groups)}문항 · {len(graded)}실행 · {esc(", ".join(agents))} ·
    {esc(stamp)} · {esc(src)}</span></h1>
  <div class="sub">{sumline}</div>
  <div class="bar">
    {cat_btns}{agent_btns}
    <span style="width:6px"></span>
    <button data-fg="flag" onclick="setF('flag','todo',this)">미채점</button>
    <button data-fg="flag" onclick="setF('flag','autofail',this)">자동실패</button>
    <button data-fg="flag" onclick="setF('flag','diagfail',this)">진단실패</button>
    <button data-fg="flag" onclick="setF('flag','noans',this)">무응답</button>
    <button data-fg="flag" onclick="setF('flag','disagree',this)">에이전트 불일치</button>
    <button data-fg="flag" onclick="setF('flag','amb',this)">안전-애매</button>
    <button data-fg="flag" onclick="setF('flag','spec',this)">방법 미지정</button>
    <span class="sp"></span>
    <button class="act" onclick="acceptAuto(false)">자동판정 일괄 채택</button>
    <button onclick="acceptAuto(true)">통과만 채택</button>
    <button onclick="clearAuto()">자동채택 취소</button>
    <button onclick="nextTodo()">다음 미채점 ▸</button>
    <button onclick="openAll(true)">전부 펼치기</button>
    <button onclick="openAll(false)">접기</button>
    <button class="act" onclick="exportJSON()">JSON</button>
    <button class="act" onclick="exportCSV()">CSV</button>
    <button class="act snap" id="snapbtn" onclick="exportSnapshot()">채점 포함 저장</button>
    <button onclick="document.getElementById('impfile').click()">채점 불러오기</button>
    <input type="file" id="impfile" accept=".json,application/json"
           style="display:none" onchange="importGrades(event)">
  </div>
  <div class="bar"><span class="tally" id="tally"></span></div>
</header>
<div class="wrap">{help_html}{cards}</div>
<script>window.__STORE_PREFIX__ = {json.dumps(STORE_PREFIX)};</script>
<script>window.__IMG__ = {json.dumps(stage_map.assets())};</script>
<script>{stage_map.HYDRATE_JS}</script>
<script>{_JS}</script>
</body></html>'''


# ── 입력 로딩 ────────────────────────────────────────────────────────────────

def _load_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("graded", data) if isinstance(data, dict) else data


# tasks.json 에서 레코드로 다시 실어 줄 '채점 설정' 필드.
# prompt/answer/tool_calls/pre_state/post_state 등 실행 데이터는 여기 없다.
_REFRESH = ("verifiers", "grading_criteria", "manual_note", "auto_gradable",
            "expected_tools", "task_kind", "capability", "category",
            "is_safety_ambiguous")


def _refresh_config(rows: list[dict]) -> int:
    """레코드에 박힌 채점 설정을 tasks.json 의 현재 값으로 갱신한다.

    run_bench 는 실행 시점의 verifiers 를 레코드에 스냅샷으로 저장한다. 그래서
    raw_runs.jsonl 을 그대로 채점하면 그 뒤에 고친 검증기가 반영되지 않는다(실측:
    AILA 자동실패가 31 vs 34 로 갈렸다). 검증기·채점기준은 '채점 설정'이므로 갱신이
    맞다 — 프롬프트·답변·툴트레이스 같은 실행 데이터는 건드리지 않는다.
    """
    tasks_path = _HERE / "tasks.json"
    if not tasks_path.exists():
        return 0
    tasks = {t.get("run_id", t["id"]): t
             for t in json.loads(tasks_path.read_text(encoding="utf-8"))}
    n = 0
    for r in rows:
        t = tasks.get(r.get("run_id") or r.get("id"))
        if not t:
            continue
        for f in _REFRESH:
            if f in t and r.get(f) != t[f]:
                r[f] = t[f]
                n += 1
    return n


def _dedupe(rows: list[dict], agents: set[str]) -> list[dict]:
    """(문항, 에이전트) 마다 한 건만. 실패보다 성공을, 같으면 나중 것을 남긴다
    raw_runs.jsonl 은 append-only 라 중단/재실행분이 그대로 쌓여 있다 — 이걸 안 하면
    같은 문항이 여러 번 집계된다."""
    def failed(r: dict) -> bool:
        return bool(r.get("http_error")) or r.get("response_type") in (None, "error")

    picked: dict[tuple, dict] = {}
    for r in rows:
        if r.get("agent") not in agents:
            continue
        key = (r.get("run_id") or r.get("id"), r["agent"])
        cur = picked.get(key)
        if cur is None:
            picked[key] = r
        elif failed(r) and not failed(cur):
            pass                                 # 성공을 실패로 덮지 않는다
        else:
            picked[key] = r                      # 실패→성공 교체, 또는 같은 등급이면 최신
    return [picked[k] for k in sorted(picked)]


def main():
    ap = argparse.ArgumentParser(description="벤치마크 채점 콘솔 HTML (AILA/CoALA 공용)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--runs", help="runs_*.json 또는 raw_runs.jsonl (여기서 바로 채점한다)")
    src.add_argument("--graded", help="grade.py 가 만든 graded_*.json (이미 채점된 것)")
    ap.add_argument("--agents", default="AILA,CoALA", help="대상 에이전트(쉼표)")
    ap.add_argument("--out", default=None, help="출력 html(기본: results/review_<시각>.html)")
    ap.add_argument("--no-embed", action="store_true",
                    help="이미지를 HTML 에 넣지 않고 상대링크로 둔다(생성이 빠르지만 "
                         "HTML 을 다른 PC 로 옮기면 그림이 깨진다)")
    args = ap.parse_args()

    inp = Path(args.graded or args.runs or (_HERE / "results" / "raw_runs.jsonl"))
    if not inp.exists():
        raise SystemExit(f"입력 파일이 없다: {inp}")
    want = [a.strip() for a in args.agents.split(",") if a.strip()]

    rows = _dedupe(_load_rows(inp), set(want))
    if not rows:
        raise SystemExit(f"{inp} 에서 에이전트 {want} 의 실행을 못 찾았다.")

    # graded 입력이면 채점 결과가 이미 있다. runs 입력이면 여기서 채점한다 —
    # 콘솔을 열기 위해 grade.py 를 따로 돌려야 하는 수고를 없앤다.
    if args.graded:
        graded = rows
    else:
        n_ref = _refresh_config(rows)
        if n_ref:
            print(f"      tasks.json 의 현재 채점설정으로 {n_ref}개 필드 갱신")
        graded = [grade_one(r) for r in rows]
    summary = summarize(graded)

    # 실제로 존재하는 에이전트만 열로 만든다(빈 열이 생기지 않게).
    have = {g.get("agent") for g in graded}
    agents = [a for a in AGENT_ORDER if a in have] + sorted(have - set(AGENT_ORDER))
    agents = [a for a in agents if a in want]

    out = Path(args.out) if args.out else (
        _HERE / "results" / f"review_{datetime.now():%Y%m%d-%H%M%S}.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    specs = _load_specs()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out.write_text(build_html(graded, summary, agents, specs, out.parent, stamp, inp.name),
                   encoding="utf-8")

    n_noans = sum(1 for g in graded
                  if _no_answer(g.get("answer") or g.get("final_report") or ""))
    st = stage_map.embed_stats()
    size = out.stat().st_size / 1e6
    if st["n"]:
        print(f"[완료] 채점 콘솔 → {out}  ({size:.1f} MB)")
        print(f"       이미지 {st['n']}장을 HTML 에 내장 ({st['bytes']/1e6:.1f} MB, "
              f"손실압축 {st['lossy']}장) - 파일 하나만 옮기면 다른 PC 에서도 그림이 보인다")
        if st["failed"]:
            print(f"       [주의] {st['failed']}장은 인코딩 실패 → 상대링크로 남았다")
    else:
        print(f"[완료] 채점 콘솔 → {out}  ({size:.1f} MB)")
        if args.no_embed:
            print("       이미지는 상대링크다 - 이 HTML 을 옮기면 그림이 깨진다")
    print(f"       문항 {len({g.get('run_id') or g.get('id') for g in graded})}개 · "
          f"실행 {len(graded)}건 · 에이전트 {agents}")
    for a, s in summary["by_agent"].items():
        print(f"       {a}: 자동통과 {s['auto_pass']} · 자동실패 {s['auto_fail']} · "
              f"수동전용 {s['manual_only']} · 오류 {s['error']}")
    print(f"       무응답(자동 오답 처리) {n_noans}건")
    if diagnostics is not None:
        try:
            tfx = json.loads((_HERE / "task_files.json").read_text(encoding="utf-8"))
        except Exception:                        # noqa: BLE001
            tfx = {}
        ds = {"pass": 0, "fail": 0}
        for g in graded:
            try:
                v = diagnostics.overall(diagnostics.run(g, g, tfx.get(g.get("id")) or {}))
            except Exception:                    # noqa: BLE001
                continue
            if v in ds:
                ds[v] += 1
        print(f"       자동 진단: 통과 {ds['pass']} · 실패 {ds['fail']} "
              f"(상단 '진단실패' 필터로 바로 이동)")
    print("       브라우저로 열고 '자동판정 일괄 채택' → 이견 필터만 손으로 확인 → JSON/CSV 내보내기")


if __name__ == "__main__":
    main()
