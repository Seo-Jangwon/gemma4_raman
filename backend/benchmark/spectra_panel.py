# -*- coding: utf-8 -*-
"""
채점 리포트용 '스펙트럼 비교 패널' 생성기.

[왜 필요한가]
report.py 는 지금까지 툴 트레이스와 최종 답변 텍스트만 보여줬다. 그런데 전처리 문항
(T037~T056 등)의 채점은 결국 "입력 스펙트럼을 요구대로 가공했는가"를 보는 일이라,
숫자 트레이스만으로는 눈으로 확인할 수가 없었다 — 스파이크가 정말 사라졌는지, 베이스라인이
정말 펴졌는지, 0-1 정규화가 됐는지는 그림을 겹쳐 봐야 1초 만에 안다.

그래서 문항마다 세 가지를 같은 x축에 위아래로 붙여 그린다:
  · 위  패널: 입력 원본 스펙트럼 (data/uploads/<날짜>/<문항ID>.csv)  — 회색
  · 아래 패널: 에이전트 산출물 (data/runs/<세션>/spectra/*.csv)      — 파랑/보라
              + 정답 레퍼런스 (backend/benchmark/task_refs/*.csv)    — 초록 파선
그리고 산출물 vs 레퍼런스의 max|Δ| / RMSE 를 표로 같이 준다(수동채점 근거).

[설계 결정]
· 그림은 matplotlib 이 아니라 '인라인 SVG' 로 직접 그린다. 리포트 HTML 한 파일만 열면
  되고(PNG 수백 장을 곁들일 필요 없음), 확대해도 깨지지 않고, 좌표만 있어 용량이 작다.
· 다운샘플은 '구간별 min/max 포장선' 방식이다. 단순 stride 로 줄이면 1점짜리 스파이크가
  통째로 사라져서 despike 채점이 불가능해진다 — 이 벤치마크에서는 치명적이다.
· 두 패널의 y축은 따로(원본은 수천 카운트, 결과는 0-1 일 수 있음), x축은 공유한다.
  x축이 어긋나면 '어디의 스파이크가 사라졌는지' 대조가 안 된다.
· 입력 파일이 스펙트럼이 아니라 맵/라이브러리(T050·T071·T072 등, x_mm/point_id/
  spectrum_id 열로 여러 스펙트럼이 세로로 쌓인 형태)면 그룹별로 나눠 최대 5개만 그린다.
  안 나누면 전부 한 줄로 이어져 톱니 쓰레기가 된다.
· 산출물 위치는 manifest(run_store) 를 1순위, 툴 트레이스를 2순위로 본다. manifest 기록은
  실패해도 실험을 막지 않도록 예외를 삼키게 되어 있어서(run_store 주석 참고) 비어 있을 수
  있는데, 그때도 save_spectrum/run_analysis 결과의 경로로 복구된다.
"""
from __future__ import annotations

import csv
import html
import json
import os
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_DATA_ROOT = _PROJECT_ROOT / "data"
_UPLOADS = _DATA_ROOT / "uploads"
_RUNS = _DATA_ROOT / "runs"
_TASK_REFS = _HERE / "task_refs"
_TASK_FILES = _HERE / "task_files.json"

MAX_POINTS = 700          # 곡선 하나당 SVG 좌표 상한(min/max 포장선 기준)
MAX_GROUPS = 5            # 맵/라이브러리 파일에서 그릴 스펙트럼 개수 상한
MAX_OUT_CURVES = 4        # 에이전트 산출물 곡선 상한(그 이상은 표에만)

_OUT_COLORS = ("#2563eb", "#7c3aed", "#db2777", "#0891b2")
_INPUT_COLOR = "#9ca3af"
_REF_COLOR = "#059669"


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


# ── task_files.json ──────────────────────────────────────────────────────────

_task_files_cache: dict | None = None


def _task_files() -> dict:
    global _task_files_cache
    if _task_files_cache is None:
        try:
            _task_files_cache = json.loads(_TASK_FILES.read_text(encoding="utf-8"))
        except Exception:
            _task_files_cache = {}
    return _task_files_cache


def _find_upload(name: str) -> Path | None:
    """data/uploads/<날짜>/<name> — 날짜 폴더가 여럿이면 가장 최근 것."""
    hits = sorted(_UPLOADS.glob(f"*/{name}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


# ── CSV 읽기 ─────────────────────────────────────────────────────────────────

_X_CANDIDATES = ("raman_shift_cm-1", "raman_shift", "wavelength_nm", "wavenumber", "pixel_index")
_GROUP_CANDIDATES = ("spectrum_id", "point_id")


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def read_curves(path: Path, max_groups: int = MAX_GROUPS) -> list[dict]:
    """CSV → [{name, x, y, xlabel}] 목록.

    · intensity 열(없으면 마지막 수치 열)을 y 로, _X_CANDIDATES 중 첫 열을 x 로 쓴다.
    · 그룹 열(spectrum_id/point_id, 또는 x_mm+y_mm)이 있으면 그룹별로 쪼갠다.
    · 헤더 앞에 붙은 '# key,value' 메타 주석행은 건너뛴다. 측정 자동저장 CSV
      (data/results/<날짜>/<세션>/*.csv)가 그 형식인데, 안 건너뛰면 첫 주석행이 헤더로
      잡혀 intensity 열을 못 찾고 '엉뚱한 열을 세기로' 그린다(라만축을 신호로 오인).
    """
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception:
        return []
    body = []
    for ln in lines:
        if not body and ln.lstrip().startswith("#"):
            continue
        body.append(ln)
    try:
        rows = list(csv.DictReader(body))
    except Exception:
        return []
    if not rows:
        return []
    headers = list(rows[0].keys())

    ycol = next((h for h in headers if h and h.strip().lower() == "intensity"), None)
    if ycol is None:                       # intensity 열이 없으면 마지막 열을 세기로 본다
        ycol = headers[-1]
    # 후보 목록 순서(라만시프트 > 파장 > 픽셀번호)가 우선순위다. headers 를 돌면 CSV 의
    # 열 순서가 우선순위를 결정해 버린다 — save_spectrum 출력은
    # 'pixel_index,raman_shift_cm-1,intensity' 라서 입력 파일(raman_shift 만 있음)과
    # 서로 다른 축에 그려지고, 두 패널의 x축 정렬이 깨진다.
    xcol = next((h for h in _X_CANDIDATES if h in headers), None)

    gcol = next((h for h in headers if h in _GROUP_CANDIDATES), None)
    gpair = ("x_mm" in headers and "y_mm" in headers)

    def gkey(r) -> str:
        if gcol:
            return str(r.get(gcol))
        if gpair:
            return f"x={r.get('x_mm')} y={r.get('y_mm')}"
        return ""

    groups: dict[str, list] = {}
    for r in rows:
        y = _to_float(r.get(ycol))
        if y is None:
            continue
        x = _to_float(r.get(xcol)) if xcol else None
        groups.setdefault(gkey(r), []).append((x, y))

    out = []
    for key, pts in list(groups.items())[:max_groups]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if any(v is None for v in xs):
            xs = list(range(len(ys)))
            xlabel = "index"
        else:
            xlabel = xcol or "index"
        out.append({"name": key, "x": xs, "y": ys, "xlabel": xlabel})
    n_more = len(groups) - len(out)
    if n_more > 0:
        out.append({"name": f"(+{n_more} more groups not drawn)", "x": [], "y": [], "xlabel": ""})
    return [c for c in out if c["y"] or c["name"].startswith("(+")]


# ── 다운샘플 & 좌표 ──────────────────────────────────────────────────────────

def decimate(x: list, y: list, max_pts: int = MAX_POINTS) -> tuple[list, list]:
    """구간별 min/max 포장선. 1점 스파이크를 반드시 살린다(despike 채점의 전제)."""
    n = len(y)
    if n <= max_pts:
        return x, y
    nb = max(1, max_pts // 2)
    step = n / nb
    rx, ry = [], []
    for b in range(nb):
        i0 = int(b * step)
        i1 = min(n, max(i0 + 1, int((b + 1) * step)))
        seg = y[i0:i1]
        imin = i0 + min(range(len(seg)), key=seg.__getitem__)
        imax = i0 + max(range(len(seg)), key=seg.__getitem__)
        for i in sorted((imin, imax)):     # 원래 x 순서를 지킨다
            rx.append(x[i]); ry.append(y[i])
    return rx, ry


def _fmt(v: float) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 1e5 or a < 1e-3:
        return f"{v:.2e}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


class _Panel:
    """왼쪽에 y축 라벨 여백을 둔 단순 선형 플롯 박스."""

    W, PAD_L, PAD_R, PAD_T, PAD_B = 1000, 62, 8, 12, 16

    def __init__(self, height: int, xr: tuple[float, float], yr: tuple[float, float]):
        self.H = height
        self.x0, self.x1 = xr
        self.y0, self.y1 = yr
        if self.x1 <= self.x0:
            self.x1 = self.x0 + 1.0
        if self.y1 <= self.y0:
            self.y1 = self.y0 + 1.0

    def sx(self, v: float) -> float:
        w = self.W - self.PAD_L - self.PAD_R
        return self.PAD_L + (v - self.x0) / (self.x1 - self.x0) * w

    def sy(self, v: float) -> float:
        h = self.H - self.PAD_T - self.PAD_B
        return self.PAD_T + (1.0 - (v - self.y0) / (self.y1 - self.y0)) * h

    def path(self, x: list, y: list) -> str:
        pts = " ".join(f"{self.sx(a):.1f},{self.sy(b):.1f}" for a, b in zip(x, y))
        return pts

    def frame(self, title: str) -> str:
        h = self.H
        return (
            f'<rect x="{self.PAD_L}" y="{self.PAD_T}" width="{self.W - self.PAD_L - self.PAD_R}" '
            f'height="{h - self.PAD_T - self.PAD_B}" fill="#fff" stroke="#e5e7eb"/>'
            f'<text x="{self.PAD_L + 4}" y="{self.PAD_T + 11}" class="ptitle">{esc(title)}</text>'
            f'<text x="{self.PAD_L - 4}" y="{self.PAD_T + 9}" class="ax" text-anchor="end">{esc(_fmt(self.y1))}</text>'
            f'<text x="{self.PAD_L - 4}" y="{h - self.PAD_B}" class="ax" text-anchor="end">{esc(_fmt(self.y0))}</text>'
        )


def _vlines(p: _Panel, xs: list) -> str:
    out = []
    for v in xs:
        if p.x0 <= v <= p.x1:
            out.append(f'<line x1="{p.sx(v):.1f}" y1="{p.PAD_T}" x2="{p.sx(v):.1f}" '
                       f'y2="{p.H - p.PAD_B}" class="mark"/>')
    return "".join(out)


# ── 산출물 탐색 ──────────────────────────────────────────────────────────────

def _sanitize_label(text: str) -> str:
    """run_store._sanitize / detail_log._sanitize 와 같은 규칙."""
    return re.sub(r"[^0-9A-Za-z_-]", "-", str(text))[:64] or "nosession"


def _rel_to_data(p: str) -> Path | None:
    """'runs/<세션>/spectra/01_x.csv' 또는 절대경로/URL → 실제 파일 경로."""
    s = str(p or "").strip()
    if not s:
        return None
    if s.startswith("/api/results/"):
        s = "results/" + s[len("/api/results/"):]
    cand = Path(s)
    if not cand.is_absolute():
        cand = _DATA_ROOT / s
    return cand if cand.exists() else None


def find_outputs(rec: dict) -> tuple[list[Path], list[str]]:
    """이 실행이 만든 스펙트럼 CSV 목록 + 에이전트 그림(상대경로) 목록.

    1순위 manifest(run_store), 2순위 툴 트레이스(save_spectrum.path /
    run_analysis.saved_files). 중복은 경로로 제거한다.
    """
    csvs: list[Path] = []
    figs: list[str] = []
    seen: set[str] = set()

    def add_csv(v):
        f = _rel_to_data(v)
        if f is not None and f.suffix.lower() == ".csv" and str(f) not in seen:
            seen.add(str(f)); csvs.append(f)

    def add_fig(v):
        f = _rel_to_data(v)
        if f is not None and f.suffix.lower() in (".png", ".jpg", ".jpeg") and str(f) not in seen:
            seen.add(str(f)); figs.append(str(f))

    label = _sanitize_label(rec.get("session_id") or "")
    mpath = _RUNS / label / "manifest.json"
    if mpath.exists():
        try:
            man = json.loads(mpath.read_text(encoding="utf-8"))
            for a in man.get("artifacts") or []:
                if a.get("kind") == "figure":
                    add_fig(a.get("path"))
                else:
                    add_csv(a.get("path"))
        except Exception:
            pass

    for c in rec.get("tool_calls") or []:
        r = c.get("result")
        if not isinstance(r, dict):
            continue
        add_csv(r.get("path"))
        for v in (r.get("saved_files") or []):
            add_csv(v.get("path") if isinstance(v, dict) else v)
        for v in (r.get("files") or []):
            if isinstance(v, str):
                add_csv(v); add_fig(v)
    return csvs, figs


# ── 통계 ─────────────────────────────────────────────────────────────────────

def _diff_stats(a: list, b: list) -> dict:
    n = min(len(a), len(b))
    if n == 0:
        return {}
    d = [abs(a[i] - b[i]) for i in range(n)]
    mx = max(d)
    rmse = (sum(v * v for v in d) / n) ** 0.5
    return {"n": n, "len_mismatch": len(a) != len(b), "max_abs": mx, "rmse": rmse,
            "argmax": d.index(mx)}


# ── 메인 ─────────────────────────────────────────────────────────────────────

def _figs_html(out_figs: list[str], out_dir: Path) -> str:
    """에이전트가 직접 그린 그림. 경로 처리는 stage_map.img_src 에 맡긴다 —
    기본이 data: URI 임베드라 HTML 한 장만 다른 PC 로 옮겨도 그림이 보인다.
    (지연 import: stage_map 은 spectra_panel 을 import 하지 않으므로 순환은 없다.)"""
    if not out_figs:
        return ""
    import stage_map
    links = []
    for f in out_figs:
        p = Path(f)
        links.append(f'<a {stage_map.img_href(p, out_dir)}>'
                     f'<img class="specfig" {stage_map.img_src(p, out_dir)} '
                     f'alt="{esc(p.name)}" loading="lazy"></a>')
    return f'<div class="specfigs">{"".join(links)}</div>'


def _curve_rows(items: list[tuple[str, str, dict]]) -> str:
    rows = []
    for role, color, c in items:
        y = c["y"]
        if not y:
            continue
        rows.append(
            f'<tr><td><span class="sw" style="background:{color}"></span>{esc(role)}</td>'
            f'<td class="fn">{esc(c["name"])}</td><td>{len(y)}</td>'
            f'<td>{esc(_fmt(min(y)))} … {esc(_fmt(max(y)))}</td></tr>'
        )
    return "".join(rows)


def build_spectra_panel(rec: dict, out_dir: Path) -> str:
    """실행 레코드 하나에 대한 스펙트럼 비교 HTML 조각. 볼 게 없으면 ''.

    out_dir: 리포트 HTML 이 저장될 디렉터리(에이전트 그림 상대링크 계산용).
    """
    tid = str(rec.get("id") or "")
    tf = _task_files().get(tid) or {}

    # 입력 원본
    inputs: list[tuple[Path, dict]] = []
    for name in tf.get("files") or []:
        p = _find_upload(name)
        if p is None:
            continue
        for c in read_curves(p):
            inputs.append((p, c))

    # 정답 레퍼런스
    refs: list[tuple[Path, dict]] = []
    for name in tf.get("reference_files") or []:
        p = _TASK_REFS / name
        if not p.exists():
            continue
        for c in read_curves(p, max_groups=1):
            refs.append((p, c))

    # 에이전트 산출물
    out_csvs, out_figs = find_outputs(rec)
    outs: list[tuple[Path, dict]] = []
    for p in out_csvs:
        for c in read_curves(p, max_groups=1):
            outs.append((p, c))

    if not (inputs or refs or outs or out_figs):
        return ""

    in_curves = [c for _, c in inputs if c["y"]]
    ref_curves = [c for _, c in refs if c["y"]]
    out_curves = [c for _, c in outs if c["y"]][:MAX_OUT_CURVES]

    # 정답 스파이크 위치(cm-1). 아래에서 축을 인덱스로 바꾸면 같이 변환해야 한다.
    marks = [float(v) for v in (tf.get("ground_truth") or {}).get("spike_positions_cm-1", [])
             if _to_float(v) is not None]
    _in_axis = list(in_curves[0]["x"]) if in_curves else []

    # x축 통일. 에이전트가 raman_shift 없이 intensity 만 저장하면(save_result 기본형)
    # 그 곡선의 x 는 0..N-1 이고 입력은 200..2000 cm-1 이다 — 그대로 겹치면 두 패널이
    # 엉뚱하게 어긋난다. 축 종류가 섞였고 점 수가 같다면 전부 인덱스로 맞춘다.
    axis_note = ""
    _all = in_curves + ref_curves + out_curves
    labels = {c["xlabel"] for c in _all}
    if len(labels) > 1:
        lens = {len(c["y"]) for c in _all}
        if len(lens) == 1:
            for c in _all:
                c["x"] = list(range(len(c["y"])))
                c["xlabel"] = "index"
            # cm-1 마커를 입력 축에서의 최근접 인덱스로 옮긴다(안 옮기면 엉뚱한 데 그려진다)
            if _in_axis:
                marks = [min(range(len(_in_axis)), key=lambda i: abs(_in_axis[i] - m)) for m in marks]
            else:
                marks = []
            axis_note = ("x축 종류가 섞여 있어(" + ", ".join(sorted(labels)) +
                         ") 점 개수가 같은 것을 확인하고 인덱스로 정렬했다.")
        else:
            axis_note = ("⚠ x축 종류(" + ", ".join(sorted(labels)) +
                         ")와 점 개수가 모두 달라 겹쳐 그린 결과가 어긋날 수 있다.")

    # x 범위는 두 패널이 공유한다 — 안 그러면 '어디가 어떻게 바뀌었나' 대조가 깨진다.
    all_x = [v for c in (in_curves + ref_curves + out_curves) for v in (c["x"][:1] + c["x"][-1:])]
    if not all_x:
        # 측정 문항처럼 CSV 는 없고 에이전트가 그린 그림만 있는 경우 — 그림만이라도 보여준다.
        figs = _figs_html(out_figs, out_dir)
        if not figs:
            return ""
        return (f'<details class="spectra" open><summary>에이전트가 그린 그림</summary>'
                f'{figs}</details>')
    xr = (min(all_x), max(all_x))
    xlabel = (in_curves or ref_curves or out_curves)[0]["xlabel"]

    def yrange(curves):
        vals = [v for c in curves for v in c["y"]]
        if not vals:
            return (0.0, 1.0)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.05 or (abs(hi) * 0.05 or 1.0)
        return (lo - pad, hi + pad)

    H1, H2, GAP = 132, 150, 10
    svg_parts = []
    y_off = 0

    # ── 위: 입력 원본. 입력 파일이 없는 문항(측정 계열 81개)은 이 패널을 아예 빼서
    #    빈 칸으로 리포트를 늘리지 않는다.
    if in_curves:
        p1 = _Panel(H1, xr, yrange(in_curves))
        body = [p1.frame(f"입력 원본 — {', '.join(sorted({p.name for p, _ in inputs}))}")]
        body.append(_vlines(p1, marks))
        for c in in_curves:
            x, y = decimate(c["x"], c["y"])
            body.append(f'<polyline points="{p1.path(x, y)}" fill="none" '
                        f'stroke="{_INPUT_COLOR}" stroke-width="1"/>')
        svg_parts.append(f'<g>{"".join(body)}</g>')
        y_off = H1 + GAP

    # ── 아래: 산출물 + 레퍼런스 (같은 y축에서 겹쳐 봐야 일치/불일치가 보인다)
    if ref_curves and out_curves:
        p2_title = "에이전트 산출물 vs 정답 레퍼런스"
    elif ref_curves:
        p2_title = "정답 레퍼런스 (에이전트 산출물 없음)"
    elif in_curves:
        p2_title = "에이전트 산출물 (비교할 레퍼런스 없음)"
    else:
        p2_title = "에이전트가 측정·저장한 스펙트럼"
    p2 = _Panel(H2, xr, yrange(out_curves + ref_curves))
    body = [p2.frame(p2_title)]
    body.append(_vlines(p2, marks))
    for c in ref_curves:
        x, y = decimate(c["x"], c["y"])
        body.append(f'<polyline points="{p2.path(x, y)}" fill="none" stroke="{_REF_COLOR}" '
                    f'stroke-width="2.2" stroke-dasharray="6 4" opacity="0.85"/>')
    for i, c in enumerate(out_curves):
        x, y = decimate(c["x"], c["y"])
        body.append(f'<polyline points="{p2.path(x, y)}" fill="none" '
                    f'stroke="{_OUT_COLORS[i % len(_OUT_COLORS)]}" stroke-width="1.2"/>')
    if not out_curves:
        body.append(f'<text x="{p2.W/2}" y="{H2/2}" class="none" text-anchor="middle">'
                    f'에이전트가 저장한 스펙트럼 없음 (파일 미저장)</text>')
    svg_parts.append(f'<g transform="translate(0,{y_off})">{"".join(body)}</g>')

    total_h = y_off + H2 + 14
    svg_parts.append(
        f'<text x="{_Panel.PAD_L}" y="{total_h - 2}" class="ax">{esc(_fmt(xr[0]))}</text>'
        f'<text x="{_Panel.W - _Panel.PAD_R}" y="{total_h - 2}" class="ax" text-anchor="end">{esc(_fmt(xr[1]))}</text>'
        f'<text x="{(_Panel.W)/2}" y="{total_h - 2}" class="ax" text-anchor="middle">{esc(xlabel)}</text>'
    )
    svg = (f'<svg class="specsvg" viewBox="0 0 {_Panel.W} {total_h}" '
           f'preserveAspectRatio="xMidYMid meet">{"".join(svg_parts)}</svg>')

    # ── 곡선 목록 표
    items = [("입력", _INPUT_COLOR, c) for c in in_curves]
    items += [("산출물", _OUT_COLORS[i % len(_OUT_COLORS)], c) for i, c in enumerate(out_curves)]
    items += [("레퍼런스", _REF_COLOR, c) for c in ref_curves]
    # 표의 '파일' 칸은 그룹명(맵 파일의 좌표 등)보다 실제 파일명이 유용하다.
    # items 와 순서가 1:1 로 맞게 같은 필터를 적용해 이름 목록을 만든다.
    names = ([p.name for p, c in inputs if c["y"]]
             + [p.name for p, c in outs if c["y"]][:MAX_OUT_CURVES]
             + [p.name for p, c in refs if c["y"]])
    items = [(role, color, {**c, "name": nm}) for (role, color, c), nm in zip(items, names)]
    curve_tbl = (f'<table class="spectbl"><thead><tr><th>역할</th><th>파일</th><th>점수</th>'
                 f'<th>y 범위</th></tr></thead><tbody>{_curve_rows(items)}</tbody></table>')

    # ── 산출물 vs 레퍼런스 오차
    diff_rows = ""
    if ref_curves and out_curves:
        for i, oc in enumerate(out_curves):
            for rc in ref_curves:
                st = _diff_stats(oc["y"], rc["y"])
                if not st:
                    continue
                warn = ' <span class="warn">길이 불일치</span>' if st["len_mismatch"] else ""
                diff_rows += (
                    f'<tr><td><span class="sw" style="background:{_OUT_COLORS[i % len(_OUT_COLORS)]}">'
                    f'</span>산출물 #{i+1}</td><td>{st["n"]}점{warn}</td>'
                    f'<td><b>{st["max_abs"]:.3e}</b></td><td>{st["rmse"]:.3e}</td>'
                    f'<td>idx {st["argmax"]}</td></tr>'
                )
    diff_tbl = ""
    if diff_rows:
        diff_tbl = (f'<table class="spectbl"><thead><tr><th>비교</th><th>길이</th>'
                    f'<th>max|Δ|</th><th>RMSE</th><th>최대오차 위치</th></tr></thead>'
                    f'<tbody>{diff_rows}</tbody></table>')
    elif ref_curves and not out_curves:
        diff_tbl = '<div class="specnote">레퍼런스는 있으나 산출물 파일이 없어 수치 비교 불가.</div>'

    figs_html = _figs_html(out_figs, out_dir)

    notes = []
    if marks:
        notes.append(f'주황 점선 = 정답 스파이크 위치 {len(marks)}개 '
                     f'(위 패널에 있고 아래 패널에 없어야 정상).')
    if axis_note:
        notes.append(axis_note)
    note = f'<div class="specnote">{" ".join(esc(n) for n in notes)}</div>' if notes else ""

    return (f'<details class="spectra" open><summary>스펙트럼 비교 (원본 → 산출물 vs 레퍼런스)</summary>'
            f'{svg}{note}<div class="spectbls">{curve_tbl}{diff_tbl}</div>{figs_html}</details>')


CSS = """
.spectra { margin-top:8px; border-top:1px dashed #e5e7eb; padding-top:6px; }
.spectra > summary { font-weight:600; color:#1f2a37; }
.specsvg { width:100%; height:auto; background:#f9fafb; border:1px solid #e5e7eb; border-radius:4px; margin:4px 0; }
.specsvg .ptitle { font:600 9px system-ui, sans-serif; fill:#4b5563; }
.specsvg .ax { font:8px ui-monospace, monospace; fill:#9ca3af; }
.specsvg .none { font:11px system-ui, sans-serif; fill:#c0c4cc; }
.specsvg .mark { stroke:#f59e0b; stroke-width:1; stroke-dasharray:2 3; opacity:.75; }
.spectbls { display:flex; gap:10px; flex-wrap:wrap; }
.spectbl { border-collapse:collapse; font-size:11px; }
.spectbl th, .spectbl td { border:1px solid #e5e7eb; padding:1px 6px; text-align:right; }
.spectbl th { background:#f3f4f6; font-weight:600; text-align:center; }
.spectbl td:first-child, .spectbl td.fn { text-align:left; }
.spectbl td.fn { font-family:ui-monospace, monospace; color:#4b5563; }
.sw { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
.specnote { font-size:11px; color:#6b7280; margin:2px 0 4px; }
.specnote .warn, .spectbl .warn { color:#b45309; }
.specnote.warn { color:#b45309; background:#fffbeb; border:1px solid #fcd34d; border-radius:4px; padding:3px 6px; }
.specfigs { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
.specfig { max-height:170px; border:1px solid #e5e7eb; border-radius:4px; background:#fff; }
"""
