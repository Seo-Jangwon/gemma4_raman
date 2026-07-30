# -*- coding: utf-8 -*-
"""채점 콘솔용 '스테이지 이동/측정 지도' + '이미지 갤러리' 생성기.

[왜 필요한가]
전처리 문항은 스펙트럼 그림(spectra_panel)으로 채점되지만, 하드웨어 문항(1~3 카테고리,
그리드 스캔·다점 측정·현미경 기반 위치 지정)은 "어디를 어떤 순서로 찍었는가"가 채점의
전부다. 그런데 그건 툴 트레이스 텍스트를 눈으로 따라가며 좌표를 머릿속에 그려야만
확인이 됐다 — 3x3 격자가 정말 등간격이었는지, 순서가 래스터였는지, 요청한 중심에
맞았는지는 XY 평면에 찍어 보면 1초에 판정된다.

그래서 이 모듈은 실행 레코드 하나에서
  · 스테이지가 간 곳 / 스펙트럼을 찍은 곳 / 그리드 스캔 점 / 저장한 점을 모두 뽑아
    XY 평면 인라인 SVG 지도로 그리고(측정점에 순번 표시),
  · 격자성 진단(행·열 개수, 간격, 등간격 여부, 중심)을 표로 내고,
  · 에이전트가 만든 이미지(현미경 스냅샷, 그리드 프리뷰, 측정 PNG, 분석 그림)를
    한 줄 갤러리로 붙인다.

[좌표를 어디서 얻는가]
툴마다 좌표가 실려 오는 자리가 다르다. acquire_spectrum 은 결과에 x/y 필드가 없고
저장 파일명(`170901_769_x37.876_y25.248.csv`)에만 좌표가 박혀 있어서 파일명에서 파싱한다.
run_grid_scan 은 result.points 에 전체 점 목록을 준다. preview_grid_scan 은 점 목록 없이
center/rows/cols/spacing 만 주므로 격자를 직접 재구성한다.

[이미지 경로]
기록된 URL 은 `/api/results/<날짜>/<파일>` 이다. 그런데 이 벤치 도중 결과 저장구조를
`<날짜>/<세션>/` 로 바꿨고 기존 파일을 세션 폴더로 옮겼기 때문에, 예전 레코드의 URL 은
그대로는 안 맞는다(실측 305개 중 76개). 그래서 '정확 경로 → 실패 시 파일명으로 재탐색'
2단계로 찾는다. 파일명에 타임스탬프+좌표가 들어가 사실상 고유하므로 오인 위험이 없다.
"""
from __future__ import annotations

import html
import io
import os
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_DATA_ROOT = _PROJECT_ROOT / "data"
_RESULTS_ROOT = _DATA_ROOT / "results"

MAX_IMAGES = 12          # 셀 하나에 붙일 이미지 상한(리포트가 수백 MB 되는 걸 막는다)
_XY_RE = re.compile(r"_x(-?\d+(?:\.\d+)?)_y(-?\d+(?:\.\d+)?)")

# 좌표 종류별 색·표시. move 는 '갔다'는 사실만, measure/grid 는 '레이저를 쐈다'는 사실.
_KIND_STYLE = {
    "measure": ("#dc2626", "측정(스펙트럼 취득)"),
    "grid":    ("#2563eb", "그리드 스캔 점"),
    "point":   ("#7c3aed", "저장한 점(save_point_data)"),
    "move":    ("#6b7280", "스테이지 이동"),
    "pixel":   ("#0891b2", "화면 클릭 이동(move_to_pixel)"),
    "preview": ("#f59e0b", "그리드 프리뷰(측정 안 함)"),
    "observe": ("#9ca3af", "위치 조회만"),
}


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


# ── 자산(이미지/CSV) 경로 해석 ────────────────────────────────────────────────

_index_cache: dict[str, Path] | None = None


def _results_index() -> dict[str, Path]:
    """data/results 아래 모든 파일의 '파일명 → 경로' 색인. 이동된 파일 재탐색용."""
    global _index_cache
    if _index_cache is None:
        idx: dict[str, Path] = {}
        if _RESULTS_ROOT.exists():
            for p in _RESULTS_ROOT.rglob("*"):
                if p.is_file():
                    idx.setdefault(p.name, p)      # 먼저 찾은 것을 남긴다
        _index_cache = idx
    return _index_cache


# ── 이미지 임베드 (다른 컴퓨터에서도 보이게) ─────────────────────────────────
#
# 상대링크로 두면 HTML 한 장만 다른 PC 로 옮겼을 때 그림이 전부 깨진다(실측 260장).
# 그래서 기본은 data: URI 임베드 — HTML 파일 하나로 완결된다.
#
# 인코딩 정책: 무손실 WebP 를 먼저 시도하고, 그게 LOSSLESS_LIMIT 을 넘을 때만
# 손실 압축으로 대체한다. 산출물의 대부분은 matplotlib 선화라서 무손실이 원본 PNG 보다
# 작고(13.4MB → 5.2MB) 축 눈금·글자가 한 픽셀도 흐려지지 않는다 — 채점에서 축 숫자를
# 읽어야 하므로 이게 중요하다. 무손실이 커지는 건 현미경 스냅샷·그리드 프리뷰처럼
# 잡음이 많은 사진성 이미지뿐이고(실측 260장 중 5장), 그것만 q88 로 줄인다.
EMBED = True
LOSSLESS_LIMIT = 120_000      # 이 바이트를 넘으면 손실 압축을 시도
LOSSY_MAX_W = 1100            # 손실 전환 시 가로 상한
# base64 는 파일에 '한 번만' 들어가야 한다. 예전에는 <img src> 와 감싸는 <a href> 에
# 같은 blob 을 두 번, 게다가 같은 그림이 여러 셀에 나오면 그만큼 더 복제해서 38MB 중
# 18MB 가 중복이었다. 그래서 URI 는 아래 목록에 한 번만 쌓고, 태그에는 색인만 넣은 뒤
# 브라우저에서 채운다(_HYDRATE_JS).
_embed_cache: dict[str, int] = {}      # 파일경로 → _assets 색인
_assets: list[str] = []                # data: URI 목록 (파일에 딱 한 번 직렬화)
_embed_stat = {"n": 0, "bytes": 0, "lossy": 0, "failed": 0}


def assets() -> list[str]:
    return _assets


def set_embed(flag: bool) -> None:
    global EMBED
    EMBED = bool(flag)


def embed_stats() -> dict:
    return dict(_embed_stat)


def _encode(path: Path) -> str | None:
    """파일 → data: URI. Pillow 가 없거나 인코딩이 실패하면 None(상대링크로 폴백)."""
    import base64
    try:
        from PIL import Image
    except Exception:                                    # noqa: BLE001
        return None
    try:
        im = Image.open(path)
        buf = io.BytesIO()
        im.save(buf, "WEBP", lossless=True, method=4)
        data = buf.getvalue()
        if len(data) > LOSSLESS_LIMIT:
            im2 = im.convert("RGB")
            if im2.width > LOSSY_MAX_W:
                im2 = im2.resize(
                    (LOSSY_MAX_W, round(im2.height * LOSSY_MAX_W / im2.width)),
                    Image.LANCZOS)
            buf2 = io.BytesIO()
            im2.save(buf2, "WEBP", quality=88, method=4)
            if len(buf2.getvalue()) < len(data):
                data = buf2.getvalue()
                _embed_stat["lossy"] += 1
    except Exception:                                    # noqa: BLE001
        _embed_stat["failed"] += 1
        return None
    _embed_stat["n"] += 1
    _embed_stat["bytes"] += len(data)
    return "data:image/webp;base64," + base64.b64encode(data).decode("ascii")


def _rel(path: Path, out_dir: Path) -> str:
    try:
        return os.path.relpath(path, out_dir).replace("\\", "/")
    except ValueError:                                   # 다른 드라이브
        return path.as_uri()


def _asset_index(path: Path) -> int | None:
    """이 파일의 data: URI 색인. 임베드가 꺼져 있거나 실패하면 None."""
    if not EMBED:
        return None
    key = str(path)
    if key not in _embed_cache:
        uri = _encode(path)
        if uri is None:
            return None
        _assets.append(uri)
        _embed_cache[key] = len(_assets) - 1
    return _embed_cache[key]


def img_src(path: Path, out_dir: Path) -> str:
    """`<img ...>` 안에 넣을 속성. 임베드면 data-img 색인, 아니면 src 상대경로."""
    i = _asset_index(path)
    return f'data-img="{i}"' if i is not None else f'src="{esc(_rel(path, out_dir))}"'


def img_href(path: Path, out_dir: Path) -> str:
    """`<a ...>` 안에 넣을 속성. 임베드면 같은 색인을 재사용한다(복제 없음)."""
    i = _asset_index(path)
    return f'data-img="{i}"' if i is not None else f'href="{esc(_rel(path, out_dir))}" target="_blank"'


# 색인 → 실제 URI 를 브라우저에서 채운다. <img> 는 화면에 들어올 때만 채워서
# 수백 장을 한꺼번에 디코딩하지 않게 한다(IntersectionObserver).
#
# data-img 색인은 채운 뒤에도 '지우지 않는다'. 리포트를 통째로 다시 저장하는
# 스냅샷 기능(review.exportSnapshot)이 직렬화 직전에 src/href 를 걷어내고 색인만
# 남겨야 하기 때문이다 — 안 그러면 같은 base64 가 __IMG__ 와 태그에 두 번 들어가
# 파일이 배로 커진다. 대신 data-hyd 로 이미 채운 걸 표시한다.
HYDRATE_JS = """
(function(){
  const M = window.__IMG__ || [];
  const fill = el => {
    const u = M[+el.dataset.img];
    if (u === undefined || el.dataset.hyd) return;
    if (el.tagName === 'IMG') el.src = u; else { el.href = u; el.target = '_blank'; }
    el.dataset.hyd = '1';
  };
  const anchors = [], imgs = [];
  document.querySelectorAll('[data-img]').forEach(el =>
    (el.tagName === 'IMG' ? imgs : anchors).push(el));
  anchors.forEach(fill);
  if (!('IntersectionObserver' in window)) { imgs.forEach(fill); return; }
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (e.isIntersecting) { fill(e.target); io.unobserve(e.target); }
  }), { rootMargin: '600px' });
  imgs.forEach(el => io.observe(el));
})();
"""


def resolve_asset(url_or_path) -> Path | None:
    """`/api/results/...` · 상대경로 · 절대경로 → 실제 파일. 없으면 파일명으로 재탐색."""
    s = str(url_or_path or "").strip()
    if not s:
        return None
    raw = s
    if s.startswith("/api/results/"):
        s = "results/" + s[len("/api/results/"):]
    p = Path(s)
    if not p.is_absolute():
        p = _DATA_ROOT / s
    if p.exists():
        return p
    return _results_index().get(Path(raw).name)


# ── 좌표 추출 ────────────────────────────────────────────────────────────────

def _xy_from_name(*vals) -> tuple[float, float] | None:
    """`..._x37.876_y25.248.png` 같은 저장 파일명에서 좌표를 뽑는다."""
    for v in vals:
        m = _XY_RE.search(str(v or ""))
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def extract_positions(rec: dict) -> list[dict]:
    """실행 레코드에서 좌표가 있는 동작을 시간순으로 뽑는다.

    반환 항목: {step, tool, kind, x, y, z, label, extra}
    kind 는 _KIND_STYLE 의 키. measure/grid 는 '실제로 레이저를 쏜 점'이다.
    """
    out: list[dict] = []

    def add(step, tool, kind, x, y, z=None, label="", **extra):
        if x is None or y is None:
            return
        out.append({"step": step, "tool": tool, "kind": kind, "x": float(x),
                    "y": float(y), "z": z, "label": label, **extra})

    for c in rec.get("tool_calls") or []:
        name = c.get("name")
        step = c.get("step")
        args = c.get("args") or {}
        res = c.get("result") if isinstance(c.get("result"), dict) else {}
        pos = res.get("position") if isinstance(res.get("position"), dict) else {}

        if name in ("move_stage", "move_stage_relative"):
            add(step, name, "move", _num(pos.get("x")) if pos else _num(args.get("x")),
                _num(pos.get("y")) if pos else _num(args.get("y")), _num(pos.get("z")))
        elif name == "move_to_pixel":
            add(step, name, "pixel", _num(pos.get("x")), _num(pos.get("y")), _num(pos.get("z")),
                label=f"px({args.get('pixel_x')},{args.get('pixel_y')})")
        elif name == "get_stage_position":
            add(step, name, "observe", _num(pos.get("x")), _num(pos.get("y")), _num(pos.get("z")))
        elif name == "acquire_spectrum":
            saved = res.get("saved") if isinstance(res.get("saved"), dict) else {}
            files = saved.get("files") if isinstance(saved.get("files"), dict) else {}
            xy = _xy_from_name(saved.get("image_url"), files.get("csv"), files.get("png"),
                               saved.get("csv_url"))
            if xy is None:
                # 저장이 실패한 측정 — 좌표를 알 수 없으니 지도에는 못 찍는다.
                continue
            add(step, name, "measure", xy[0], xy[1],
                label=f"{res.get('exposure_time','?')}s / {res.get('laser_power_pct','?')}%",
                imax=res.get("max_intensity"))
        elif name == "run_grid_scan":
            for pt in res.get("points") or []:
                add(step, name, "grid", _num(pt.get("x")), _num(pt.get("y")),
                    label=f"#{pt.get('i')}", imax=pt.get("max_intensity"))
        elif name == "preview_grid_scan":
            ctr = res.get("center") if isinstance(res.get("center"), dict) else {}
            cx, cy = _num(ctr.get("x")), _num(ctr.get("y"))
            rows, cols = _num(res.get("rows")), _num(res.get("cols"))
            sp = _num(res.get("spacing_mm"))
            if None not in (cx, cy, rows, cols, sp):
                for j in range(int(rows)):
                    for i in range(int(cols)):
                        add(step, name, "preview",
                            cx + (i - (cols - 1) / 2) * sp,
                            cy + (j - (rows - 1) / 2) * sp,
                            label=f"prev {j},{i}")
        elif name == "save_point_data":
            p = args.get("position") if isinstance(args.get("position"), dict) else {}
            add(step, name, "point", _num(p.get("x")), _num(p.get("y")), _num(p.get("z")),
                label=str(args.get("point_id") or ""))
    return out


# ── 격자성 진단 ──────────────────────────────────────────────────────────────

def _uniq(vals: list[float], tol: float = 1e-3) -> list[float]:
    """부동소수 오차를 흡수해 서로 다른 좌표값만 남긴다."""
    out: list[float] = []
    for v in sorted(vals):
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def grid_stats(pts: list[dict]) -> dict:
    """'레이저를 쏜 점'의 격자성. 그리드 문항의 채점 근거가 된다.

    등간격 판정은 인접 좌표 간격의 (최대-최소)/평균 으로 본다 — 절대값 기준을 쓰면
    0.05mm 격자와 0.5mm 격자에 같은 잣대를 대게 된다.
    """
    shot = [p for p in pts if p["kind"] in ("measure", "grid")]
    if not shot:
        return {}
    xs, ys = [p["x"] for p in shot], [p["y"] for p in shot]
    ux, uy = _uniq(xs), _uniq(ys)

    def spacing(u: list[float]) -> tuple[float | None, float | None]:
        if len(u) < 2:
            return None, None
        d = [u[i + 1] - u[i] for i in range(len(u) - 1)]
        mean = sum(d) / len(d)
        spread = (max(d) - min(d)) / mean if mean else None
        return mean, spread

    dx, sx = spacing(ux)
    dy, sy = spacing(uy)
    return {
        "n": len(shot), "n_cols": len(ux), "n_rows": len(uy),
        "x_range": (min(xs), max(xs)), "y_range": (min(ys), max(ys)),
        "center": ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2),
        "dx": dx, "dy": dy, "dx_spread": sx, "dy_spread": sy,
        "is_grid": len(ux) > 1 and len(uy) > 1 and len(shot) == len(ux) * len(uy),
        "uniform": (sx is None or sx < 0.02) and (sy is None or sy < 0.02),
        "cols": ux, "rows": uy,
    }


# ── SVG 지도 ─────────────────────────────────────────────────────────────────

_W, _H = 430, 300
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 48, 12, 14, 30


def _svg_map(pts: list[dict], gs: dict) -> str:
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    # 점이 한 곳에 모여 있으면(단일 측정) 범위가 0 이 되어 나누기가 깨진다.
    px = (x1 - x0) * 0.12 or 0.05
    py = (y1 - y0) * 0.12 or 0.05
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py

    def sx(v):
        return _PAD_L + (v - x0) / (x1 - x0) * (_W - _PAD_L - _PAD_R)

    def sy(v):   # y 는 위로 증가 = 화면 좌표 반전
        return _PAD_T + (1 - (v - y0) / (y1 - y0)) * (_H - _PAD_T - _PAD_B)

    body = [f'<rect x="{_PAD_L}" y="{_PAD_T}" width="{_W-_PAD_L-_PAD_R}" '
            f'height="{_H-_PAD_T-_PAD_B}" fill="#fff" stroke="#e5e7eb"/>']

    # 격자 보조선 — 등간격인지 눈으로 보이게 실제 좌표값에 선을 긋는다
    for v in (gs.get("cols") or []):
        body.append(f'<line x1="{sx(v):.1f}" y1="{_PAD_T}" x2="{sx(v):.1f}" y2="{_H-_PAD_B}" class="gl"/>')
    for v in (gs.get("rows") or []):
        body.append(f'<line x1="{_PAD_L}" y1="{sy(v):.1f}" x2="{_W-_PAD_R}" y2="{sy(v):.1f}" class="gl"/>')

    # 이동 경로 — 순서를 보여준다(래스터인지 지그재그인지)
    path = [p for p in pts if p["kind"] != "observe"]
    if len(path) > 1:
        body.append('<polyline points="' +
                    " ".join(f"{sx(p['x']):.1f},{sy(p['y']):.1f}" for p in path) +
                    '" fill="none" class="trail"/>')

    order = 0
    for p in pts:
        color = _KIND_STYLE.get(p["kind"], ("#6b7280", ""))[0]
        cx, cy = sx(p["x"]), sy(p["y"])
        if p["kind"] in ("measure", "grid"):
            order += 1
            body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{color}" '
                        f'fill-opacity="0.9"/><text x="{cx:.1f}" y="{cy+2.6:.1f}" '
                        f'class="onum" text-anchor="middle">{order}</text>')
        elif p["kind"] == "preview":
            body.append(f'<rect x="{cx-3:.1f}" y="{cy-3:.1f}" width="6" height="6" '
                        f'fill="none" stroke="{color}" stroke-width="1.2"/>')
        else:
            body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.6" fill="{color}" '
                        f'fill-opacity="0.8"/>')
        body.append(f'<title>step {p["step"]} · {p["tool"]} · '
                    f'({p["x"]:.4f}, {p["y"]:.4f}) {p["label"]}</title>')

    body.append(
        f'<text x="{_PAD_L}" y="{_H-10}" class="ax">{x0:.3f}</text>'
        f'<text x="{_W-_PAD_R}" y="{_H-10}" class="ax" text-anchor="end">{x1:.3f}</text>'
        f'<text x="{(_W)/2:.0f}" y="{_H-10}" class="ax" text-anchor="middle">X (mm)</text>'
        f'<text x="{_PAD_L-4}" y="{_PAD_T+9}" class="ax" text-anchor="end">{y1:.3f}</text>'
        f'<text x="{_PAD_L-4}" y="{_H-_PAD_B}" class="ax" text-anchor="end">{y0:.3f}</text>'
        f'<text x="10" y="{(_H)/2:.0f}" class="ax" transform="rotate(-90 10 {(_H)/2:.0f})" '
        f'text-anchor="middle">Y (mm)</text>')
    return (f'<svg class="mapsvg" viewBox="0 0 {_W} {_H}" '
            f'preserveAspectRatio="xMidYMid meet">{"".join(body)}</svg>')


def _grid_table(gs: dict) -> str:
    if not gs:
        return ""
    def f(v, nd=4):
        return "—" if v is None else f"{v:.{nd}f}"

    ok = lambda b: "ok" if b else "bad"      # noqa: E731
    rows = [
        f'<tr><td class="l">쏜 점 개수</td><td>{gs["n"]}</td></tr>',
        f'<tr><td class="l">서로 다른 X / Y 좌표</td><td>{gs["n_cols"]} × {gs["n_rows"]}</td></tr>',
        f'<tr><td class="l">완전 격자 여부</td>'
        f'<td class="{ok(gs["is_grid"])}">{"예" if gs["is_grid"] else "아니오(격자 아님/불완전)"}</td></tr>',
        f'<tr><td class="l">X 간격 (편차)</td><td>{f(gs["dx"])} mm '
        f'({"—" if gs["dx_spread"] is None else f"{gs['dx_spread']*100:.2f}%"})</td></tr>',
        f'<tr><td class="l">Y 간격 (편차)</td><td>{f(gs["dy"])} mm '
        f'({"—" if gs["dy_spread"] is None else f"{gs['dy_spread']*100:.2f}%"})</td></tr>',
        f'<tr><td class="l">등간격</td><td class="{ok(gs["uniform"])}">'
        f'{"예 (간격편차 2% 미만)" if gs["uniform"] else "아니오"}</td></tr>',
        f'<tr><td class="l">스캔 중심</td><td>({gs["center"][0]:.4f}, {gs["center"][1]:.4f})</td></tr>',
        f'<tr><td class="l">X / Y 범위</td>'
        f'<td>{gs["x_range"][0]:.4f}~{gs["x_range"][1]:.4f} / '
        f'{gs["y_range"][0]:.4f}~{gs["y_range"][1]:.4f}</td></tr>',
    ]
    return f'<table class="maptbl"><tbody>{"".join(rows)}</tbody></table>'


def _pos_table(pts: list[dict]) -> str:
    rows = ""
    order = 0
    for p in pts:
        shot = p["kind"] in ("measure", "grid")
        if shot:
            order += 1
        color = _KIND_STYLE.get(p["kind"], ("#6b7280", ""))[0]
        rows += (f'<tr><td>{order if shot else ""}</td><td>{p["step"]}</td>'
                 f'<td class="l"><span class="dot" style="background:{color}"></span>'
                 f'{esc(p["tool"])}</td>'
                 f'<td class="mono">{p["x"]:.4f}</td><td class="mono">{p["y"]:.4f}</td>'
                 f'<td class="mono">{"—" if p["z"] is None else f"{p['z']:.4f}"}</td>'
                 f'<td class="l">{esc(p["label"])}'
                 f'{"" if p.get("imax") is None else f" · Imax={esc(p['imax'])}"}</td></tr>')
    return (f'<table class="maptbl pos"><thead><tr><th>순번</th><th>step</th><th class="l">툴</th>'
            f'<th>X</th><th>Y</th><th>Z</th><th class="l">비고</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def build_stage_map(rec: dict) -> str:
    """실행 레코드 하나의 좌표 지도 HTML. 좌표가 없으면 ''."""
    pts = extract_positions(rec)
    if not pts:
        return ""
    gs = grid_stats(pts)
    kinds = {p["kind"] for p in pts}
    legend = "".join(
        f'<span class="lg"><span class="dot" style="background:{_KIND_STYLE[k][0]}"></span>'
        f'{esc(_KIND_STYLE[k][1])}</span>' for k in _KIND_STYLE if k in kinds)
    n_shot = sum(1 for p in pts if p["kind"] in ("measure", "grid"))
    return (f'<details class="stagemap" open><summary>스테이지 지도 — '
            f'좌표 {len(pts)}개 / 레이저 쏜 점 {n_shot}개</summary>'
            f'<div class="maprow">{_svg_map(pts, gs)}'
            f'<div class="mapside"><div class="legend">{legend}</div>{_grid_table(gs)}</div></div>'
            f'<details class="postbl"><summary>좌표 전체 목록 ({len(pts)})</summary>'
            f'{_pos_table(pts)}</details></details>')


# ── 이미지 갤러리 ────────────────────────────────────────────────────────────

_IMG_LABEL = {
    "capture_scene": "현미경 스냅샷",
    "capture_camera_frame": "카메라 프레임",
    "analyze_microscope_image": "현미경 분석 화면",
    "preview_grid_scan": "그리드 프리뷰",
    "run_grid_scan": "그리드 스캔",
    "acquire_spectrum": "측정 스펙트럼 PNG",
    "run_analysis": "분석 그림",
    "save_spectrum": "저장 스펙트럼",
}
_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _session_dirs(rec: dict) -> list[Path]:
    """이 실행의 결과 폴더 `data/results/<날짜>/<세션>/`.
    세션명 정규화 규칙은 run_store._sanitize / detail_log._sanitize 와 같아야 한다."""
    label = re.sub(r"[^0-9A-Za-z_-]", "-", str(rec.get("session_id") or ""))[:64]
    if not label:
        return []
    return [p for p in _RESULTS_ROOT.glob(f"*/{label}") if p.is_dir()]


def collect_images(rec: dict) -> list[dict]:
    """에이전트가 만든 이미지 전부. [{path, tool, step, label}] (중복 제거, 시간순).

    트레이스에 URL 이 실려 오는 것만 모으면 빠지는 게 있다 — run_grid_scan 은 점마다
    스펙트럼 PNG 를 자동저장하지만 결과에는 CSV 파일명만 담는다(그리드 문항은 정작
    점별 스펙트럼을 봐야 채점된다). 그래서 세션 결과 폴더를 훑어 보완한다.
    """
    out: list[dict] = []
    seen: set[str] = set()

    def add(url, tool, step):
        p = resolve_asset(url)
        if p is None or p.suffix.lower() not in _IMG_EXT or str(p) in seen:
            return
        seen.add(str(p))
        out.append({"path": p, "tool": tool, "step": step,
                    "label": _IMG_LABEL.get(tool, tool or "이미지")})

    for c in rec.get("tool_calls") or []:
        tool, step = c.get("name"), c.get("step")
        r = c.get("result")
        if not isinstance(r, dict):
            continue
        add(r.get("image_url"), tool, step)
        saved = r.get("saved") if isinstance(r.get("saved"), dict) else {}
        add(saved.get("image_url"), tool, step)
        files = saved.get("files") if isinstance(saved.get("files"), dict) else {}
        add(files.get("png"), tool, step)
        for v in (r.get("images") or []):
            add(v, tool, step)
        for v in (r.get("files") or []):
            if isinstance(v, str):
                add(v, tool, step)
        for v in (r.get("saved_files") or []):
            add(v.get("path") if isinstance(v, dict) else v, tool, step)
        # run_grid_scan: 점별 파일명(csv)만 오므로 같은 이름의 png 로 바꿔 찾는다
        for pt in (r.get("points") or []):
            f = pt.get("file") if isinstance(pt, dict) else None
            if f:
                add(str(Path(str(f)).with_suffix(".png")), tool, step)

    for d in _session_dirs(rec):
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMG_EXT and str(p) not in seen:
                seen.add(str(p))
                out.append({"path": p, "tool": "session", "step": "—",
                            "label": "세션 폴더 자동저장"})
    return out


def build_images(rec: dict, out_dir: Path) -> str:
    """이미지 갤러리 HTML. 경로는 img_src/img_href 가 처리한다 — 기본은 data: URI
    임베드라 HTML 한 장만 옮겨도 그림이 살아 있다(--no-embed 일 때만 상대링크)."""
    imgs = collect_images(rec)
    if not imgs:
        return ""
    shown, extra = imgs[:MAX_IMAGES], max(0, len(imgs) - MAX_IMAGES)
    cells = []
    for im in shown:
        cells.append(
            f'<figure class="shot"><a {img_href(im["path"], out_dir)}>'
            f'<img {img_src(im["path"], out_dir)} loading="lazy" '
            f'alt="{esc(im["path"].name)}"></a>'
            f'<figcaption>{esc(im["label"])} <span class="sstep">step {im["step"]}</span><br>'
            f'<span class="sname">{esc(im["path"].name)}</span></figcaption></figure>')
    more = f'<div class="specnote">그림 {extra}개 더 있음(생략).</div>' if extra else ""
    return (f'<details class="shots" open><summary>화면·이미지 ({len(imgs)})</summary>'
            f'<div class="shotrow">{"".join(cells)}</div>{more}</details>')


CSS = """
.stagemap, .shots { margin-top:8px; border-top:1px dashed #e5e7eb; padding-top:6px; }
.stagemap > summary, .shots > summary { font-weight:600; color:#1f2a37; }
.maprow { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-start; margin-top:5px; }
.mapsvg { width:430px; max-width:100%; height:auto; background:#f9fafb; border:1px solid #e5e7eb; border-radius:4px; }
.mapsvg .gl { stroke:#e5e7eb; stroke-width:1; stroke-dasharray:3 3; }
.mapsvg .trail { stroke:#cbd5e1; stroke-width:1.2; stroke-dasharray:4 3; }
.mapsvg .onum { font:600 8px system-ui, sans-serif; fill:#fff; }
.mapsvg .ax { font:8px ui-monospace, monospace; fill:#9ca3af; }
.mapside { flex:1; min-width:250px; }
.legend { display:flex; gap:9px; flex-wrap:wrap; font-size:11px; color:#4b5563; margin-bottom:4px; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:3px; vertical-align:-1px; }
.maptbl { border-collapse:collapse; font-size:11.5px; width:100%; }
.maptbl td, .maptbl th { border:1px solid #e5e7eb; padding:2px 6px; text-align:right; }
.maptbl th { background:#f3f4f6; text-align:center; }
.maptbl td.l, .maptbl th.l { text-align:left; }
.maptbl .ok { color:#059669; font-weight:600; }
.maptbl .bad { color:#dc2626; font-weight:600; }
.maptbl .mono, .maptbl td.mono { font-family:ui-monospace, monospace; }
.postbl { margin-top:5px; }
.postbl .pos { max-height:none; }
.shotrow { display:flex; gap:8px; flex-wrap:wrap; margin-top:5px; }
.shot { margin:0; width:190px; }
.shot img { width:100%; height:130px; object-fit:contain; background:#fff; border:1px solid #e5e7eb; border-radius:4px; }
.shot figcaption { font-size:10px; color:#6b7280; line-height:1.35; margin-top:2px; }
.shot .sstep { color:#9ca3af; }
.shot .sname { font-family:ui-monospace, monospace; font-size:9px; color:#9ca3af; word-break:break-all; }
"""
