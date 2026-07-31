"""
[역할]
  라만 측정 결과(acquire_spectrum 반환 dict)를 로컬에 '날짜/시간별'로 저장하고
  채팅 인라인 표시용 PNG 를 렌더한다. CoALA·AILA 두 에이전트가 공유하는 순수
  유틸(하드웨어 import 없음) — acquire_spectrum 이 측정 직후 이걸 호출한다.

  저장 레이아웃 — 개별 측정은 '날짜/세션' 아래로 모은다:
    data/results/<YYYY-MM-DD>/<세션>/<HHMMSS_mmm>_<tag>.png   ← 플롯(채팅 표시용)
    data/results/<YYYY-MM-DD>/<세션>/<HHMMSS_mmm>_<tag>.csv   ← 데이터(엑셀 호환)
    data/results/<YYYY-MM-DD>/<세션>/<HHMMSS_mmm>_<tag>.json  ← 원본 결과+메타
    data/results/<YYYY-MM-DD>/<세션>/fig<HHMMSS_mmm>_<i>.png  ← run_analysis 그림
  세션 경계를 넘는 것들만 날짜 폴더 직속에 남는다('_' 접두라 측정 목록에 안 잡힌다):
    _scene_*.npz/.png     현미경 장면(스테이지 좌표계 기준이라 세션 무관)
    _combined_*.png / _summary_*.csv / _bundle_*.zip / _<tag>_*.png  집계·미리보기

  <세션> 은 run_store 의 세션 라벨(= 벤치의 'bench_<문항>_<에이전트>_<시각>')이다.
  세션이 없으면 '_unassigned'. 예전에는 개별 측정이 날짜 폴더에 그대로 쏟아져서
  (하루 85개 파일) 어느 문항·어느 에이전트의 측정인지 파일만 보고는 알 수 없었다.

  하위호환: list_results 는 세션 폴더와 '날짜 폴더 직속(구버전)' 을 모두 읽는다 —
  이미 쌓인 결과가 갑자기 안 보이면 안 되므로. 구버전 파일을 세션 폴더로 정리하려면
  backend/tools/migrate_results.py 를 쓴다.

  PNG/JSON URL 은 '/api/results/...' 로 서빙된다(vite proxy 가 /api 만 통과시키므로
  결과 파일도 /api 아래에 둔다). server.py 가 이 경로를 StaticFiles 로 정적 서빙하고,
  StaticFiles 는 하위 폴더를 그대로 서빙하므로 URL 에 세션이 한 단계 끼어도 문제없다.
"""
from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # GUI 없는 서버 스레드에서 렌더 (반드시 pyplot import 전)
import matplotlib.pyplot as plt  # noqa: E402

# 제목·라벨에 한글이 들어가므로 한글 글리프 있는 폰트로(없으면 네모 □ 로 깨짐).
# Windows 기본 'Malgun Gothic', 없으면 DejaVu Sans 폴백. 마이너스 기호 깨짐 방지.
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# data/results (세션 저장소 data/sessions 와 같은 부모) — server.py 가 /api/results 로 서빙
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "data" / "results"
URL_PREFIX = "/api/results"      # 프론트에서 접근하는 공개 경로
UNASSIGNED = "_unassigned"       # 세션 없이 나온 산출물이 떨어지는 폴더


def session_folder() -> str:
    """개별 측정을 담을 하위 폴더명 = 현재 run_store 세션 라벨.

    import 를 함수 안에서 하는 이유: spectrum_store 는 '하드웨어/에이전트 import 없는
    순수 유틸' 이라는 계약이 있어 모듈 수준 의존을 만들지 않는다. 세션 조회가 실패해도
    측정 저장은 반드시 성공해야 하므로 예외는 삼키고 _unassigned 로 떨어뜨린다.
    """
    try:
        from backend.agents import run_store
        return run_store.current().get("label") or UNASSIGNED
    except Exception:
        return UNASSIGNED


# ──────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼 — 결과 dict 의 두 저장 형태(single/accumulate vs kinetic)를 흡수
# ──────────────────────────────────────────────────────────────────────────────
def _safe_tag(meta: dict | None) -> str:
    """파일명 tag 를 메타에서 뽑아 파일시스템 안전 문자열로 정리한다."""
    raw = ""
    if meta:
        # 우선순위: 명시 tag > 좌표(스캔) > 시료명
        if meta.get("tag"):
            raw = str(meta["tag"])
        elif meta.get("x") is not None and meta.get("y") is not None:
            raw = f"x{meta['x']}_y{meta['y']}"
        elif meta.get("sample"):
            raw = str(meta["sample"])
    if not raw:
        raw = "spectrum"
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in raw)
    return safe[:60].strip("_") or "spectrum"


def _x_axis(result: dict) -> tuple[list, str]:
    """보정 여부에 따라 x축(라만 shift 또는 pixel)과 라벨을 고른다."""
    if result.get("calibrated") and result.get("raman_shift_cm-1"):
        return list(result["raman_shift_cm-1"]), "Raman shift (cm$^{-1}$)"
    intensity = _intensity_of(result)
    n = result.get("length") or len(intensity)
    return list(range(n)), "Pixel"


def _intensity_of(result: dict) -> list:
    """single/accumulate 결과에서 세기 배열을 꺼낸다('data' 우선, 하위호환 'intensity')."""
    return list(result.get("data") or result.get("intensity") or [])


def _series(result: dict) -> list[tuple[str, list, list, str]]:
    """(라벨, x, y, xlabel) 시리즈 목록. kinetic 은 프레임별 여러 시리즈, 그 외 단일."""
    if result.get("mode") == "kinetic" and result.get("frames"):
        out = []
        for fr in result["frames"]:
            if fr.get("calibrated") and fr.get("raman_shift_cm-1"):
                x, xlabel = list(fr["raman_shift_cm-1"]), "Raman shift (cm$^{-1}$)"
            else:
                y0 = list(fr.get("intensity") or [])
                x, xlabel = list(range(len(y0))), "Pixel"
            out.append((f"frame {fr.get('frame_index', '?')}",
                        x, list(fr.get("intensity") or []), xlabel))
        return out
    x, xlabel = _x_axis(result)
    return [("", x, _intensity_of(result), xlabel)]


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────────────────────────────────────
def make_title(result: dict, meta: dict | None = None) -> str:
    """실험 조건으로 사람이 읽을 제목을 생성한다(합본/파일 공용).

    예) 'x3_y5 · 20% · 0.2s'  (스캔 좌표 있을 때)
        '라만 · 40% · 0.5s · accumulate×4'
    """
    parts = []
    if meta and meta.get("x") is not None and meta.get("y") is not None:
        parts.append(f"({meta['x']}, {meta['y']})")
    elif meta and meta.get("sample"):
        parts.append(str(meta["sample"]))
    pwr = result.get("laser_power_pct")
    if pwr is not None:
        parts.append(f"{pwr}%")
    exp = result.get("exposure_time")
    if exp is not None:
        parts.append(f"{exp}s")
    mode = result.get("mode")
    if mode == "accumulate" and result.get("num_accumulations"):
        parts.append(f"accum×{result['num_accumulations']}")
    elif mode == "kinetic" and result.get("kinetic_count"):
        parts.append(f"kinetic×{result['kinetic_count']}")
    return " · ".join(parts) if parts else "라만 스펙트럼"


def render_png(result: dict, path: Path, title: str) -> None:
    """스펙트럼을 PNG 로 렌더한다(채팅 인라인 표시용)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=110)
    series = _series(result)
    xlabel = series[0][3] if series else "Pixel"
    multi = len(series) > 1
    for label, x, y, xl in series:
        if not x or not y:
            continue
        ax.plot(x, y, linewidth=0.9, label=label if multi else None)
        xlabel = xl
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Intensity (ADU)", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.margins(x=0.01)
    if multi:
        ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── 스펙트럼 CSV 저장 (프로젝트 전체의 단일 포맷) ─────────────────────────────
#
# [왜 한 곳으로 모았는가 — 2026-07-30]
# 스펙트럼 CSV 를 쓰는 코드가 네 벌 있었고, 헤더가 서로 달랐다:
#   spectrum_store._write_csv                pixel, raman_shift_cm-1, wavelength_nm, intensity
#   analysis_sandbox.save_result             pixel_index, raman_shift_cm-1, wavelength_nm, intensity
#   raman_tools.apply_background_subtraction pixel_index, raman_shift_cm-1,
#                                            corrected_intensity, background_intensity
#   save_csv.save_spectrum_csv               pixel, ...  (테스트 스크립트 전용)
# 그 결과 load_spectrum 이 세기 열 이름을 추측해야 했고('intensity' 인지
# 'corrected_intensity' 인지), 벤치마크 채점기도 x축 후보 이름을 나열해 두어야 했다.
# 이제 아래 한 함수만이 스펙트럼 CSV 를 쓴다.
#
# 포맷: pixel_index, [raman_shift_cm-1,] [wavelength_nm,] intensity, [background_intensity]
#   · 첫 열 이름은 'pixel_index' 로 통일한다(분석 산출물·채점기가 이미 쓰던 이름).
#   · 배경 제거 결과의 세기 열도 'intensity' 다 — '보정된 세기'라는 사실은 파일명과
#     background_intensity 열의 존재로 드러난다. 열 이름을 바꾸면 같은 데이터를 읽는
#     코드가 두 갈래로 갈린다(그래서 갈렸었다).
#   · 메타는 '# key,value' 주석행. load_spectrum 이 이 행들을 건너뛰고 metadata 로 준다.

_CSV_META_KEYS = ("laser_power_pct", "exposure_time", "laser_nm", "mode")


def write_spectrum_csv(path, intensity, raman_shift=None, wavelength_nm=None,
                       background=None, meta: dict | None = None,
                       encoding: str = "utf-8-sig") -> None:
    """스펙트럼 1개를 표준 포맷 CSV 로 쓴다. 프로젝트의 유일한 스펙트럼 CSV writer.

    Parameters
    ----------
    path        : 저장 경로
    intensity   : 세기 배열(필수)
    raman_shift : 라만 shift 배열(cm-1). None 이면 열 자체를 만들지 않는다.
    wavelength_nm : 파장 배열(nm). 없으면 열 생략.
    background  : 배경 배열. 배경 제거 결과일 때만 준다(열이 하나 늘어난다).
    meta        : 헤더 앞에 '# key,value' 주석행으로 남길 측정 조건.
    encoding    : 기본 utf-8-sig(엑셀 호환). load_spectrum 이 utf-8-sig 로 읽는다.
    """
    ints = [float(v) for v in intensity]
    cols = ["pixel_index"]
    extra = []
    if raman_shift is not None:
        cols.append("raman_shift_cm-1")
        extra.append(("raman_shift_cm-1", [float(v) for v in raman_shift], "{:.3f}"))
    if wavelength_nm is not None:
        cols.append("wavelength_nm")
        extra.append(("wavelength_nm", [float(v) for v in wavelength_nm], "{:.4f}"))
    cols.append("intensity")
    bg = [float(v) for v in background] if background is not None else None
    if bg is not None:
        cols.append("background_intensity")

    for name, seq, _fmt in extra:
        if len(seq) != len(ints):
            raise ValueError(f"{name} has {len(seq)} points but intensity has {len(ints)}.")
    if bg is not None and len(bg) != len(ints):
        raise ValueError(f"background has {len(bg)} points but intensity has {len(ints)}.")

    with open(path, "w", newline="", encoding=encoding) as f:
        for k, v in (meta or {}).items():
            if v is not None:
                f.write(f"# {k},{v}\n")
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(ints)):
            row = [i]
            for _name, seq, fmt in extra:
                row.append(fmt.format(seq[i]))
            row.append(ints[i])
            if bg is not None:
                row.append(bg[i])
            w.writerow(row)


def _write_csv(result: dict, path: Path) -> None:
    """측정 결과(acquire_spectrum 반환)를 CSV 로 저장. kinetic 만 프레임 나열 포맷."""
    meta = {k: result.get(k) for k in _CSV_META_KEYS}
    if result.get("mode") == "kinetic" and result.get("frames"):
        # 프레임이 여러 벌이라 위의 단일 스펙트럼 포맷에 담기지 않는다.
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            for k, v in meta.items():
                if v is not None:
                    f.write(f"# {k},{v}\n")
            w = csv.writer(f)
            w.writerow(["frame_index", "pixel_index", "intensity"])
            for fr in result["frames"]:
                for px, val in enumerate(fr.get("intensity") or []):
                    w.writerow([fr.get("frame_index", ""), px, val])
        return
    calibrated = bool(result.get("calibrated") and result.get("raman_shift_cm-1"))
    write_spectrum_csv(
        path,
        intensity=_intensity_of(result),
        raman_shift=result["raman_shift_cm-1"] if calibrated else None,
        wavelength_nm=result.get("wavelength_nm") if calibrated else None,
        meta=meta,
    )


def spectrum_event(tool_result: dict | None) -> dict | None:
    """acquire_spectrum 결과에서 프론트 인라인 표시용 'spectrum' 이벤트를 만든다.

    저장(save_spectrum)이 성공해 result['saved'] 가 있을 때만 이벤트를 돌려주고,
    아니면 None. 두 에이전트(CoALA·AILA)의 stream_experiment 가 공통으로 호출한다.
    """
    saved = (tool_result or {}).get("saved")
    if not saved:
        return None
    # 이미지(개별/합본) 또는 다운로드 산출물(요약 CSV·zip) 중 하나라도 있으면 표시.
    if not any(saved.get(k) for k in ("image_url", "csv_url", "zip_url")):
        return None
    return {
        "type": "spectrum",
        "image_url": saved.get("image_url"),
        "title": saved.get("title", ""),
        "csv_url": saved.get("csv_url"),
        "json_url": saved.get("json_url"),
        "zip_url": saved.get("zip_url"),
    }


def save_spectrum(result: dict, meta: dict | None = None) -> dict:
    """측정 결과를 날짜/시간 폴더에 png+csv+json 으로 저장한다.

    Returns
    -------
    dict — {ok, title, timestamp, session, dir, files:{png,csv,json},
            image_url, csv_url, json_url}
           실패해도 측정 자체를 막지 않도록 예외를 삼키고 {ok: False, error} 를 돌려준다.
    """
    if not result or not result.get("ok"):
        return {"ok": False, "error": "No valid measurement result to save."}
    try:
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
        tag = _safe_tag(meta)
        base = f"{stamp}_{tag}"
        sess = session_folder()

        out_dir = RESULTS_ROOT / date_dir / sess
        out_dir.mkdir(parents=True, exist_ok=True)

        png_path = out_dir / f"{base}.png"
        csv_path = out_dir / f"{base}.csv"
        json_path = out_dir / f"{base}.json"

        title = make_title(result, meta)
        render_png(result, png_path, title)
        _write_csv(result, csv_path)
        # json: 원본 결과 + 메타 + 제목 (합본 재렌더·검색용)
        # session 을 안에도 적어 둔다 — 파일을 옮겨도 귀속이 남는다.
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"title": title, "meta": meta or {}, "session": sess,
                       "timestamp": now.isoformat(), "result": result},
                      f, ensure_ascii=False)

        def url(p: Path) -> str:
            return f"{URL_PREFIX}/{date_dir}/{sess}/{p.name}"

        # 세션 manifest 에 측정을 등록한다 — 에이전트가 list_session_artifacts 로
        # '내가 이번 세션에 뭘 측정했는지'를 스스로 찾을 수 있어야 하므로.
        try:
            from backend.agents import run_store
            run_store.record(run_store.KIND_MEASUREMENT, f"results/{date_dir}/{sess}/{base}.csv",
                             title=title, tag=tag, image=f"{base}.png")
        except Exception:
            pass          # 부기 실패가 측정 저장을 깨뜨리면 안 된다

        return {
            "ok": True,
            "title": title,
            "timestamp": now.isoformat(),
            "session": sess,
            "dir": str(out_dir),
            "files": {"png": str(png_path), "csv": str(csv_path), "json": str(json_path)},
            "image_url": url(png_path),
            "csv_url": url(csv_path),
            "json_url": url(json_path),
        }
    except Exception as e:
        return {"ok": False, "error": f"Failed to save results: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# 저장된 여러 측정 정리 — 합본 렌더(#2) / 요약 CSV(#3) / 묶음 다운로드(#4)
#   전부 저장 산출물(data/results/<날짜>/*.json)만 읽어 동작(하드웨어 불필요).
#   생성물(합본/요약/zip)은 '_' 로 시작해 재귀 포함되지 않게 한다.
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_date(date: str | None) -> str:
    return date or datetime.now().strftime("%Y-%m-%d")


def _url(date: str, name: str) -> str:
    return f"{URL_PREFIX}/{date}/{name}"


def list_results(date: str | None = None, scope: str = "session") -> list[dict]:
    """해당 날짜(기본 오늘)에 저장된 개별 측정 목록을 시각 순으로 돌려준다.

    각 항목: {base, session, date, title, timestamp, meta, result, png/csv/json(Path)}.
    합본/요약/zip 등 생성물('_' 접두 파일)은 제외한다.

    scope
    -----
    "session" (기본) 현재 run_store 세션의 측정만. 세션이 없으면 전체와 같다.
    "all"            그 날짜의 모든 세션.

    기본을 session 으로 둔 이유: 이 목록이 run_analysis 샌드박스에 spectra 로 주입되고
    combine/aggregate/bundle 의 대상이 된다. 전체를 주면 벤치마크에서 앞선 문항들의
    측정이 다음 문항 코드에 섞여 들어가(문항 간 오염) 채점이 무의미해지고, 하루가
    쌓일수록 주입량도 같이 커진다. 여러 세션을 일부러 합칠 때만 scope="all" 을 준다.

    구버전(날짜 폴더 직속) 파일도 함께 읽는다 — session 은 빈 문자열이 되고,
    scope="session" 이어도 '귀속 불명' 이므로 배제하지 않는다(안 보이면 유실로 보인다).
    """
    d = _resolve_date(date)
    day_dir = RESULTS_ROOT / d
    if not day_dir.exists():
        return []

    cur = session_folder() if scope == "session" else ""
    items = []
    # 세션 하위 폴더 + 구버전(날짜 폴더 직속). '_' 접두 파일은 생성물이라 제외한다.
    for jp in list(day_dir.glob("*/*.json")) + list(day_dir.glob("*.json")):
        if jp.stem.startswith("_"):
            continue
        sess = jp.parent.name if jp.parent != day_dir else ""
        # 구버전 파일(sess=="")은 귀속 불명이므로 세션 필터에서 걸러내지 않는다.
        if cur and sess and sess != cur:
            continue
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append({
            "base": jp.stem, "session": sess, "date": d,
            "title": payload.get("title", ""),
            "timestamp": payload.get("timestamp", ""),
            "meta": payload.get("meta", {}) or {},
            "result": payload.get("result", {}) or {},
            "png": jp.with_suffix(".png"),
            "csv": jp.with_suffix(".csv"),
            "json": jp,
        })
    # 시각 순. 파일명 정렬로는 세션 폴더가 먼저 묶여 시간 순서가 깨진다.
    items.sort(key=lambda it: (it["timestamp"] or "", it["base"]))
    return items


def _select(date: str | None, names: list[str] | None,
            scope: str = "session") -> tuple[str, list[dict]]:
    d = _resolve_date(date)
    items = list_results(d, scope=scope)
    if names:
        want = set(names)
        # base(=파일 stem)로 고른다 — 세션이 폴더로 갈렸어도 모델이 보는 이름은 그대로다.
        items = [it for it in items if it["base"] in want]
    return d, items


def combine_spectra(date: str | None = None, names: list[str] | None = None,
                    out_name: str | None = None, max_cols: int = 4,
                    scope: str = "session") -> dict:
    """저장된 여러 스펙트럼을 한 장(격자)으로 렌더한다(#2). 제목은 각 측정의
    저장 제목(좌표·조건 자동 생성)을 그대로 쓴다. 반환 saved.image_url 로 채팅 표시."""
    d, items = _select(date, names, scope)
    if not items:
        return {"ok": False, "error": f"No measurement results to combine on {d}."}
    n = len(items)
    cols = max(1, min(max_cols, n))
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5),
                             dpi=110, squeeze=False)
    for idx, it in enumerate(items):
        ax = axes[idx // cols][idx % cols]
        xlabel = "Pixel"
        for _lbl, x, y, xl in _series(it["result"]):
            if x and y:
                ax.plot(x, y, linewidth=0.7)
                xlabel = xl
        ax.set_title(it["title"] or it["base"], fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel(xlabel, fontsize=6)
    for j in range(n, rows * cols):           # 남는 칸 숨김
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{d} · 측정 {n}개 합본", fontsize=12)
    fig.tight_layout()

    day_dir = RESULTS_ROOT / d
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    base = (out_name or f"combined_{stamp}").lstrip("_")
    png = day_dir / f"_{base}.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {"ok": True, "count": n,
            "saved": {"title": f"Combined · {n} measurements", "image_url": _url(d, png.name)}}


def aggregate_spectra_csv(date: str | None = None, names: list[str] | None = None,
                          out_name: str | None = None, scope: str = "session") -> dict:
    """저장된 여러 측정을 실험당 한 행으로 요약한 CSV 를 만든다(#3).
    열: 날짜/시각/제목/좌표/파워/노출/모드/최대세기/총세기/피크위치."""
    d, items = _select(date, names, scope)
    if not items:
        return {"ok": False, "error": f"No measurement results to summarize on {d}."}
    day_dir = RESULTS_ROOT / d
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    base = (out_name or f"summary_{stamp}").lstrip("_")
    csv_path = day_dir / f"_{base}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "time", "title", "x", "y", "power_pct", "exposure_s",
                    "mode", "max_intensity", "sum_intensity", "peak_raman_shift_cm-1"])
        for it in items:
            res, meta = it["result"], it["meta"]
            inten = _intensity_of(res)
            peak = ""
            if res.get("calibrated") and res.get("raman_shift_cm-1") and inten:
                pk = max(range(len(inten)), key=lambda i: inten[i])
                peak = f"{res['raman_shift_cm-1'][pk]:.2f}"
            w.writerow([
                d, (it["timestamp"][11:19] if len(it["timestamp"]) >= 19 else ""),
                it["title"], meta.get("x", ""), meta.get("y", ""),
                res.get("laser_power_pct", ""), res.get("exposure_time", ""),
                res.get("mode", ""),
                f"{max(inten):.1f}" if inten else "",
                f"{sum(inten):.1f}" if inten else "", peak,
            ])
    return {"ok": True, "count": len(items),
            "saved": {"title": f"Summary table · {len(items)} measurements", "csv_url": _url(d, csv_path.name)}}


def save_scene(image, extent: list | None = None, meta: dict | None = None) -> dict:
    """현미경(카메라) 이미지 한 장을 저장한다 — 분석 코드가 피크맵을 이 위에 오버레이한다.

    image  : numpy 배열 (H,W) 또는 (H,W,3, RGB).
    extent : [xmin,xmax,ymin,ymax] 스테이지 좌표(mm) 범위. imshow(extent=)로 좌표 정합.
    npz(정확 픽셀+extent)와 png(미리보기)를 함께 남긴다.
    """
    import numpy as np
    try:
        arr = np.asarray(image)
        day = datetime.now().strftime("%Y-%m-%d")
        # 측정과 같은 세션 폴더에 둔다. '_scene_' 접두는 유지 — list_results 가 '_'
        # 접두를 개별 측정에서 제외하고, bundle 에도 안 실리게 하는 표식이다.
        sess = session_folder()
        out_dir = RESULTS_ROOT / day / sess
        out_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
        npz_path = out_dir / f"_scene_{stamp}.npz"
        png_path = out_dir / f"_scene_{stamp}.png"
        ext_arr = np.asarray(extent, dtype=float) if extent else np.asarray([], dtype=float)
        np.savez_compressed(npz_path, image=arr, extent=ext_arr)
        # 미리보기 png
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None,
                  extent=(extent if extent else None))
        ax.set_title("현미경 이미지")
        fig.tight_layout()
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        return {"ok": True, "image_url": f"{URL_PREFIX}/{day}/{sess}/{png_path.name}",
                "session": sess, "scene_npz": str(npz_path), "extent": extent}
    except Exception as e:
        return {"ok": False, "error": f"Failed to save microscope image: {e}"}


def save_preview_png(png_bytes: bytes, tag: str = "preview") -> dict:
    """이미 인코딩된 PNG 바이트(오버레이 등)를 재렌더 없이 그대로 저장하고 image_url을 돌려준다.
    save_scene(matplotlib 재렌더·제목·축)과 달리 원본 PNG를 보존한다 — 그리드 미리보기처럼
    오버레이를 정확히 채팅에 보여줄 때 쓴다. '_' 접두라 list_results(개별 측정 목록)에는 안 잡힌다.
    """
    try:
        day = datetime.now().strftime("%Y-%m-%d")
        out_dir = RESULTS_ROOT / day
        out_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
        safe = "".join(c for c in str(tag) if c.isalnum() or c in "-_") or "preview"
        name = f"_{safe}_{stamp}.png"
        (out_dir / name).write_bytes(png_bytes)
        return {"ok": True, "image_url": _url(day, name)}
    except Exception as e:
        return {"ok": False, "error": f"Failed to save preview image: {e}"}


def latest_scene(date: str | None = None, scope: str = "session") -> str | None:
    """가장 최근 scene npz 경로. 없으면 None.

    이 값이 run_analysis 의 `microscope_image` 로 주입되므로 '남의 세션 장면'을
    돌려주면 안 된다 — 다른 문항이 찍은 현미경 사진 위에 피크맵을 겹치게 되고,
    그건 없는 것보다 나쁘다(조용히 틀린 결과가 나온다).

    구버전 폴백은 '날짜 폴더 직속'에만 적용한다. 리팩터 이전 장면들이 모여 있는
    단일 풀이고, migrate_results 로 정리하면 자연히 비어서 폴백이 죽는다. 형제
    세션 폴더는 절대 뒤지지 않는다.
    """
    day_dir = RESULTS_ROOT / _resolve_date(date)
    if not day_dir.exists():
        return None
    if scope == "session":
        sess_dir = day_dir / session_folder()
        scenes = sorted(sess_dir.glob("_scene_*.npz")) if sess_dir.exists() else []
        if scenes:
            return str(scenes[-1])
        scenes = sorted(day_dir.glob("_scene_*.npz"))        # 구버전 폴백
        return str(scenes[-1]) if scenes else None
    # scope="all": 그날 아무 세션에서든 가장 최근 것(수동 조사용)
    scenes = sorted(day_dir.glob("_scene_*.npz")) + sorted(day_dir.glob("*/_scene_*.npz"))
    scenes.sort(key=lambda p: p.name)
    return str(scenes[-1]) if scenes else None


def bundle_results(date: str | None = None, names: list[str] | None = None,
                   include: tuple[str, ...] = ("png", "csv", "json"),
                   scope: str = "session") -> dict:
    """저장된 측정 파일들을 zip 하나로 묶어 다운로드 URL 을 돌려준다(#4)."""
    d, items = _select(date, names, scope)
    if not items:
        return {"ok": False, "error": f"No measurement results to bundle on {d}."}
    day_dir = RESULTS_ROOT / d
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    zip_name = f"_bundle_{stamp}.zip"
    zip_path = day_dir / zip_name
    n_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for it in items:
            for ext in include:
                p = it[ext]
                if isinstance(p, Path) and p.exists():
                    # 세션을 zip 안에서도 폴더로 유지한다 — scope="all" 로 여러 세션을
                    # 묶으면 같은 이름이 서로 덮어쓸 수 있고, 풀었을 때 귀속도 사라진다.
                    arc = f"{it['session']}/{p.name}" if it.get("session") else p.name
                    z.write(p, arcname=arc)
                    n_files += 1
    return {"ok": True, "count": len(items), "files": n_files,
            "saved": {"title": f"Bundle (zip) · {len(items)} measurements / {n_files} files",
                      "zip_url": _url(d, zip_name)}}
