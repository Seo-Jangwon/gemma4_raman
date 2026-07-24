"""
[역할]
  라만 측정 결과(acquire_spectrum 반환 dict)를 로컬에 '날짜/시간별'로 저장하고
  채팅 인라인 표시용 PNG 를 렌더한다. CoALA·AILA 두 에이전트가 공유하는 순수
  유틸(하드웨어 import 없음) — acquire_spectrum 이 측정 직후 이걸 호출한다.

  저장 레이아웃:
    data/results/<YYYY-MM-DD>/<HHMMSS_mmm>_<tag>.png   ← 스펙트럼 플롯(채팅 표시용)
    data/results/<YYYY-MM-DD>/<HHMMSS_mmm>_<tag>.csv   ← 스펙트럼 데이터(엑셀 호환)
    data/results/<YYYY-MM-DD>/<HHMMSS_mmm>_<tag>.json  ← 원본 결과+메타(합본/재렌더용)

  PNG/JSON URL 은 '/api/results/...' 로 서빙된다(vite proxy 가 /api 만 통과시키므로
  결과 파일도 /api 아래에 둔다). server.py 가 이 경로를 정적 서빙한다.
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


def _write_csv(result: dict, path: Path) -> None:
    """스펙트럼을 CSV 로 저장(보정 시 4열, 미보정 시 2열). 메타는 주석행으로."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        for k in ("laser_power_pct", "exposure_time", "laser_nm", "mode"):
            if result.get(k) is not None:
                f.write(f"# {k},{result[k]}\n")
        w = csv.writer(f)
        if result.get("mode") == "kinetic" and result.get("frames"):
            w.writerow(["frame_index", "pixel", "intensity"])
            for fr in result["frames"]:
                for px, val in enumerate(fr.get("intensity") or []):
                    w.writerow([fr.get("frame_index", ""), px, val])
            return
        intensity = _intensity_of(result)
        if result.get("calibrated") and result.get("raman_shift_cm-1"):
            w.writerow(["pixel", "raman_shift_cm-1", "wavelength_nm", "intensity"])
            shift = result["raman_shift_cm-1"]
            wl = result.get("wavelength_nm") or [""] * len(intensity)
            for i in range(len(intensity)):
                w.writerow([i, f"{shift[i]:.3f}", f"{wl[i]:.4f}" if wl[i] != "" else "",
                            intensity[i]])
        else:
            w.writerow(["pixel", "intensity"])
            for px, val in enumerate(intensity):
                w.writerow([px, val])


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
    dict — {ok, title, timestamp, dir, files:{png,csv,json}, image_url, csv_url, json_url}
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

        out_dir = RESULTS_ROOT / date_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        png_path = out_dir / f"{base}.png"
        csv_path = out_dir / f"{base}.csv"
        json_path = out_dir / f"{base}.json"

        title = make_title(result, meta)
        render_png(result, png_path, title)
        _write_csv(result, csv_path)
        # json: 원본 결과 + 메타 + 제목 (합본 재렌더·검색용)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"title": title, "meta": meta or {},
                       "timestamp": now.isoformat(), "result": result},
                      f, ensure_ascii=False)

        def url(p: Path) -> str:
            return f"{URL_PREFIX}/{date_dir}/{p.name}"

        return {
            "ok": True,
            "title": title,
            "timestamp": now.isoformat(),
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


def list_results(date: str | None = None) -> list[dict]:
    """해당 날짜(기본 오늘)에 저장된 개별 측정 목록을 시각 순으로 돌려준다.

    각 항목: {base, date, title, timestamp, meta, result, png/csv/json(Path)}.
    합본/요약/zip 등 생성물('_' 접두)은 제외한다.
    """
    d = _resolve_date(date)
    day_dir = RESULTS_ROOT / d
    if not day_dir.exists():
        return []
    items = []
    for jp in sorted(day_dir.glob("*.json")):
        if jp.stem.startswith("_"):
            continue
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append({
            "base": jp.stem, "date": d,
            "title": payload.get("title", ""),
            "timestamp": payload.get("timestamp", ""),
            "meta": payload.get("meta", {}) or {},
            "result": payload.get("result", {}) or {},
            "png": jp.with_suffix(".png"),
            "csv": jp.with_suffix(".csv"),
            "json": jp,
        })
    return items


def _select(date: str | None, names: list[str] | None) -> tuple[str, list[dict]]:
    d = _resolve_date(date)
    items = list_results(d)
    if names:
        want = set(names)
        items = [it for it in items if it["base"] in want]
    return d, items


def combine_spectra(date: str | None = None, names: list[str] | None = None,
                    out_name: str | None = None, max_cols: int = 4) -> dict:
    """저장된 여러 스펙트럼을 한 장(격자)으로 렌더한다(#2). 제목은 각 측정의
    저장 제목(좌표·조건 자동 생성)을 그대로 쓴다. 반환 saved.image_url 로 채팅 표시."""
    d, items = _select(date, names)
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
                          out_name: str | None = None) -> dict:
    """저장된 여러 측정을 실험당 한 행으로 요약한 CSV 를 만든다(#3).
    열: 날짜/시각/제목/좌표/파워/노출/모드/최대세기/총세기/피크위치."""
    d, items = _select(date, names)
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
        out_dir = RESULTS_ROOT / day
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
        return {"ok": True, "image_url": _url(day, png_path.name),
                "scene_npz": str(npz_path), "extent": extent}
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


def latest_scene(date: str | None = None) -> str | None:
    """해당 날짜(기본 오늘)의 가장 최근 scene npz 경로. 없으면 None."""
    day_dir = RESULTS_ROOT / _resolve_date(date)
    if not day_dir.exists():
        return None
    scenes = sorted(day_dir.glob("_scene_*.npz"))
    return str(scenes[-1]) if scenes else None


def bundle_results(date: str | None = None, names: list[str] | None = None,
                   include: tuple[str, ...] = ("png", "csv", "json")) -> dict:
    """저장된 측정 파일들을 zip 하나로 묶어 다운로드 URL 을 돌려준다(#4)."""
    d, items = _select(date, names)
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
                    z.write(p, arcname=p.name)
                    n_files += 1
    return {"ok": True, "count": len(items), "files": n_files,
            "saved": {"title": f"Bundle (zip) · {len(items)} measurements / {n_files} files",
                      "zip_url": _url(d, zip_name)}}
