# -*- coding: utf-8 -*-
"""배경 제거(IPBSA)와 스펙트럼 CSV 읽기.

하드웨어를 직접 만지지 않지만 '직전 측정(source=last)'을 세션 상태에서 읽으므로
hw_core 의 세션 캐시를 공유한다. 계산 자체는 backend.util.spectro_math 단일 출처다.
"""
from __future__ import annotations

import csv
import json

from backend.service.store.spectrum_store import write_spectrum_csv as _store_write_csv
from backend.tools.result import fail, ok
from pydantic import Field
from typing import Annotated, Optional
from backend.tools.hw_tools.hw_tools.hw_core import _sstate


# ──────────────────────────────────────────
# 데이터 저장 / 로드
# ──────────────────────────────────────────

# data/ 위치는 service.store 단일 출처. 예전에는 여기서 __file__ 로 되짚었는데, 이 파일이
# hw_tools/hw_tools/ 로 한 단계 내려가면서 <루트>/data 대신 backend/data 를 가리키게 됐다.
# 에러가 아니라 "File not found"로만 드러나는 종류라 발견이 늦다.


def _resolve_data_path(source: str):
    """파일 지시자를 실제 경로로. (경로, 못 찾았을 때의 설명) 을 돌려준다.

    [해석기를 넷에서 하나로 — 2026-08-12]
    이 함수는 예전에 자기 규칙을 갖고 있었다: data/ 기준 상대경로를 먼저 보고, 안 되면
    upload_store 로 넘겼다. 같은 일을 하는 함수가 프로젝트에 넷이었고(여기, upload_store,
    view_image, run_analysis) 저마다 아는 뿌리가 달랐다. 이제 규칙은
    backend/service/store/paths.py 하나이고, 이 함수는 그리로 넘기는 껍데기다.

    남겨 둔 이력 — 이 자리가 왜 아팠는지:
    2026-08-10 이전에는 _DATA_DIR 기준으로만 풀어서, list_uploaded_files 가 준 file_id 를
    그대로 넘긴 호출이 **구조적으로 전부 실패**했다(data/uploads/<날짜>/ 를 봐야 하는데
    data/<날짜>/ 를 봤다 — `uploads/` 한 칸 차이).

        "File not found: ...\\data\\2026-08-07\\N05.csv"   ← 실제 파일은 data/uploads/2026-08-07/

    목록 도구가 준 이름을 그대로 쓴 것이라 모델이 빠져나갈 방법이 없었고, 2026-08-07
    실행에서 N05·T039·T064·T070·T112·T115 가 여기서 막혔다. 막힌 뒤에는 run_analysis 로
    로딩·보정을 손수 구현하게 되는데, 그렇게 길어진 코드가 tool call JSON 유실
    (2,200자 이상에서 9.1%)을 부르는 경로이기도 했다.
    """
    from backend.service.store.paths import try_resolve

    # kind 를 주지 않는다 — 이 함수는 csv·json·절대경로를 모두 나르는 범용 통로이고,
    # 확장자 판정은 부르는 쪽(load_spectrum 은 .csv, 배경제거는 .csv/.json)이 한다.
    return try_resolve(source)


# ──────────────────────────────────────────
# 배경 제거 (IPBSA)
# ──────────────────────────────────────────

def _ipbsa(intensity, poly_order=5, max_iterations=100, threshold=0.001):
    """IPBSA — 구현은 backend.util.spectro_math 한 곳에만 있다.

    run_analysis 샌드박스도 같은 함수를 ipbsa() 라는 이름으로 주입받는다. 예전처럼
    양쪽이 각자 구현하면, 도구로 푼 실행과 코드로 푼 실행이 미세하게 다른 답을 내고
    그 차이가 에이전트 성적에 실린다 — 에이전트가 통제할 수 없는 차이다.
    """
    from backend.service.analyse.spectro_math import ipbsa_detail
    corrected, bg, iterations, converged = ipbsa_detail(
        intensity, order=poly_order, max_iterations=max_iterations, threshold=threshold)
    return corrected.tolist(), bg.tolist(), iterations, converged


def apply_background_subtraction(
    poly_order: Annotated[int, Field(ge=2, le=10, description="Polynomial order (2-10) - REQUIRED, decide it yourself from the shape of this spectrum's background. A low order (2-3) fits only a gentle slope and will leave a curved background behind; a high order (8-10) can bend enough to follow the peaks and subtract away real signal. Mid orders (4-6) suit the moderate fluorescence curvature that is typical, but confirm it against the data rather than assuming.")],
    max_iterations: Annotated[Optional[int], Field(ge=10, le=500, description='Maximum number of iterations (10-500). Default 100.')] = 100,
    threshold: Annotated[Optional[float], Field(ge=0.001, le=1.0, description='Convergence criterion - relative L2 change of the background curve between iterations (0.001-1.0). Default 0.001. Smaller means stricter convergence.')] = 0.001,
    source: Annotated[Optional[str], Field(description="Source spectrum to background-subtract. 'last': use the most recent acquire_spectrum() result (default). Otherwise a JSON or CSV file_id in the form '<area>:<path>' (e.g. 'uploads:2026-08-07/N05.csv', 'results:2026-08-07/<session>/1408_x37.csv'), exactly as the listing tools return it; paths without the prefix and absolute paths also work. TWO LIMITS ON 'last'. (1) It only works on Single/Accumulate spectra - a Kinetic measurement has no single intensity array and is REJECTED. (2) run_grid_scan acquires internally at every point, so right after a grid scan 'last' means ONLY the final point of that grid, not the grid. To baseline-correct a whole grid, pass file paths one at a time (get them from list_results), or write the loop yourself in run_analysis.")] = 'last',
    version_label: Annotated[Optional[str], Field(description="Version name to attach to this result. e.g. 'v1_poly5', 'v2_poly7'. Calling again with the same name overwrites it. Default 'default'.")] = 'default',
    save_result: Annotated[Optional[bool], Field(description="If True, also write the corrected spectrum to this session's folder as a CSV (standard format: pixel_index, raman_shift_cm-1, intensity, background_intensity) and return its file_id in saved_path, which open_file and run_analysis(file_ids=[...]) both accept. Default false - without it the result exists only in memory for this conversation. SET IT TO TRUE IF YOU WILL PLOT OR ANALYSE THE RESULT: run_analysis reads files, not this conversation's memory, so an unsaved version is invisible to it and there is no way to hand the arrays over afterwards. Leave it false only when you are just comparing poly_order settings via list_bg_versions.")] = False,
) -> dict:
    """IPBSA(반복 다항식 배경 제거)를 수행하고 결과를 이 세션의 버전 목록에 저장한다."""
    _st = _sstate()
    _last_spectrum = _st["last_spectrum"]

    if not (2 <= poly_order <= 10):
        return fail(f"poly_order must be 2-10 (got: {poly_order})")
    if not (10 <= max_iterations <= 500):
        return fail(f"max_iterations must be 10-500 (got: {max_iterations})")
    if not (0.001 <= threshold <= 1.0):
        return fail(f"threshold must be 0.001-1.0 (got: {threshold})")

    intensity: list = []
    raman_shift = None

    if source == "last":
        if _last_spectrum is None:
            return fail("No saved spectrum. Call acquire_spectrum() first.")
        if "data" not in _last_spectrum:
            return fail("The last spectrum is in Kinetic mode. This applies only to Single/Accumulate spectra.")
        intensity = _last_spectrum["data"]
        raman_shift = _last_spectrum.get("raman_shift_cm-1")
    else:
        filepath, why = _resolve_data_path(source)
        if filepath is None:
            return fail(why)
        try:
            if filepath.suffix.lower() == ".json":
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if "data" in loaded:
                    intensity = loaded["data"]
                elif "corrected_data" in loaded:
                    intensity = loaded["corrected_data"]
                else:
                    return fail("The JSON file has no 'data' or 'corrected_data' key.")
                raman_shift = loaded.get("raman_shift_cm-1")
            elif filepath.suffix.lower() == ".csv":
                # CSV 읽기는 load_spectrum 에 위임한다 — 여기서 csv.DictReader 를 직접
                # 쓰던 예전 코드는 '# key,value' 메타 주석행을 건너뛰지 않아서, 측정
                # 자동저장 CSV(data/results/...)를 source 로 주면 첫 주석행이 헤더로 잡혀
                # "intensity 열이 없다"로 실패했다(BOM 도 처리하지 않았다).
                loaded = load_spectrum(str(filepath))
                if not loaded.get("ok"):
                    return fail(f"File load error: {loaded.get('error')}")
                intensity = loaded["intensity"]
                raman_shift = loaded.get("raman_shift_cm-1")
            else:
                return fail("Unsupported file format (only JSON or CSV allowed).")
        except Exception as e:
            return fail(f"File load error: {e}")

    if not intensity:
        return fail("The spectrum intensity array is empty.")
    if len(intensity) < poly_order + 1:
        return fail(f"Spectrum length ({len(intensity)}) is smaller than poly_order+1 ({poly_order + 1}). "
                    "Lower the polynomial order.")

    try:
        corrected, background, iterations_run, converged = _ipbsa(
            intensity=intensity,
            poly_order=poly_order,
            max_iterations=max_iterations,
            threshold=threshold,
        )
    except Exception as e:
        return fail(f"IPBSA algorithm error: {e}")

    saved_path = None
    if save_result:
        # run_store 세션 폴더에 저장한다. 예전에는 data/ 최상위에 bg_corrected_<label>.csv 로
        # 떨궈서, 같은 라벨을 쓰면 이전 결과를 덮어썼고 어느 과제 산출물인지도 알 수 없었다
        # (save_spectrum 이 세션 폴더로 옮겨진 것과 같은 이유 — run_store.py 참고).
        try:
            from backend.service.store import run_store
            save_filepath, rel = run_store.new_spectrum_path(f"bg_corrected_{version_label}")
            # 저장 포맷은 spectrum_store.write_spectrum_csv 단일 출처를 쓴다. 예전에는
            # 여기서 csv.writer 를 직접 돌리며 세기 열을 'corrected_intensity' 로 썼는데,
            # 같은 폴더에 run_analysis 의 save_result 가 'intensity' 로 쓰고 있어서
            # 한 폴더 안에 두 포맷이 섞였다(load_spectrum 이 열 이름을 추측해야 했던 이유).
            _store_write_csv(
                save_filepath,
                intensity=corrected,
                raman_shift=raman_shift,
                background=background,
                meta={"kind": "background_subtracted", "version_label": version_label,
                      "poly_order": poly_order, "iterations_run": iterations_run},
            )
            run_store.record(run_store.KIND_SPECTRA, rel, num_points=len(corrected),
                             kind_detail="background_subtracted", version_label=version_label)
            # 상대경로를 준다 — 이 문자열을 그대로 load_spectrum 에 넘겨 다시 읽을 수 있다.
            saved_path = rel
        except Exception:
            pass

    result = ok(version_label=version_label,
                poly_order=poly_order,
                max_iterations=max_iterations,
                threshold=threshold,
                iterations_run=iterations_run,
                converged=converged,
                max_corrected_intensity=float(max(corrected)) if corrected else 0.0,
                max_background_intensity=float(max(background)) if background else 0.0,
                corrected_data=corrected,
                background_data=background)
    if raman_shift is not None:
        result["raman_shift_cm-1"] = raman_shift
    if saved_path is not None:
        result["saved_path"] = saved_path

    _st["bg_versions"][version_label] = result.copy()
    return result


def list_bg_versions() -> dict:
    """저장된 모든 배경 제거 결과 버전의 목록과 주요 통계를 반환한다."""
    _bg_versions = _sstate()["bg_versions"]
    if not _bg_versions:
        return ok(count=0,
                  versions=[],
                  message="No saved versions. Call apply_background_subtraction() first.")
    summaries = []
    for label, v in _bg_versions.items():
        summaries.append({
            "version_label":            label,
            "poly_order":               v.get("poly_order"),
            "max_iterations":           v.get("max_iterations"),
            "threshold":                v.get("threshold"),
            "iterations_run":           v.get("iterations_run"),
            "converged":                v.get("converged"),
            "max_corrected_intensity":  v.get("max_corrected_intensity"),
            "max_background_intensity": v.get("max_background_intensity"),
            "has_raman_shift":          "raman_shift_cm-1" in v,
            "data_length":              len(v.get("corrected_data", [])),
        })
    return ok(count=len(summaries), versions=summaries)


def get_bg_version(
    version_label: Annotated[str, Field(description="Version name to query. e.g. 'v1_poly5'")],
) -> dict:
    """특정 버전의 배경 제거 결과 전체 데이터를 반환한다."""
    _bg_versions = _sstate()["bg_versions"]
    if version_label not in _bg_versions:
        return fail(f"Version '{version_label}' not found.",
                    available_versions=list(_bg_versions.keys()))
    return ok(**_bg_versions[version_label])


# [제거됨 — save_spectrum 툴, 2026-07-30]
# "강도 배열을 인자로 받아 CSV 로 저장하는" 툴이었다. 세 가지 이유로 정당한 용례가 없다:
#
#  1) 에이전트는 그 배열을 애초에 갖고 있지 않다. 관측 축약기(_slim, single_agent_AILA.py)가
#     길이 32 초과 리스트를 통째로 버리므로 acquire_spectrum 의 data(1024~2048점)는 모델에
#     도달하지 않는다. 즉 이 툴을 호출하려면 배열을 지어내거나 잘라내야 한다.
#  2) 원측정 데이터는 이미 자동 저장된다(_persist_spectrum → data/results/<날짜>/<세션>/).
#  3) 가공한 배열은 run_analysis 안의 save_result 훅이 담당한다. 그 훅은 정확히 이 왕복을
#     없애려고 만들어졌다(analysis_sandbox.py 상단 주석: 1801점 스펙트럼이 컨텍스트를
#     2만 토큰씩 왕복해 생성이 잘리던 문제). 시스템 프롬프트도 이미 "print 해서
#     save_spectrum 에 넣지 말라"고 금지하고 있었다.
#
# 대체: 측정 결과 → 자동 저장 / 가공 결과 → run_analysis + save_result(...) /
#       다시 읽기 → load_spectrum(path) / 측정점 묶기 → save_measurement_point(...)


def load_spectrum(
    filename: Annotated[str, Field(description="A file_id, in the form '<area>:<path>' where area is uploads, results or runs - exactly as the listing tools return it (e.g. 'uploads:2026-08-07/N05.csv', 'runs:<session>/spectra/01_corrected.csv'). Paths without the area prefix and absolute paths are also accepted.")],
) -> dict:
    """
    저장된 스펙트럼 CSV 파일 또는 업로드된 입력 파일을 로드한다.
    받는 형태는 세 가지다 — 절대 경로, data/ 기준 상대 경로,
    그리고 list_uploaded_files 가 돌려주는 file_id("<YYYY-MM-DD>/<파일명>").

    이 프로젝트가 쓰는 스펙트럼 CSV 는 모두 같은 포맷이다(spectrum_store.write_spectrum_csv):
        pixel_index, [raman_shift_cm-1,] [wavelength_nm,] intensity, [background_intensity]
    헤더 앞에는 '# key,value' 메타 주석행이 붙는다. 그걸 건너뛰지 않으면 첫 주석행이
    헤더로 잡혀 'intensity' 열을 못 찾는다 — 측정 결과를 다시 읽는 경로가 통째로 막힌다.
    encoding 은 utf-8-sig: 같은 파일이 BOM 을 달고 저장된다(엑셀 호환).

    측정 자동저장분(data/results/...), run_analysis 의 save_result 산출물, 배경 제거
    산출물(data/runs/...)을 모두 같은 방식으로 읽는다. 'corrected_intensity' 는 포맷
    통일 이전(2026-07-30 이전) 파일에만 남아 있는 옛 이름이라 하위호환으로만 인정한다.
    """
    try:
        if not filename.endswith(".csv"):
            filename += ".csv"
        filepath, why = _resolve_data_path(filename)
        if filepath is None:
            return fail(why)

        comments: dict = {}
        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            lines = f.read().splitlines()
        body = []
        for ln in lines:
            if not body and ln.lstrip().startswith("#"):
                bits = ln.lstrip().lstrip("#").strip().split(",", 1)
                if len(bits) == 2:
                    comments[bits[0].strip()] = bits[1].strip()
                continue
            body.append(ln)

        reader = csv.DictReader(body)
        headers = reader.fieldnames or []
        rows = list(reader)
        # ── Kinetic 파일 방어 — 2026-08-01 ────────────────────────────────────
        # Kinetic 측정은 spectrum_store._write_csv 가 'frame_index, pixel_index, intensity'
        # 롱포맷으로 저장한다(프레임 N개가 세로로 이어 붙는다). 아래 파서는 1D 스펙트럼만
        # 가정하므로, 그 파일을 그냥 읽으면 에러 없이 **N프레임을 한 배열로 이어붙인**
        # 길이 N×픽셀 짜리 배열이 나온다. 조용히 틀린 데이터를 주는 쪽이 실패보다 나쁘다.
        if "frame_index" in headers:
            n_frames = len({r.get("frame_index") for r in rows})
            return fail(f"{filepath.name} is a Kinetic measurement ({n_frames} frames stored as "
                        f"frame_index,pixel_index,intensity). load_spectrum only reads 1D spectra "
                        f"(Single/Accumulate) and would concatenate the frames into one wrong array. "
                        f"Analyse it with run_analysis instead: kinetic measurements arrive there as "
                        f"spectra[i]['frames'], a (n_frames, n_pixels) array.")
        # 세기 열은 'intensity' 로 통일돼 있다. 'corrected_intensity' 는 포맷 통일
        # 이전에 배경 제거 툴이 쓰던 이름이라 옛 파일을 위해서만 남긴다.
        col = next((c for c in ("intensity", "corrected_intensity") if c in headers), None)
        if col is None:
            return fail(f"No 'intensity' (or 'corrected_intensity') column in "
                        f"{filepath.name}. Columns: {headers}")

        intensity = [float(r[col]) for r in rows]
        result: dict = ok(filename=str(filepath),
                          num_points=len(intensity),
                          headers=headers,
                          intensity_column=col,
                          intensity=intensity)
        if comments:
            result["metadata"] = comments        # laser_power_pct / exposure_time / mode 등
        if "raman_shift_cm-1" in headers:
            result["raman_shift_cm-1"] = [float(r["raman_shift_cm-1"]) for r in rows]
        if "wavelength_nm" in headers:
            result["wavelength_nm"] = [float(r["wavelength_nm"]) for r in rows]
        if "background_intensity" in headers:
            result["background_intensity"] = [float(r["background_intensity"]) for r in rows]
        return result
    except Exception as e:
        return fail(str(e))
