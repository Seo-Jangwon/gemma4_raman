"""
[역할] 분석 전용 코드 샌드박스.
  에이전트가 "저장된 측정 데이터에 대한 계산·시각화"를 코드로 풀어야 할 때, 그 코드를
  '안전한 범위'에서만 실행한다. 핵심 안전 경계:

    · 하드웨어 격리 — 생성 코드는 별도 서브프로세스에서 돌고, 레이저/스테이지/CCD 등
      어떤 하드웨어 객체도 노출하지 않는다(전용 모듈이라 raman_tools 전역을 안 물음).
    · import 화이트리스트 — numpy/scipy/matplotlib/math 등 계산·플롯 라이브러리만.
      os/sys/subprocess/socket/open 등은 AST 검사 단계에서 거부.
    · 제한된 builtins — 파일/네트워크/eval/exec/__import__/getattr 류 차단.
    · 타임아웃 — 무한 루프 방지(부모가 프로세스 kill).
    · 파일 출력은 data/results 로만 — matplotlib 그림만 자동 저장되어 채팅에 표시.

  실행 컨텍스트에 미리 주입되는 것:
    spectra : list[dict]  — 저장된 측정들. 각 dict:
        base, title, x, y, power, exposure, mode,
        raman_shift(np.ndarray|None), intensity(np.ndarray)
    files   : list[dict]  — 사용자가 채팅에 첨부한 파일(file_ids 로 지정된 것만). 각 dict:
        file_id, filename, sheet, columns(list[str]), n_rows,
        table(dict: 컬럼명 → 숫자면 np.ndarray, 문자면 list[str])
    np      : numpy
    plt     : matplotlib.pyplot   (그림을 만들면 자동 저장됨)

  ※ 첨부 파일도 microscope_image 와 똑같은 원칙으로 들어온다 — 파일을 '읽는' 것은
    신뢰된 부모(upload_store)이고, 생성된 코드는 이미 파싱된 변수만 만진다.
    위 import 화이트리스트/BANNED 목록 때문에 생성 코드가 스스로 파일을 열 방법은 없다.

  최악의 오작동도 "그림/계산이 틀림"에 그치고 하드웨어에는 닿지 않는다.
"""
from __future__ import annotations

import ast
import builtins as _builtins
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── 정책 ──────────────────────────────────────────────────────────────────────
_ALLOWED_IMPORT_ROOTS = {
    "numpy", "scipy", "math", "statistics", "matplotlib", "pandas",
    "json", "itertools", "functools", "collections", "cmath", "random",
}
_BANNED_NAMES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open", "eval",
    "exec", "compile", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "exit", "quit", "breakpoint", "help",
    "memoryview", "classmethod", "staticmethod", "super", "object", "type",
}
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "bytes", "chr", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "int", "isinstance",
    "issubclass", "len", "list", "map", "max", "min", "ord", "pow", "print",
    "range", "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "abs", "hasattr",
    # 예외 계열 — 코드가 try/except 를 쓸 수 있게
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "RuntimeError", "StopIteration", "ArithmeticError",
]


def validate_code(code: str) -> None:
    """실행 전에 AST를 훑어 위험 구문을 거부한다(허용 안 되면 ValueError)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"Disallowed import: '{a.name}'. "
                                     f"Allowed: {sorted(_ALLOWED_IMPORT_ROOTS)}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Disallowed import: '{node.module}'")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith("__"):
                raise ValueError(f"Disallowed attribute access: '{node.attr}' "
                                 "(dunder attributes forbidden)")
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES:
                raise ValueError(f"Disallowed name use: '{node.id}'")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global/nonlocal not allowed")


def _safe_builtins() -> dict:
    d = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(_builtins, name):
            d[name] = getattr(_builtins, name)
    # __import__ 를 화이트리스트 기반으로 제공 — import 문이 동작하되 허용 모듈만.
    def _guarded_import(name, *args, **kwargs):
        if name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"Disallowed import: {name}")
        return _builtins.__import__(name, *args, **kwargs)
    d["__import__"] = _guarded_import
    return d


# ── 서브프로세스 진입점: python -m backend.analysis_sandbox <payload.json> ──────
def _main(payload_path: str) -> None:
    import io
    import contextlib
    import traceback
    from datetime import datetime

    from backend.spectrum_store import RESULTS_ROOT, URL_PREFIX

    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    code = payload["code"]

    buf = io.StringIO()
    result: dict = {"ok": False}
    try:
        validate_code(code)

        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        spectra = []
        for it in payload["spectra"]:
            d = dict(it)
            d["intensity"] = np.asarray(it.get("intensity") or [], dtype=float)
            rs = it.get("raman_shift")
            d["raman_shift"] = np.asarray(rs, dtype=float) if rs else None
            spectra.append(d)

        # 현미경 이미지(있으면) 주입 — 코드가 피크맵을 이 위에 오버레이할 수 있게.
        # 로드는 신뢰된 하버스(여기)가 하고, 사용자 코드는 변수만 쓴다(파일 접근 없음).
        microscope_image = None
        image_extent = None
        scene_path = payload.get("scene_path")
        if scene_path:
            with np.load(scene_path) as z:
                microscope_image = z["image"]
                ext = z["extent"]
                image_extent = [float(v) for v in ext] if ext.size == 4 else None

        # 첨부 파일(있으면) 주입 — 숫자 컬럼만 np 배열로 승격하고 문자 컬럼은 리스트 그대로.
        # 어느 컬럼이 무슨 의미인지는 여기서 판단하지 않는다(그건 에이전트 몫).
        files = []
        for up in payload.get("uploads") or []:
            table = {}
            for cname, vals in (up.get("numeric") or {}).items():
                table[cname] = np.asarray(
                    [np.nan if v is None else v for v in vals], dtype=float)
            for cname, vals in (up.get("text") or {}).items():
                table[cname] = list(vals)
            files.append({
                "file_id": up.get("file_id"),
                "filename": up.get("filename"),
                "sheet": up.get("sheet"),
                "columns": up.get("columns") or [],
                "n_rows": up.get("n_rows", 0),
                "table": table,
            })

        ns = {
            "__builtins__": _safe_builtins(),
            "np": np, "plt": plt, "spectra": spectra, "files": files,
            "microscope_image": microscope_image, "image_extent": image_extent,
        }
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<analysis>", "exec"), ns)  # noqa: S102 — 샌드박스됨

        # 생성된 모든 그림을 data/results 에 저장
        day = datetime.now().strftime("%Y-%m-%d")
        out_dir = RESULTS_ROOT / day
        out_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
        images = []
        for i, num in enumerate(plt.get_fignums()):
            fig = plt.figure(num)
            name = f"_analysis_{stamp}_{i}.png"
            fig.savefig(out_dir / name, bbox_inches="tight")
            images.append(f"{URL_PREFIX}/{day}/{name}")
        plt.close("all")

        result = {"ok": True, "stdout": buf.getvalue(), "images": images}
    except Exception as e:
        result = {
            "ok": False,
            "stdout": buf.getvalue(),
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1200:],
        }
    sys.stdout.write("\n__ANALYSIS_RESULT__" + json.dumps(result, ensure_ascii=False))


# ── 부모(서버) 쪽 오케스트레이터 ───────────────────────────────────────────────
def run_analysis(code: str, date: str | None = None, names: list[str] | None = None,
                 title: str | None = None, timeout_sec: int = 60,
                 file_ids: list[str] | None = None) -> dict:
    """저장된 측정 데이터(+첨부 파일)를 대상으로 분석/시각화 코드를 안전 실행한다.

    file_ids 를 주면 그 업로드 파일들이 파싱되어 샌드박스의 `files` 변수로 들어간다.
    측정 데이터(spectra)와 첨부 파일(files)은 동시에 쓸 수 있다 — 예를 들어 사용자가
    올린 참조 스펙트럼과 방금 측정한 스펙트럼을 한 그림에 겹쳐 그리는 식.

    Returns
    -------
    dict — {ok, stdout, image_count, saved?{title,image_url}, images?, error?}
           그림이 생기면 saved.image_url 로 채팅에 표시된다(spectrum_event 재사용).
    """
    import subprocess
    import tempfile
    from backend.spectrum_store import list_results, latest_scene
    from backend.upload_store import load_upload

    items = list_results(date)
    if names:
        want = set(names)
        items = [it for it in items if it["base"] in want]

    spectra = []
    for it in items:
        res = it["result"]
        intensity = res.get("data") or res.get("intensity") or []
        spectra.append({
            "base": it["base"], "title": it["title"],
            "x": it["meta"].get("x"), "y": it["meta"].get("y"),
            "power": res.get("laser_power_pct"), "exposure": res.get("exposure_time"),
            "mode": res.get("mode"),
            "raman_shift": res.get("raman_shift_cm-1"),
            "intensity": list(intensity),
        })

    # 첨부 파일을 '여기서(신뢰된 부모)' 읽어 둔다 — 샌드박스 코드는 파일을 열 수 없으므로
    # 이 단계에서 파싱된 값만이 생성 코드가 볼 수 있는 전부다.
    uploads = []
    for fid in (file_ids or []):
        try:
            uploads.append(load_upload(str(fid)))
        except Exception as e:
            return {"ok": False,
                    "error": f"Could not load the attached file '{fid}': {type(e).__name__}: {e}",
                    "hint": "Check the exact file_id with list_uploaded_files."}

    # 실행 전에 문법·정책을 부모에서도 1차 검사(빠른 실패, 명확한 에러)
    try:
        validate_code(code)
    except ValueError as e:
        return {"ok": False, "error": f"Code policy violation: {e}",
                "hint": "Analysis only with numpy/scipy/matplotlib. No hardware/file/network access."}

    payload = {"code": code, "spectra": spectra, "scene_path": latest_scene(date),
               "uploads": uploads}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "backend.analysis_sandbox", payload_path],
            capture_output=True, text=True, timeout=timeout_sec,
            cwd=str(_PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Execution timed out ({timeout_sec}s) - infinite loop or too heavy a computation."}
    finally:
        try:
            Path(payload_path).unlink()
        except OSError:
            pass

    marker = "__ANALYSIS_RESULT__"
    out = proc.stdout or ""
    if marker not in out:
        # 결과 마커 없이 죽음 — 프로세스 자체 크래시(예: 세그폴트)나 정책 외 종료
        tail = (proc.stderr or out)[-800:]
        return {"ok": False, "error": "The analysis process exited without a result.", "detail": tail}
    payload = json.loads(out.split(marker, 1)[1])

    if not payload.get("ok"):
        return {"ok": False, "error": payload.get("error", "Analysis failed"),
                "stdout": payload.get("stdout", ""), "trace": payload.get("trace", "")}

    images = payload.get("images", [])
    resp = {"ok": True, "stdout": payload.get("stdout", ""),
            "image_count": len(images), "images": images}
    if images:
        resp["saved"] = {"title": title or "Analysis result", "image_url": images[0]}
    return resp


if __name__ == "__main__":
    _main(sys.argv[1])
