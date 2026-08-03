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
    · 파일 출력은 data/ 아래로만 — matplotlib 그림은 data/results 에 자동 저장되어
      채팅에 표시되고, 계산된 배열은 save_result() 로만 data/ 에 쓸 수 있다.
      생성 코드가 경로를 직접 만질 방법은 없다(파일명은 훅이 basename 으로 강제).

  실행 컨텍스트에 미리 주입되는 것:
    spectra : list[dict]  — 저장된 측정들. 각 dict:
        base, title, x, y, power, exposure, mode,
        raman_shift(np.ndarray|None), intensity(np.ndarray)
        mode=='kinetic' 이면 추가로:
          frames(np.ndarray, shape (n_frames, n_pixels)), n_frames, n_frames_total,
          intensity 는 프레임 평균이고 intensity_is_frame_mean=True 가 함께 붙는다,
          잘렸으면 frames_truncated(str), 프레임 길이가 다르면 frames_ragged=True
          (이때 frames 는 배열들의 리스트)
    files   : list[dict]  — 사용자가 채팅에 첨부한 파일(file_ids 로 지정된 것만). 각 dict:
        file_id, filename, sheet, columns(list[str]), n_rows,
        table(dict: 컬럼명 → 숫자면 np.ndarray, 문자면 list[str])
    np      : numpy
    plt     : matplotlib.pyplot   (그림을 만들면 자동 저장됨)
    save_result : 계산 결과 배열을 CSV 로 저장하는 훅 (아래 참고)

  [save_result 가 왜 있는가 — 컨텍스트 왕복 제거]
  이 샌드박스는 파일을 못 쓰므로, 원래는 계산 결과를 밖으로 빼는 통로가 print() 뿐이었다.
  그래서 "보정해서 저장하라"류 과제에서 모델은 ① 배열 전체를 print 하고 ② 그걸 읽어서
  ③ 저장 툴 인자로 통째로 다시 뱉어야 했다(당시의 save_spectrum 툴 — 2026-07-30 제거).
  1801점짜리 스펙트럼 하나가
  컨텍스트를 2만 토큰씩 왕복하는 셈이라, 프롬프트(30k)만으로 창(32k)이 차고 생성이
  중간에 잘려 빈 응답이 나왔다("Failed to generate a response."의 실제 원인).
  save_result 는 그 왕복을 없앤다 — 숫자는 샌드박스 밖으로 나가지 않고, 모델은 저장된
  경로 한 줄만 받는다. 부수 효과로 정밀도 손실도 사라진다(모델이 float 을 옮겨 적지 않으므로).

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


# stdout 상한. 이걸 넘으면 잘라내고 "대신 save_result 를 쓰라"고 안내한다.
# 원래 무제한이라, 모델이 배열을 통째로 print 하면 그 문자열이 그대로 다음 프롬프트에
# 실려 컨텍스트를 고갈시켰다. 4000자면 진단용 print(피크 위치, 통계 등)는 다 들어간다.
_MAX_STDOUT_CHARS = 4000

# save_result 로 한 번의 실행에서 만들 수 있는 파일 수 상한(폭주 방지).
# saved_files 는 에이전트의 _SLIM_KEEP_KEYS 에 있어 길이와 무관하게 모델에 전달되므로
# _slim 의 32 제한에 묶이지 않는다 — 이 값은 순수하게 '한 번의 실행에서 디스크를
# 어지럽히지 않게' 하는 상한이다.
_MAX_SAVED_FILES = 20


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


def _clip_stdout(s: str) -> tuple[str, bool]:
    """stdout 을 상한까지 자른다. (잘린 문자열, 잘렸는지) 반환.

    잘린 경우 안내문을 덧붙인다 — 모델이 "출력이 왜 끊겼지" 하고 같은 코드를 다시
    돌리는 대신, 배열은 save_result 로 저장해야 한다는 걸 알게 하려는 것.
    """
    if len(s) <= _MAX_STDOUT_CHARS:
        return s, False
    return (
        s[:_MAX_STDOUT_CHARS]
        + f"\n\n...[stdout truncated at {_MAX_STDOUT_CHARS} chars of {len(s)} total]\n"
          "Do NOT print large arrays - the output is cut off and it wastes the context window. "
          "Save arrays with save_result(filename, intensity, raman_shift=...) instead; "
          "it returns the saved path. print() only small summaries (counts, peak positions, statistics).",
        True,
    )


def _sanitize_filename(name) -> str:
    """save_result 에 넘어온 파일명을 data/ 바로 아래의 단일 .csv 파일명으로 강제한다.

    생성 코드는 신뢰할 수 없으므로 디렉터리 성분을 통째로 버린다 — '../../x',
    'C:/Windows/x', 'sub/dir/x' 모두 마지막 조각만 남기고, 거기서 다시 경로
    구분자와 위험 문자를 제거한다. 즉 어떤 입력이 와도 data/<safe>.csv 밖으로는
    나갈 수 없다.
    """
    raw = str(name or "").strip()
    if not raw:
        raise ValueError("save_result: filename is empty.")
    # 구분자를 통일한 뒤 마지막 조각만 취한다(드라이브 문자·상위참조 제거).
    tail = raw.replace("\\", "/").rstrip("/").split("/")[-1]
    tail = tail.replace(":", "_")
    safe = "".join(c for c in tail if c.isalnum() or c in "._- ").strip(" .")
    if not safe:
        raise ValueError(f"save_result: filename {raw!r} has no usable characters.")
    if not safe.lower().endswith(".csv"):
        safe += ".csv"
    return safe


def _make_save_result(data_dir, saved: list, rel_prefix: str = "", start_index: int = 1):
    """샌드박스에 주입할 save_result 훅을 만든다.

    이 함수 자체는 '신뢰된 부모 코드'라서 open/Path 를 자유롭게 쓴다 — AST 금지 목록은
    exec 되는 생성 코드의 텍스트에만 적용된다. 생성 코드가 통제하는 것은 인자뿐이고,
    파일명은 _sanitize_filename 이, 경로는 data_dir 고정이 막는다.

    CSV 포맷(pixel_index, [raman_shift_cm-1,] [wavelength_nm,] intensity)은 고정이다 —
    같은 과제를 어느 경로로 풀든 산출물이 같아야 채점(참조 스펙트럼과의 수치 비교)이
    성립하고, load_spectrum 으로 그대로 다시 읽힌다. 실제 쓰기는
    spectrum_store.write_spectrum_csv 가 담당한다(2026-07-30: 측정 저장·배경 제거·이
    훅이 각자 헤더를 쓰던 것을 한 함수로 모았다).

    [rel_prefix / start_index — 세션 귀속]
    샌드박스는 별도 서브프로세스라 run_store 의 '현재 세션'을 공유하지 못한다. 그래서
    부모가 세션 디렉터리(data_dir)와 data/ 기준 상대경로 접두사(rel_prefix), 그리고
    이 세션에서 이미 쓴 개수(start_index)를 payload 로 넘겨준다. 파일명 앞에 순번을
    붙이는 이유는 목록만 보고 '어느 것이 최종 산출물인가'를 알 수 있게 하려는 것.
    """
    from backend.spectrum_store import write_spectrum_csv as _write_spectrum_csv

    def save_result(filename, intensity, raman_shift=None, wavelength_nm=None,
                    metadata=None) -> str:
        """계산된 스펙트럼을 세션 폴더에 저장하고 data/ 기준 상대경로를 돌려준다.

        intensity 는 필수, raman_shift / wavelength_nm 은 선택. 값은 float 그대로
        기록한다(포맷팅하지 않는다) — repr 왕복 정밀도가 유지되어야 하므로.
        """
        if len(saved) >= _MAX_SAVED_FILES:
            raise RuntimeError(
                f"save_result: too many files in one run (max {_MAX_SAVED_FILES}).")

        stem = _sanitize_filename(filename)[:-4]          # .csv 제거
        safe = f"{start_index + len(saved):02d}_{stem}.csv"

        def _col(v, label):
            """(아래 검증은 write_spectrum_csv 에도 있지만, 여기서 먼저 걸러야 생성
            코드에 '어느 인자가 문제인지' 알려 줄 수 있다.)"""
            if v is None:
                return None
            seq = [float(x) for x in v]
            if not seq:
                raise ValueError(f"save_result: {label} is empty.")
            return seq

        ints = _col(intensity, "intensity")
        if ints is None:
            raise ValueError("save_result: intensity is required.")
        shift = _col(raman_shift, "raman_shift")
        wl = _col(wavelength_nm, "wavelength_nm")
        for label, seq in (("raman_shift", shift), ("wavelength_nm", wl)):
            if seq is not None and len(seq) != len(ints):
                raise ValueError(
                    f"save_result: {label} has {len(seq)} points but intensity has "
                    f"{len(ints)} - they must match.")

        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / safe
        # 같은 세션 폴더에 apply_background_subtraction 도 파일을 쓴다. 두 쪽이 순번을
        # 따로 세므로(이쪽은 start_index+len(saved), 저쪽은 manifest 개수) 이름이
        # 겹칠 수 있다 — 덮어쓰지 않고 접미사를 붙인다.
        n = 2
        while path.exists():
            path = data_dir / f"{safe[:-4]}-{n}.csv"
            n += 1
        safe = path.name
        _write_spectrum_csv(path, intensity=ints, raman_shift=shift, wavelength_nm=wl)

        if metadata:
            # dict 가 아닌 게 와도 죽지 않게 — 저장 자체가 실패하면 안 된다.
            try:
                path.with_suffix(".json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
            except Exception:
                pass

        # data/ 기준 상대경로를 돌려준다 — 이 문자열을 그대로 load_spectrum 에 넘겨
        # 다시 읽을 수 있다(부모가 manifest 에도 이 경로로 인덱싱한다).
        rel = f"{rel_prefix}{safe}" if rel_prefix else safe
        saved.append({"path": rel, "num_points": len(ints),
                      "has_raman_shift": shift is not None})
        return rel

    return save_result


# ── 서브프로세스 진입점: python -m backend.analysis_sandbox <payload.json> ──────
def _main(payload_path: str) -> None:
    import io
    import contextlib
    import traceback
    from datetime import datetime

    from backend.spectrum_store import RESULTS_ROOT, URL_PREFIX

    # 자식의 stdout/stderr 는 Windows 에서 cp949 다. 여기에 cp949 로 표현 못 하는 문자가
    # 하나라도 섞이면 write 자체가 UnicodeEncodeError 로 죽고, 부모는 진짜 원인 대신
    # "결과 없이 종료"만 보게 된다(생성 코드의 에러 메시지가 통째로 사라진다).
    # 인코딩 불가 문자는 이스케이프로 흘려보내고 프로세스는 절대 죽지 않게 한다.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass

    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    code = payload["code"]

    buf = io.StringIO()
    result: dict = {"ok": False}
    # try 바깥에서 만든다 — save_result 로 파일을 쓴 뒤 코드가 죽어도, 무엇이
    # 이미 저장됐는지는 에러 응답에도 실려야 모델이 같은 파일을 또 쓰지 않는다.
    saved_files: list = []
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
            wl = it.get("wavelength_nm")
            if wl:
                d["wavelength_nm"] = np.asarray(wl, dtype=float)
            # Kinetic: (n_frames, n_pixels) 2D 배열로 올린다 — frames.mean(axis=0),
            # frames[:, px] 같은 자연스러운 코드가 바로 되도록. 프레임 길이가 서로 다르면
            # (frames_ragged) 2D 로 못 쌓으므로 리스트의 리스트 그대로 둔다.
            if "frames" in d:
                fr = d["frames"]
                if it.get("frames_ragged"):
                    d["frames"] = [np.asarray(f, dtype=float) for f in fr]
                elif fr:
                    d["frames"] = np.asarray(fr, dtype=float).reshape(len(fr), -1)
                    # kinetic 은 단일 intensity 가 없다. 프레임 평균을 채워 두면 '스펙트럼
                    # 하나'를 기대하는 기존 코드(피크 찾기 등)가 그대로 돌아간다.
                    # 평균이라는 사실은 플래그로 밝혀 둔다 — 모르고 쓰면 시간 정보가
                    # 사라진 것을 눈치채지 못한다.
                    if d["intensity"].size == 0:
                        d["intensity"] = d["frames"].mean(axis=0)
                        d["intensity_is_frame_mean"] = True
                else:
                    d["frames"] = np.zeros((0, 0), dtype=float)
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

        # save_result 훅 — 계산 결과 배열을 컨텍스트에 태우지 않고 바로 파일로 뺀다.
        # 저장 위치는 부모가 정한 세션 폴더(run_store) 다. 부모가 안 넘겼으면(단독 실행
        # 이나 구버전 payload) data/ 최상위로 떨어뜨려 최소한 유실은 막는다.
        _sdir = payload.get("spectra_dir")
        save_result = _make_save_result(
            Path(_sdir) if _sdir else (_PROJECT_ROOT / "data"),
            saved_files,
            rel_prefix=payload.get("spectra_rel_prefix") or "",
            start_index=int(payload.get("spectra_start_index") or 1),
        )

        ns = {
            "__builtins__": _safe_builtins(),
            "np": np, "plt": plt, "spectra": spectra, "files": files,
            "microscope_image": microscope_image, "image_extent": image_extent,
            "save_result": save_result,
        }
        with contextlib.redirect_stdout(buf):
            # 이 줄의 주석은 ASCII 로만 쓴다 — traceback 에 소스 줄이 그대로 실려
            # 모델에게 전달되므로, 비ASCII가 있으면 에러 메시지가 이스케이프로 더럽혀진다.
            exec(compile(code, "<analysis>", "exec"), ns)  # noqa: S102 (sandboxed)

        # 생성된 모든 그림을 data/results/<date>/<세션>/ 에 저장한다.
        # 세션 폴더 아래(측정 png/csv/json 과 같은 자리)에 두면 '이 문항이 만든 것'이
        # 폴더 하나로 모인다. URL_PREFIX 서빙은 StaticFiles 라 하위 폴더도 그대로
        # 나가므로 채팅 표시 계약은 그대로다.
        day = datetime.now().strftime("%Y-%m-%d")
        sess = payload.get("session_folder") or "_unassigned"
        out_dir = RESULTS_ROOT / day / sess
        out_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
        images = []
        for i, num in enumerate(plt.get_fignums()):
            fig = plt.figure(num)
            name = f"fig{stamp}_{i}.png"
            fig.savefig(out_dir / name, bbox_inches="tight")
            images.append(f"{URL_PREFIX}/{day}/{sess}/{name}")
        plt.close("all")

        out, clipped = _clip_stdout(buf.getvalue())
        result = {"ok": True, "stdout": out, "images": images,
                  "saved_files": saved_files}
        if clipped:
            result["stdout_truncated"] = True
    except Exception as e:
        out, clipped = _clip_stdout(buf.getvalue())
        result = {
            "ok": False,
            "stdout": out,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-1200:],
            "saved_files": saved_files,
        }
        if clipped:
            result["stdout_truncated"] = True
    # ensure_ascii 는 기본값(True) 그대로 둔다 — 비ASCII를 \uXXXX 로 이스케이프해
    # 순수 ASCII 로만 내보낸다. 부모의 json.loads 가 원래 문자로 복원하므로 무손실이고,
    # cp949 파이프를 타도 절대 깨지지 않는다. 여기서 ensure_ascii=False 를 쓰면
    # traceback 에 섞여 들어온 한글 주석 한 줄이 프로세스를 죽인다.
    sys.stdout.write("\n__ANALYSIS_RESULT__" + json.dumps(result))


# ── Kinetic 프레임 주입 ───────────────────────────────────────────────────────
# [왜 필요했는가 — 2026-08-01]
# 예전에는 주입식이 res['data'] 만 봤다. Kinetic 측정은 'data' 가 없고 'frames' 에
# 프레임별 배열이 들어가므로, spectra 항목은 mode='kinetic' 에 intensity=[] 인
# **조용히 빈 항목**이 됐다. 모델은 항목이 멀쩡히 보이니 자기 코드가 틀린 줄 알고
# ("zero-size array to reduction operation maximum") 고쳐 쓰기를 반복했다.
# 저장된 kinetic 을 사후 분석할 경로가 이것 말고 없다(load_spectrum 은 1D 전용).
#
# [축을 공유하는 이유]
# acquire_spectrum 은 프레임마다 raman_shift_cm-1 / wavelength_nm 을 각각 복사해 둔다.
# 두 축은 픽셀→파장 매핑이라 프레임이 달라도 값이 같다 — 실측으로 프레임당 43.7KB 중
# 약 2/3 가 이 중복이었다(3프레임 저장 파일 131KB). 주입할 때 축 1벌만 싣고 세기만
# 프레임별로 쌓으면 프레임당 ~14.6KB 로 떨어진다. 저장 포맷은 건드리지 않는다 —
# render_png / _write_csv 가 프레임별 축을 읽고 있으므로 여기(주입)에서만 정규화한다.
_MAX_KINETIC_FRAMES = 200          # 측정 1건당 주입 상한(≈3MB). 넘으면 잘라내고 알린다.


def _kinetic_payload(res: dict) -> dict:
    """kinetic 결과에서 샌드박스에 실을 {frames, raman_shift?, ...} 를 만든다.

    kinetic 이 아니면 빈 dict — 호출부가 그대로 update 하면 된다.
    """
    if res.get("mode") != "kinetic":
        return {}
    frames = res.get("frames") or []
    if not frames:
        return {"n_frames": 0, "frames": []}

    total = len(frames)
    kept = frames[:_MAX_KINETIC_FRAMES]
    # 축은 첫 프레임 것 1벌만 싣는다(위 주석). 프레임마다 길이가 다르면 2D 로 못 쌓으므로
    # 그때는 축을 싣지 않고 프레임도 짧은 쪽에 맞추지 않는다 — 판단은 모델에게 넘긴다.
    lengths = {len(fr.get("intensity") or []) for fr in kept}
    out: dict = {
        "n_frames": len(kept),
        "n_frames_total": total,
        "frames": [list(fr.get("intensity") or []) for fr in kept],
    }
    if len(lengths) == 1:
        first = kept[0]
        # 상위 raman_shift 가 비어 있으면(kinetic 결과에는 보통 없다) 프레임 것을 올린다.
        if not res.get("raman_shift_cm-1") and first.get("raman_shift_cm-1"):
            out["raman_shift"] = list(first["raman_shift_cm-1"])
        if first.get("wavelength_nm"):
            out["wavelength_nm"] = list(first["wavelength_nm"])
    else:
        out["frames_ragged"] = True
    if total > len(kept):
        out["frames_truncated"] = (
            f"Only the first {len(kept)} of {total} frames were loaded "
            f"(limit {_MAX_KINETIC_FRAMES} per measurement).")
    return out


# ── 부모(서버) 쪽 오케스트레이터 ───────────────────────────────────────────────
def run_analysis(code: str, date: str | None = None, names: list[str] | None = None,
                 title: str | None = None, timeout_sec: int = 60,
                 file_ids: list[str] | None = None) -> dict:
    """저장된 측정 데이터(+첨부 파일)를 대상으로 분석/시각화 코드를 안전 실행한다.

    file_ids 를 주면 그 업로드 파일들이 파싱되어 샌드박스의 `files` 변수로 들어간다.
    측정 데이터(spectra)와 첨부 파일(files)은 동시에 쓸 수 있다 — 예를 들어 사용자가
    올린 참조 스펙트럼과 방금 측정한 스펙트럼을 한 그림에 겹쳐 그리는 식.

    [인자 정규화를 여기서 하는 이유 — 2026-07-30]
    이 함수로 오는 길이 둘이다: file_tools.FILE_DISPATCH(에이전트가 실제로 타는 길)와
    raman_tools.TOOL_DISPATCH. 예전에는 file_tools 쪽만 '모델이 리스트 대신 문자열
    하나를 준 경우'를 흡수해서, 같은 인자가 어느 길로 오느냐에 따라 통하기도 하고
    TypeError 로 죽기도 했다. 정규화를 함수 안으로 들여 두 경로를 같게 만든다.

    Returns
    -------
    dict — {ok, stdout, image_count, saved?{title,image_url}, images?, error?}
           그림이 생기면 saved.image_url 로 채팅에 표시된다(spectrum_event 재사용).
    """
    import subprocess
    import tempfile
    from backend.spectrum_store import (list_results, latest_scene,
                                        session_folder as _session_folder)
    from backend.upload_store import load_upload

    # ── 인자 정규화(두 호출 경로 공통) ──
    # 모델은 리스트 자리에 문자열 하나를 주거나, 빈 문자열로 '없음'을 표현하기도 한다.
    # 빈 값을 걸러내지 않으면 ''를 파일 id 로 알고 조회하다 실패한다.
    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v if str(x).strip()]

    if not isinstance(code, str) or not code.strip():
        return {"ok": False, "error": "code is empty. Provide the Python analysis code to run."}
    date = (date.strip() or None) if isinstance(date, str) else date
    title = (title.strip() or None) if isinstance(title, str) else title
    names = _as_list(names) or None
    file_ids = _as_list(file_ids)

    items = list_results(date)
    if names:
        want = set(names)
        items = [it for it in items if it["base"] in want]

    spectra = []
    for it in items:
        res = it["result"]
        intensity = res.get("data") or res.get("intensity") or []
        entry = {
            "base": it["base"], "title": it["title"],
            "x": it["meta"].get("x"), "y": it["meta"].get("y"),
            "power": res.get("laser_power_pct"), "exposure": res.get("exposure_time"),
            "mode": res.get("mode"),
            "raman_shift": res.get("raman_shift_cm-1"),
            "intensity": list(intensity),
        }
        entry.update(_kinetic_payload(res))
        spectra.append(entry)

    # 첨부 파일을 '여기서(신뢰된 부모)' 읽어 둔다 — 샌드박스 코드는 파일을 열 수 없으므로
    # 이 단계에서 파싱된 값만이 생성 코드가 볼 수 있는 전부다.
    uploads = []
    for fid in file_ids:
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

    # 세션 귀속 정보를 자식에게 넘긴다 — 자식은 별도 프로세스라 run_store 의 현재
    # 세션을 못 보므로, 저장 디렉터리·상대경로 접두사·시작 순번을 부모가 계산해 준다.
    from backend.agents import run_store
    _spectra_dir = run_store.session_dir() / run_store.KIND_SPECTRA
    _rel_prefix = f"{run_store.current()['rel_dir']}/{run_store.KIND_SPECTRA}/"
    payload = {"code": code, "spectra": spectra, "scene_path": latest_scene(date),
               "uploads": uploads,
               "spectra_dir": str(_spectra_dir),
               "spectra_rel_prefix": _rel_prefix,
               "spectra_start_index": run_store.next_index(run_store.KIND_SPECTRA),
               "session_folder": _session_folder()}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        payload_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "backend.analysis_sandbox", payload_path],
            # errors="replace": 자식 stderr 에 로케일로 못 읽는 바이트가 섞여도 부모의
            # 읽기 스레드가 UnicodeDecodeError 로 죽지 않게 한다
            # (encoding 은 자식과 같은 로케일 기본값을 그대로 쓴다).
            capture_output=True, text=True, errors="replace", timeout=timeout_sec,
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
        # 실패해도 saved_files 는 실어 보낸다 — 이미 쓴 파일을 모델이 알아야 한다.
        err = {"ok": False, "error": payload.get("error", "Analysis failed"),
               "stdout": payload.get("stdout", ""), "trace": payload.get("trace", "")}
        if payload.get("saved_files"):
            err["saved_files"] = payload["saved_files"]
            # 코드가 죽기 전에 이미 쓴 파일도 manifest 에 남긴다 — 디스크에는 있는데
            # 인덱스에는 없는 '유령 파일'이 생기지 않게.
            for _sf in payload["saved_files"]:
                run_store.record(run_store.KIND_SPECTRA, _sf.get("path", ""),
                                 num_points=_sf.get("num_points"), partial=True)
        if payload.get("stdout_truncated"):
            err["stdout_truncated"] = True
        return err

    images = payload.get("images", [])
    resp = {"ok": True, "stdout": payload.get("stdout", ""),
            "image_count": len(images), "images": images}
    if payload.get("stdout_truncated"):
        resp["stdout_truncated"] = True
    # save_result 로 쓴 파일 목록. 여기 경로가 곧 load_spectrum 이 읽는 경로다.
    if payload.get("saved_files"):
        resp["saved_files"] = payload["saved_files"]
        for _sf in payload["saved_files"]:
            run_store.record(run_store.KIND_SPECTRA, _sf.get("path", ""),
                             num_points=_sf.get("num_points"),
                             has_raman_shift=_sf.get("has_raman_shift"))
    for _img in images:
        run_store.record(run_store.KIND_FIGURE, _img, title=title or "Analysis result")
    if images:
        resp["saved"] = {"title": title or "Analysis result", "image_url": images[0]}
    return resp


if __name__ == "__main__":
    _main(sys.argv[1])
