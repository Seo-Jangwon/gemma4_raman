# -*- coding: utf-8 -*-
"""
[역할] 사용자가 채팅에 첨부한 데이터 파일(csv/tsv/txt/dat/xlsx/xls)의 저장·파싱 계층.

  spectrum_store.py 가 '장비가 측정한 결과'를 다루는 것과 대칭으로, 이 모듈은
  '사람이 밖에서 가져온 데이터'를 다룬다. 하드웨어 import 가 전혀 없어서 장비
  미연결 개발 PC 에서도 그대로 동작한다.

  저장 레이아웃 (spectrum_store 와 동일한 날짜 폴더 규칙):
    data/uploads/<YYYY-MM-DD>/<HHMMSS_mmm>_<원본파일명>

  file_id 는 그 상대 키다: "<YYYY-MM-DD>/<HHMMSS_mmm>_<원본파일명>".
  절대경로를 노출하지 않으므로 LLM 이 임의 경로를 읽어달라고 할 방법이 없다.

  [왜 세션이 아니라 날짜로 묶는가]
  프론트의 session_id 는 '첫 응답을 받은 뒤에야' 서버에서 발급된다. 즉 첫 메시지에
  파일을 붙이는 순간에는 세션이 아직 없다. 날짜 폴더면 그 부트스트랩 문제가 아예
  없고, 이미 프로젝트 전체가 쓰는 규칙이라 결과물 위치도 예측 가능하다.

[설계 원칙 — 파서는 멍청해야 한다]
  이 모듈은 "1열이 라만 shift 다" 같은 판단을 절대 하지 않는다. 표를 있는 그대로
  읽어 형태·컬럼명·통계만 돌려주고, '스펙트럼인지 기타 정보인지'는 에이전트(LLM)가
  inspect_upload 결과를 보고 스스로 판단한다.

  여기에 휴리스틱을 넣으면 single_agent_AILA 의 baseline 설계 원칙(비-LLM 판단
  코드를 두지 않는다)이 깨지고, AILA↔CoALA 비교의 독립변수가 '내가 넣은 컬럼 추측
  로직'으로 오염된다. 판단은 전부 모델 몫이다.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

# data/uploads — spectrum_store.RESULTS_ROOT(data/results) 와 같은 부모.
UPLOADS_ROOT = Path(__file__).resolve().parent.parent / "data" / "uploads"

# 허용 확장자 — '표 형태 데이터'만. 논문/프로토콜 문서(pdf/md)는 지식베이스로
# 가야 하므로 여기가 아니라 /api/kb/upload 담당이다.
ALLOWED_SUFFIXES = {".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls"}
_EXCEL_SUFFIXES = {".xlsx", ".xls"}

# 병적으로 큰 파일이 payload JSON 을 통째로 메모리에 올리는 것을 막는 상한.
_MAX_LOAD_ROWS = 200_000
# 컨텍스트 보호: inspect 결과에 실을 컬럼 수 상한. 에이전트의 _slim()이 길이 32
# 초과 리스트를 통째로 버리므로, 잘려서 사라지느니 여기서 먼저 줄이고 알린다.
_MAX_REPORT_COLS = 32


# ──────────────────────────────────────────────────────────────────────────────
# 경로 · 이름
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_date(date: str | None) -> str:
    return date or datetime.now().strftime("%Y-%m-%d")


def _safe_name(raw: str) -> str:
    """원본 파일명을 파일시스템 안전 문자열로. 확장자는 보존한다(종류 판별에 쓰임)."""
    p = Path(raw or "")
    suffix = p.suffix.lower()
    stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in p.stem)
    stem = stem.strip("_")[:60] or "upload"
    return stem + suffix


def _resolve_path(file_id: str) -> Path:
    """file_id → 실제 경로. 경로 탈출을 여기서 전부 막는다.

    받는 형태는 두 가지:
      "<YYYY-MM-DD>/<name>"  정식 file_id
      "<name>"               날짜를 빠뜨린 경우 — 최근 날짜부터 역순으로 찾아준다.
                             (모델이 날짜 접두사를 흘리는 일이 흔해서 허용한다)
    """
    raw = str(file_id or "").replace("\\", "/")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ValueError(f"Invalid file_id: {file_id!r}")

    if len(parts) == 1:
        name = parts[0]
        if name != Path(name).name:
            raise ValueError(f"Invalid file_id: {file_id!r}")
        if not UPLOADS_ROOT.exists():
            raise FileNotFoundError(f"No uploaded file named {name!r}.")
        for day_dir in sorted(UPLOADS_ROOT.iterdir(), reverse=True):
            cand = day_dir / name
            if day_dir.is_dir() and cand.is_file():
                return cand
        raise FileNotFoundError(f"No uploaded file named {name!r}.")

    if len(parts) != 2:
        raise ValueError(f"Invalid file_id: {file_id!r} (expected '<YYYY-MM-DD>/<filename>')")

    date, name = parts
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or name != Path(name).name:
        raise ValueError(f"Invalid file_id: {file_id!r} (expected '<YYYY-MM-DD>/<filename>')")

    path = UPLOADS_ROOT / date / name
    if not path.is_file():
        raise FileNotFoundError(f"Uploaded file not found: {file_id}")
    return path


def _kind(path: Path) -> str:
    return "excel" if path.suffix.lower() in _EXCEL_SUFFIXES else "text"


# ──────────────────────────────────────────────────────────────────────────────
# 저장 · 목록
# ──────────────────────────────────────────────────────────────────────────────

def save_upload(filename: str, data: bytes) -> dict:
    """업로드된 바이트를 오늘 날짜 폴더에 저장하고 file_id 를 돌려준다."""
    name = _safe_name(filename)
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )

    day = _resolve_date(None)
    out_dir = UPLOADS_ROOT / day
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    stamp = now.strftime("%H%M%S_") + f"{now.microsecond // 1000:03d}"
    stored = f"{stamp}_{name}"
    path = out_dir / stored
    path.write_bytes(data)

    return {
        "ok": True,
        "file_id": f"{day}/{stored}",
        "filename": name,
        "bytes": path.stat().st_size,
    }


def list_uploads(date: str | None = None) -> list[dict]:
    """해당 날짜(기본 오늘)에 올라온 파일 목록을 시각 순으로."""
    day = _resolve_date(date)
    day_dir = UPLOADS_ROOT / day
    if not day_dir.exists():
        return []
    items = []
    for p in sorted(day_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        items.append({
            "file_id": f"{day}/{p.name}",
            "filename": p.name,
            "kind": _kind(p),
            "bytes": p.stat().st_size,
        })
    return items


# ──────────────────────────────────────────────────────────────────────────────
# 파싱 — pandas 로 표를 '있는 그대로' 읽는다 (의미 해석은 하지 않는다)
# ──────────────────────────────────────────────────────────────────────────────

def _looks_numeric(values) -> bool:
    """이 행의 값들이 전부 숫자로 읽히는가 — 헤더 유무 판정에만 쓴다."""
    seen = False
    for v in values:
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            continue
        try:
            float(s)
        except ValueError:
            return False
        seen = True
    return seen


def _read_text_table(path: Path):
    """csv/tsv/txt/dat 를 DataFrame 으로. 구분자와 헤더 유무를 시도해서 맞춘다.

    라만 데이터는 쉼표·탭·공백 구분에 헤더가 있기도 없기도 해서, 하나로 못 박으면
    자주 깨진다. 후보 구분자를 순서대로 시도하고 '열이 가장 많이 쪼개진' 해석을
    채택한다. '#' 로 시작하는 주석 줄은 계측기 파일에서 흔하므로 건너뛴다.
    """
    import pandas as pd

    attempts = (
        {"sep": None, "engine": "python"},     # 스니핑(쉼표/탭/세미콜론 등)
        {"sep": r"\s+", "engine": "python"},   # 공백 구분
        {"sep": ",", "engine": "python"},
    )
    best = None
    last_err: Exception | None = None
    for kwargs in attempts:
        try:
            probe = pd.read_csv(path, nrows=1, header=None, comment="#", **kwargs)
            header = None if _looks_numeric(probe.iloc[0].tolist()) else 0
            df = pd.read_csv(path, header=header, comment="#", **kwargs)
        except Exception as e:
            last_err = e
            continue
        if df.shape[1] == 0 or df.shape[0] == 0:
            continue
        if header is None:
            df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
        if best is None or df.shape[1] > best.shape[1]:
            best = df
        if best.shape[1] > 1:
            break

    if best is None:
        raise ValueError(f"Could not parse as a table: {last_err}")
    return best


def _read_excel_table(path: Path, sheet: str | None):
    """xlsx/xls 의 한 시트를 DataFrame 으로. 반환: (df, 전체 시트명, 실제 읽은 시트명)."""
    import pandas as pd

    xls = pd.ExcelFile(path)
    sheets = list(xls.sheet_names)
    target = sheet if (sheet and sheet in sheets) else (sheets[0] if sheets else 0)

    probe = pd.read_excel(xls, sheet_name=target, header=None, nrows=1)
    header = None if (len(probe) and _looks_numeric(probe.iloc[0].tolist())) else 0
    df = pd.read_excel(xls, sheet_name=target, header=header)
    if header is None:
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
    return df, sheets, str(target)


def _read_table(path: Path, sheet: str | None):
    if _kind(path) == "excel":
        return _read_excel_table(path, sheet)
    return _read_text_table(path), [], None


def _is_numeric_col(series) -> bool:
    import pandas as pd
    return bool(pd.api.types.is_numeric_dtype(series))


def _cell(v):
    """head 표시에 쓸 값 하나를 JSON 안전한 형태로(긴 실수는 반올림, 나머지는 문자열)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)[:80]
    if f != f:          # NaN
        return None
    return round(f, 6)


# ──────────────────────────────────────────────────────────────────────────────
# 공개 API — 도구 계층(agents/file_tools.py)이 쓰는 두 함수
# ──────────────────────────────────────────────────────────────────────────────

def inspect_upload(file_id: str, sheet: str | None = None, max_rows: int = 5) -> dict:
    """파일의 '구조'만 요약해서 돌려준다 — 전체 데이터는 절대 싣지 않는다.

    에이전트가 이걸 보고 "어느 컬럼이 라만 shift 인지, 스펙트럼이 들어있긴 한지,
    아니면 메타데이터 표인지"를 스스로 판단한다. 실제 계산은 run_analysis 에서
    file_ids 로 전체 데이터를 받아 수행한다.

    컨텍스트 폭발 방지: head 는 max_rows 행, 컬럼 정보는 32개까지만.
    """
    path = _resolve_path(file_id)
    df, sheets, used_sheet = _read_table(path, sheet)

    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    cols = [str(c) for c in df.columns]
    shown = cols[:_MAX_REPORT_COLS]

    dtypes = {c: ("numeric" if _is_numeric_col(df[orig]) else "text")
              for c, orig in zip(shown, list(df.columns)[:_MAX_REPORT_COLS])}

    stats = {}
    for c, orig in zip(shown, list(df.columns)[:_MAX_REPORT_COLS]):
        if dtypes[c] != "numeric":
            continue
        s = df[orig].dropna()
        if s.empty:
            continue
        stats[c] = {
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
            "mean": round(float(s.mean()), 6),
            "n_valid": int(s.size),
        }

    head_rows = []
    for _, row in df.head(max(1, min(int(max_rows or 5), 20))).iterrows():
        head_rows.append([_cell(row[orig]) for orig in list(df.columns)[:_MAX_REPORT_COLS]])

    out = {
        "ok": True,
        "file_id": file_id,
        "filename": path.name,
        "kind": _kind(path),
        "bytes": path.stat().st_size,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": shown,
        "dtypes": dtypes,
        "stats": stats,
        "head": head_rows,
        "note": (
            "This is a structural summary only. Decide yourself what each column means "
            "(e.g. whether a numeric column is Raman shift in cm-1, intensity, wavelength, "
            "a coordinate, or unrelated metadata) - nothing here has been interpreted for you. "
            "To compute on the full data, call run_analysis with file_ids=['" + str(file_id) + "']."
        ),
    }
    if sheets:
        out["sheets"] = sheets[:_MAX_REPORT_COLS]
        out["sheet"] = used_sheet
    if n_cols > _MAX_REPORT_COLS:
        out["columns_truncated"] = True
        out["note"] += f" (Only the first {_MAX_REPORT_COLS} of {n_cols} columns are listed here.)"
    return out


def load_upload(file_id: str, sheet: str | None = None) -> dict:
    """샌드박스 주입용 전체 데이터. 숫자 컬럼은 float 리스트, 나머지는 문자열 리스트.

    반환 dict 는 그대로 JSON payload 에 실려 analysis_sandbox 자식 프로세스로 건너간다
    (거기서 숫자 컬럼만 np.ndarray 로 승격된다). 이 경로가 '신뢰된 부모가 읽어서
    변수로 주입한다'는 샌드박스 원칙을 지키는 지점이다 — 생성된 코드는 파일을 열지 않는다.
    """
    path = _resolve_path(file_id)
    df, sheets, used_sheet = _read_table(path, sheet)

    truncated = False
    if df.shape[0] > _MAX_LOAD_ROWS:
        df = df.head(_MAX_LOAD_ROWS)
        truncated = True

    numeric: dict[str, list] = {}
    text: dict[str, list] = {}
    for orig in df.columns:
        name = str(orig)
        col = df[orig]
        if _is_numeric_col(col):
            numeric[name] = [None if v != v else float(v) for v in col.tolist()]
        else:
            text[name] = ["" if v != v else str(v) for v in col.tolist()]

    return {
        "file_id": file_id,
        "filename": path.name,
        "sheet": used_sheet,
        "sheets": sheets,
        "columns": [str(c) for c in df.columns],
        "n_rows": int(df.shape[0]),
        "numeric": numeric,
        "text": text,
        "truncated": truncated,
    }
