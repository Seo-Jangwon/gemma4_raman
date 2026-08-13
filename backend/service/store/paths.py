# -*- coding: utf-8 -*-
"""파일 지시자(file_id) → 실제 경로. **해석기는 이 파일 하나뿐이다.**

┌─────────────────────────────────────────────────────────────────────────────┐
│  file_id 형식 —  "<뿌리>:<뿌리 기준 상대경로>"                                │
│                                                                             │
│    uploads:2026-08-12/N05.csv              data/uploads/2026-08-12/N05.csv  │
│    results:2026-08-12/_microscope_1401.png data/results/2026-08-12/...      │
│    results:2026-08-12/<세션>/1408_x37.csv  data/results/2026-08-12/<세션>/…  │
│    runs:<세션>/spectra/01_corrected.csv    data/runs/<세션>/spectra/…        │
└─────────────────────────────────────────────────────────────────────────────┘

[왜 뿌리를 id 에 박는가 — 2026-08-12]
예전에는 뿌리가 셋인데 id 는 전부 "<날짜>/<이름>" 한 모양이었다. 그래서 **눈으로 구별이
안 되는 두 id 가 서로 다른 폴더를 가리켰다**:

    2026-08-12/230456_data.csv            → data/uploads/
    2026-08-12/_microscope_224141.png     → data/results/

그리고 도구마다 자기가 아는 뿌리만 뒤졌다 — 해석기가 넷이었다(load_spectrum,
inspect_file, view_image, run_analysis). 목록 도구가 준 id 를 그대로 다른 도구에 넘기면
"File not found" 가 났고, 모델 입장에서는 빠져나갈 방법이 없었다. 2026-08-07 실행에서
N05·T039·T064·T070·T112·T115 여섯 문항이 여기서 막혔다. 막힌 뒤에는 run_analysis 로
로딩을 손수 구현하게 되는데, 그렇게 길어진 코드가 tool call JSON 유실(2,200자 이상에서
9.1%)을 부르는 경로이기도 했다.

접두를 박으면 id 만 봐도 어디 것인지 알 수 있고, 해석이 한 곳으로 모인다. 도구를 새로
만들어도 해석기가 늘지 않는다.

[접두 없는 옛 id 도 그대로 받는다]
기존 호출·프롬프트·벤치 로그가 전부 접두 없는 형태다. 그래서 접두가 없으면 예전 규칙대로
data/ → uploads → results → runs 순으로 찾아 준다. **동작이 바뀌는 호출은 없다.**

[못 찾거나 종류가 다르면 '어느 도구를 쓰라'고 말해 준다]
모델이 읽는 메시지다. "File not found" 만 주면 같은 호출을 반복하거나 스스로 로더를
짜기 시작한다. 어디를 뒤졌는지, 그 파일이 어떤 종류인지, 그러면 어느 도구를 써야 하는지
까지 적는다.

    python backend/service/store/paths.py      자체 검사
"""
from __future__ import annotations

import re
from pathlib import Path

from backend.service.store import DATA_ROOT

# ── 뿌리 3종. 다른 모듈은 이걸 import 해서 쓴다(각자 DATA_ROOT/"..." 를 다시 적지 않는다) ──
UPLOADS_ROOT = DATA_ROOT / "uploads"      # 사용자가 채팅에 첨부한 것
RESULTS_ROOT = DATA_ROOT / "results"      # 측정 결과·플롯·현미경 캡처
RUNS_ROOT = DATA_ROOT / "runs"            # 에이전트가 세션 중 만든 산출물

ROOTS: dict[str, Path] = {
    "uploads": UPLOADS_ROOT,
    "results": RESULTS_ROOT,
    "runs": RUNS_ROOT,
}

#: 프론트가 결과 파일을 받아가는 공개 URL 접두. 모델이 도구 결과에서 image_url 을 집어
#: 오는 일이 흔해서, file_id 자리에 URL 을 줘도 같은 파일로 풀리게 한다.
URL_PREFIX = "/api/results"

#: 확장자 → 종류. **open_file 이 무엇을 돌려줄지가 여기서 갈린다** — 분기표는 이 하나뿐이다.
#: 순서가 곧 우선순위다(.csv 처럼 겹칠 수 있는 확장자를 위해 dict 순서로 못 박는다).
KIND_SUFFIXES: dict[str, set[str]] = {
    "image": {".png", ".jpg", ".jpeg"},
    "table": {".csv", ".tsv", ".txt", ".dat", ".xlsx", ".xls"},
    "json":  {".json"},
}

#: 그 종류를 읽는 도구. 지금은 전부 open_file 이다 — 종류를 모델이 고르지 않게 하려고
#: 합쳤기 때문이다(2026-08-12). 예전에는 view_image / inspect_file / load_spectrum 셋이었고,
#: 모델이 id 만 보고 어느 도구인지 맞혀야 해서 "id 는 맞는데 도구가 틀린" 실패가 났다.
KIND_TOOL = {k: "open_file" for k in KIND_SUFFIXES}


def kind_of(path: Path | str) -> str | None:
    """확장자로 종류를 정한다. 모르는 확장자면 None.

    KIND_SUFFIXES 의 **선언 순서가 우선순위**다. '.csv' 처럼 여러 종류에 걸칠 수 있는
    확장자가 있어도 결과가 하나로 정해진다 — 분기가 호출부마다 갈리지 않게.
    """
    suf = Path(path).suffix.lower()
    return next((k for k, sufs in KIND_SUFFIXES.items() if suf in sufs), None)

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


class FileIdError(ValueError):
    """file_id 를 풀 수 없다. **메시지는 모델이 그대로 읽는다** — 다음 행동까지 적을 것."""


def make_id(root: str, *parts: str) -> str:
    """뿌리와 상대경로 조각들로 file_id 를 만든다. 저장하는 쪽이 쓴다.

    make_id("results", "2026-08-12", "_microscope_1401.png")
        -> "results:2026-08-12/_microscope_1401.png"
    """
    if root not in ROOTS:
        raise ValueError(f"Unknown root {root!r} (expected one of {sorted(ROOTS)})")
    rel = "/".join(str(p).strip("/\\") for p in parts if str(p).strip("/\\"))
    return f"{root}:{rel}"


def split_id(file_id: str) -> tuple[str | None, str]:
    """file_id 를 (뿌리, 상대경로) 로 나눈다. 접두가 없으면 (None, 원문).

    윈도우 드라이브 문자("C:\\...")를 뿌리 접두로 오해하지 않는다 — ROOTS 에 있는
    이름일 때만 접두로 본다.
    """
    raw = str(file_id or "").strip().replace("\\", "/")
    head, sep, tail = raw.partition(":")
    if sep and head.strip().lower() in ROOTS:
        return head.strip().lower(), tail.lstrip("/")
    return None, raw


def _safe_parts(rel: str) -> list[str]:
    """상대경로를 조각으로. 경로 탈출을 여기서 전부 막는다."""
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or p != Path(p).name for p in parts):
        raise FileIdError(f"Invalid file_id: {rel!r} (it must be a plain relative path)")
    return parts


def _known_suffixes() -> str:
    return ", ".join(sorted({s for sufs in KIND_SUFFIXES.values() for s in sufs}))


def _id_of(path: Path) -> str:
    """실제 경로 → 정식 file_id. 어느 뿌리 밑인지 되짚는다(못 찾으면 절대경로 그대로)."""
    for name, root in ROOTS.items():
        try:
            return f"{name}:{path.relative_to(root).as_posix()}"
        except ValueError:
            continue
    return str(path)


def _check_kind(path: Path, kind: str | None, file_id: str) -> Path:
    """요구한 종류와 확장자가 맞는지.

    kind 를 주는 곳은 이제 내부 호출뿐이다(예: upload_store 가 표 파서를 부르기 전에).
    모델이 부르는 open_file 은 kind 를 주지 않고 kind_of() 로 **알아서 분기**한다 —
    종류를 고르는 부담을 모델에게 지우지 않는 것이 이 설계의 요점이다.
    """
    if kind is None:
        return path
    allowed = KIND_SUFFIXES.get(kind)
    if allowed is None:
        raise ValueError(f"Unknown kind {kind!r} (expected one of {sorted(KIND_SUFFIXES)})")
    if path.suffix.lower() in allowed:
        return path
    raise FileIdError(
        f"'{file_id}' is a {path.suffix or 'no-extension'} file, but a "
        f"{kind} file was required here (accepted: {', '.join(sorted(allowed))}).")


def _owner_session(path: Path) -> tuple[str, str | None]:
    """경로가 어느 뿌리의 누구 것인지 → (뿌리 이름, 소유 세션 라벨).

    세션 라벨이 None 이면 '세션 소유가 아님'(uploads), 빈 문자열이면 '귀속 불명'
    (results 날짜 폴더 직속 — 구버전 파일). 뿌리 밖이면 ("", "").
    """
    for name, root in ROOTS.items():
        try:
            parts = path.relative_to(root).parts
        except ValueError:
            continue
        if name == "uploads":
            return name, None                      # 사용자 첨부 — 세션 소유가 아니다
        if name == "runs":
            return name, parts[0] if parts else ""
        # results/<날짜>/<세션>/<이름> 이면 세션이 있고, results/<날짜>/<이름> 이면 없다
        return name, parts[1] if len(parts) >= 3 else ""
    return "", ""


def _accept(path: Path, kind: str | None, file_id: str) -> Path:
    """resolve 의 **유일한 반환 지점** — 확장자 검사 + 세션 격리 검사.

    resolve 는 반환 지점이 여럿이라(접두 있는 id·data/ 상대·절대경로·이름만 폴백)
    검사를 각 지점에 흩어 두면 폴백 하나가 늘 때마다 우회로가 생긴다. 전부 여기를
    지나게 해서, 어떤 경로로 찾아냈든 격리 규칙이 똑같이 걸리게 한다.
    """
    path = _check_kind(path, kind, file_id)

    from backend.service.store import run_store          # 순환 import 회피(지연)
    mine = run_store.isolated_label()
    if mine is None:                                     # 격리 OFF(llm_config 로 끈 경우)
        return path

    root, owner = _owner_session(path)
    if owner is None:                                    # uploads — 항상 허용
        return path
    if owner == mine:
        return path
    if not root:
        raise FileIdError(
            f"'{file_id}' points outside the managed data folders, and this run may only read "
            f"its own session's files. Use an id from list_results(), list_session_artifacts() "
            f"or list_uploaded_files().")
    raise FileIdError(
        f"'{file_id}' belongs to "
        + (f"another session ({owner})" if owner else "no session (an older file)")
        + f", and this run may only read its own session `{mine}`. "
          f"List your own files with list_results() or list_session_artifacts(), "
          f"or measure it yourself in this run.")


def resolve(file_id: str, kind: str | None = None) -> Path:
    """file_id → 실제 경로. **읽기 전용 — 파일이 없으면 올린다.**

    Parameters
    ----------
    file_id : "<뿌리>:<상대경로>" 가 정식. 아래도 전부 받는다(하위호환):
              · 접두 없는 "<날짜>/<이름>"  — data/ → uploads → results → runs 순으로 탐색
              · data/ 기준 상대경로 "results/2026-08-12/<세션>/x.csv"
              · 절대경로
              · 공개 URL "/api/results/2026-08-12/x.png"
    kind    : "image" | "table" | "spectrum" | None. 주면 확장자를 검사하고, 어긋나면
              어느 도구를 써야 하는지 알려 준다.

    Raises
    ------
    FileIdError — 형식 오류 · 못 찾음 · 종류 불일치. 메시지가 곧 모델에게 갈 문장이다.
    """
    raw = str(file_id or "").strip()
    if not raw:
        raise FileIdError("file_id is empty. Get one from list_uploaded_files, "
                          "list_results or an earlier tool result.")

    # 공개 URL 형태 → results 뿌리
    if raw.replace("\\", "/").startswith(URL_PREFIX):
        raw = "results:" + raw.replace("\\", "/")[len(URL_PREFIX):].lstrip("/")

    root, rel = split_id(raw)

    # 절대경로는 그대로 통과시킨다(load_spectrum 이 예전부터 받아 온 형태).
    if root is None and Path(rel).is_absolute():
        p = Path(rel)
        if not p.is_file():
            raise FileIdError(f"File not found: {p}")
        return _accept(p, kind, file_id)

    # 접두가 있으면 그 뿌리 하나만 본다 — 남의 뿌리를 뒤지지 않는다.
    if root is not None:
        path = ROOTS[root].joinpath(*_safe_parts(rel))
        if not path.is_file():
            raise FileIdError(
                f"Not found under '{root}': {rel}. The id names the {root} area but no such "
                f"file is there. List what exists with "
                f"{'list_uploaded_files' if root == 'uploads' else 'list_results' if root == 'results' else 'list_session_artifacts'}().")
        return _accept(path, kind, file_id)

    # ── 접두 없음: 옛 규칙 그대로, data/ 를 먼저 보고 뿌리들을 차례로 본다 ──────────
    parts = _safe_parts(rel)
    local = DATA_ROOT.joinpath(*parts)
    if local.is_file():
        return _accept(local, kind, file_id)

    for name, root_dir in ROOTS.items():
        cand = root_dir.joinpath(*parts)
        if cand.is_file():
            return _accept(cand, kind, file_id)

    # 날짜 폴더를 빠뜨린 이름만 넘어오는 일이 잦다 — 최근 날짜부터 되짚어 준다.
    if len(parts) == 1:
        for name, root_dir in ROOTS.items():
            if not root_dir.is_dir():
                continue
            for day in sorted((d for d in root_dir.iterdir() if d.is_dir()), reverse=True):
                cand = day / parts[0]
                if cand.is_file():
                    return _accept(cand, kind, file_id)

    raise FileIdError(
        f"No file matches the id {file_id!r}. Looked under data/ and in "
        f"{', '.join(ROOTS)}. Ids look like 'uploads:2026-08-12/N05.csv' or "
        f"'results:2026-08-12/_microscope_1401.png' - get an exact one from "
        f"list_uploaded_files(), list_results() or list_session_artifacts().")


def try_resolve(file_id: str, kind: str | None = None) -> tuple[Path | None, str]:
    """resolve 의 예외 없는 판. (경로, 실패사유) 를 돌려준다 — 둘 중 하나는 항상 빈다."""
    try:
        return resolve(file_id, kind), ""
    except (FileIdError, ValueError) as e:
        return None, str(e)


__all__ = ["DATA_ROOT", "UPLOADS_ROOT", "RESULTS_ROOT", "RUNS_ROOT", "ROOTS", "URL_PREFIX",
           "KIND_SUFFIXES", "KIND_TOOL", "FileIdError",
           "make_id", "split_id", "kind_of", "resolve", "try_resolve"]


# ──────────────────────────────────────────────────────────────────────────────
# 자체 점검:  python backend/service/store/paths.py
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="paths_selfcheck_"))
    try:
        # 진짜 data/ 를 건드리지 않고 임시 트리로 갈아 끼운다.
        DATA_ROOT = tmp                                          # noqa: F811
        UPLOADS_ROOT, RESULTS_ROOT, RUNS_ROOT = tmp / "uploads", tmp / "results", tmp / "runs"
        ROOTS = {"uploads": UPLOADS_ROOT, "results": RESULTS_ROOT, "runs": RUNS_ROOT}

        for p, body in [
            (UPLOADS_ROOT / "2026-08-12" / "N05.csv", "a,b\n1,2\n"),
            (RESULTS_ROOT / "2026-08-12" / "_microscope_1401.png", "PNG"),
            (RESULTS_ROOT / "2026-08-12" / "sess" / "1408_x37.csv", "x\n"),
            (RUNS_ROOT / "sess" / "spectra" / "01_corrected.csv", "y\n"),
        ]:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")

        # 접두 있는 정식 id
        assert resolve("uploads:2026-08-12/N05.csv").name == "N05.csv"
        assert resolve("results:2026-08-12/_microscope_1401.png", "image").suffix == ".png"
        assert resolve("results:2026-08-12/sess/1408_x37.csv", "table").name == "1408_x37.csv"
        assert resolve("runs:sess/spectra/01_corrected.csv").parent.name == "spectra"

        # 하위호환 — 접두 없는 옛 id 가 여전히 풀린다
        assert resolve("2026-08-12/N05.csv").name == "N05.csv"           # uploads 로 폴백
        assert resolve("results/2026-08-12/sess/1408_x37.csv").name == "1408_x37.csv"  # data/ 상대
        assert resolve("N05.csv").name == "N05.csv"                      # 날짜 생략
        assert resolve(str(UPLOADS_ROOT / "2026-08-12" / "N05.csv")).name == "N05.csv"  # 절대경로
        assert resolve("/api/results/2026-08-12/_microscope_1401.png").suffix == ".png"  # URL

        # make_id ↔ resolve 왕복
        mid = make_id("results", "2026-08-12", "_microscope_1401.png")
        assert mid == "results:2026-08-12/_microscope_1401.png", mid
        assert resolve(mid).is_file()
        assert _id_of(resolve(mid)) == mid

        # 경로 탈출
        for bad in ("uploads:../../../etc/passwd", "../../secrets", "uploads:a/../../b"):
            try:
                resolve(bad)
                raise AssertionError(f"경로 탈출이 통과했다: {bad}")
            except FileIdError:
                pass

        # 확장자 → 종류 분기. open_file 이 무엇을 돌려줄지가 여기서 정해진다.
        assert kind_of("a/b.png") == "image"
        assert kind_of("a/b.JPG") == "image"
        assert kind_of("a/b.csv") == "table"          # .csv 는 표로 고정(우선순위)
        assert kind_of("a/b.xlsx") == "table"
        assert kind_of("a/b.json") == "json"
        assert kind_of("a/b.zip") is None             # 모르는 확장자

        # kind 를 명시하면 확장자를 검사한다(내부 호출용 — 모델은 이 경로를 안 쓴다)
        try:
            resolve("uploads:2026-08-12/N05.csv", "image")
            raise AssertionError("csv 를 image 로 받았다")
        except FileIdError as e:
            assert "image file was required" in str(e), e

        # 뿌리를 지정하면 남의 뿌리는 안 뒤진다
        try:
            resolve("uploads:2026-08-12/_microscope_1401.png")
            raise AssertionError("uploads 접두인데 results 파일을 찾았다")
        except FileIdError as e:
            assert "uploads" in str(e) and "list_uploaded_files" in str(e), e

        # 빈 값·없는 파일 — 메시지에 다음 행동이 있는가
        for miss in ("", "   ", "2026-08-12/nope.csv"):
            p, why = try_resolve(miss)
            assert p is None and ("list_" in why), (miss, why)

        print("통과: 정식 id 4 · 하위호환 5 · 왕복 · 경로탈출 3 · 종류안내 2 · 뿌리격리 · 실패안내 3")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
