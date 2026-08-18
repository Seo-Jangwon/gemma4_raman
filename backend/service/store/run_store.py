# -*- coding: utf-8 -*-
"""세션 단위 산출물 저장소 — AILA/CoALA 두 에이전트 공용.

[왜 필요한가 — 2026-07-29]
이전에는 에이전트가 만든 파일이 전부 data/ 최상위에 '모델이 정한 이름'으로 흩어졌다.
실측된 상태:

    data/T039_despiked.csv          <- 그래도 과제 ID 가 있는 경우
    data/T046_processed.csv
    data/processed_spectrum.csv     <- 어느 과제 산출물인지 알 수 없음
    data/bg_corrected_default.csv   <- 동상. 두 번 돌리면 조용히 덮어씀

문제가 셋이었다:
  1. 귀속 불가 — 파일만 보고 어느 과제/어느 에이전트/몇 번째 실행인지 알 수 없다.
     채점은 "T046 의 저장 결과를 레퍼런스와 비교"인데 그 파일을 특정할 수 없다.
  2. 충돌 — 같은 과제를 AILA/CoALA 로 두 번 돌리면 뒤가 앞을 덮는다. 파일명이
     모델 재량이라 두 에이전트가 같은 이름을 고르는 일이 실제로 일어난다.
  3. 자기 참조 불가 — 에이전트가 "내가 방금/직전 턴에 무엇을 저장했는지" 물어볼
     수단이 없었다. 멀티턴 과제에서 직전 산출물을 이어받지 못한다.

[구조]
    data/runs/<session_label>/
        manifest.json          세션 메타 + 모든 산출물 인덱스(다른 곳에 있는 것까지)
        spectra/NN_<name>.csv  처리된 스펙트럼 (run_analysis 의 save_result)
        points/NN_<point>.json 측정점 기록 (save_measurement_point)

session_label 은 session_id 를 파일명 안전화한 것이다. 벤치마크는
`bench_<run_id>_<agent>_<stamp>`(예: bench_T046_AILA_20260729_131500)를 넘기므로
디렉터리 이름만으로 과제·에이전트·실행시각이 전부 드러난다.

[data/results 쪽은 spectrum_store 가 담당한다]
acquire_spectrum 자동 저장물과 run_analysis 그림은 프론트 서빙(URL_PREFIX)·분석 주입
(list_results) 계약이 걸려 있어 data/runs 로 끌어오지 않는다. 대신 그쪽도 세션별로
갈라져 있다 — data/results/<date>/<session_label>/ (spectrum_store 문서 참고). 이 모듈은
manifest 에 그 '포인터'를 기록해, 세션 하나만 보면 흩어진 산출물을 다 찾을 수 있게 한다.
즉 "새로 쓰는 것은 세션 디렉터리로, 계약이 걸린 것은 제자리에서 세션별로 + manifest
인덱싱"이 이 모듈의 방침이다.

[실패 격리]
저장소 부기(manifest)가 실패해도 실험은 굴러가야 한다 — 모든 I/O 는 예외를 삼키고
stderr 경고만 낸다. 반대로 '실제 산출물 쓰기'는 삼키지 않는다(그건 결과 유실이므로).
"""
from __future__ import annotations

import json
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

# data/ 위치는 service.store 패키지가 단 한 번만 계산하고(__init__.py 머리말),
# 뿌리 이름과 file_id 규칙은 paths.py 가 단독으로 정한다(그 파일 머리말).
from backend.service.store import DATA_ROOT
from backend.service.store.paths import RUNS_ROOT

# 산출물 종류. 디렉터리를 갖는 것은 spectra/points 이고 나머지는 '제자리 + 인덱싱'이다.
KIND_SPECTRA = "spectra"
KIND_FIGURE = "figure"
KIND_MEASUREMENT = "measurement"
# 측정점 기록(save_measurement_point). 한 지점에서 얻은 스펙트럼·현미경 이미지·좌표를
# 하나로 묶은 JSON 이다. 실제 스펙트럼/이미지 파일은 각자의 자리에 그대로 두고
# 여기서는 그 '포인터'만 모은다 — 이 모듈의 기본 방침(제자리 + 인덱싱)과 같다.
# 값이 그대로 하위 디렉터리 이름이 된다 — KIND_SPECTRA("spectra")와 같이 복수형으로 둔다.
KIND_POINT = "points"
_DIR_KINDS = {KIND_SPECTRA, KIND_POINT}

_LOCK = threading.Lock()

# 현재 턴의 세션. **스레드로컬**이다 — 2026-07-31.
#
# [왜 전역 dict 에서 바꿨는가]
# 예전 주석은 "단일 사용자 로컬 도구라 전역 1개로 충분하다"였는데, 서버가
# ThreadPoolExecutor(max_workers=4) 로 돌기 때문에 그 전제가 성립하지 않았다.
# 채팅 탭을 두 개 열면 /api/experiment/stream 이 워커 두 개에서 동시에 돌고,
# 뒤에 시작한 세션의 begin_session 이 이 dict 를 덮어쓴다. 그러면 먼저 돌던 세션의
# 이후 산출물이 **남의 세션 폴더로 저장된다**(spectrum_store 의 저장 경로와
# analysis_sandbox 의 save_result 경로가 모두 이 label 을 읽는다). 에러가 나지 않고
# 파일 위치만 조용히 틀리는 종류라 발견이 늦다.
#
# 스레드로컬이면 되는 이유: 에이전트 실행 1회 = 워커 스레드 1개이고, begin_session 은
# 매 턴 그 스레드 안에서 가장 먼저 호출된다(stream_experiment / run_experiment 머리).
# 따라서 같은 세션이 다음 턴에 다른 워커로 배정돼도 그 스레드에서 다시 설정된다.
# (contextvars 가 아니라 threading.local 인 이유: ThreadPoolExecutor 는 제출 시점의
#  컨텍스트를 워커로 자동 전파하지 않으므로 ContextVar 로는 이 구조가 안 잡힌다.)
#
# 주의: begin_session 을 부른 적 없는 스레드(예: 서버의 상태 폴링 엔드포인트)에서
# current() 를 읽으면 빈 label 이 나온다 — 이는 예전 전역 방식에서도 '세션 시작 전'과
# 같은 상태이고, 호출부는 이미 그 경우를 다룬다(session_dir 의 '_unassigned' 폴백).
class _Current(threading.local):
    def __init__(self):
        self.session_id = ""
        self.agent = ""
        self.label = ""
        self.isolated = False


_current = _Current()


def _sanitize(text: str) -> str:
    """파일/디렉터리명에 안전한 형태로. detail_log._sanitize 와 같은 규칙 —
    같은 세션이 DetailLog 와 data/runs 에서 같은 이름으로 보여야 대조가 쉽다."""
    return re.sub(r"[^0-9A-Za-z_-]", "-", str(text))[:64] or "nosession"


def _safe_name(name, fallback: str = "result") -> str:
    """모델이 준 산출물 이름을 한 조각짜리 안전한 stem 으로 강제한다.
    디렉터리 성분은 통째로 버린다 — 생성 코드/모델 인자는 신뢰할 수 없다."""
    raw = str(name or "").strip()
    tail = raw.replace("\\", "/").rstrip("/").split("/")[-1]
    tail = tail.replace(":", "_")
    if tail.lower().endswith(".csv"):
        tail = tail[:-4]
    safe = "".join(c for c in tail if c.isalnum() or c in "._- ").strip(" .")
    safe = re.sub(r"\s+", "_", safe)
    return safe[:60] or fallback


# ── 세션 경계 ────────────────────────────────────────────────────────────────
def begin_session(session_id: str, agent: str = "", isolated: bool = False) -> dict:
    """턴 시작 시 에이전트 루프가 호출한다. 같은 session_id 로 다시 불러도
    같은 디렉터리를 계속 쓴다(멀티턴 세션의 산출물이 한곳에 모여야 하므로).

    isolated=True 면 이 턴은 **자기 세션의 파일만** 읽는다. 자세한 규칙은 isolated_label()
    참고. 벤치는 항상 True 로 부르고, 대화는 llm_config.CHAT_SESSION_ISOLATED 를 그대로
    넘긴다(기본 True).
    """
    label = _sanitize(session_id)
    _current.session_id = str(session_id or "")
    _current.agent = str(agent or "")
    _current.label = label
    _current.isolated = bool(isolated)
    try:
        d = RUNS_ROOT / label
        d.mkdir(parents=True, exist_ok=True)
        mpath = d / "manifest.json"
        if not mpath.exists():
            # virtual/scene 을 남기는 이유: 가상 장비로 돈 측정이 data/results 에 실측과
            # **똑같은 형식**으로 쌓인다. 파일만 봐서는 구분할 방법이 없고, 섞인 뒤에는
            # 되돌릴 수도 없다. 채점·비교 전에 걸러낼 수 있는 유일한 표식이다.
            from backend.llm_config import VIRTUAL_HW, VIRTUAL_SCENE
            _write_manifest({
                "session_id": _current.session_id,
                "agent": _current.agent,
                "label": label,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "virtual": VIRTUAL_HW,
                "scene": VIRTUAL_SCENE if VIRTUAL_HW else None,
                "artifacts": [],
            })
    except Exception as e:
        print(f"[warn] run_store.begin_session: {type(e).__name__}: {e}", file=sys.stderr)
    return current()


def current() -> dict:
    """현재 세션 정보. label 이 빈 문자열이면 아직 세션이 열리지 않은 상태."""
    label = _current.label
    return {
        "session_id": _current.session_id,
        "agent": _current.agent,
        "label": label,
        "dir": str(RUNS_ROOT / label) if label else "",
        "rel_dir": f"runs/{label}" if label else "",
    }


def isolated_label() -> str | None:
    """격리 모드면 현재 세션 라벨, 아니면 None. **세션 격리의 단일 판정 지점.**

    [왜 있는가]
    세션은 문항/대화마다 새로 열리지만 파일은 날짜 폴더를 공유한다. 앞 세션이 남긴 측정을
    뒤 세션이 읽으면 채점은 '이번에 측정했는가'가 아니라 '앞 문항이 뭘 남겼는가'에 달리고,
    대화에서는 한 세션의 좌표·파워·노출이 다른 세션에 그대로 보인다.

    [누가 켜는가]
    벤치는 항상 켠다. 대화는 llm_config.CHAT_SESSION_ISOLATED 가 정하고 기본이 켜짐이다 —
    끄면 지난 세션 결과를 물어보는 사용이 가능해지는 대신 세션 경계가 사라진다.

    [무엇을 막는가]
      results:<날짜>/<세션>/…   세션이 내 라벨일 때만
      runs:<라벨>/…             라벨이 내 것일 때만
      results:<날짜>/<이름>      세션 세그먼트 없는 구버전 → 격리 모드에서는 막는다
      uploads:…                 항상 허용 — 사용자 첨부는 세션 소유가 아니다(2026-08-13 결정)

    [왜 스레드로컬인가] _Current 와 같은 이유 — 채팅 탭 두 개와 벤치가 같은
    ThreadPoolExecutor 의 서로 다른 워커에서 동시에 돌 수 있다.
    """
    return _current.label if _current.isolated and _current.label else None


def session_dir() -> Path:
    """현재 세션 디렉터리. 세션이 없으면 '_unassigned' 로 떨어뜨린다 —
    산출물을 버리는 것보다 한곳에 모아두는 편이 낫다(원인 추적 가능)."""
    label = _current.label or "_unassigned"
    d = RUNS_ROOT / label
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── manifest ─────────────────────────────────────────────────────────────────
def _manifest_path() -> Path:
    return session_dir() / "manifest.json"


def _write_manifest(doc: dict) -> None:
    _manifest_path().write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def manifest() -> dict:
    try:
        p = _manifest_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] run_store.manifest: {type(e).__name__}: {e}", file=sys.stderr)
    return {"session_id": _current.session_id, "label": _current.label, "artifacts": []}


def record(kind: str, rel_path: str, **meta) -> dict:
    """산출물 1건을 manifest 에 인덱싱한다. rel_path 는 data/ 기준 상대경로
    (그대로 open_file / run_analysis(file_ids) 에 넘길 수 있는 형태) 또는 서빙 URL.

    같은 rel_path 를 다시 기록하면 덮어쓴다 — 덮어쓴 파일이 두 줄로 남지 않게.
    """
    entry = {"kind": kind, "path": rel_path,
             "saved_at": datetime.now().strftime("%H:%M:%S"), **meta}
    try:
        with _LOCK:
            doc = manifest()
            arts = [a for a in doc.get("artifacts", []) if a.get("path") != rel_path]
            arts.append(entry)
            doc["artifacts"] = arts
            doc.setdefault("session_id", _current.session_id)
            doc.setdefault("label", _current.label)
            doc["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _write_manifest(doc)
    except Exception as e:
        print(f"[warn] run_store.record: {type(e).__name__}: {e}", file=sys.stderr)
    return entry


def next_index(kind: str) -> int:
    """이 종류의 다음 순번(1부터). 파일명 앞에 붙여 저장 순서를 이름에 새긴다 —
    '어느 것이 최종 산출물인가'를 파일 목록만 보고 알 수 있게."""
    return sum(1 for a in manifest().get("artifacts", []) if a.get("kind") == kind) + 1


# ── 산출물 경로 발급 ──────────────────────────────────────────────────────────
def new_artifact_path(kind: str, name, ext: str = ".csv") -> tuple[Path, str]:
    """kind 별 세션 하위 디렉터리에 순번이 붙은 저장 경로를 발급한다.

    Returns (절대경로, data/ 기준 상대경로).
    상대경로는 그 문자열 그대로 다시 읽을 수 있는 형태다(open_file 이 받는다).
    """
    if kind not in _DIR_KINDS:
        raise ValueError(f"kind must be one of {sorted(_DIR_KINDS)} (got {kind!r})")
    d = session_dir() / kind
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{next_index(kind):02d}_{_safe_name(name)}"
    p = d / f"{stem}{ext}"
    # 같은 순번+이름이 이미 있으면(같은 턴 내 중복) 뒤에 -2, -3 을 붙인다.
    n = 2
    while p.exists():
        p = d / f"{stem}-{n}{ext}"
        n += 1
    label = _current.label or "_unassigned"
    return p, f"runs/{label}/{kind}/{p.name}"


def new_spectrum_path(name, ext: str = ".csv") -> tuple[Path, str]:
    """처리된 스펙트럼(run_analysis 의 save_result) 저장 경로를 발급한다."""
    return new_artifact_path(KIND_SPECTRA, name, ext)


def new_point_path(name, ext: str = ".json") -> tuple[Path, str]:
    """측정점 기록(save_measurement_point) 저장 경로를 발급한다."""
    return new_artifact_path(KIND_POINT, name, ext)


# ── 에이전트용 조회 ───────────────────────────────────────────────────────────
def list_artifacts(kind: str | None = None) -> list[dict]:
    arts = manifest().get("artifacts", [])
    if kind:
        arts = [a for a in arts if a.get("kind") == kind]
    return arts


def summary_for_prompt() -> str:
    """시스템 프롬프트에 실을 짧은 세션 요약. 토큰을 아끼려 최근 것만, 한 줄씩."""
    cur = current()
    if not cur["label"]:
        return ""
    arts = list_artifacts()
    head = (f"Your session label is `{cur['label']}`. Files you save go under "
            f"`{cur['rel_dir']}/`, and spectra you measure are auto-saved under "
            f"`results/<date>/{cur['label']}/`. Both are listed below and both are "
            f"readable with open_file('<path>').")
    # 격리 모드에서는 왜 남의 파일이 안 열리는지 미리 말해 준다 — 모르면 차단된 id 를
    # 여러 번 재시도하다 사이클을 태운다(에러 메시지만으로는 한 번 부딪혀야 안다).
    if isolated_label():
        head += (" This run is ISOLATED: you can only read files from your own session "
                 "(and files the user attached). Measurements from other sessions are not "
                 "available - if you need data, measure it yourself in this run.")
    if not arts:
        return head + " You have not saved any artifact in this session yet."
    lines = [f"  - [{a.get('kind')}] {a.get('path')}"
             + (f"  ({a.get('num_points')} pts)" if a.get("num_points") else "")
             for a in arts[-12:]]
    more = "" if len(arts) <= 12 else f"  ... and {len(arts) - 12} earlier artifact(s)\n"
    return (head + f" Artifacts you have already saved in this session ({len(arts)}):\n"
            + more + "\n".join(lines)
            + "\nRead any of these back with open_file('<path>') or run_analysis.")
