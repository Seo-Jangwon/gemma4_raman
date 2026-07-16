# -*- coding: utf-8 -*-
"""
지식베이스 색인기 — json/pdf/txt/스펙트럼을 읽어 Chroma에 넣는다.

실행:
    python -m backend.agents.kb_ingest              # 전체 재색인
    python -m backend.agents.kb_ingest --status     # 색인 안 하고 상태만 확인
    python -m backend.agents.kb_ingest --caption    # PDF 페이지를 gemma4로 캡션(느림)
    python -m backend.agents.kb_ingest --no-spectra # 스펙트럼 색인 건너뛰기

┌─────────────────────────────────────────────────────────────────────────────┐
│ knowledge.py = 읽기(검색) 전용,  kb_ingest.py = 쓰기(색인) 전용             │
│ 이 분리 덕분에 런타임(서버/에이전트)은 fitz 같은 무거운 파서를 import하지    │
│ 않는다. 색인은 오프라인 1회, 검색은 매 요청 — 수명주기가 다르다.            │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
[소스별로 처리가 다른 이유 — 이건 한 종류의 문제가 아니다]

  knowledge_base.json  1항목 = 1청크 그대로.
                       이미 사람이 정제한 지식이라 손댈 게 없다.

  kb_sources/*.pdf     페이지 단위 청크.
                       ⚠ 문서 전체를 한 청크로 넣으면 안 된다. 실측: MantaRay
                       퀵가이드는 36p/8,501자인데 통째로 넣고 "silicon calibration
                       520"으로 검색하니 실리콘 내용이 없는데도 2위로 올라왔다
                       (긴 문서일수록 아무 단어나 하나는 걸린다). 게다가 히트하면
                       8,501자가 통째로 컨텍스트에 쏟아진다.

  kb_sources/*.txt     문단 단위 청크. 일반 텍스트 문서용.

  스펙트럼 .txt        헤더만 텍스트 청크로, 데이터 배열은 스펙트럼 컬렉션으로.
                       ⚠ X/Y 배열 1024점을 텍스트 KB에 넣으면 안 된다. 그건 지식이
                       아니라 측정 데이터이고, single_agent의 _slim()이 컨텍스트에서
                       막고 있는 바로 그 물건이다. 반면 헤더(Rayleigh 532.021nm /
                       Grating 1200gr / 노출 0.2s / ND 20% / 렌즈 MPLFLN20x)는
                       "이 장비에서 실제로 쓰인 파라미터" 기록이라 진짜 쓸모가 있다.

  Config.txt           색인하지 않는다.
                       INI 형식 장비 설정은 지식이 아니라 "현재 상태"다. 정적 KB에
                       박제하면 설정이 바뀌는 순간 KB가 거짓말을 시작한다. 필요하면
                       KB가 아니라 도구(실시간 조회)로 노출해야 한다.

═══════════════════════════════════════════════════════════════════════════════
[재색인은 멱등(idempotent)이다]

문서 ID를 "출처파일:페이지" 같은 결정론적 문자열로 만들고 upsert한다. 같은 소스를
두 번 색인해도 중복이 쌓이지 않고 덮어쓴다. 소스에서 삭제된 문서를 지우려면
--reset으로 컬렉션을 비우고 다시 넣는다(기본 동작).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

# knowledge.py에서 설정을 그대로 가져온다 — 색인과 검색이 같은 상수를 봐야
# 컬렉션 이름/차원/임베딩 모델이 어긋나지 않는다.
from backend.agents.knowledge import (
    DOCS_COLLECTION,
    EMBED_MODEL,
    KB_SOURCES_DIR,
    SPECTRA_COLLECTION,
    SPECTRA_DIM,
    _KB_JSON_PATH,
    _ollama_host,
    embed_texts,
    get_client,
    get_collection,
    kb_status,
    preprocess_spectrum,
)

# 스펙트럼 원본이 쌓이는 곳(backend/spectra/<세션>/*.txt).
_SPECTRA_ROOT = Path(__file__).resolve().parent.parent / "spectra"

# 임베딩 배치 크기. 너무 크면 Ollama가 타임아웃, 너무 작으면 왕복이 잦다.
_EMBED_BATCH = 32

# 텍스트 청크 최소 길이. 이보다 짧으면 색인해봐야 노이즈다(페이지 번호만 있는
# 표지/간지 등). 실측상 MantaRay PDF의 35페이지가 0자, 33페이지가 27자였다.
_MIN_CHUNK_CHARS = 40


# ══════════════════════════════════════════════════════════════════════════════
# 청크 자료구조
# ══════════════════════════════════════════════════════════════════════════════

def _make_chunk(
    doc_id: str,
    title: str,
    text: str,
    source: str,
    *,
    keywords: list[str] | None = None,
    recommended_params: dict | None = None,
    page: int | None = None,
) -> dict:
    """Chroma에 넣을 청크 하나를 만든다.

    [메타데이터가 JSON 문자열인 이유]
    Chroma 메타데이터는 str/int/float/bool만 허용한다. keywords(리스트)와
    recommended_params(딕트)를 그대로 넣으면 거부당한다. 그래서 직렬화해서 넣고,
    검색 시 knowledge._meta_to_entry가 되돌린다. 이 왕복 덕분에 search_kb의
    반환 모양이 예전 knowledge_base.json 구조와 똑같이 유지되고,
    single_agent.py와 planner.py는 한 줄도 안 고쳐도 된다.
    """
    meta: dict[str, Any] = {"title": title, "source": source}
    if keywords:
        meta["keywords"] = json.dumps(keywords, ensure_ascii=False)
    if recommended_params:
        meta["recommended_params"] = json.dumps(recommended_params, ensure_ascii=False)
    if page is not None:
        meta["page"] = page
    return {"id": doc_id, "document": text, "metadata": meta}


# ══════════════════════════════════════════════════════════════════════════════
# 로더 1 — knowledge_base.json (손 큐레이션)
# ══════════════════════════════════════════════════════════════════════════════

def load_curated() -> list[dict]:
    """knowledge_base.json의 각 항목을 청크 하나로 변환한다.

    항목이 이미 짧고(수백 자) 주제가 하나씩이라 더 쪼갤 필요가 없다.
    검색 대상 텍스트에는 title/keywords도 함께 넣는다 — 임베딩이 "그래핀"이라는
    제목 신호까지 흡수해야 "graphene 측정법" 질의에 잘 걸린다.
    """
    if not _KB_JSON_PATH.exists():
        print(f"  [건너뜀] {_KB_JSON_PATH.name} 없음")
        return []

    try:
        with open(_KB_JSON_PATH, encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as e:
        # 여기서만큼은 조용히 넘어가면 안 된다 — 색인은 사람이 보고 있는 작업이다.
        print(f"  [오류] {_KB_JSON_PATH.name} 파싱 실패: {e}", file=sys.stderr)
        return []

    chunks = []
    for i, e in enumerate(entries):
        title = e.get("title", f"항목 {i}")
        content = e.get("content", "")
        keywords = e.get("keywords", [])
        # 임베딩용 텍스트 = 제목 + 키워드 + 본문
        embed_text = f"{title}\n{' '.join(keywords)}\n{content}"
        chunks.append(_make_chunk(
            doc_id=f"curated:{i}",
            title=title,
            text=embed_text,
            source=_KB_JSON_PATH.name,
            keywords=keywords,
            recommended_params=e.get("recommended_params"),
        ))
    print(f"  knowledge_base.json → {len(chunks)}청크")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 로더 2 — PDF
# ══════════════════════════════════════════════════════════════════════════════

def load_pdf(path: Path, caption: bool = False) -> list[dict]:
    """PDF를 페이지 단위 청크로 변환한다.

    [레이아웃 붕괴 경고 — 이 PDF의 한계]
    PyMuPDF의 get_text()는 읽기 순서를 보장하지 않는다. 실측: MantaRay 퀵가이드
    19페이지의 표에서 '명칭' 열과 '설명' 열이 분리되어, "Homing"과 "Grating 초기화
    버튼"의 짝이 끊어진 채 추출된다. 즉 이 PDF에서 뽑은 텍스트는 '어떤 항목들이
    있다'는 사실은 알려주지만 '무엇이 무엇인지'는 왜곡될 수 있다.

    caption=True면 페이지를 이미지로 렌더해 gemma4에게 설명시킨다. 이 PDF는
    36페이지에 임베디드 이미지가 141개라 내용의 대부분이 그림 안에 있고, 캡션이
    그걸 살리는 유일한 방법이자 레이아웃 붕괴 우회로다. 대신 페이지당 VLM 호출
    1회라 느리다(색인 시 1회만, 런타임 비용은 0).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(f"  [건너뜀] {path.name}: PyMuPDF 미설치 (pip install pymupdf)")
        return []

    chunks = []
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"  [오류] {path.name} 열기 실패: {e}", file=sys.stderr)
        return []

    skipped = 0
    for pno in range(doc.page_count):
        page = doc[pno]
        text = page.get_text().strip()

        if caption:
            cap = _caption_page(page, path.name, pno + 1)
            if cap:
                # 캡션을 앞에 둔다 — 임베딩이 문장 앞쪽에 더 민감하고,
                # 캡션이 원문 텍스트보다 의미적으로 정돈되어 있기 때문.
                text = f"{cap}\n\n[페이지 원문 텍스트]\n{text}"

        if len(text) < _MIN_CHUNK_CHARS:
            skipped += 1
            continue

        chunks.append(_make_chunk(
            doc_id=f"pdf:{path.name}:{pno + 1}",
            title=f"{path.stem} p.{pno + 1}",
            text=text,
            source=path.name,
            page=pno + 1,
        ))

    doc.close()
    note = f" (내용 없어 건너뛴 페이지 {skipped})" if skipped else ""
    print(f"  {path.name} → {len(chunks)}청크{note}")
    return chunks


def _caption_page(page: Any, fname: str, pno: int) -> str:
    """PDF 페이지를 PNG로 렌더해 gemma4에게 설명시킨다(실패하면 빈 문자열).

    single_agent.py가 현미경 이미지를 gemma4에 넘길 때 쓰는 것과 같은 경로다
    (ollama의 images 필드). 여기선 채팅이 아니라 색인 전처리로 쓴다.
    """
    import base64
    import requests

    try:
        # 2배 확대 — 기본 72dpi로는 UI 스크린샷의 작은 라벨을 못 읽는다.
        pix = page.get_pixmap(matrix=__import__("fitz").Matrix(2, 2))
        png = pix.tobytes("png")
    except Exception as e:
        print(f"    [캡션 실패] {fname} p.{pno} 렌더: {e}", file=sys.stderr)
        return ""

    prompt = (
        "이것은 라만 분광기(MantaRay) 사용 설명서의 한 페이지입니다. "
        "이 페이지가 설명하는 기능과 조작 절차를 한국어로 요약하세요. "
        "표가 있으면 각 항목과 그 설명을 짝지어 정확히 옮기세요. "
        "화면에 보이는 버튼/입력란 이름은 원문 그대로 유지하세요. "
        "페이지에 내용이 없으면 '내용 없음'이라고만 답하세요."
    )
    try:
        resp = requests.post(
            f"{_ollama_host().rstrip('/')}/api/generate",
            json={
                "model": "gemma4:31b",
                "prompt": prompt,
                "images": [base64.b64encode(png).decode()],
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        out = (resp.json().get("response") or "").strip()
    except Exception as e:
        print(f"    [캡션 실패] {fname} p.{pno}: {e}", file=sys.stderr)
        return ""

    if "내용 없음" in out[:20]:
        return ""
    print(f"    캡션 {fname} p.{pno}: {len(out)}자")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 로더 3 — 일반 텍스트
# ══════════════════════════════════════════════════════════════════════════════

def load_txt(path: Path) -> list[dict]:
    """텍스트 파일을 문단 단위 청크로 변환한다(빈 줄 2개 = 문단 경계)."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  [오류] {path.name} 읽기 실패: {e}", file=sys.stderr)
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if len(p.strip()) >= _MIN_CHUNK_CHARS]
    chunks = [
        _make_chunk(
            doc_id=f"txt:{path.name}:{i}",
            title=f"{path.stem} ({i + 1}번째 문단)",
            text=p,
            source=path.name,
        )
        for i, p in enumerate(paras)
    ]
    print(f"  {path.name} → {len(chunks)}청크")
    return chunks


def load_json_source(path: Path) -> list[dict]:
    """kb_sources/에 드랍된 임의 JSON을 색인한다.

    knowledge_base.json과 같은 구조(title/keywords/content)면 그대로 쓰고,
    아니면 통째로 예쁘게 찍어서 텍스트 청크 하나로 만든다.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [오류] {path.name} 파싱 실패: {e}", file=sys.stderr)
        return []

    chunks = []
    if isinstance(data, list) and data and isinstance(data[0], dict) and "content" in data[0]:
        for i, e in enumerate(data):
            title = e.get("title", f"{path.stem} 항목 {i}")
            kws = e.get("keywords", [])
            chunks.append(_make_chunk(
                doc_id=f"json:{path.name}:{i}",
                title=title,
                text=f"{title}\n{' '.join(kws)}\n{e.get('content', '')}",
                source=path.name,
                keywords=kws,
                recommended_params=e.get("recommended_params"),
            ))
    else:
        chunks.append(_make_chunk(
            doc_id=f"json:{path.name}:0",
            title=path.stem,
            text=json.dumps(data, ensure_ascii=False, indent=2)[:4000],
            source=path.name,
        ))
    print(f"  {path.name} → {len(chunks)}청크")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 로더 4 — 스펙트럼 (텍스트 헤더 + 시그널, 둘로 갈라진다)
# ══════════════════════════════════════════════════════════════════════════════

def parse_spectrum(path: Path) -> tuple[dict, list[float], list[float]] | None:
    """스펙트럼 txt를 (헤더딕트, X축, Y축)으로 파싱한다. 형식이 아니면 None.

    실측한 파일 구조(backend/spectra/260508-2/레이즈온스펙트럼.txt):

        System                              ← 섹션 헤더(값 없음)
        Type\tMANTARAY                      ← 탭 구분 key/value
        Unit\tWAVENUMBER
        Rayleigh\t532.021nm
        Grating\t1200gr
                                            ← 빈 줄
        Measuring Condition                 ← 섹션 헤더
        Acquisition Mode\tSINGLE
        Exposure Time\t0.2s
        Laser Filter\t532nm
        ND Filter\t20%
        Lens Magnification\tMPLFLN20x
        X Pixels\t1024\tY Pixels\t255       ← 한 줄에 key/value 두 쌍
                                            ← 빈 줄 2개
        X Axis Data\tY Axis Data            ← 데이터 시작 표식
        -196.22\t99                         ← 여기부터 1024줄
        ...
    """
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None

    header: dict[str, str] = {}
    data_start = None

    for i, line in enumerate(lines):
        if line.startswith("X Axis Data"):
            data_start = i + 1
            break
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        # "X Pixels 1024 Y Pixels 255"처럼 한 줄에 쌍이 여러 개일 수 있어
        # 2개씩 끊어 읽는다. 홀수 개(섹션 헤더 등)는 무시된다.
        for j in range(0, len(parts) - 1, 2):
            header[parts[j]] = parts[j + 1]

    if data_start is None:
        return None   # 스펙트럼 파일이 아니다(예: hsspeed.txt)

    xs: list[float] = []
    ys: list[float] = []
    for line in lines[data_start:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            continue

    if not xs:
        return None
    return header, xs, ys


def _header_summary(header: dict, path: Path, n_points: int) -> str:
    """헤더를 에이전트가 읽을 자연어 문장으로 만든다.

    key=value 나열보다 문장이 임베딩에 유리하다 — 질의도 자연어이기 때문이다.
    """
    # 세션 폴더명이 보통 날짜다(예: 260508-2 → 2026-05-08 2번째 세션).
    session = path.parent.name
    bits = [f"과거 라만 측정 기록 (세션 {session}, 파일 {path.name})."]

    def add(label: str, key: str) -> None:
        if header.get(key):
            bits.append(f"{label} {header[key]}")

    add("장비", "Type")
    add("단위", "Unit")
    add("레일리 파장", "Rayleigh")
    add("그레이팅", "Grating")
    add("획득 모드", "Acquisition Mode")
    add("노출 시간", "Exposure Time")
    add("레이저 필터", "Laser Filter")
    add("ND 필터", "ND Filter")
    add("렌즈 배율", "Lens Magnification")
    bits.append(f"데이터 포인트 {n_points}개.")
    return " ".join(bits)


def load_spectra() -> tuple[list[dict], list[dict]]:
    """backend/spectra/ 아래 모든 스펙트럼을 (텍스트청크, 시그널청크)로 나눈다.

    같은 파일이 두 컬렉션에 나뉘어 들어간다:
      헤더 → raman_docs    ("어떤 파라미터로 찍었나" — 에이전트가 읽는 지식)
      배열 → raman_spectra ("어떤 모양이었나"      — 시그널 유사도 검색용)
    """
    if not _SPECTRA_ROOT.exists():
        print(f"  [건너뜀] {_SPECTRA_ROOT} 없음")
        return [], []

    doc_chunks: list[dict] = []
    sig_chunks: list[dict] = []
    skipped = 0

    for path in sorted(_SPECTRA_ROOT.rglob("*.txt")):
        parsed = parse_spectrum(path)
        if parsed is None:
            skipped += 1
            continue
        header, xs, ys = parsed
        rel = str(path.relative_to(_SPECTRA_ROOT))
        title = f"측정기록 {path.parent.name}/{path.stem}"
        summary = _header_summary(header, path, len(xs))

        doc_chunks.append(_make_chunk(
            doc_id=f"spectrum_meta:{rel}",
            title=title,
            text=summary,
            source=rel,
        ))

        try:
            vec = preprocess_spectrum(xs, ys)
        except Exception as e:
            print(f"  [오류] {rel} 전처리 실패: {e}", file=sys.stderr)
            continue

        sig_chunks.append({
            "id": f"spectrum:{rel}",
            "document": summary,       # 검색 결과에 보여줄 사람이 읽는 설명
            "metadata": {"title": title, "source": rel},
            "embedding": vec,          # ← 임베딩 모델을 안 거치고 직접 넣는다
        })

    note = f" (스펙트럼 형식 아님 {skipped}개 건너뜀)" if skipped else ""
    print(f"  spectra/ → 헤더 {len(doc_chunks)}청크, 시그널 {len(sig_chunks)}벡터{note}")
    return doc_chunks, sig_chunks


# ══════════════════════════════════════════════════════════════════════════════
# 색인
# ══════════════════════════════════════════════════════════════════════════════

def _batched(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def index_docs(chunks: list[dict], reset: bool = True) -> int:
    """텍스트 청크를 임베딩해 raman_docs에 넣는다. 넣은 개수를 반환."""
    if not chunks:
        return 0
    client = get_client()
    if client is None:
        raise RuntimeError("Chroma를 열 수 없습니다 (pip install chromadb)")

    if reset:
        # 소스에서 지운 문서가 인덱스에 유령으로 남는 걸 막는다.
        try:
            client.delete_collection(DOCS_COLLECTION)
        except Exception:
            pass

    coll = get_collection(DOCS_COLLECTION)
    if coll is None:
        raise RuntimeError(f"컬렉션 '{DOCS_COLLECTION}'을 만들 수 없습니다")

    total = 0
    for batch in _batched(chunks, _EMBED_BATCH):
        embs = embed_texts([c["document"] for c in batch])
        coll.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["document"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=embs,
        )
        total += len(batch)
        print(f"    ... {total}/{len(chunks)}")
    return total


def index_spectra(chunks: list[dict], reset: bool = True) -> int:
    """시그널 벡터를 raman_spectra에 넣는다.

    임베딩 모델을 거치지 않는다 — 이미 preprocess_spectrum이 512차원 벡터를
    만들어 뒀다. 스펙트럼은 텍스트가 아니라서 bge-m3에 넣을 대상이 아니다.
    """
    if not chunks:
        return 0
    client = get_client()
    if client is None:
        raise RuntimeError("Chroma를 열 수 없습니다 (pip install chromadb)")

    if reset:
        try:
            client.delete_collection(SPECTRA_COLLECTION)
        except Exception:
            pass

    coll = get_collection(SPECTRA_COLLECTION, dim_hint=SPECTRA_DIM)
    if coll is None:
        raise RuntimeError(f"컬렉션 '{SPECTRA_COLLECTION}'을 만들 수 없습니다")

    for batch in _batched(chunks, _EMBED_BATCH):
        coll.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["document"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
        )
    return len(chunks)


def collect_sources(caption: bool = False) -> list[dict]:
    """knowledge_base.json + kb_sources/의 모든 파일을 청크로 모은다."""
    chunks = load_curated()

    KB_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(KB_SOURCES_DIR.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            chunks += load_pdf(path, caption=caption)
        elif suffix in (".txt", ".md"):
            chunks += load_txt(path)
        elif suffix == ".json":
            chunks += load_json_source(path)
        else:
            print(f"  [건너뜀] {path.name}: 지원하지 않는 형식({suffix})")
    return chunks


def ingest(caption: bool = False, with_spectra: bool = True) -> dict:
    """전체 색인 파이프라인."""
    print("═" * 70)
    print(f"KB 색인 시작 — 임베딩 모델: {EMBED_MODEL} @ {_ollama_host()}")
    print("═" * 70)

    print("\n[1/3] 문서 수집")
    doc_chunks = collect_sources(caption=caption)

    sig_chunks: list[dict] = []
    if with_spectra:
        print("\n[2/3] 스펙트럼 수집")
        spec_docs, sig_chunks = load_spectra()
        doc_chunks += spec_docs
    else:
        print("\n[2/3] 스펙트럼 건너뜀 (--no-spectra)")

    print(f"\n[3/3] 색인 — 텍스트 {len(doc_chunks)}청크, 시그널 {len(sig_chunks)}벡터")
    n_docs = index_docs(doc_chunks)
    n_sigs = index_spectra(sig_chunks) if sig_chunks else 0

    print("\n" + "═" * 70)
    print(f"완료 — raman_docs: {n_docs},  raman_spectra: {n_sigs}")
    print("═" * 70)
    return {"docs": n_docs, "spectra": n_sigs}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    ap = argparse.ArgumentParser(description="라만 KB를 Chroma에 색인한다.")
    ap.add_argument("--status", action="store_true", help="색인하지 않고 상태만 출력")
    ap.add_argument("--caption", action="store_true",
                    help="PDF 페이지를 gemma4로 캡션 (느림, Ollama 필요)")
    ap.add_argument("--no-spectra", action="store_true", help="스펙트럼 색인 건너뛰기")
    args = ap.parse_args()

    if args.status:
        st = kb_status()
        print(json.dumps(st, ensure_ascii=False, indent=2))
        # retriever가 keyword면 벤치마크를 돌리면 안 된다 — 종료코드로 알린다.
        return 0 if st["retriever"] == "chroma" else 1

    try:
        ingest(caption=args.caption, with_spectra=not args.no_spectra)
    except Exception as e:
        print(f"\n[색인 실패] {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print("\n상태 확인:")
    print(json.dumps(kb_status(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
