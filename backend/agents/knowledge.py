# -*- coding: utf-8 -*-
"""
지식베이스 검색 — Chroma 벡터 검색(+키워드 폴백). LLM(채팅) 모델은 쓰지 않는다.

┌─────────────────────────────────────────────────────────────────────────────┐
│ 이 모듈의 공개 API (이 세 개만 밖에서 쓴다)                                  │
│   search_kb(query, top_k)        → 텍스트 지식 검색   [시그니처 불변]        │
│   search_spectra(x, y, top_k)    → 과거 스펙트럼 유사도 검색                 │
│   kb_status()                    → 현재 어떤 검색기가 살아있는지 진단        │
└─────────────────────────────────────────────────────────────────────────────┘

[이 파일이 따로 존재하는 이유 — 단일/다중 에이전트 공정 비교]
다중 에이전트의 Planner는 계획 직전에 KB를 검색해 결과를 프롬프트에 "자동 주입"
받는다(planner._search_kb). 반면 단일 에이전트는 같은 지식에 접근할 수단이 아예
없었다. 그래서 다중은 KB에서 권장 파워/노출을 근거로 받아오고 단일은 맨땅에서
추측하는 — 아키텍처와 무관한 능력 격차가 있었다.

"단일 vs 다중" 비교 실험에서 독립변수는 오케스트레이션 방식 하나여야 한다.
지식 접근 여부가 함께 달라지면 성능 차이가 아키텍처 때문인지 지식 때문인지
분리할 수 없다. 그래서 양쪽이 "같은 인덱스를, 같은 검색기로" 조회하도록
이 모듈로 분리했다.

[중요 — 다중 에이전트 쪽도 이 모듈을 쓰도록 바꿔야 한다]
planner.py에는 아직 키워드 매칭 사본(_load_kb / _search_kb)이 남아 있다.
planner.py를 복구할 때 그 사본을 지우고 여기서 import할 것:

    from backend.agents.knowledge import search_kb          # planner.py 상단
    # 기존 _load_kb / _search_kb / _KB_PATH / _kb_cache 정의는 삭제
    # 호출부 _search_kb(query, top_k=3) → search_kb(query, top_k=3)

사본을 그대로 두면 다중은 키워드 매칭, 단일은 벡터 검색을 쓰는 상태가 되어
비교가 조용히 오염된다. 이제는 알고리즘 자체가 달라지므로 더 치명적이다.

[의도적으로 남긴 차이 — 이건 제거하면 안 된다]
Planner는 KB 결과를 강제로 주입받고, 단일 에이전트는 search_knowledge_base 도구를
"스스로 호출할지 말지 판단"한다. 이 차이는 버그가 아니라 측정 대상이다 —
지식 접근성(access)은 동등하게 맞추되, 활용 판단(judgment)은 각 아키텍처의
고유 특성으로 남긴다.

[이 모듈에 채팅 LLM import가 없어야 하는 이유]
backend.agents.planner를 import하면 모듈 로드 시점에
`_llm = ChatAnthropic(model="claude-opus-4-8")`가 실행되어 ANTHROPIC_API_KEY를
요구한다. 단일 에이전트는 로컬 gemma4만으로 완결되어야 하므로(그게 이 프로젝트의
논지다) planner를 import할 수 없다. 이 모듈은 임베딩 모델만 HTTP로 호출하고
채팅 모델은 건드리지 않는다.

═══════════════════════════════════════════════════════════════════════════════
[저장 구조 — 무엇이 어디에 있는가]

  backend/agents/
  ├── knowledge.py            ← 이 파일. 검색만 한다(읽기 전용).
  ├── kb_ingest.py            ← 색인기. 로더/청킹/임베딩/쓰기 담당.
  ├── knowledge_base.json     ← 손으로 큐레이션한 지식 (원본, git 추적)
  ├── kb_sources/             ← pdf/txt/json을 여기 드랍하면 색인 대상 (git 추적)
  └── knowledge_base/         ← Chroma 영속 디렉터리 (git 무시, 재생성 가능)
      ├── chroma.sqlite3      ← 메타데이터 + 문서 원문 + 임베딩
      └── <uuid>/             ← HNSW 벡터 인덱스 바이너리

원본(json/pdf/txt)만 git에 넣고 인덱스는 무시한다. 인덱스는 `python -m
backend.agents.kb_ingest`로 언제든 똑같이 재생성되는 파생물이기 때문이다.
인덱스를 커밋하면 바이너리 diff로 저장소가 부풀고, 재현성은 오히려 원본 +
결정론적 색인 절차로 보장하는 편이 낫다.

═══════════════════════════════════════════════════════════════════════════════
[컬렉션이 두 개인 이유 — 절대 합치지 말 것]

  raman_docs     : 텍스트 청크. bge-m3 임베딩. 1024차원.
  raman_spectra  : 스펙트럼 시그널. 공통 파수 그리드 리샘플. 512차원.

라만 스펙트럼의 X Pixels도 1024다(스펙트럼 파일 헤더 확인). bge-m3 임베딩 역시
1024차원이라, 둘을 한 컬렉션에 넣으면 Chroma가 차원 검사를 통과시켜 버린다.
에러 없이 들어가서 "그래핀 프로토콜 알려줘"에 raw 스펙트럼 벡터가 튀어나오는
사고가 난다 — 숫자 모양만 같고 의미 공간은 완전히 무관하기 때문이다.

그래서 (1) 컬렉션을 분리하고, (2) 스펙트럼은 일부러 512차원으로 리샘플한다.
512는 해상도 손실 없이(피크폭 10~20cm⁻¹ vs 빈 간격 ~6cm⁻¹) 차원을 다르게 만들어,
혹시 컬렉션을 잘못 지정해도 Chroma가 차원 불일치로 즉시 터지게 하는 안전장치다.
"조용한 오답"보다 "시끄러운 에러"가 낫다.

═══════════════════════════════════════════════════════════════════════════════
[폴백은 반드시 시끄러워야 한다]

chromadb가 없거나 Ollama 임베딩 서버가 죽어 있으면 키워드 매칭으로 폴백한다.
KB는 "있으면 더 잘하는" 참고자료이지 필수 의존성이 아니라서, 검색이 안 된다고
측정 자체를 막으면 안 되기 때문이다(planner의 기존 정책).

하지만 이게 조용하면 벤치마크가 통째로 무효가 된다 — 키워드 검색으로 돌아간
결과를 벡터 검색 결과로 착각하게 되니까. 그래서:
  1. 폴백 전환 시 stderr에 경고를 찍고 (프로세스당 1회)
  2. 모든 결과 항목에 "_retriever": "chroma" | "keyword" 를 박아
     사후에 로그만 봐도 어느 검색기가 답했는지 추적 가능하게 한다.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# ══════════════════════════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════════════════════════

_HERE = Path(__file__).parent

# 손 큐레이션 원본. 키워드 폴백은 이 파일만 본다(Chroma가 없어도 동작해야 하므로).
_KB_JSON_PATH = _HERE / "knowledge_base.json"

# Chroma 영속 디렉터리. 여기 chroma.sqlite3가 생긴다.
_CHROMA_DIR = _HERE / "knowledge_base"

# 색인 대상 파일을 드랍하는 폴더(pdf/txt/json).
KB_SOURCES_DIR = _HERE / "kb_sources"

DOCS_COLLECTION = "raman_docs"
SPECTRA_COLLECTION = "raman_spectra"

# 임베딩 모델. 한/영 혼용 KB라 다국어 모델이 필요하다 — "그래핀"과 "graphene"이
# 같은 공간에 매핑되어야 keywords 수작업 없이도 교차 검색이 된다.
#   계측 PC에서 1회: ollama pull bge-m3
EMBED_MODEL = os.environ.get("KB_EMBED_MODEL", "bge-m3")

# 스펙트럼 리샘플 그리드. 위 주석의 "차원을 일부러 다르게" 안전장치.
SPECTRA_DIM = 512
SPECTRA_MIN_CM = 200.0
SPECTRA_MAX_CM = 3200.0


def _ollama_host() -> str:
    """Ollama 서버 주소. hw_manager와 같은 값을 쓰되 import 실패에 대비한다.

    hardware_manager는 pythoncom/장비 SDK를 끌고 들어올 수 있어서, 개발 PC에서
    이 모듈만 단독으로 쓸 때 import가 터질 수 있다. 그래서 try로 감싸고 기본값을 둔다.
    """
    try:
        from backend.hardware_manager import OLLAMA_HOST  # type: ignore
        return OLLAMA_HOST
    except Exception:
        return os.environ.get("OLLAMA_HOST", "http://192.168.1.15:11434")


# ══════════════════════════════════════════════════════════════════════════════
# 경고 (프로세스당 1회만)
# ══════════════════════════════════════════════════════════════════════════════

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """같은 경고를 도구 호출마다 반복해서 로그를 도배하지 않도록 1회만 찍는다.

    [머리표가 ASCII인 이유 — 이모지를 쓰면 안 된다]
    윈도우 콘솔의 stderr는 cp949이고 파이썬은 인코딩 불가 문자에 backslashreplace를
    적용한다. "⚠"를 쓰면 실제로 "\\u26a0"이라는 글자가 찍힌다(확인함). 한글은 cp949에
    있어서 멀쩡하지만 이모지는 아니다. 이 경고는 "지금 벡터 검색이 아니라 키워드
    폴백으로 돌고 있다"는 실험 유효성 경보라서, 어떤 콘솔에서도 읽혀야 한다.
    """
    if key in _warned:
        return
    _warned.add(key)
    print(f"[knowledge][경고] {msg}", file=sys.stderr, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# 임베딩 — Ollama HTTP 직접 호출
# ══════════════════════════════════════════════════════════════════════════════
#
# langchain_ollama.OllamaEmbeddings를 쓰지 않고 requests로 직접 때리는 이유:
# 이 모듈은 단일/다중 양쪽이 import하는 공용 모듈이라 의존성을 최소로 유지해야
# 한다. requests는 이미 benchmark_runner가 쓰고 있어 추가 비용이 0이다.

def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """텍스트 목록을 임베딩 벡터 목록으로 변환한다.

    Raises
    ------
    RuntimeError — Ollama 서버 접근 실패, 모델 미설치 등. 호출자가 폴백을 결정한다.
    """
    import requests

    if not texts:
        return []

    url = f"{_ollama_host().rstrip('/')}/api/embed"
    try:
        resp = requests.post(
            url,
            json={"model": EMBED_MODEL, "input": list(texts)},
            timeout=120,   # 색인 시 수십 청크를 한 번에 보내므로 넉넉히
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"임베딩 실패({url}, model={EMBED_MODEL}): {e}") from e

    # /api/embed는 {"embeddings": [[...], ...]} 형태.
    embs = data.get("embeddings")
    if not embs:
        raise RuntimeError(
            f"임베딩 응답에 embeddings가 없음. 모델이 없을 수 있다 → "
            f"계측 PC에서 `ollama pull {EMBED_MODEL}` 실행 필요. 응답: {str(data)[:200]}"
        )
    return embs


# ══════════════════════════════════════════════════════════════════════════════
# Chroma 클라이언트
# ══════════════════════════════════════════════════════════════════════════════

_client_cache: Any = None
_client_failed = False


def get_client() -> Any:
    """Chroma PersistentClient를 반환한다(불가하면 None).

    None을 반환하는 경우:
      - chromadb 미설치
      - knowledge_base/ 디렉터리를 못 만들거나 sqlite를 못 여는 경우
    둘 다 키워드 폴백으로 처리하고 예외를 밖으로 던지지 않는다.
    """
    global _client_cache, _client_failed
    if _client_cache is not None:
        return _client_cache
    if _client_failed:
        return None

    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        _client_failed = True
        _warn_once(
            "no_chromadb",
            "chromadb가 설치되어 있지 않습니다 → 키워드 매칭으로 폴백합니다. "
            "벡터 검색을 쓰려면: pip install chromadb",
        )
        return None

    try:
        _CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client_cache = chromadb.PersistentClient(
            path=str(_CHROMA_DIR),
            # 텔레메트리 off — 계측 PC는 외부망이 막혀 있을 수 있고,
            # 켜두면 기동 시 조용히 지연이 생긴다.
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    except Exception as e:
        _client_failed = True
        _warn_once("chroma_open_fail", f"Chroma 열기 실패 → 키워드 폴백: {e}")
        return None
    return _client_cache


def get_collection(name: str, dim_hint: int | None = None) -> Any:
    """컬렉션 핸들을 반환한다(없으면 생성). Chroma가 없으면 None.

    Parameters
    ----------
    dim_hint : 문서화 목적으로 메타데이터에 차원을 남긴다. Chroma가 강제하진
               않지만, 나중에 컬렉션을 열었을 때 무슨 벡터가 든 통인지 알 수 있다.
    """
    client = get_client()
    if client is None:
        return None
    try:
        meta: dict[str, Any] = {
            # 코사인 거리. 텍스트 임베딩·정규화된 스펙트럼 둘 다 코사인이 맞다.
            # (L2로 두면 스펙트럼은 세기 스케일에, 텍스트는 문서 길이에 끌려간다.)
            "hnsw:space": "cosine",
        }
        if dim_hint is not None:
            meta["dim"] = dim_hint
        return client.get_or_create_collection(name=name, metadata=meta)
    except Exception as e:
        _warn_once(f"coll_fail_{name}", f"컬렉션 '{name}' 열기 실패 → 폴백: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 키워드 폴백 (Chroma/Ollama 없을 때)
# ══════════════════════════════════════════════════════════════════════════════

_kb_cache: list[dict] | None = None


def load_kb() -> list[dict]:
    """knowledge_base.json을 읽어 항목 리스트를 반환한다(실패 시 빈 리스트).

    [주의 — 프로세스 수명 동안 캐시된다]
    파일을 고쳐도 서버를 재시작해야 반영된다. 핫 리로드가 필요하면
    knowledge._kb_cache = None 으로 캐시를 비우거나 /api/kb/reload를 호출한다.

    [왜 예외를 삼키는가]
    KB는 참고자료이지 필수 의존성이 아니다. JSON이 깨졌다고 측정을 막으면 안 된다.
    다만 조용히 비면 벤치마크가 무효가 되므로 경고는 반드시 남긴다.
    """
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    try:
        if _KB_JSON_PATH.exists():
            with open(_KB_JSON_PATH, encoding="utf-8") as f:
                _kb_cache = json.load(f)
        else:
            _kb_cache = []
            _warn_once("no_kb_json", f"{_KB_JSON_PATH.name} 없음 → KB가 비었습니다.")
    except Exception as e:
        # JSON 문법 오류(트레일링 콤마 등)가 여기로 온다. 예전엔 완전히 조용해서
        # KB가 통째로 사라져도 아무도 몰랐다 — 그게 벤치마크 최악의 시나리오다.
        _kb_cache = []
        _warn_once("kb_json_broken", f"{_KB_JSON_PATH.name} 파싱 실패 → KB가 비었습니다: {e}")
    return _kb_cache


def _keyword_search(query: str, top_k: int) -> list[dict]:
    """부분문자열 매칭 폴백 — 기존 planner._search_kb와 동일한 알고리즘.

    질의를 공백으로 쪼갠 각 단어가 (title + content + keywords)를 소문자로 이어붙인
    텍스트에 들어있으면 1점. 점수 내림차순 상위 top_k개. 0점 항목은 제외.

    한계(그래서 Chroma로 갈아탄 것): "탄소나노튜브"처럼 keywords에 없는 표현은
    놓친다. 한/영 교차 검색이 안 되어 keywords에 양쪽을 손으로 다 적어야 한다.
    """
    kb = load_kb()
    if not kb:
        return []

    query_words = set(query.lower().split())
    scored: list[tuple[int, dict]] = []

    for entry in kb:
        text = (
            entry.get("title", "") + " "
            + entry.get("content", "") + " "
            + " ".join(entry.get("keywords", []))
        ).lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[dict] = []
    for score, entry in scored[:top_k]:
        e = dict(entry)
        e["_retriever"] = "keyword"
        e["_score"] = score
        e["_source"] = _KB_JSON_PATH.name
        out.append(e)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API
# ══════════════════════════════════════════════════════════════════════════════

def _meta_to_entry(doc: str, meta: dict, distance: float) -> dict:
    """Chroma 결과를 knowledge_base.json 항목과 같은 모양으로 되돌린다.

    [왜 이 변환이 필요한가]
    Chroma 메타데이터는 str/int/float/bool만 담을 수 있다. keywords(리스트)와
    recommended_params(딕트)를 그대로 넣을 수 없어서 색인할 때 JSON 문자열로
    직렬화해 뒀다(kb_ingest._to_metadata 참고). 여기서 되돌린다.

    이 함수 덕분에 search_kb의 반환 모양이 예전과 같아서, single_agent.py와
    planner.py는 한 줄도 고칠 필요가 없다.
    """
    entry: dict[str, Any] = {
        "title": meta.get("title", ""),
        "content": doc,
    }

    kw = meta.get("keywords")
    if kw:
        try:
            entry["keywords"] = json.loads(kw)
        except Exception:
            entry["keywords"] = []

    rp = meta.get("recommended_params")
    if rp:
        try:
            params = json.loads(rp)
            if params:
                entry["recommended_params"] = params
        except Exception:
            pass

    entry["_retriever"] = "chroma"
    # 코사인 거리 → 유사도. 0=무관, 1=동일. 에이전트가 근거의 신뢰도를 가늠하고
    # 벤치마크 로그에서 검색 품질을 사후 분석할 수 있게 노출한다.
    entry["_score"] = round(1.0 - float(distance), 4)
    entry["_source"] = meta.get("source", "")
    if meta.get("page"):
        entry["_page"] = meta["page"]
    return entry


def search_kb(query: str, top_k: int = 3) -> list[dict]:
    """질의와 의미적으로 가까운 지식 청크를 최대 top_k개 반환한다.

    ⚠ 이 시그니처는 바꾸지 말 것 — single_agent.py와 planner.py가 그대로 쓴다.

    동작 순서:
      1. Chroma + Ollama 임베딩으로 벡터 검색 시도
      2. 어느 하나라도 실패하면 → 경고 후 키워드 폴백
      3. 인덱스가 비어 있으면(색인 안 함) → 키워드 폴백

    Returns
    -------
    list[dict] — knowledge_base.json 원본과 같은 구조 + 추적용 언더스코어 필드:
        {"title": str, "content": str,
         "keywords": [str],                      (있을 때만)
         "recommended_params": {...},            (있을 때만)
         "_retriever": "chroma" | "keyword",     ← 어느 검색기가 답했나
         "_score": float,                        ← 유사도(chroma) / 매칭 단어수(keyword)
         "_source": str,                         ← 출처 파일명
         "_page": int}                           (PDF일 때만)
    """
    query = (query or "").strip()
    if not query:
        return []

    coll = get_collection(DOCS_COLLECTION, dim_hint=None)
    if coll is None:
        return _keyword_search(query, top_k)

    try:
        if coll.count() == 0:
            _warn_once(
                "empty_index",
                "Chroma 인덱스가 비어 있습니다 → 키워드 폴백. "
                "색인하려면: python -m backend.agents.kb_ingest",
            )
            return _keyword_search(query, top_k)

        qvec = embed_texts([query])[0]
        res = coll.query(
            query_embeddings=[qvec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        # 임베딩 서버가 죽었거나 모델이 없는 경우가 대부분. 측정을 막지 않고 폴백.
        _warn_once("vec_search_fail", f"벡터 검색 실패 → 키워드 폴백: {e}")
        return _keyword_search(query, top_k)

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    return [
        _meta_to_entry(d, m or {}, dist)
        for d, m, dist in zip(docs, metas, dists)
    ]


def search_spectra(
    wavenumbers: Sequence[float],
    intensities: Sequence[float],
    top_k: int = 3,
) -> list[dict]:
    """주어진 스펙트럼과 모양이 비슷한 과거 측정을 찾는다.

    ⚠ 이건 search_kb와 목적이 다르다. search_kb는 "파라미터를 정할 근거"를 찾고,
    이건 "이 스펙트럼이 뭔지"를 과거 측정과 대조해 좁힌다(물질 동정). 에이전트가
    파라미터를 고를 때 쓰는 도구가 아니다.

    전처리는 preprocess_spectrum()이 담당한다 — 왜 필요한지는 그쪽 주석 참고.

    Returns
    -------
    list[dict] — {"title", "content"(헤더 요약), "_score", "_source", "_retriever"}
                  Chroma가 없거나 인덱스가 비면 빈 리스트(키워드 폴백 불가 —
                  시그널 검색은 텍스트로 대체할 수 있는 게 아니다).
    """
    coll = get_collection(SPECTRA_COLLECTION, dim_hint=SPECTRA_DIM)
    if coll is None:
        return []

    try:
        if coll.count() == 0:
            _warn_once(
                "empty_spectra_index",
                "스펙트럼 인덱스가 비어 있습니다 → python -m backend.agents.kb_ingest",
            )
            return []
        vec = preprocess_spectrum(wavenumbers, intensities)
        res = coll.query(
            query_embeddings=[vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        _warn_once("spectra_search_fail", f"스펙트럼 검색 실패: {e}")
        return []

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out = []
    for d, m, dist in zip(docs, metas, dists):
        m = m or {}
        out.append({
            "title": m.get("title", ""),
            "content": d,
            "_retriever": "chroma",
            "_score": round(1.0 - float(dist), 4),
            "_source": m.get("source", ""),
        })
    return out


def preprocess_spectrum(
    wavenumbers: Sequence[float],
    intensities: Sequence[float],
) -> list[float]:
    """스펙트럼을 비교 가능한 고정 길이 벡터로 만든다.

    [raw 배열을 그대로 넣으면 안 되는 이유]
    1) X축 그리드가 측정마다 다르다. grating/center 설정에 따라 시작점도 간격도
       달라진다(예: 관측된 파일은 -196.22 cm⁻¹에서 시작). 길이가 같아도 i번째
       원소가 같은 파수를 가리키지 않으므로 벡터 비교 자체가 성립하지 않는다.
    2) 베이스라인과 세기 스케일이 유사도를 지배한다. 노출/파워가 다르면 같은
       물질도 다르게, 베이스라인만 비슷하면 다른 물질도 같게 나온다.

    그래서 세 단계를 거친다:
      리샘플 → 공통 파수 그리드(200~3200 cm⁻¹, 512빈)로 선형보간해 축을 통일
      베이스라인 제거 → 하위 5% 분위수를 빼서 오프셋 제거
      L2 정규화 → 세기 스케일 제거. 코사인 거리가 "피크 패턴"만 보게 된다.

    측정 범위 밖(그리드에는 있지만 측정엔 없는 구간)은 0으로 채운다 — 정규화
    후이므로 "신호 없음"을 뜻하고 코사인에 기여하지 않는다.
    """
    import numpy as np

    x = np.asarray(wavenumbers, dtype=float)
    y = np.asarray(intensities, dtype=float)

    if x.size == 0 or x.size != y.size:
        raise ValueError(f"스펙트럼 축 길이 불일치: x={x.size}, y={y.size}")

    # np.interp는 x가 오름차순이어야 한다. 장비에 따라 내림차순으로 저장된다.
    order = np.argsort(x)
    x, y = x[order], y[order]

    grid = np.linspace(SPECTRA_MIN_CM, SPECTRA_MAX_CM, SPECTRA_DIM)
    # left/right=0.0 → 측정 범위 밖은 신호 없음 처리
    resampled = np.interp(grid, x, y, left=0.0, right=0.0)

    baseline = np.percentile(resampled, 5)
    resampled = resampled - baseline
    # 베이스라인 아래로 내려간 노이즈는 음수 피크가 아니라 잡음이므로 잘라낸다.
    resampled = np.clip(resampled, 0.0, None)

    norm = np.linalg.norm(resampled)
    if norm > 0:
        resampled = resampled / norm

    return resampled.tolist()


def kb_status() -> dict:
    """KB 상태 진단 — 어느 검색기가 살아있고 인덱스에 뭐가 몇 개 들었는지.

    벤치마크를 돌리기 전에 반드시 확인할 것. retriever가 "keyword"인데 그걸
    모르고 실험하면 결과가 통째로 무효다.
    """
    status: dict[str, Any] = {
        "chroma_dir": str(_CHROMA_DIR),
        "chroma_available": False,
        "embed_model": EMBED_MODEL,
        "embed_available": False,
        "docs_count": 0,
        "spectra_count": 0,
        "kb_json_entries": len(load_kb()),
        "retriever": "keyword",
    }

    coll = get_collection(DOCS_COLLECTION)
    if coll is not None:
        status["chroma_available"] = True
        try:
            status["docs_count"] = coll.count()
        except Exception:
            pass

    sp = get_collection(SPECTRA_COLLECTION, dim_hint=SPECTRA_DIM)
    if sp is not None:
        try:
            status["spectra_count"] = sp.count()
        except Exception:
            pass

    try:
        embed_texts(["ping"])
        status["embed_available"] = True
    except Exception as e:
        status["embed_error"] = str(e)

    if status["chroma_available"] and status["embed_available"] and status["docs_count"] > 0:
        status["retriever"] = "chroma"

    return status


def reload_kb() -> None:
    """캐시를 비워 다음 조회 때 디스크에서 다시 읽게 한다(/api/kb/reload용)."""
    global _kb_cache, _client_cache, _client_failed
    _kb_cache = None
    _client_cache = None
    _client_failed = False
    _warned.clear()
