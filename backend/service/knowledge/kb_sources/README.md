# kb_sources — 지식베이스

에이전트가 `search_knowledge_base` 로 검색하는 지식은 **이 폴더의
`knowledge_base.json` 하나**입니다. 색인기도 벡터 DB도 없습니다.

```bash
# 1. knowledge_base.json 에 항목 추가
# 2. 서버가 떠 있으면 캐시만 비우면 반영됩니다(재기동 불필요)
curl -X POST http://localhost:8000/api/kb/reload

# 잘 걸리는지 확인 — 에이전트와 똑같은 경로로 검색해 봅니다
curl "http://localhost:8000/api/kb/search?q=graphene"
curl http://localhost:8000/api/kb/status      # kb_json_entries 가 0 이면 KB 가 죽은 것
```

## 항목 형식

```json
{
  "title": "Graphene Raman Standard Protocol",
  "keywords": ["graphene", "그래핀", "D band", "G band", "carbon"],
  "content": "Graphene Raman analysis: the D band (~1350 cm-1) indicates defects, ...",
  "recommended_params": {"laser_power_pct": 20, "exposure_s": 2.0}
}
```

검색은 **질의 단어가 `title + content + keywords` 안에 있는지**만 봅니다. 그래서
`keywords` 에 부르는 이름을 전부 적어야 합니다 — 한글·영어·약어·별칭 모두요.
"탄소나노튜브"를 안 적어두면 그 질의는 못 찾습니다.

`content` 는 그대로 모델 컨텍스트에 실립니다. **영어로 쓰세요** — 논문 산출물에
한국어가 섞이면 안 됩니다.

## 벡터 검색(Chroma)이 없어진 이유 — 2026-08-12

붙어 있던 Chroma 인덱스가 문서 0개였습니다. 도입 이래 한 번도 답한 적이 없고
모든 검색이 조용히 키워드 매칭으로 처리되고 있었습니다. 그래서 실제로 도는 것만
남기고 나머지(chromadb + onnxruntime ~200MB, bge-m3 임베딩, pymupdf, 색인기,
`/api/kb/{upload,reindex}`)를 걷어냈습니다. 자세한 근거는
`backend/service/knowledge/search.py` 머리말에 있습니다.

**되돌릴 시점:** 매뉴얼 PDF 처럼 수백 쪽짜리 원본을 넣기로 할 때입니다. 항목마다
`keywords` 를 손으로 달 수 없는 규모가 되면 임베딩이 값을 합니다. 지금은 5항목이라
손으로 적는 편이 더 정확하고 디버깅도 됩니다.

## 여기 넣으면 안 되는 것

**측정 데이터** (스펙트럼 X/Y 배열, 이미지 raw)
→ 1024점 배열이 통째로 LLM 컨텍스트에 쏟아집니다. 측정 결과는 `data/results/` 에
자동 저장되고 `list_results` / `load_spectrum` 으로 조회합니다.

**장비 설정 파일** (`Config.txt` 등)
→ 그건 지식이 아니라 *현재 상태*입니다. 정적 문서에 박제하면 설정이 바뀌는 순간
KB 가 거짓말을 시작합니다. 필요하면 KB 가 아니라 실시간 조회 도구로 노출하세요.

**비밀정보** (API 키, 계정 정보)
→ 이 폴더는 git 에 추적됩니다. 그리고 KB 내용은 LLM 컨텍스트로 들어갑니다.
