# kb_sources — 지식베이스 원본 문서 드랍 폴더

여기에 파일을 넣고 색인하면 에이전트(단일·다중 양쪽)가 검색할 수 있게 됩니다.

```bash
# 1. 파일을 이 폴더에 복사
# 2. 색인
python -m backend.agents.kb_ingest

# 서버가 떠 있다면 HTTP로도 가능
curl -F "file=@내문서.pdf" http://localhost:8000/api/kb/upload
curl -X POST http://localhost:8000/api/kb/reindex
```

## 지원 형식

| 형식 | 청킹 단위 | 비고 |
|---|---|---|
| `.pdf` | 페이지 1장 = 1청크 | `--caption`으로 gemma4 페이지 캡션 추가 가능 |
| `.txt` / `.md` | 빈 줄 2개로 나눈 문단 | |
| `.json` | `[{title, keywords, content}]` 구조면 항목별, 아니면 통째로 1청크 | |

40자 미만 청크는 노이즈로 보고 버립니다.

## 여기 넣으면 안 되는 것

**측정 데이터** (스펙트럼 X/Y 배열, 이미지 raw)
→ `backend/spectra/`에 두면 색인기가 알아서 처리합니다. 헤더는 텍스트 지식으로,
배열은 별도 시그널 컬렉션으로 갈라 넣습니다. 여기 `.txt`로 넣으면 1024점 배열이
통째로 LLM 컨텍스트에 쏟아집니다.

**장비 설정 파일** (`Config.txt` 등)
→ 그건 지식이 아니라 *현재 상태*입니다. 정적 색인에 박제하면 설정이 바뀌는 순간
KB가 거짓말을 시작합니다. 필요하면 KB가 아니라 실시간 조회 도구로 노출하세요.

**비밀정보** (API 키, 계정 정보)
→ 이 폴더는 git에 추적됩니다. 그리고 KB 내용은 LLM 컨텍스트로 들어갑니다.

## PDF 주의사항

스캔/스크린샷 위주 PDF는 텍스트만 뽑으면 거의 못 씁니다. 실측 예 —
`backend/hw_tools/docs/MantaRay_QuickGuide_KR.pdf`는 36페이지에 임베디드 이미지가
141개인데 추출되는 텍스트는 전부 8,501자(페이지당 236자)뿐이고, 표의 항목명과
설명이 서로 분리되어 나옵니다. 이런 문서는 `--caption`이 사실상 필수입니다:

```bash
python -m backend.agents.kb_ingest --caption   # 페이지당 VLM 1회 — 느림
```

캡션은 색인 시 1회만 돌고 결과가 Chroma에 저장되므로 런타임 비용은 0입니다.
단 계측 PC(gemma4 접근 가능한 곳)에서 실행해야 합니다.
