# app/ 멀티 에이전트 설계 개선 — 설계 문서

작성일: 2026-09-19
대상: `app/` (LangGraph 구현). `app2/`(Claude Agent SDK)는 이번 범위 밖.

## 1. 배경

현재 `app/` 은 2단 구조다.

```
질문 → query_classifier → [complex | general | unknown] → 단일 ReAct 에이전트 → 답변
```

`complex` 경로가 단일 에이전트라, 서버·인프라와 AI가 섞인 질문("MAU 10만 사내 챗봇의
게이트웨이·대화이력·RAG·모델서빙·모니터링을 잡는다면")에서 한 에이전트가 전 영역을
얕게 훑는다. 참조 소스도 Pinecone 논문 하나뿐이라 인프라/설계 질문에 쓸 근거가 없다.

## 2. 목표 구조

```
사용자 ─ Streamlit
   │
   ▼
Orchestrator (Opus)
   ├─ 질문 유형 판단
   └─ 모호하면 interrupt 로 재질문 (Human in the Loop)
   │
   ├─[complex]──▶ 판단 Agent (분해)
   │                 서브질문 N개로 분해: server_infra / system_design / ai
   │                      │
   │                      ├─ Send fan-out (병렬) ─┬─ 서버·인프라 Agent ─▶ 옵시디언 MCP
   │                      │                        ├─ 시스템설계 Agent ─▶ 옵시디언 MCP
   │                      │                        └─ AI Agent        ─▶ Pinecone 벡터스토어
   │                      ▼
   │                 판단 Agent (병합) — 중복 제거·상충 조정·Guardrail
   │
   ├─[general]──▶ IT 트렌드 Agent (Sonnet 5) ─▶ 웹 서칭
   └─[unknown]──▶ Orchestrator 직접 응답
   │
   ▼
최종 답변 + Langfuse 트레이싱
```

## 3. 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 도메인 에이전트 수 | **3개** (서버·인프라 / 시스템설계 / AI) | 사용자 지정 |
| 옵시디언 접근 | **Local REST API with MCP 플러그인** | 사용자 지정. 설치·연결 확인 완료 |
| Memory | **SummarizationMiddleware + InMemorySaver** | 요약 기반. 별도 저장소 불필요 |
| 재질문(HITL) | **LangGraph `interrupt` 기반, 이번에 포함** | 사용자 지정 |

## 4. 모델 배정

그림의 배정을 따르되, `models.py` 가 현재 인자를 무시하고 항상 `claude-sonnet-5` 를
반환하는 문제를 먼저 고친다.

| 역할 | 모델 | 이유 |
|---|---|---|
| Orchestrator | `claude-opus-5` | 분류 + 모호성 판단 + 재질문 결정 |
| 판단 Agent (분해/병합) | `claude-opus-5` | 병합 품질이 최종 답변 품질을 좌우 |
| 도메인 Agent 3종 | `claude-sonnet-5` | 병렬 3개 동시 실행, 비용 지배적 |
| IT 트렌드 Agent | `claude-sonnet-5` | 단순 개념·트렌드 |

그림의 "fable(3.5배)" 는 판단 Agent 후보로 적혀 있으나, Fable 은 Opus 대비 입력·출력
모두 2배 단가이고 병합 작업이 그 비용을 정당화할 만큼 어렵지 않다. **Opus 5 로 시작하고,
병합 품질이 부족하면 그때 올린다.** 전부 `AGENT_*_MODEL` 환경변수로 교체 가능하게 한다.

## 5. 참조 소스 배분

| Agent | 소스 | 접근 |
|---|---|---|
| 서버·인프라 | 옵시디언 볼트 | MCP |
| 시스템설계 | 옵시디언 볼트 | MCP |
| AI | Pinecone `testdb` (Gemini 1.5 논문, 벡터 2,710개) + 옵시디언 | MCP + `vector_search_tool` |

### 옵시디언 MCP — 실측 결과

볼트: `~/OneDrive/문서/Obsidian Vault` (.md 214개)

**Local REST API with MCP** (Adam Coddington, `obsidian-local-rest-api` v5.1.0) 설치 완료.
플러그인이 MCP 서버를 내장하므로 별도 브릿지 패키지가 필요 없다.

| 항목 | 값 |
|---|---|
| 엔드포인트 | `https://127.0.0.1:27124/mcp/` (Streamable HTTP) |
| 프로토콜 | `2025-06-18` — initialize 200, 세션 발급 확인 |
| 인증 | `Authorization: Bearer <apiKey>` |
| HTTP(27123) | 비활성 (`enableInsecureServer=false`) |
| 인증서 | 자체서명 → TLS 검증 비활성화 필요 |
| 노출 도구 | **16종** |

핵심 도구: `search_simple`(Obsidian 내장 검색), `search_query`(JsonLogic 메타데이터 질의),
`vault_read`, `vault_list`, `vault_get_document_map`(헤딩 트리·블록 참조), `tag_list`.

**전제조건**: Obsidian 앱이 실행 중이어야 한다. 꺼져 있으면 해당 도메인 에이전트가
자료를 조회할 수 없으므로, 조회 실패 시 "내부 문서를 조회할 수 없었다"고 명시하고
진행하도록 프롬프트에 규정한다.

### 볼트 내용 적합성 — 확인됨

`search_simple` 본문 검색 실측:

| 검색어 | 히트 | 검색어 | 히트 |
|---|---|---|---|
| 아키텍처 | 117 | 데이터베이스 | 45 |
| 서버 | 62 | RAG | 41 |
| 캐시 | 62 | 배포 | 36 |
| API | 60 | 쿠버네티스 | 32 |
| 트랜잭션 | 27 | LLM | 20 |

**MOC(Map of Content) 노트가 33개** 있어 도메인 허브 역할을 한다. 도메인별 진입점:

- 서버·인프라: `컨테이너 MOC`, `네트워크 MOC`, `모니터링 MOC`, `99. 도커 아키텍처 구조`
- 시스템설계: `시스템 아키텍처 MOC`, `관계형 데이터베이스 MOC`, `NoSQL MOC`, `메시징 MOC`
- AI: `00~02. 테디노트 RAG (기본/심화/에이전트)`, `Part 1. LLM과 Llama 3.pdf`

**태그 라우팅은 쓰지 않는다.** 전체 태그 27개가 `MOC(33)`, `농업(10)`, `공고문(8)`,
`경제(5)` 계열이고 기술 태그가 없다. 대신 **MOC 노트 진입 + 본문 검색**으로 설계한다.

## 6. Memory 설계

LangChain 1.x 에서 `ConversationSummaryBufferMemory` 계열은 제거됐다
(`langchain.memory` 모듈 자체가 없고 `langchain_community.memory` 는 심볼 0개). 현행 대체제를 쓴다.

| 계층 | 수단 | 비고 |
|---|---|---|
| 대화 요약 | **`langchain.agents.middleware.SummarizationMiddleware`** | langchain 1.4.0 에 포함, 추가 설치 불필요 |
| 세션 상태 | **`langgraph.checkpoint.memory.InMemorySaver`** | HITL interrupt 에 필요 |

```python
SummarizationMiddleware(
    model=get_model("judge"),
    trigger=("tokens", 8000),   # 임계치 도달 시 요약으로 치환
    keep=("messages", 20),      # 최근 20개는 원문 유지
)
```

오래된 메시지를 요약으로 치환하고 최근 메시지는 원문으로 남긴다. AI/Tool 메시지 쌍이
끊기지 않게 처리하며, 요약 호출 실패 시 3회 재시도 후 에러를 올린다(요약을 날조하지 않음).

### Redis/Postgres 를 쓰지 않는 이유

- **단기 기억**은 checkpointer 가 thread_id 별로 이미 보관한다. Redis 는 중복이다.
- **장기 기억·중복질문 탐지**는 세션을 넘어선 영속성이 필요하지만 이번 범위에서 제외한다.
  요약 메모리만으로 "긴 대화 토큰 절약 + 문맥 유지" 목표는 달성된다.
- 나중에 필요해지면 `langgraph.store.postgres.PostgresStore`(pgvector 필요)를 붙인다.
  `InMemorySaver` → `PostgresSaver` 교체도 한 줄이다.

## 7. Human in the Loop

Orchestrator 가 질문이 모호하다고 판단하면 `interrupt({"clarifying_question": ...})` 로
멈춘다. Streamlit 이 되묻고, 사용자의 답을 `Command(resume=...)` 로 넣어 재개한다.

`interrupt` 는 checkpointer 가 필수다. `InMemorySaver` 를 쓴다. Streamlit 의
`@st.cache_resource` 가 프로세스 수명 동안 인스턴스를 유지하므로 재실행(rerun) 사이에도
상태가 보존된다. 앱을 재시작하면 진행 중이던 대화는 사라지는데, 토이프로젝트 범위에서는
허용 가능한 트레이드오프다.

**모호함의 기준** (프롬프트에 명시):
- 설계 요청인데 규모·트래픽·제약이 전혀 없어 답이 완전히 달라지는 경우
- 요구가 서로 상충해 한쪽을 고르지 않으면 답할 수 없는 경우
- 그 외에는 **재질문하지 않고** 가정을 명시하고 진행한다 (과도한 되묻기 방지)

## 8. Guardrail / 관측

- **Guardrail**: 기존 `fact_check_tool` 을 병합 직후 1회 실행. `overall_accuracy < 0.8`
  이면 판단 Agent 가 1회 재작성. 현재 `fact_check_tool` 이 context 없을 때 Gemini 스니펫을
  하드코딩 대체하는 버그가 있으므로 제거한다.
- **관측**: Langfuse 는 이미 동작 중(3300). 각 노드에 `run_name` 을 붙여 병렬
  서브에이전트를 트레이스에서 구분할 수 있게 한다.
- **LiteLLM 경유**: 모델별 배정이 생기므로 게이트웨이(4000)를 경유해 모델 라우팅·비용
  집계를 한곳에서 본다. `models.py` 가 `LITELLM_BASE_URL` 이 있으면 그쪽으로 붙는다.

## 9. 범위 밖

- `app2/`(Agent SDK) 에 같은 구조를 이식하는 것
- 장기 기억 / 중복질문 탐지 (Postgres + pgvector)
- Qdrant 로의 마이그레이션 (Pinecone 유지). 안 쓰던 Qdrant 컨테이너는 2026-09-19 제거
