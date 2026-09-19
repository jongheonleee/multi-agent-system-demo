# app — 멀티 에이전트 IT 어시스턴트 (LangGraph)

질문을 도메인별로 쪼개 전문 에이전트 3종이 병렬로 조사하고, 판단 Agent 가 하나로
합쳐서 답한다. 근거는 옵시디언 볼트(내부 문서)와 Pinecone(논문), 웹 검색에서 가져온다.

## 실행

```powershell
cd app
..\.venv\Scripts\python.exe -m streamlit run app.py
```

반드시 `app/` 안에서 실행한다. 모듈들이 평탄 import(`from state import ...`)를 쓴다.

### 전제조건

| 항목 | 확인 방법 |
|---|---|
| **Obsidian 앱 실행 중** | `curl -sk https://127.0.0.1:27124/` 가 200 |
| 인프라 스택 | `cd infra && docker compose up -d` |
| LiteLLM 게이트웨이 | `curl http://localhost:4000/health/liveliness` |

Obsidian 이 꺼져 있으면 서버·인프라/시스템설계 에이전트가 볼트를 못 읽는다.
앱이 죽지는 않고 "내부 문서를 조회할 수 없었다"고 답한다.

## 구조

```
사용자 ─ Streamlit
   │
   ▼
Orchestrator (Opus 5)  ─ 의도 분류, 모호하면 interrupt 로 재질문
   │
   ├─[complex]──▶ 판단 Agent 분해 ─(Send fan-out, 병렬)─┬─ 서버·인프라 ─▶ 옵시디언 MCP
   │                                                      ├─ 시스템설계  ─▶ 옵시디언 MCP
   │                                                      └─ AI         ─▶ Pinecone + 옵시디언
   │                             └─▶ 판단 Agent 병합 ─▶ Guardrail(fact_check)
   │
   ├─[general]──▶ IT 트렌드 Agent ─▶ Serper 웹 검색
   └─[unknown]──▶ 직접 응답
   │
   ▼
finalize ─▶ 답변 (+ Langfuse 트레이싱)
```

### 파일

| 파일 | 역할 |
|---|---|
| `main_agent.py` | 그래프 배선 + Orchestrator(의도 분류·재질문) |
| `judge_agent.py` | 판단 Agent — 질문 분해, 답변 병합, Guardrail |
| `domain_agents.py` | 도메인 전문 에이전트 3종 |
| `rag_tools.py` | 공용 도구 — Pinecone 벡터 검색, 팩트체크 |
| `obsidian_tools.py` | 옵시디언 MCP 연결 (읽기 전용) |
| `general_it_trend_topic_agent.py` | 웹 검색 기반 IT 트렌드 에이전트 |
| `models.py` | 역할별 모델 배정 + 요약 미들웨어 |
| `state.py` | 그래프 상태 정의 |
| `async_utils.py` | 동기 코드에서 코루틴 실행 |
| `app.py` | Streamlit UI |
| `pre_process_save_document.py` | PDF → Pinecone 색인 (최초 1회) |

## 옵시디언 — 읽기 전용

볼트는 **절대 읽기 전용**이다. 플러그인이 쓰기/삭제 도구까지 16종을 노출하지만,
`OBSIDIAN_TOOL_NAMES` 화이트리스트에 있는 5종만 에이전트에 전달한다.

```
통과: search_simple, search_query, vault_read, vault_list, vault_get_document_map
차단: vault_write, vault_append, vault_patch, vault_delete, vault_move,
      vault_copy, command_execute, open_file, command_list, tag_list,
      active_file_get_path
```

블랙리스트가 아니라 화이트리스트라, 플러그인이 새 쓰기 도구를 추가해도 자동 차단된다.
테스트 2개로 고정되어 있다.

### MOC 진입점

볼트에 기술 태그가 없다(태그 27개가 전부 농업/경제/공고문 계열). 대신 `#MOC` 태그가
붙은 Map of Content 노트 33개가 도메인 허브라서, 이걸 진입점으로 쓴다.
경로는 `obsidian_tools.MOC_ENTRYPOINTS` 에 있고 실존 여부를 테스트로 검증한다.

## 환경변수

`app/.env` 에서 읽는다.

| 변수 | 기본값 | 용도 |
|---|---|---|
| `OBSIDIAN_MCP_URL` | `https://127.0.0.1:27124/mcp/` | 플러그인 MCP 엔드포인트 |
| `OBSIDIAN_API_KEY` | — | 플러그인 설정에서 복사 |
| `AGENT_ORCHESTRATOR_MODEL` | `claude-opus-5` | 의도 분류 |
| `AGENT_JUDGE_MODEL` | `claude-opus-5` | 분해·병합·팩트체크 |
| `AGENT_DOMAIN_MODEL` | `claude-sonnet-5` | 도메인 3종 |
| `AGENT_TREND_MODEL` | `claude-sonnet-5` | 웹 검색 |
| `SUMMARY_TRIGGER_TOKENS` | `120000` | 요약 발동 임계치 |
| `SUMMARY_KEEP_MESSAGES` | `40` | 요약 후 원문 유지 개수 |
| `LITELLM_BASE_URL` | — | 있으면 게이트웨이 경유 |

`get_model()` 은 정의된 role(`orchestrator`/`judge`/`domain`/`trend`/`default`)만 받는다.
`"Opus"` 같은 이름을 넘기면 조용히 기본값으로 떨어지므로 가드 테스트로 막아둔다.

## 관측 (Langfuse)

`http://localhost:3300` 에서 두 계층의 기록을 볼 수 있다.

| 계층 | 트레이스 이름 | 범위 |
|---|---|---|
| 앱 | `main_agent` | 그래프 노드와 병렬 도메인 스팬(`domain:server_infra` 등)까지 |
| 게이트웨이 | `litellm-acompletion` | LiteLLM 을 통과한 **모든** LLM 호출 (앱 밖 호출 포함) |

게이트웨이 계층은 `infra/litellm.config.yaml` 의 `success_callback: ["langfuse"]` 로 켠다.
중복이 아니라 관점이 다르다 — 앱 트레이스는 에이전트 흐름을, 게이트웨이 트레이스는
모델별 호출량·비용을 본다.

## 테스트

```powershell
# LLM/외부 API 호출 없는 테스트
..\.venv\Scripts\python.exe -m pytest -q

# 실제 API 를 호출하는 테스트 (Obsidian 실행 필요, 비용 발생)
..\.venv\Scripts\python.exe -m pytest -q -m llm
```

## 알아둘 것

- **요약 임계치를 낮추지 말 것.** MOC 노트 1개가 ~852 토큰이라, 8000 으로 두면
  노트 9~10개만 읽어도 요약이 발동해 조사 중인 도구 결과가 날아간다.
- **노드는 부분 dict 를 반환할 것.** state 객체를 통째로 반환하면 `operator.add`
  리듀서가 중복 누적한다(도메인 답변 3개가 12개가 됐다).
- **Pinecone 인덱스는 `Index(host=...)` 로 열 것.** 위치 인자로 주면 인덱스 이름으로
  해석되어 404 가 난다.
- **팩트체크는 `max_tokens` 를 크게 줄 것.** 문장마다 판정을 내므로 출력이 길다.
  기본 4096 이면 잘려서 파싱이 실패하고, 그 실패가 정확도 0.0 으로 반환되어
  멀쩡한 답변을 재작성시킨다. `[검증불가]` 접두사로 실패와 부정확을 구분한다.
- **재질문 상태는 프로세스 수명까지만 유지된다.** `InMemorySaver` 를 쓰므로 앱을
  재시작하면 진행 중이던 대화가 사라진다. 영속이 필요하면 `PostgresSaver` 로 교체.
