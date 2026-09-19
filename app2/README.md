# app2 — Claude Agent SDK 버전

`app/` 의 LangGraph 에이전트 시스템을 **Claude Agent SDK** 로 다시 구현한 것.
화면과 동작은 같고 내부 구조만 다르다.

## 실행

```powershell
cd app2
..\.venv\Scripts\python.exe -m streamlit run app.py
```

전제조건 2가지:

- `claude` CLI 설치 (Agent SDK 가 이 프로세스를 띄운다) — 이미 설치되어 있음
- API 키 — `app2/.env` 가 있으면 그걸, 없으면 `app/.env` 를 그대로 읽는다

## 구조

| 파일 | 역할 | app/ 의 대응 |
|---|---|---|
| `tools.py` | 커스텀 도구 4종을 in-process MCP 서버로 노출 | 각 에이전트의 `@tool` 들 |
| `agents.py` | 라우터 프롬프트 + 서브에이전트 2종 정의 | `main_agent.py` + 서브그래프 2개 |
| `runner.py` | async `query()` 를 Streamlit(동기)에서 쓰도록 감싼 어댑터 | `graph.stream(...)` 루프 |
| `app.py` | Streamlit UI | `app/app.py` |

### 라우팅

LangGraph 버전은 `query_classifier` 노드가 구조화 출력으로 의도를 분류하고
조건부 엣지로 서브그래프를 골랐다. Agent SDK 에서는 라우터 에이전트가
시스템 프롬프트의 분류 기준을 읽고 `Agent`(Task) 도구로 서브에이전트에
위임한다. 분류 기준(3종 의도와 예시)은 `main_agent.py` 의 것을 그대로 옮겼다.

```
사용자 질문
  └─ 라우터 (ROUTER_PROMPT)
       ├─ complex_discussion_topic  → complex-discussion-topic 서브에이전트
       │                               vector_search (Pinecone + Cohere 리랭크)
       │                               fact_check    (Anthropic 구조화 출력)
       ├─ general_it_trend_topic    → general-it-trend-topic 서브에이전트
       │                               web_search    (Serper)
       │                               fetch_webpage (WebBaseLoader)
       └─ unknown                   → 라우터가 직접 응답
```

### 도구

`@tool` 로 정의하면 `mcp__research__<도구명>` 으로 노출된다. 핸들러는 async 여야
하므로 Pinecone/langchain 같은 동기 호출은 `asyncio.to_thread` 로 감싼다.

## app/ 과의 의도적 차이

- **질의 확장 도구 없음.** `app/` 에는 질문을 여러 개로 불리는 `get_multiple_queries`
  / `expand_query` 가 있었고, 이들은 내부에서 별도 LLM 을 호출했다. Agent SDK 에서는
  에이전트 자신이 LLM 이라 같은 일을 하려고 LLM 을 또 부르는 건 낭비다. 대신
  서브에이전트 프롬프트에서 "질의를 바꿔가며 여러 번 검색하라"고 지시한다.
  실제로 한 질문에 `vector_search` 를 13회까지 호출한다.
- **모델이 Opus 5.** `app/` 은 `claude-sonnet-5` 를 하드코딩했다. `AGENT_MODEL`
  환경변수로 바꿀 수 있다.
- **대화 이어가기.** `ResultMessage.session_id` 를 저장해 다음 턴에 `resume` 으로
  넘긴다. `app/` 은 매 턴 전체 메시지 배열을 다시 보냈다.

## 환경변수

| 변수 | 기본값 | 용도 |
|---|---|---|
| `AGENT_MODEL` | `claude-opus-5` | 라우터·서브에이전트 모델 |
| `AGENT_EFFORT` | `high` | 사고 깊이. `low`/`medium` 으로 낮추면 비용이 크게 준다 |
| `AGENT_MAX_TURNS` | `40` | 한 질문당 최대 턴 |
| `FACT_CHECK_MODEL` | `claude-opus-5` | `fact_check` 도구가 쓰는 모델 |

## 비용

`effort=high` + Opus 5 기준 실측:

| 질문 유형 | 턴 | 비용 |
|---|---|---|
| 인사말 (위임 없음) | 1 | $0.09 |
| 웹 검색 (도구 7회) | 2 | $0.55 |
| 논문 RAG (도구 14회) | 2 | $0.78 |

비용이 부담되면 `AGENT_EFFORT=medium` 또는 `AGENT_MODEL=claude-sonnet-5` 로 낮춘다.
