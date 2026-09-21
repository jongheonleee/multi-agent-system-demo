# app2 — 멀티 에이전트 IT 어시스턴트 (Claude Agent SDK)

`app/`(LangGraph)과 **기능·화면·프롬프트가 같은** 앱을 Claude Agent SDK 방식으로 다시 만든 것.
그래프를 코드로 짜지 않고, 오케스트레이터 에이전트 하나에 서브에이전트·도구·훅을 쥐여준다.

## 실행

```powershell
cd app2
..\.venv\Scripts\python.exe -m streamlit run app.py
```

- 키는 `app2/.env` 가 있으면 그것을, 없으면 `app/.env` 를 읽는다.
- `LITELLM_BASE_URL` 이 있으면 게이트웨이를 경유한다(app/ 과 같은 규칙). Claude Code 는
  Anthropic Messages API(`/v1/messages`)로 붙는데, LiteLLM 이 같은 포트에서 제공한다.
- API 키도 게이트웨이도 없으면 Claude Code 는 이 PC 의 Claude 로그인을 쓴다.
- 전제조건은 app/ 과 같다: Obsidian 실행 중, infra 스택(LiteLLM·Langfuse).

## 구조

```
사용자 ─ Streamlit (app.py)
   │  stream_turn(prompt, session_id)       ← 세션 resume 으로 대화가 이어진다
   ▼
오케스트레이터 (Opus 5, main_agent.py)       ← 분류·분해·병합 규칙은 프롬프트, 강제 규칙은 훅
   │  Agent 도구를 한 응답에서 여러 번 → Claude Code 가 서브에이전트를 병렬 실행
   ├─ server_infra  ──▶ 옵시디언 MCP (이 서브에이전트에만 연결)
   ├─ system_design ──▶ 옵시디언 MCP
   ├─ ai            ──▶ vector_search_tool (in-process) + 옵시디언 MCP
   ├─ general_it_trend ─▶ expand_query / web_search_tool / fetch_webpage_content (in-process)
   ├─ fact_check (in-process)                ← 병합 답 검증, 1회 재작성
   └─ request_clarification (in-process)     ← 재질문 후 턴 종료, 다음 턴에 resume
```

### LangGraph → Agent SDK 대응

| app/ (LangGraph) | app2/ (Agent SDK) |
|---|---|
| `classify` 노드 + `route_by_intent` | 오케스트레이터 프롬프트(의도 정의·재질문 기준은 app/ 문장 그대로) |
| `decompose` + `Send` fan-out | 한 응답에서 `Agent` 도구 여러 번 호출 → 병렬 서브에이전트 |
| `create_agent` 도메인 에이전트 3종 | `AgentDefinition` 3종 (`domain_agents.py`) |
| 트렌드 서브그래프 | `AgentDefinition` 1개 (`general_it_trend_topic_agent.py`) |
| `merge` LLM 호출 | 오케스트레이터가 직접 병합 (병합 규칙은 app/ 문장 그대로) |
| `decompose` 의 `[:3]` | PreToolUse 훅 — 도메인당 1회, 최대 3개 (실패한 위임은 재시도 허용) |
| `_guard` (fact_check → 1회 재작성) | `fact_check` 도구 + PreToolUse(1회 제한) + Stop 훅(검증 누락 차단) |
| `interrupt()` + `Command(resume)` | `request_clarification` 도구 + 세션 resume |
| `InMemorySaver` (thread_id) | Claude Code 세션 (`session_id`) |
| `langchain_mcp_adapters` 변환 | MCP 서버 설정(dict)만 넘김 — Claude Code 가 MCP 클라이언트 |
| `SummarizationMiddleware` | Claude Code 자동 컴팩션 (임계치 120k 로 맞춤) |
| `graph.stream(subgraphs=True)` | `ClaudeSDKClient` 메시지 스트림 (`parent_tool_use_id` 로 중첩 구분) |
| LangChain `CallbackHandler` (Langfuse) | 메시지 스트림으로 trace 구성 (`app.py` 의 `LangfuseTurn`) |
| LangSmith 자동 추적 | `langsmith.integrations.claude_agent_sdk` 공식 연동 |

### 파일

app/ 과 파일 이름이 같다. 역할도 같은 자리에 있다.

| 파일 | 역할 |
|---|---|
| `main_agent.py` | 오케스트레이터 프롬프트·옵션·훅 배선, `stream_turn` |
| `judge_agent.py` | 분해/병합 규칙, `fact_check` 도구, guardrail 훅 |
| `domain_agents.py` | 도메인 서브에이전트 3종 (프롬프트는 app/ 과 글자까지 같음) |
| `general_it_trend_topic_agent.py` | 트렌드 서브에이전트 + 웹 도구 3종 |
| `rag_tools.py` | `vector_search_tool`, 팩트체크 (Pinecone/OpenAI/Cohere 공식 클라이언트) |
| `obsidian_tools.py` | 옵시디언 MCP 설정, 읽기 전용 훅 |
| `models.py` | 역할별 모델, 게이트웨이/실행 환경, 단발 `query()` 헬퍼 |
| `state.py` | 턴 상태(TurnState), UI 이벤트 |
| `async_utils.py` | Streamlit(동기)에서 async 스트림 소비 |
| `app.py` | Streamlit UI (CSS·테마·문구 app/ 과 동일) |
| `pre_process_save_document.py` | PDF → Pinecone 색인 (app/ 과 동일 파일) |

## Agent SDK 로 바꾸며 얻은 것

- **오케스트레이션 코드가 사라졌다.** 노드·엣지·리듀서·Send·체크포인터가 없다. 분기 규칙은
  프롬프트에, 반드시 지켜야 하는 규칙만 훅에 있다.
- **리듀서 버그가 구조적으로 없다.** app/ 은 같은 thread 에서 `clarification` 이 다음 턴까지 남고
  `domain_answers`(operator.add)가 턴마다 누적되어 이전 턴 답까지 병합된다(실측). app2 는 턴 상태를
  매 턴 새로 만들고, 대화 맥락은 세션이 들고 있다.
- **도구 스코핑이 선언적이다.** 옵시디언 MCP 는 `AgentDefinition.mcpServers` 에 인라인으로 넣어
  도메인 서브에이전트에게만 보인다. 오케스트레이터 컨텍스트에는 볼트 도구가 아예 없다.
- **권한이 세 겹이다.** `AgentDefinition.tools`(보이는 도구) + `allowed_tools`/`dontAsk`(실행 허가)
  + PreToolUse 훅(화이트리스트 밖 deny). 모두 화이트리스트라 플러그인이 쓰기 도구를 추가해도 막힌다.
- **TLS 검증을 끄지 않는다.** app/ 은 httpx `verify=False` 였다. app2 는 플러그인 인증서를 받아
  `NODE_EXTRA_CA_CERTS` 로 그 인증서만 추가 신뢰한다.
- **컨텍스트 격리.** 서브에이전트는 자기 컨텍스트에서 노트 수십 개를 읽고 최종 답만 돌려준다.
  오케스트레이터 컨텍스트가 도구 결과로 불어나지 않는다.

## 실측으로 알게 된 것 (함정)

- **`tools=[]` 는 `Agent` 도구까지 끈다.** 오케스트레이터가 위임을 못 하고 직접 도구를 부른다.
  `tools=["Agent"]` 로 연다.
- **서브에이전트가 백그라운드로 뜬다.** 기본값이면 "Async agent launched" 만 받고 결과 전에 턴이
  끝난다. `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` + `AgentDefinition(background=False)`.
- **출력 한도 초과는 에러다.** app/ 은 4096 에서 잘린 답을 돌려주지만 Claude Code 는 서브에이전트를
  종료시킨다("response exceeded the 4096 output token maximum"). 에이전트 세션은 16000 으로 둔다.
  도구 안의 단발 호출은 app/ 과 같은 4096(팩트체크 16000).
- **분해 규칙은 프롬프트만으로 안 지켜진다.** 재질문 후 이어간 턴에서 오케스트레이터가 답을 받은 뒤
  같은 도메인에 보충 조사를 다시 맡겨 위임이 5회가 됐다($8.5, 15분). 도메인당 1회·최대 3개를 훅으로 강제한다.
- **최종 답은 코드가 고른다.** 오케스트레이터에게 "받은 답을 그대로 전달하라"고 하면 다시 받아쓰거나
  요약을 덧붙인다(실측: PASS 뒤 "앞서 드린 정리가 최종 답변입니다…"). 서브에이전트 답은 PostToolUse
  훅으로 모으고, app/ 의 merge/finalize 규칙대로 `final_answer()` 가 고른다.
  단일 답 → 그대로 / 병합+PASS → 검증한 병합본 / REWRITE·안내문 → 오케스트레이터의 마지막 답.
- **`st.session_state` 는 스크립트 스레드에서만 읽힌다.** 백그라운드 스레드의 람다 안에서 읽으면
  매 턴 오류가 난다(UI 테스트로 발견).
- **사용자 PC 설정이 섞인다.** 기본값이면 `~/.claude` 의 CLAUDE.md·훅·플러그인·스킬이 에이전트에 붙는다.
  `setting_sources=[]`, `skills=[]`, `strict_mcp_config=True` 로 격리한다.

## app/ 과 의도적으로 다른 점

- **리듀서 누적 동작을 재현하지 않았다.** 위 "리듀서 버그" 참고. 버그라서 옮기지 않았다.
- **트렌드 에이전트가 이전 대화를 직접 보지 않는다.** app/ 은 전체 메시지를 넘기고, app2 는
  오케스트레이터가 필요한 맥락을 위임 prompt 에 덧붙인다(대화 자체는 오케스트레이터 세션에 있다).
- **도구 결과 형식.** 벡터/웹 결과를 LangChain `Document` repr 대신 JSON 으로 준다. 내용은 같다
  (벡터 검색은 같은 문서·같은 순서·같은 점수 — 라이브 parity 테스트로 확인).
- **모델이 보는 도구 이름**은 `mcp__<서버>__<도구>` 형태다(예: `mcp__obsidian__vault_read`).
  UI 에는 app/ 처럼 `vault_read` 로 보인다.

## 테스트

```powershell
cd app2
# 오프라인 (Claude Code/외부 API 호출 없음)
..\.venv\Scripts\python.exe -m pytest -q

# 실제 Claude Code 로 end-to-end (비용 발생). API 키 대신 이 PC 의 Claude 로그인으로:
$env:APP2_TEST_USE_CLAUDE_LOGIN="1"; ..\.venv\Scripts\python.exe -m pytest -q -m llm

# 실행 중인 Obsidian 이 필요한 테스트
..\.venv\Scripts\python.exe -m pytest -q -m obsidian
```

| 파일 | 확인하는 것 |
|---|---|
| `test_parity.py` | app/ 을 별도 프로세스로 import 해 프롬프트·도구 스키마·팩트체크 스키마·상수·UI 문자열이 **바이트 단위로 같은지**, 벡터 검색 결과가 같은지(llm) |
| `test_main_agent.py` | 옵션 배선, 조사 도구 차단 훅, 메시지 스트림 → UI 이벤트, 최종 답 선택 규칙 |
| `test_judge_agent.py` | PASS/REWRITE/검증불가 분기, 분해 1회·최대 3개, 검증 1회 제한, Stop 훅 |
| `test_obsidian_tools.py` | 화이트리스트 훅, MCP 설정, 인증서 신뢰 |
| `test_app_ui.py` | Streamlit AppTest — 질문·도구 과정 표시·재질문→resume·오류·초기화 |
| `test_live.py` | (llm) 인사말 고정 응답 / 개념 질문→트렌드 위임 / 모호한 요청→재질문 |
