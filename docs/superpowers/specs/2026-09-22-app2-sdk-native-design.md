# app2 — Claude Agent SDK 네이티브 재설계

날짜: 2026-09-22
대상: `app2/` (Claude Agent SDK 버전). `app/`(LangGraph)은 건드리지 않는다.

## 배경

app2 는 app/ 의 LangGraph 그래프를 Agent SDK 위에 "옮겨 놓은" 상태다. 메인 에이전트에게
분류표·분해 규칙·병합 규칙을 프롬프트로 다시 적어 주고, "전달 완료" 프로토콜로 모델이 답을
쓰지 못하게 한 뒤 코드(`final_answer()`)가 최종 답을 고른다. 팩트체크는 모델이 잊지 않고
도구를 부르도록 PreToolUse·Stop 훅으로 감시한다. 재질문은 커스텀 도구로 턴을 끝낸 뒤
다음 턴에 세션을 resume 한다. 도구 안에서 LLM 이 필요하면 Claude Code 프로세스를 매번 띄운다.

공식 문서(code.claude.com/docs/en/agent-sdk)가 말하는 방식은 다르다.

- "Claude automatically decides when to invoke subagents based on the task and each
  subagent's `description`." — 오케스트레이터를 따로 쓰지 않는다. `query()`/`ClaudeSDKClient`
  의 메인 에이전트가 곧 오케스트레이터다.
- 재질문은 내장 `AskUserQuestion` 도구가 `can_use_tool` 콜백을 부르고, 앱이
  `PermissionResultAllow(updated_input={"questions": …, "answers": …})` 로 답을 돌려준다.
- 훅은 "코드가 강제해야 하는 규칙"에 쓴다. Stop 훅의 `stop_hook_active` 는 "한 번만 다시
  시킨다"는 의미론을 내장한다.
- 같은 프로세스 안의 멀티턴 대화는 `ClaudeSDKClient` 하나에 `query()` 를 반복한다.
- 웹 검색/페이지 읽기는 내장 `WebSearch`/`WebFetch` 도구가 있다.

이 스펙은 app2 를 그 방식으로 다시 짠다. 사용자 결정(2026-09-22):
**SDK 다움 > app/ 과의 문장·동작 일치**, 재질문은 **AskUserQuestion + 상주 client**,
트렌드 웹 도구는 **내장 WebSearch/WebFetch**.

## 목표

1. 오케스트레이션 로직(분류·분해·병합·최종 답 선택)이 코드와 프롬프트에서 사라지고
   서브에이전트 `description` 과 Claude Code 의 위임에 맡겨진다.
2. 재질문·팩트체크·권한이 각각 SDK 의 1차 메커니즘(AskUserQuestion, Stop 훅,
   can_use_tool) 하나로 구현된다.
3. 대화 하나 = `ClaudeSDKClient` 하나. resume 코드가 없다.
4. 도구 안에서 Claude Code 프로세스를 띄우는 곳은 팩트체크(턴당 최대 1회) 하나뿐이다.
5. 화면·CSS·문구, 도메인 서브에이전트 프롬프트, 팩트체크 프롬프트/스키마, 벡터 검색 도구,
   옵시디언 MCP 설정은 app/ 과 같게 유지한다.

## 비목표

- app/ 의 세부 동작 재현: 단일 서브에이전트 답을 글자 그대로 전달, "unknown" 고정 안내문의
  바이트 일치, decompose `[:3]`. 모델의 판단에 맡긴다.
- 오케스트레이션 규칙 문장의 app/ 과의 바이트 일치 검사.
- Serper/BeautifulSoup 기반 웹 도구 유지.
- LangSmith/Langfuse 연동 방식 변경(그대로 둔다).

## 구조

```
사용자 ─ Streamlit (app.py)
   │  AgentSession.send(prompt) → Event 스트림        (채팅 세션당 client 1개, 턴마다 client.query)
   ▼
메인 에이전트 (Opus 5, main_agent.py)                  ← 짧은 시스템 프롬프트. 위임 판단은 Claude Code
   │  tools=["Agent", "AskUserQuestion"]
   │  can_use_tool = 유일한 권한 게이트 (재질문 대기 / 화이트리스트 / 서브에이전트 전용 / 도메인당 1회)
   │  hooks: PostToolUse(Agent) 답 수집, Stop 팩트체크
   ├─ server_infra   ──▶ 옵시디언 MCP (인라인, 이 서브에이전트에만)
   ├─ system_design  ──▶ 옵시디언 MCP
   ├─ ai             ──▶ mcp__rag__vector_search_tool + 옵시디언 MCP
   └─ general_it_trend ─▶ 내장 WebSearch / WebFetch
```

### 메인 에이전트 (`main_agent.py`)

`build_options(turn)` 이 `ClaudeAgentOptions` 를 만든다. `isolated_options()` 는 유지.

- `system_prompt`: 다음만 담는다. 의도 분류표·분해 규칙·"전달 완료"·처리 과정 서술 금지 규칙은
  뺀다.
  - 역할: "IT 질문에 답하는 어시스턴트. 조사는 전문 서브에이전트에게 맡기고, 돌아온 답을
    하나의 글로 합쳐 사용자에게 답한다."
  - `AMBIGUITY_RULES` (app/ 문장 그대로). 재질문할 때는 AskUserQuestion 을 쓴다는 한 줄.
  - `MERGE_RULES` (app/ 문장 그대로). 서브에이전트가 하나면 그 답을 바탕으로 답한다는 한 줄.
  - 인사말·잡담이면 도구 없이 `GENERAL_REPLY` 문장으로 답한다는 한 줄.
  - 서브 질문이 여러 개면 한 응답에서 Agent 를 여러 번 불러 병렬로 맡긴다는 한 줄.
  - 직접 조사하지 않는다는 한 줄. 한국어로 답한다.
- `tools=["Agent", "AskUserQuestion"]` (내장 도구는 이 둘만).
- `agents=` 도메인 3종 + 트렌드 1종.
- `mcp_servers={"rag": rag_server()}` 만. `web`, `judge`, `orchestrator` 서버는 없어진다.
- `permission_mode="default"`, `allowed_tools=["Agent"]`. 나머지는 전부 `can_use_tool` 로 온다.
- `can_use_tool=turn.permission_gate` (아래).
- `hooks={"PostToolUse": [HookMatcher("Agent|Task", [turn.record_subagent_answer])],
  "Stop": [HookMatcher(hooks=[turn.fact_check_on_stop])]}`.
- `max_budget_usd=float(os.getenv("AGENT_MAX_BUDGET_USD", "3"))` — 위임 폭주 시 비용 상한.
- `env`: 지금과 같다(`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`, 출력 한도 16000, 게이트웨이,
  옵시디언 TLS 인증서). `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` 은 WebSearch 를 막지 않는지
  구현 중 실측하고, 막으면 뺀다.
- 최종 답 = `ResultMessage.result`. 비어 있으면 `EMPTY_REPLY`("답변을 생성하지 못했습니다.").

`stream_turn()` 은 없어지고 `AgentSession` (아래 "세션") 이 그 자리를 맡는다.
`short_tool_name()` 은 유지.

### 권한 게이트 (`can_use_tool`)

`TurnState.permission_gate(tool_name, input_data, context)` 한 함수가 모든 비허용 호출을 받는다.
순서대로 판정한다.

1. `AskUserQuestion`: `input_data["questions"]` 를 UI 로 보내고(Event `question`), 사용자의 답을
   `await` 한 뒤 `PermissionResultAllow(updated_input={"questions": 원본, "answers": {질문문: 답}})`.
   질문이 여러 개여도 UI 는 한 번에 하나씩 묻는다. 사용자가 대화를 초기화하면 대기가 취소되고
   client 가 닫힌다.
2. `Agent`/`Task` 호출인데 `subagent_type` 이 이번 턴에 이미 답한 도메인이면
   `PermissionResultDeny("… 답은 이미 받았습니다. 받은 답으로 병합하세요.")`. 실패해서 답이
   없는 도메인은 다시 허용한다. 트렌드 에이전트는 제한하지 않는다.
3. 조사 도구(`mcp__rag__*`, `mcp__obsidian__*`, `WebSearch`, `WebFetch`)는 `context.agent_id` 가
   있을 때만 허용. 없으면 `PermissionResultDeny("조사는 서브에이전트의 몫입니다. Agent 로 위임하세요.")`.
4. `mcp__obsidian__*` 는 `OBSIDIAN_TOOLS` 화이트리스트 5종만 허용(읽기 전용 보장. 지금의
   `deny_vault_writes` 훅이 여기로 옮겨 온다).
5. 그 외는 전부 `PermissionResultDeny`.

`allowed_tools=["Agent"]` 만 자동 승인이므로 위 게이트가 빠지는 호출은 없다.
`research_only_in_subagents`, `deny_vault_writes`, `limit_decomposition`, `limit_fact_check`,
`require_fact_check` 훅과 `request_clarification`, `fact_check` 도구는 삭제한다.

### 팩트체크 guardrail (`judge_agent.py`)

`TurnState.fact_check_on_stop(input_data, tool_use_id, context)` — Stop 훅.

- `input_data["stop_hook_active"]` 가 True 면 `{}` (두 번째 Stop. 재작성은 1회).
- 이번 턴에 수집된 도메인 답이 2개 미만이면 `{}`.
- 아니면 `input_data["last_assistant_message"]` 의 텍스트를 `answer`, 도메인 답들을 `context` 로
  `rag_tools.fact_check()` 를 부른다(구조화 출력 단발 `query()`, 지금과 같다).
  - `[검증불가]` 코멘트 → `{}` (검증 실패는 오답이 아니다).
  - `overall_accuracy >= 0.8` → `{}`.
  - 미만 → `{"decision": "block", "reason": FIX_RULES + "\n\n> 검증 결과\n" + 코멘트}`.
    모델은 이 이유를 보고 한 번 고쳐 쓴 뒤 다시 Stop 하고, 그때는 `stop_hook_active` 로 통과한다.
- 판정·점수는 `TurnState.fact_check` 에 남겨 UI/Langfuse 가 쓴다.

`last_assistant_message` 는 CLI 가 Stop 훅 입력에 넣어 주는 필드다(SDK 의 TypedDict 에는 없지만
dict 로 그대로 전달된다). 구현 첫 단계에서 실측으로 형식(문자열인지 블록 리스트인지)을 확인하고
`_response_text()` 로 텍스트를 뽑는다. 없으면 `transcript_path` 의 마지막 assistant 항목으로
대체한다.

`TurnState.record_subagent_answer` — PostToolUse(Agent). `agent_id` 가 없는(메인의) 호출만,
`subagent_type` 이 `DOMAINS` 면 `DomainAnswer(domain, prompt, 답 텍스트)` 를 모은다.
트렌드 답도 `trend_answers` 에 모은다(UI 표시용).

`judge_agent.py` 에 남는 것: `MERGE_RULES`, `FIX_RULES`, `GUARDRAIL_THRESHOLD`, `_DOMAIN_LABELS`,
`format_answers`, `_response_text`, `decide()` (PASS/REWRITE 판정 함수, Stop 훅이 쓴다).
`DECOMPOSE_RULES`, `Judge` 클래스, `FACT_CHECK_TOOL`, `JUDGE_SERVER` 는 삭제.

### 서브에이전트

- `domain_agents.py`: 프롬프트 3종은 app/ 과 같게 유지. `description` 은 자동 위임의 근거이므로
  "언제 부르는가"를 넣어 다듬는다(예: server_infra — "배포·컨테이너·네트워크·DB 운영·모니터링·
  장애 진단·용량 등 **운영 중인 시스템을 돌리는 층**의 질문이나 서브 질문을 맡긴다. 개인 옵시디언
  볼트만 근거로 답한다."). 도메인 정의 문장은 app/ DECOMPOSE_PROMPT 의 것을 그대로 쓴다.
  `tools`/`mcpServers`/`model`/`background=False` 는 지금과 같다.
- `general_it_trend_topic_agent.py`: `AgentDefinition(tools=["WebSearch", "WebFetch"])`.
  프롬프트는 app/ 문장을 바탕으로 도구 단계만 바꾼다(expand_query → "질문을 2~3개의 검색어로
  넓혀 WebSearch 를 여러 번", fetch_webpage_content → "WebFetch 로 본문 확인"). 나머지 역할·제약·
  Sources 규칙은 그대로. `expand_query`, `web_search_tool`, `fetch_webpage_content`, `web_server()`,
  `QUERY_EXPANSION_TEMPLATE`, `search_web`, `load_page`, `fetch_pages` 는 삭제한다.
  `WEB_TOOLS = ["WebSearch", "WebFetch"]`.
- `models.py`: `ask()` 삭제(호출처가 없어진다). `ask_structured()`, `one_shot_options()` 유지
  (팩트체크). 나머지 그대로.
- `rag_tools.py`: 그대로.
- `obsidian_tools.py`: `deny_vault_writes` 훅 삭제. 나머지(설정, 화이트리스트, TLS) 그대로.

### 세션과 UI

`async_utils.py` 를 `AgentSession` 으로 다시 쓴다.

```python
class AgentSession:
    """채팅 세션 하나. 백그라운드 스레드의 이벤트 루프에서 ClaudeSDKClient 를 상주시킨다."""
    def __init__(self, make_options: Callable[[TurnState], ClaudeAgentOptions]) -> None
    def send(self, prompt: str) -> Iterator[Event]      # 턴 실행. 다음 question 또는 done|error 까지 이벤트를 내준다
    def answer(self, text: str) -> None                 # AskUserQuestion 대기를 푼다
    def resume(self) -> Iterator[Event]                 # answer() 뒤 같은 턴의 나머지 이벤트를 이어 받는다
    def close(self) -> None                             # client 종료, 대기 취소, 스레드 종료
    @property
    def waiting_question(self) -> str | None            # UI 가 그릴 질문(있으면)
```

- 첫 `send()` 에서 스레드·루프·client 를 만들고 `connect()` 한다. 이후 `send()` 는 같은 client 에
  `query()` 한다(공식 문서의 멀티턴 패턴). 세션 ID 는 `ResultMessage.session_id` 로 관찰만 한다.
- `TurnState` 는 `send()` 마다 새로 만들고, 훅·게이트는 "현재 턴" 참조를 통해 그것을 본다
  (client 는 옵션을 한 번만 받으므로 훅 함수는 세션 소유, 상태는 턴 소유).
- 이벤트 종류: `tool_call`, `tool_result`, `answer`(메인의 텍스트 블록), `question`
  (AskUserQuestion 이 기다리는 중), `done`, `error`. `clarification` 은 `question` 으로 대체.
- `question` 이벤트가 오면 `send()` 의 이터레이터는 **끝난다**(UI 가 질문을 그리고 입력을 받아야
  하므로). 사용자가 답하면 UI 는 `answer(text)` 를 부르고 `resume()` 으로 같은 턴의 나머지 이벤트를
  이어 받는다. 즉 `send()`/`resume()` 둘 다 "다음 question 또는 done/error 까지" 를 내준다.
- `app.py`: `st.session_state.agent` 에 `AgentSession` 을 둔다. `pending_clarification`·
  `resume_value`·`agent_session_id`·`clarification_prompt()` 는 없어진다. 재질문 답이 들어오면
  `agent.answer(text)` 후 `agent.resume()` 을 그린다. "대화 초기화"는 `agent.close()`.
  화면 문구·CSS·상태 카드·Langfuse trace 구성은 그대로. 사이드바 캡션은
  "메인 에이전트 → 서브에이전트 자동 위임 · 병렬 → 병합 (Claude Agent SDK)".
- Streamlit 은 스크립트를 매번 다시 실행하지만 세션 상태의 스레드는 살아 있다. `st.*` 는
  스크립트 스레드에서만 부른다(지금 규칙 유지).

### 오류 처리

- `ResultMessage.is_error` 또는 `subtype != "success"` → `error` 이벤트. `error_max_budget_usd` 는
  "비용 상한(…$)에 도달했습니다" 로 풀어 쓴다.
- client 프로세스가 죽으면 `AgentSession` 은 다음 `send()` 에서 새 client 를 만든다(대화 맥락은
  잃는다. UI 에 "이전 대화 맥락을 잃었습니다" 한 줄).
- 팩트체크 `query()` 실패는 `[검증불가]` 로 돌아와 Stop 을 막지 않는다(지금과 같다).
- AskUserQuestion 대기 중 초기화 → 대기 Future 취소 → 게이트는 `PermissionResultDeny` 를 돌려주고
  client 를 닫는다.

## 테스트

오프라인(기본):

| 파일 | 확인하는 것 |
|---|---|
| `test_main_agent.py` | 옵션 배선(tools/agents/allowed_tools/permission_mode/hooks/max_budget_usd), 시스템 프롬프트에 분류표·"전달 완료"가 없고 AMBIGUITY/MERGE 규칙이 있음, `permission_gate` 5단계 판정, 메시지 스트림 → 이벤트(nested 구분), 빈 결과 → EMPTY_REPLY, 오류 결과 → error |
| `test_judge_agent.py` | `decide()` PASS/REWRITE/검증불가, Stop 훅: 도메인 답 <2 통과 / `stop_hook_active` 통과 / 미달 시 block+FIX_RULES / `last_assistant_message` 문자열·블록 두 형식, PostToolUse 수집(메인 호출만, 도메인·트렌드 구분) |
| `test_domain_agents.py` | 프롬프트 app/ 과 동일, description 에 도메인 정의 문장 포함, 도구·MCP 배선 |
| `test_general_it_trend.py` | `tools == ["WebSearch", "WebFetch"]`, 프롬프트에 역할·제약·Sources 규칙 유지, 커스텀 웹 도구 부재 |
| `test_async_utils.py` (신규) | 가짜 client 로 `AgentSession`: 두 번 `send()` 가 같은 client 에 `query()`, question → 이터레이터 종료 → `answer()`+`resume()` 로 이어짐, `close()` 가 대기를 취소 |
| `test_app_ui.py` | AppTest: 질문·도구 과정 표시·재질문(question) → 답 → 같은 턴 이어짐·오류·초기화 |
| `test_parity.py` | 유지: 도메인 프롬프트, 볼트 상수, 모델 역할, 팩트체크 프롬프트/스키마, `vector_search_tool` 스키마, AMBIGUITY/MERGE/FIX 문장, threshold, UI. 삭제: INTENT/DECOMPOSE 검사, 고정 안내문 검사, 웹 도구 스키마 검사, 트렌드 프롬프트 바이트 일치(→ "역할·제약 문단 포함" 으로 완화) |
| `test_models.py`, `test_rag_tools.py`, `test_obsidian_tools.py` | 삭제된 심볼(`ask`, `deny_vault_writes`) 검사만 제거 |

라이브(`-m llm`, 비용 발생) `test_live.py`:

1. 인사말 → 도구 호출 없이 안내 문장으로 답한다(문장 포함 검사, 바이트 일치 아님).
2. 개념 질문 → `general_it_trend` 에 위임되고 WebSearch 가 호출된다.
3. 규모·제약이 전혀 없는 설계 요청 → AskUserQuestion 이 게이트에 도착한다(테스트 게이트가 답을
   넣어 주면 턴이 이어진다).
4. (신규) 두 도메인에 걸친 질문 → Agent 가 2회 이상 병렬 호출되고 Stop 훅이 팩트체크를 1회 돌린다.

## 구현 순서(개요)

1. 실측 스파이크: `tools=["Agent","AskUserQuestion"]` + `permission_mode="default"` 에서
   `can_use_tool` 이 AskUserQuestion 을 받는지, Stop 훅 입력의 `last_assistant_message` 형식,
   `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` 하에서 WebSearch 동작. 결과를 README "실측" 절에 적는다.
2. `state.py`(TurnState·Event), `judge_agent.py`(Stop 훅·수집), `main_agent.py`(옵션·게이트).
3. 서브에이전트 정의 수정(트렌드 → 내장 웹 도구, description).
4. `async_utils.py` → `AgentSession`, `app.py` 연결.
5. 테스트 정리·추가, README 갱신(구조도·대응표·"실측" 절).

## README 대응표(갱신본 요지)

| app/ (LangGraph) | app2/ (Agent SDK) |
|---|---|
| `classify` + `route_by_intent` | 없음. 서브에이전트 `description` 을 보고 Claude Code 가 위임 |
| `decompose` + `Send` fan-out | 메인 에이전트가 한 응답에서 Agent 여러 번 → 병렬 |
| `merge` LLM 호출 | 메인 에이전트의 응답 자체 |
| `_guard` (1회 재작성) | Stop 훅 + `stop_hook_active` |
| `interrupt()` / `Command(resume)` | `AskUserQuestion` + `can_use_tool` 대기 |
| `InMemorySaver` | `ClaudeSDKClient` 상주(대화당 1개) |
| 웹 도구 3종 | 내장 `WebSearch` / `WebFetch` |
| `[:3]`, 읽기 전용, 조사 도구 격리 | `can_use_tool` 한 함수 |
