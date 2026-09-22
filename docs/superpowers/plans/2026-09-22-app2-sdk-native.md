# app2 Agent SDK 네이티브 재설계 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** app2 에서 손으로 짠 오케스트레이션(분류·분해·병합 프롬프트, 코드의 최종 답 선택, 커스텀 재질문 도구, 팩트체크 도구+감시 훅)을 걷어내고 Claude Agent SDK 의 1차 메커니즘(서브에이전트 자동 위임, AskUserQuestion + can_use_tool, Stop 훅, 상주 ClaudeSDKClient, 내장 WebSearch/WebFetch)으로 같은 앱을 다시 만든다.

**Architecture:** 메인 에이전트(`ClaudeSDKClient`)가 곧 오케스트레이터다. 코드는 (1) 짧은 시스템 프롬프트, (2) `AgentDefinition` 4종, (3) 권한 게이트 한 함수 `can_use_tool`, (4) guardrail 훅 2개(PostToolUse 수집, Stop 팩트체크)만 준다. 대화 하나는 백그라운드 스레드의 이벤트 루프에서 상주하는 client 하나이고, Streamlit 은 그 세션 객체에 턴을 보내고 이벤트를 그린다.

**Tech Stack:** Python 3.12, `claude-agent-sdk` 0.2.152 (번들 CLI 2.1.259), Streamlit, pytest (+ `streamlit.testing.v1.AppTest`), pydantic. 가상환경 `..\.venv` (app2 에서 `..\.venv\Scripts\python.exe -m pytest -q`).

**Spec:** `docs/superpowers/specs/2026-09-22-app2-sdk-native-design.md`

## Global Constraints

- 파일 이름은 app/ 과 같게 유지한다 (`test_parity.py::test_same_module_filenames_as_app`). 새 파일은 테스트만 추가한다.
- 유지(바이트 일치): `domain_agents.DOMAIN_PROMPTS`, `obsidian_tools` 상수, `models._ROLE_DEFAULTS`/`SUMMARY_TRIGGER_TOKENS`, `rag_tools.FACT_CHECK_PROMPT`/`FactCheckResult`/`VERIFY_FAILED_COMMENT_PREFIX`, `vector_search_tool` 스키마, `main_agent.AMBIGUITY_RULES`, `judge_agent.MERGE_RULES`/`FIX_RULES`/`GUARDRAIL_THRESHOLD`/`_DOMAIN_LABELS`, `GENERAL_REPLY`, `EMPTY_REPLY`, app.py 의 `CSS`/`EXAMPLE_QUESTIONS`/`set_page_config`, `.streamlit/config.toml`.
- `st.*` 는 스크립트 스레드에서만 부른다. 백그라운드 스레드에서는 `st.session_state` 를 읽지 않는다.
- 모든 Claude Code 실행은 `models.isolated_options()` 를 거친다(`setting_sources=[]`, `skills=[]`, `strict_mcp_config=True`).
- 서브에이전트는 동기 실행(`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`, `background=False`), 출력 한도 16000.
- 커밋 메시지 끝에 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` 를 붙인다.
- 오프라인 테스트 명령: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q` (llm/obsidian 마커는 기본 제외).
- 라이브 테스트: `$env:APP2_TEST_USE_CLAUDE_LOGIN="1"; ..\.venv\Scripts\python.exe -m pytest -q -m llm` (비용 발생. 자격 증명이 없으면 건너뛰고 그 사실을 보고한다).

---

## 파일 구조

| 파일 | 책임 (변경 후) |
|---|---|
| `app2/state.py` | `DomainAnswer`, `TurnState`(한 턴 동안 훅이 모으는 값), `Event`(UI 이벤트) |
| `app2/judge_agent.py` | 병합/재작성 규칙 문장, `decide()`, `response_text()`, `last_assistant_text()`, `Guardrail`(PostToolUse 수집 + Stop 팩트체크) |
| `app2/main_agent.py` | `SYSTEM_PROMPT`, `MainAgent`(권한 게이트, 옵션, 메시지→이벤트), `build_options()`, `new_session()` |
| `app2/async_utils.py` | `AgentSession`(상주 스레드/루프/client, send/answer/resume/close), `TurnDriver` 프로토콜, `SessionClosed` |
| `app2/domain_agents.py` | 도메인 서브에이전트 3종(프롬프트 유지, description 다듬음) |
| `app2/general_it_trend_topic_agent.py` | 트렌드 서브에이전트(내장 WebSearch/WebFetch) |
| `app2/models.py` | 모델 배정, 실행 환경, `ask_structured()` (팩트체크용). `ask()` 삭제 |
| `app2/obsidian_tools.py` | MCP 설정, 화이트리스트, `is_allowed_vault_tool()`. 훅 삭제 |
| `app2/rag_tools.py` | 변경 없음 |
| `app2/app.py` | Streamlit UI. `AgentSession` 을 세션 상태에 두고 question/answer/resume 흐름 |
| `app2/tests/test_async_utils.py` | (신규) `AgentSession` 가짜 client 테스트 |
| 나머지 테스트 | 각 태스크에서 갱신 |

---

### Task 1: 라이브 스파이크 — SDK 가 실제로 주는 것 확인

설계가 기대는 세 가지 사실을 실측한다. 자격 증명이 없으면 이 태스크는 "실행 못 함"으로 보고하고 다음으로 간다(코드는 폴백을 갖는다).

**Files:**
- Modify: `app2/tests/test_live.py` (파일 끝에 추가)

**Interfaces:**
- Consumes: `models.isolated_options`, `models.base_env`
- Produces: 없음 (사실 확인). 결과는 Task 8 에서 README 에 적는다.

- [ ] **Step 1: 스파이크 테스트 추가**

`app2/tests/test_live.py` 끝에 추가:

```python
# ---------------------------------------------------------------------------
# SDK 실측 — 설계가 기대는 사실. 실패하면 설계를 고쳐야 한다.
# ---------------------------------------------------------------------------
def test_ask_user_question_reaches_can_use_tool_in_default_mode():
    """tools=["AskUserQuestion"] + permission_mode="default" 면 AskUserQuestion 이 can_use_tool 로 온다."""
    import asyncio

    from claude_agent_sdk import ClaudeSDKClient, PermissionResultAllow, ResultMessage

    from models import AGENT_MAX_TOKENS, base_env, isolated_options

    seen = {}

    async def gate(tool_name, input_data, context):
        seen["tool"] = tool_name
        seen["input"] = input_data
        questions = input_data.get("questions") or []
        return PermissionResultAllow(
            updated_input={"questions": questions, "answers": {q["question"]: "파란색" for q in questions}}
        )

    async def run():
        options = isolated_options(
            model="claude-sonnet-5",
            system_prompt="사용자에게 좋아하는 색을 AskUserQuestion 으로 물은 뒤, 그 색을 한 단어로 답하세요.",
            tools=["AskUserQuestion"],
            permission_mode="default",
            can_use_tool=gate,
            env=base_env(AGENT_MAX_TOKENS),
        )
        async with ClaudeSDKClient(options=options) as client:
            await client.query("시작")
            async for m in client.receive_response():
                if isinstance(m, ResultMessage):
                    return m

    result = asyncio.run(run())
    assert seen.get("tool") == "AskUserQuestion", seen
    assert seen["input"]["questions"][0]["question"]
    assert "파란" in (result.result or "")


def test_stop_hook_input_carries_last_assistant_message():
    """Stop 훅 입력에 last_assistant_message(또는 transcript_path)가 있다."""
    import asyncio

    from claude_agent_sdk import ClaudeSDKClient, HookMatcher, ResultMessage

    from models import AGENT_MAX_TOKENS, base_env, isolated_options

    seen = {}

    async def on_stop(input_data, tool_use_id, context):
        seen.update(input_data)
        return {}

    async def run():
        options = isolated_options(
            model="claude-sonnet-5",
            system_prompt="숫자만 답하세요.",
            tools=[],
            hooks={"Stop": [HookMatcher(hooks=[on_stop])]},
            env=base_env(AGENT_MAX_TOKENS),
        )
        async with ClaudeSDKClient(options=options) as client:
            await client.query("1+1은?")
            async for m in client.receive_response():
                if isinstance(m, ResultMessage):
                    return m

    asyncio.run(run())
    assert "stop_hook_active" in seen
    assert "last_assistant_message" in seen or "transcript_path" in seen, sorted(seen)
    print("STOP HOOK INPUT KEYS:", sorted(seen), "last_assistant_message=", repr(seen.get("last_assistant_message"))[:300])
```

- [ ] **Step 2: 실행**

Run: `cd app2; $env:APP2_TEST_USE_CLAUDE_LOGIN="1"; ..\.venv\Scripts\python.exe -m pytest -q -m llm tests/test_live.py -k "ask_user_question or stop_hook_input" -s`
Expected: 2 passed. 출력의 `STOP HOOK INPUT KEYS` 줄에서 `last_assistant_message` 의 형식(문자열인지 `{"role":..., "content":[...]}` 인지)을 적어 둔다 → Task 3 의 `response_text` 가 두 형식을 모두 처리하므로 코드 변경은 없지만 Task 8 README 에 기록한다.

실행할 수 없으면(자격 증명 없음): 그대로 두고 다음 태스크로. Task 3 의 `last_assistant_text()` 는 `transcript_path` 폴백을 갖는다.

- [ ] **Step 3: 커밋**

```bash
git add app2/tests/test_live.py
git commit -m "test(app2): SDK 실측 스파이크 — AskUserQuestion 게이트, Stop 훅 입력

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `AgentSession` — 상주 client 와 question/answer/resume

**Files:**
- Rewrite: `app2/async_utils.py`
- Create: `app2/tests/test_async_utils.py`

**Interfaces:**
- Consumes: `state.Event`, `claude_agent_sdk.ClaudeSDKClient`, `claude_agent_sdk.ResultMessage`
- Produces:
  - `class TurnDriver(Protocol)`: `options(ask_user) -> ClaudeAgentOptions`, `begin_turn() -> Any`, `events(message) -> list[Event]`, `result(result: ResultMessage | None) -> Event`
  - `AskUser = Callable[[list[dict]], Awaitable[dict[str, str]]]`
  - `class SessionClosed(RuntimeError)`
  - `class AgentSession(driver: TurnDriver, client_factory=ClaudeSDKClient)`: `send(prompt) -> Iterator[Event]`, `answer(text) -> None`, `resume() -> Iterator[Event]`, `close() -> None`, `async ask_user(questions) -> dict[str, str]`, 속성 `waiting_question: str | None`, `session_id: str | None`, `lost_context: bool`
  - `run_coro(coro)` 유지 (pre_process_save_document 등 다른 곳에서 쓸 수 있다)

- [ ] **Step 1: 실패하는 테스트 작성**

`app2/tests/test_async_utils.py`:

```python
"""AgentSession — 상주 client 위에서 send / question → answer → resume / close 가 맞게 도는지.

Claude Code 없이 가짜 client 로 돈다. 가짜 client 의 스크립트 항목이 callable 이면
"모델이 AskUserQuestion 을 불렀다"는 뜻으로 options 를 넘겨 await 한다.
"""
import threading

import pytest
from claude_agent_sdk import ResultMessage

from async_utils import AgentSession, SessionClosed
from state import Event


class Driver:
    """TurnDriver 최소 구현. 문자열 메시지는 answer 이벤트로 바꾼다."""

    def __init__(self):
        self.turns = 0
        self.ask_user = None

    def options(self, ask_user):
        self.ask_user = ask_user
        return {"ask_user": ask_user}

    def begin_turn(self):
        self.turns += 1

    def events(self, message):
        return [Event("answer", body=message)] if isinstance(message, str) else []

    def result(self, result):
        if result is None:
            return Event("error", body="결과 없음", is_error=True)
        return Event("done", body=result.result, data={"session_id": result.session_id})


class FakeClient:
    instances: list = []
    scripts: list = []

    def __init__(self, options):
        self.options = options
        self.prompts = []
        FakeClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self.prompts.append(prompt)

    async def receive_response(self):
        for item in FakeClient.scripts.pop(0):
            if callable(item):
                await item(self.options)
            else:
                yield item

    async def interrupt(self):
        pass


def _result(text, sid="sess-1"):
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                         num_turns=1, session_id=sid, result=text, total_cost_usd=0.01)


@pytest.fixture
def session():
    FakeClient.instances = []
    FakeClient.scripts = []
    s = AgentSession(Driver(), client_factory=FakeClient)
    yield s
    s.close()


def test_two_turns_reuse_one_client(session):
    FakeClient.scripts = [["첫 답", _result("첫 답")], ["둘째 답", _result("둘째 답", sid="sess-1")]]
    first = list(session.send("q1"))
    second = list(session.send("q2"))
    assert [e.kind for e in first] == ["answer", "done"]
    assert [e.kind for e in second] == ["answer", "done"]
    assert len(FakeClient.instances) == 1
    assert FakeClient.instances[0].prompts == ["q1", "q2"]
    assert session.session_id == "sess-1"
    assert session._driver.turns == 2


def test_question_pauses_turn_until_answer(session):
    box = {}

    async def ask(options):
        box["answers"] = await options["ask_user"]([{"question": "예상 트래픽 규모는?", "header": "규모"}])

    FakeClient.scripts = [["시작", ask, "이어서", _result("MAU 10만 기준 설계")]]
    first = list(session.send("아키텍처 잡아줘"))
    assert [(e.kind, e.body) for e in first] == [("answer", "시작"), ("question", "예상 트래픽 규모는?")]
    assert session.waiting_question == "예상 트래픽 규모는?"

    session.answer("MAU 10만")
    rest = list(session.resume())
    assert [(e.kind, e.body) for e in rest] == [("answer", "이어서"), ("done", "MAU 10만 기준 설계")]
    assert box["answers"] == {"예상 트래픽 규모는?": "MAU 10만"}
    assert session.waiting_question is None


def test_send_while_waiting_is_an_error(session):
    async def ask(options):
        await options["ask_user"]([{"question": "규모?"}])

    FakeClient.scripts = [[ask, _result("x")]]
    list(session.send("q"))
    with pytest.raises(RuntimeError):
        list(session.send("또 질문"))
    session.answer("답")
    list(session.resume())


def test_client_failure_yields_error_and_next_turn_gets_new_client(session):
    async def boom(options):
        raise RuntimeError("Claude Code 프로세스 종료")

    FakeClient.scripts = [[boom], ["복구", _result("복구")]]
    first = list(session.send("q1"))
    assert [e.kind for e in first] == ["error"]
    assert "프로세스 종료" in first[0].body
    second = list(session.send("q2"))
    assert [e.kind for e in second] == ["answer", "done"]
    assert len(FakeClient.instances) == 2
    assert session.lost_context is True


def test_close_cancels_waiting_question_and_stops_thread(session):
    box = {}

    async def ask(options):
        try:
            await options["ask_user"]([{"question": "규모?"}])
        except SessionClosed as e:
            box["closed"] = str(e)
            raise

    FakeClient.scripts = [[ask, _result("x")]]
    list(session.send("q"))
    thread = session._thread
    session.close()
    assert box["closed"]
    assert not thread.is_alive()
    assert session.waiting_question is None
    assert session._loop is None


def test_close_before_start_is_noop():
    AgentSession(Driver(), client_factory=FakeClient).close()
```

- [ ] **Step 2: 실패 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_async_utils.py`
Expected: ImportError (`AgentSession`, `SessionClosed` 없음).

- [ ] **Step 3: `async_utils.py` 구현**

`app2/async_utils.py` 전체를 다음으로 바꾼다:

```python
"""Streamlit(동기)에서 상주하는 Claude Code 세션.

공식 문서의 멀티턴 패턴은 ClaudeSDKClient 하나에 query() 를 반복하는 것이다. Streamlit 은
스크립트를 매번 다시 실행하므로 client 를 st.session_state 에 두고, 백그라운드 스레드의
이벤트 루프 안에서 코루틴 하나(_main)가 client 를 소유한다. 턴 요청은 asyncio.Queue 로
넣고, 이벤트는 queue.Queue 로 받는다. st.* 는 소비하는 쪽(스크립트 스레드)만 부른다.

AskUserQuestion 은 can_use_tool 콜백이 사용자 답을 받아야 끝난다. 콜백(루프 쪽)은
question 이벤트를 내고 Future 를 기다린다. UI 는 send()/resume() 이터레이터가 question 에서
멈추면 질문을 그리고, 사용자 입력을 answer() 로 넘긴 뒤 resume() 으로 같은 턴을 이어 받는다.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Iterator, Protocol

from claude_agent_sdk import ClaudeSDKClient, ResultMessage

from state import Event

logger = logging.getLogger(__name__)

AskUser = Callable[[list[dict[str, Any]]], Awaitable[dict[str, str]]]


class TurnDriver(Protocol):
    """앱 쪽 정책. main_agent.MainAgent 가 구현한다."""

    def options(self, ask_user: AskUser) -> Any: ...
    def begin_turn(self) -> Any: ...
    def events(self, message: Any) -> list[Event]: ...
    def result(self, result: ResultMessage | None) -> Event: ...


class SessionClosed(RuntimeError):
    """재질문을 기다리는 동안 대화가 초기화됐다."""


_PAUSE = object()  # question 을 냈다. UI 가 답을 받아야 한다.
_END = object()    # 턴이 끝났다 (done 또는 error 뒤)
_CLOSE = object()  # 세션 종료 요청


def run_coro(coro):
    """실행 중인 루프가 있든 없든 코루틴을 끝까지 돌리고 결과를 반환한다."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class AgentSession:
    """채팅 세션 하나 = client 하나. 스크립트 스레드에서 send/answer/resume/close 를 부른다."""

    def __init__(self, driver: TurnDriver, client_factory: Callable[..., Any] = ClaudeSDKClient) -> None:
        self._driver = driver
        self._client_factory = client_factory
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._requests: asyncio.Queue | None = None
        self._events: queue.Queue = queue.Queue()
        self._reply: asyncio.Future | None = None
        self._client: Any = None
        self._closing = False
        self.waiting_question: str | None = None
        self.session_id: str | None = None
        self.lost_context = False  # client 가 죽어 새로 만들었다(대화 맥락을 잃었다)

    # ---- 스크립트 스레드 API ------------------------------------------------
    def send(self, prompt: str) -> Iterator[Event]:
        """턴을 시작한다. 다음 question 또는 done/error 까지의 이벤트를 내준다."""
        if self.waiting_question is not None:
            raise RuntimeError("재질문에 먼저 answer() 로 답해야 합니다")
        self._ensure_running()
        assert self._loop is not None and self._requests is not None
        self._loop.call_soon_threadsafe(self._requests.put_nowait, prompt)
        return self._drain()

    def answer(self, text: str) -> None:
        """기다리는 AskUserQuestion 에 답을 준다. 이어서 resume() 을 부른다."""
        if self._reply is None or self._loop is None:
            raise RuntimeError("기다리는 질문이 없습니다")
        fut = self._reply
        self._loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(text))
        self.waiting_question = None

    def resume(self) -> Iterator[Event]:
        """answer() 뒤 같은 턴의 나머지 이벤트를 이어 받는다."""
        return self._drain()

    def close(self) -> None:
        """client 를 닫고 스레드를 끝낸다. 재질문 대기는 취소된다."""
        if self._loop is None:
            return
        loop, thread = self._loop, self._thread
        assert thread is not None and self._requests is not None
        self._closing = True
        if self._reply is not None:
            fut = self._reply
            loop.call_soon_threadsafe(lambda: fut.done() or fut.cancel())
        loop.call_soon_threadsafe(self._requests.put_nowait, _CLOSE)
        thread.join(timeout=15)
        if thread.is_alive():
            logger.warning("에이전트 세션 스레드가 15초 안에 끝나지 않았습니다")
        self._loop = self._thread = self._requests = None
        self._reply = None
        self._client = None
        self.waiting_question = None
        self._closing = False

    def _drain(self) -> Iterator[Event]:
        while True:
            item = self._events.get()
            if item is _PAUSE or item is _END:
                return
            yield item

    # ---- 루프 쪽 ------------------------------------------------------------
    def _ensure_running(self) -> None:
        if self._loop is not None:
            return
        loop = asyncio.new_event_loop()
        started = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(loop)
            self._requests = asyncio.Queue()
            started.set()
            try:
                loop.run_until_complete(self._main())
            finally:
                loop.close()

        self._loop = loop
        # 호출한 쪽의 컨텍스트(LangSmith 부모 run 등)를 스레드로 넘긴다.
        self._thread = threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True)
        self._thread.start()
        started.wait()

    async def _main(self) -> None:
        """client 를 소유하는 유일한 코루틴. 턴 요청을 순서대로 처리한다."""
        assert self._requests is not None
        prompt = await self._requests.get()
        while prompt is not _CLOSE:
            try:
                async with self._client_factory(options=self._driver.options(self.ask_user)) as client:
                    self._client = client
                    prompt = await self._serve(client, prompt)
            except Exception as e:  # noqa: BLE001 — client 프로세스가 죽었다. 다음 턴에 새로 만든다.
                self._client = None
                self._reply = None
                self.waiting_question = None
                if self._closing:
                    return
                logger.error("에이전트 세션 오류: %s", e)
                self.lost_context = True
                self._events.put(Event("error", body=str(e), is_error=True))
                self._events.put(_END)
                prompt = await self._requests.get()

    async def _serve(self, client: Any, prompt: Any) -> Any:
        """같은 client 로 _CLOSE 가 올 때까지 턴을 돈다. 마지막에 읽은 요청(_CLOSE)을 돌려준다."""
        assert self._requests is not None
        while prompt is not _CLOSE:
            await self._turn(client, prompt)
            prompt = await self._requests.get()
        return prompt

    async def _turn(self, client: Any, prompt: str) -> None:
        self._driver.begin_turn()
        result: ResultMessage | None = None
        await client.query(prompt)
        async for message in client.receive_response():
            for ev in self._driver.events(message):
                self._events.put(ev)
            if isinstance(message, ResultMessage):
                result = message
        if result is not None and result.session_id:
            self.session_id = result.session_id
        self._events.put(self._driver.result(result))
        self._events.put(_END)

    async def ask_user(self, questions: list[dict[str, Any]]) -> dict[str, str]:
        """can_use_tool(AskUserQuestion) 이 부른다. 질문마다 UI 의 답을 기다린다."""
        answers: dict[str, str] = {}
        for q in questions:
            text = q.get("question", "")
            self._reply = asyncio.get_running_loop().create_future()
            self.waiting_question = text
            self._events.put(Event("question", body=text, data={"question": q}))
            self._events.put(_PAUSE)
            try:
                answers[text] = await self._reply
            except asyncio.CancelledError:
                raise SessionClosed("대화가 초기화되었습니다") from None
            finally:
                self._reply = None
        return answers
```

- [ ] **Step 4: 통과 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_async_utils.py`
Expected: 6 passed.

`iterate_in_thread` 가 없어졌으므로 아직 그것을 쓰는 곳(`app.py`, `test_live.py`)이 있다. 이번 태스크에서는 `app.py` 의 import 만 깨진다. 전체 스위트를 돌리면 `test_app_ui.py` 가 실패한다 — Task 6 에서 고친다. 이 태스크의 검증은 위 명령으로 한다.

- [ ] **Step 5: 커밋**

```bash
git add app2/async_utils.py app2/tests/test_async_utils.py
git commit -m "feat(app2): AgentSession — 대화당 상주 ClaudeSDKClient, AskUserQuestion 대기

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `Guardrail` — PostToolUse 수집 + Stop 팩트체크

기존 `Judge` 는 Task 7 에서 지운다. 여기서는 새 것을 **추가**만 한다.

**Files:**
- Modify: `app2/state.py` (`TurnState` 에 `fact_check` 필드 추가, `Event` 종류에 `question` 추가)
- Modify: `app2/judge_agent.py` (`response_text`, `last_assistant_text`, `Guardrail` 추가)
- Modify: `app2/tests/test_judge_agent.py` (끝에 추가)

**Interfaces:**
- Consumes: `rag_tools.fact_check(text, context) -> dict`, `state.TurnState`, `domain_agents.DOMAINS`, `general_it_trend_topic_agent.TREND_AGENT`
- Produces:
  - `state.TurnState.fact_check: dict | None` — `{"verdict": "PASS"|"REWRITE", "score": float|None, "comment": str}`
  - `judge_agent.response_text(value) -> str` (기존 `_response_text` 를 공개 이름으로; `_response_text = response_text` 별칭 유지)
  - `judge_agent.last_assistant_text(input_data: dict) -> str`
  - `judge_agent.Guardrail(turn_of: Callable[[], TurnState])` with `async record_subagent_answer(input_data, tool_use_id, context) -> dict`, `async fact_check_on_stop(input_data, tool_use_id, context) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

`app2/tests/test_judge_agent.py` 끝에 추가:

```python
# ---------------------------------------------------------------------------
# Guardrail — Stop 훅 하나로 "1회 재작성"
# ---------------------------------------------------------------------------
from judge_agent import Guardrail, last_assistant_text, response_text  # noqa: E402


def _guard(turn):
    return Guardrail(lambda: turn)


def test_response_text_handles_str_blocks_and_message_dict():
    assert response_text("x") == "x"
    assert response_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert response_text({"content": [{"type": "text", "text": "c"}]}) == "c"
    assert response_text({"role": "assistant", "content": [{"type": "text", "text": "d"}]}) == "d"
    assert response_text({"message": {"role": "assistant", "content": [{"type": "text", "text": "e"}]}}) == "e"
    assert response_text(None) == ""


def test_last_assistant_text_prefers_hook_field_then_transcript(tmp_path):
    assert last_assistant_text({"last_assistant_message": "최종 답"}) == "최종 답"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"user","message":{"role":"user","content":"q"}}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"첫 답"}]}}\n'
        'not json\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"마지막 답"}]}}\n',
        encoding="utf-8",
    )
    assert last_assistant_text({"transcript_path": str(transcript)}) == "마지막 답"
    assert last_assistant_text({}) == ""


def test_guardrail_collects_only_top_level_domain_and_trend_answers():
    turn = TurnState()
    g = _guard(turn)
    ok = {"tool_name": "Agent", "tool_input": {"subagent_type": "ai", "prompt": "RAG?"},
          "tool_response": {"content": [{"type": "text", "text": "RAG 답"}]}}
    trend = {"tool_name": "Agent", "tool_input": {"subagent_type": "general_it_trend", "prompt": "q"}, "tool_response": "웹 답"}
    nested = {**ok, "agent_id": "sub-1"}
    for data in (ok, trend, nested):
        run(g.record_subagent_answer(data, "t", None))
    assert [(a.domain, a.question, a.answer) for a in turn.domain_answers] == [("ai", "RAG?", "RAG 답")]
    assert turn.trend_answers == ["웹 답"]


def test_stop_hook_passes_with_fewer_than_two_domain_answers(report):
    turn = TurnState(domain_answers=_answers()[:1])
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": False, "last_assistant_message": "답"}, None, None)) == {}
    assert report["context"] is None  # 팩트체크를 부르지 않았다
    assert turn.fact_check is None


def test_stop_hook_blocks_once_with_fix_rules_when_below_threshold(report):
    report["report"] = {"overall_accuracy": 0.3, "overall_accuracy_comment": "3번 문장 근거 없음"}
    turn = TurnState(domain_answers=_answers())
    out = run(_guard(turn).fact_check_on_stop({"stop_hook_active": False, "last_assistant_message": "병합 답"}, None, None))
    assert out["decision"] == "block"
    assert judge_agent.FIX_RULES in out["reason"] and "3번 문장 근거 없음" in out["reason"]
    assert report["context"] == "Nginx 를 앞단에 둔다.\n\nPinecone 에 문서를 색인한다."
    assert turn.fact_check == {"verdict": "REWRITE", "score": 0.3, "comment": "3번 문장 근거 없음"}
    # 재작성 뒤의 두 번째 Stop 은 stop_hook_active 라서 통과 (SDK 내장 1회 의미론)
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": True, "last_assistant_message": "재작성"}, None, None)) == {}


def test_stop_hook_passes_when_accurate_or_unverifiable(report):
    turn = TurnState(domain_answers=_answers())
    report["report"] = {"overall_accuracy": 0.9, "overall_accuracy_comment": "ok"}
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": False, "last_assistant_message": "답"}, None, None)) == {}
    assert turn.fact_check["verdict"] == "PASS"
    report["report"] = {"overall_accuracy": 0.0, "overall_accuracy_comment": "[검증불가] 응답이 잘림"}
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": False, "last_assistant_message": "답"}, None, None)) == {}


def test_stop_hook_passes_when_final_text_is_empty(report):
    turn = TurnState(domain_answers=_answers())
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": False}, None, None)) == {}
    assert report["context"] is None
```

- [ ] **Step 2: 실패 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_judge_agent.py`
Expected: ImportError (`Guardrail` 없음).

- [ ] **Step 3: `state.py` 수정**

`TurnState` 에 필드를 추가하고 `EventKind` 에 `question` 을 넣는다(기존 필드는 Task 7 까지 둔다):

```python
@dataclass
class TurnState:
    # 도메인 서브에이전트(server_infra/system_design/ai)가 돌려준 답
    domain_answers: list[DomainAnswer] = field(default_factory=list)
    # 트렌드 서브에이전트가 돌려준 답
    trend_answers: list[str] = field(default_factory=list)
    # Stop 훅의 팩트체크 결과 {"verdict": "PASS"|"REWRITE", "score": float|None, "comment": str}. 안 돌았으면 None
    fact_check: dict[str, Any] | None = None
    fact_checks: int = 0
    # fact_check 에 넘어온 병합 답과 판정(PASS / REWRITE)
    checked_answer: str = ""
    verdict: str = ""
    clarifying_question: str = ""


EventKind = Literal["tool_call", "tool_result", "answer", "question", "clarification", "done", "error"]
```

- [ ] **Step 4: `judge_agent.py` 에 추가**

`_response_text` 를 다음으로 바꾸고(별칭 유지), 파일 끝에 `last_assistant_text` 와 `Guardrail` 을 추가한다. import 에 `import json` 과 `from pathlib import Path`, `from typing import Any, Callable` 를 추가한다.

```python
def response_text(response: Any) -> str:
    """도구 결과·훅 입력에서 텍스트만 뽑는다.

    받는 형식: str | 블록 리스트 | {"content": ...} | {"message": {...}} | {"result": ...} | {"text": ...}
    (Agent 도구 결과, Stop 훅의 last_assistant_message, transcript 항목이 모두 이 중 하나다).
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("message", "content", "result"):
            if key in response:
                return response_text(response[key])
        return response.get("text", "")
    if isinstance(response, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in response
            if not isinstance(b, dict) or b.get("type", "text") == "text"
        )
    return str(response or "")


_response_text = response_text


def last_assistant_text(input_data: dict[str, Any]) -> str:
    """Stop 훅 입력에서 최종 답 텍스트. CLI 가 주는 last_assistant_message 를 쓰고, 없으면 transcript 를 읽는다."""
    text = response_text(input_data.get("last_assistant_message"))
    if text.strip():
        return text
    path = input_data.get("transcript_path")
    if not path or not Path(path).is_file():
        return ""
    last = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("type") == "assistant":
            candidate = response_text(entry.get("message"))
            if candidate.strip():
                last = candidate
    return last


class Guardrail:
    """한 대화의 guardrail 훅. 현재 턴의 TurnState 는 turn_of() 로 받는다(턴마다 바뀐다)."""

    def __init__(self, turn_of: Callable[[], TurnState]) -> None:
        self.turn_of = turn_of

    async def record_subagent_answer(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        """PostToolUse(Agent). 서브에이전트가 돌려준 답을 모은다(실패한 위임은 PostToolUse 가 안 온다)."""
        if input_data.get("agent_id"):
            return {}
        turn = self.turn_of()
        tool_input = input_data.get("tool_input") or {}
        agent = tool_input.get("subagent_type")
        text = response_text(input_data.get("tool_response"))
        if agent in DOMAINS:
            turn.domain_answers.append(DomainAnswer(domain=agent, question=tool_input.get("prompt", ""), answer=text))
        elif agent == TREND_AGENT:
            turn.trend_answers.append(text)
        return {}

    async def fact_check_on_stop(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        """Stop. 도메인 답이 2개 이상이면 최종 답을 검증하고, 미달이면 한 번 고쳐 쓰게 한다.

        두 번째 Stop 은 stop_hook_active 가 True 라 그대로 통과한다 — app/ 의 "1회 재작성"이
        SDK 내장 의미론으로 구현된다.
        """
        turn = self.turn_of()
        if input_data.get("stop_hook_active") or len(turn.domain_answers) < 2:
            return {}
        answer = last_assistant_text(input_data)
        if not answer.strip():
            return {}
        verdict, report = await decide(answer, turn.domain_answers)
        comment = report.get("overall_accuracy_comment", "")
        turn.fact_check = {"verdict": verdict, "score": report.get("overall_accuracy"), "comment": comment}
        if verdict == "PASS":
            return {}
        return {"decision": "block", "reason": f"{FIX_RULES}\n\n> 검증 결과\n{comment}"}
```

- [ ] **Step 5: 통과 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_judge_agent.py tests/test_main_agent.py`
Expected: 모두 통과 (기존 `Judge` 테스트 포함).

- [ ] **Step 6: 커밋**

```bash
git add app2/state.py app2/judge_agent.py app2/tests/test_judge_agent.py
git commit -m "feat(app2): Guardrail — Stop 훅 팩트체크(1회 재작성은 stop_hook_active)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: 트렌드 서브에이전트 → 내장 WebSearch/WebFetch

**Files:**
- Rewrite: `app2/general_it_trend_topic_agent.py`
- Modify: `app2/main_agent.py` (import 와 `mcp_servers` 에서 `web` 제거 — 임시. Task 5 에서 전부 다시 쓴다)
- Rewrite: `app2/tests/test_general_it_trend.py`
- Modify: `app2/tests/test_main_agent.py:42` (`{"rag", "web", "judge", "orchestrator"}` → `{"rag", "judge", "orchestrator"}`)
- Modify: `app2/.env.example` (SERPER_API_KEY, USER_AGENT 줄 삭제)

**Interfaces:**
- Produces: `WEB_TOOLS = ["WebSearch", "WebFetch"]`, `TREND_AGENT = "general_it_trend"`, `system_prompt: str`, `build_trend_agent() -> AgentDefinition`
- 삭제: `expand_query`, `web_search_tool`, `fetch_webpage_content`, `web_server`, `WEB_SERVER`, `QUERY_EXPANSION_TEMPLATE`, `expand`, `search_web`, `load_page`, `fetch_pages`

- [ ] **Step 1: 실패하는 테스트 작성**

`app2/tests/test_general_it_trend.py` 전체:

```python
"""트렌드 서브에이전트 — 내장 WebSearch/WebFetch 만 쓴다."""
import general_it_trend_topic_agent as trend


def test_trend_agent_uses_builtin_web_tools_only():
    agent = trend.build_trend_agent()
    assert agent.prompt == trend.system_prompt
    assert agent.model == "claude-sonnet-5"
    assert agent.tools == ["WebSearch", "WebFetch"]
    assert agent.mcpServers is None
    assert agent.background is False
    assert trend.WEB_TOOLS == ["WebSearch", "WebFetch"]


def test_description_says_when_to_delegate():
    d = trend.build_trend_agent().description
    assert "개념" in d and "트렌드" in d and "웹 검색" in d


def test_prompt_keeps_app_role_and_constraints_and_names_builtin_tools():
    p = trend.system_prompt
    assert "당신은 웹에서 정보를 찾아 제공하는 데 특화된 유능한 AI 어시스턴트임" in p
    assert "오직 도구를 통해 검색한 정보만을 근거로" in p
    assert "[Source 1], [Source 2] 형식의 인라인 인용" in p
    assert '"Sources" 섹션' in p
    assert "WebSearch" in p and "WebFetch" in p
    for gone in ("expand_query", "web_search_tool", "fetch_webpage_content"):
        assert gone not in p


def test_custom_web_tools_are_gone():
    for name in ("expand_query", "web_search_tool", "fetch_webpage_content", "web_server", "search_web", "load_page"):
        assert not hasattr(trend, name), name
```

- [ ] **Step 2: 실패 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_general_it_trend.py`
Expected: FAIL (`agent.tools` 가 `mcp__web__*`).

- [ ] **Step 3: `general_it_trend_topic_agent.py` 전체 교체**

```python
"""웹 검색 기반 IT 트렌드 서브에이전트.

app/ 은 도구 3종(expand_query, web_search_tool, fetch_webpage_content)을 만들어 ReAct 에이전트에
쥐여 주고, 도구 안에서 다시 LLM 을 불러 질의를 확장하고 페이지를 요약했다. Claude Code 에는
WebSearch / WebFetch 가 내장돼 있으므로 AgentDefinition 에 그 둘만 열어 준다. 질의 확장과
본문 읽기는 도구가 아니라 서브에이전트 자신이 한다. 역할·제약·Sources 규칙은 app/ 과 같다.
"""
from __future__ import annotations

from claude_agent_sdk import AgentDefinition

from models import resolve_model_id

TREND_AGENT = "general_it_trend"
WEB_TOOLS = ["WebSearch", "WebFetch"]

system_prompt = """
> 역할
- 1. 당신은 웹에서 정보를 찾아 제공하는 데 특화된 유능한 AI 어시스턴트임
- 2. 당신의 목표는 오직 도구를 통해 검색한 정보만을 근거로 간결하고 정확하며 잘 구조화된 답변을 작성하는 것 

> 지시사항
- 1. 질문 이해: 사용자가 어떤 정보를 요구하는지 분석함
- 2. 다중 쿼리 생성: 질문의 서로 다른 측면이나 세부사항에 초점을 맞춘 검색어를 2~3개 만듦
- 3. 정보 검색: 검색어마다 WebSearch 로 웹에서 관련 정보를 조회함
- 4. 본문 수집: 답에 필요한 페이지는 WebFetch 로 전체 내용을 조회함
- 5. 정보 종합: 검색된 문서들의 정보를 종합하여 질문에 직접 답함
- 6. 출처 인용: 답변 말미에 "Sources" 섹션을 두고 사용한 URL을 나열함 

> 답변 제약 사항
- 1. 찾아낸 정보를 근거로 항상 상세한 답변을 제공할 것 
- 2. 특정 정보를 인용할 때 [Source 1], [Source 2] 형식의 인라인 인용을 포함할 것
- 3. 주어진 정보를 답할 수 없는 질문이라면 그 사실을 명확히 밝힐 것 
- 4. 별도 지시가 없는 한 한국어로 답변할 것 
"""


def build_trend_agent() -> AgentDefinition:
    return AgentDefinition(
        description=(
            "단순 개념 질의 또는 최신 IT 트렌드·시세를 웹 검색으로 답하는 에이전트. "
            "트레이드오프 판단이 아니라 '무엇인가', '차이가 뭔가', '지금 얼마인가' 류의 질문을 맡긴다. "
            "예) JWT 와 세션 인증의 차이, 트랜잭션 격리 수준, 현재 맥북 가격"
        ),
        prompt=system_prompt,
        tools=list(WEB_TOOLS),
        model=resolve_model_id("trend"),
        background=False,
    )
```

- [ ] **Step 4: `main_agent.py` 임시 패치, 테스트·env 수정**

`app2/main_agent.py`:
- `from general_it_trend_topic_agent import TREND_AGENT, WEB_TOOLS, build_trend_agent, web_server` → `web_server` 를 뺀다.
- `mcp_servers={...}` 에서 `"web": web_server(),` 줄을 지운다.

`app2/tests/test_main_agent.py:42`: `assert set(opts.mcp_servers) == {"rag", "judge", "orchestrator"}`

`app2/.env.example`: `# ── 웹 검색 ──…` 블록(3줄: 제목, SERPER_API_KEY, USER_AGENT)을 지우고, 파일 끝 "모델 배정 오버라이드" 블록에 다음 한 줄을 추가한다:
```
AGENT_MAX_BUDGET_USD=3           # 턴당 비용 상한(USD). 위임 폭주 방지
```

- [ ] **Step 5: 통과 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_general_it_trend.py tests/test_main_agent.py tests/test_judge_agent.py`
Expected: 모두 통과.

- [ ] **Step 6: 커밋**

```bash
git add app2/general_it_trend_topic_agent.py app2/main_agent.py app2/tests/test_general_it_trend.py app2/tests/test_main_agent.py app2/.env.example
git commit -m "feat(app2): 트렌드 서브에이전트를 내장 WebSearch/WebFetch 로

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `main_agent.py` — 메인 에이전트, 권한 게이트, 세션 생성

여기서 오케스트레이터 프롬프트·"전달 완료"·`final_answer()`·`stream_turn()`·커스텀 도구가 사라진다. `test_app_ui.py` 와 `test_live.py` 는 이 태스크 뒤에 실패 상태가 되며 Task 6/8 에서 고친다. 이 태스크는 `tests/test_main_agent.py` 로만 검증한다.

**Files:**
- Rewrite: `app2/main_agent.py`
- Rewrite: `app2/tests/test_main_agent.py`
- Modify: `app2/obsidian_tools.py` (변경 없음 — `is_allowed_vault_tool` 은 이미 있다)

**Interfaces:**
- Consumes: `async_utils.AgentSession`, `async_utils.SessionClosed`, `async_utils.AskUser`, `judge_agent.Guardrail`, `judge_agent.MERGE_RULES`, `domain_agents.DOMAINS/build_domain_agents`, `general_it_trend_topic_agent.TREND_AGENT/WEB_TOOLS/build_trend_agent`, `obsidian_tools.OBSIDIAN_SERVER/is_allowed_vault_tool/tls_env`, `rag_tools.RAG_SERVER/rag_server`, `models.*`
- Produces:
  - 상수 `AGENT_TOOL_NAMES = ("Agent", "Task")`, `BUILTIN_TOOLS = ["Agent", "AskUserQuestion"]`, `GENERAL_REPLY`, `EMPTY_REPLY`, `AMBIGUITY_RULES`, `SYSTEM_PROMPT`
  - `is_research_tool(name) -> bool`, `short_tool_name(name) -> str`, `max_budget_usd() -> float`
  - `message_events(message, names: dict[str, str]) -> list[Event]`, `result_event(result: ResultMessage | None, turn: TurnState) -> Event`
  - `class MainAgent`: `turn: TurnState`, `guardrail: Guardrail`, `ask_user: AskUser | None`, `options(ask_user)`, `begin_turn()`, `events(message)`, `result(result)`, `async permission_gate(tool_name, input_data, context) -> PermissionResultAllow | PermissionResultDeny`
  - `build_options(agent: MainAgent) -> ClaudeAgentOptions`
  - `new_session(client_factory=ClaudeSDKClient) -> AgentSession`

- [ ] **Step 1: 실패하는 테스트 작성**

`app2/tests/test_main_agent.py` 전체:

```python
"""메인 에이전트 — 옵션 배선, 권한 게이트, 메시지 → 이벤트."""
import asyncio

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

import main_agent
from main_agent import MainAgent
from state import DomainAnswer, TurnState


@pytest.fixture(autouse=True)
def no_obsidian(monkeypatch):
    import domain_agents

    monkeypatch.setattr(domain_agents, "obsidian_mcp_server", lambda: None)
    monkeypatch.setattr(main_agent, "tls_env", lambda: {})


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 프롬프트와 옵션
# ---------------------------------------------------------------------------
def test_system_prompt_has_no_hand_written_orchestration():
    p = main_agent.SYSTEM_PROMPT
    assert main_agent.AMBIGUITY_RULES in p
    assert main_agent.MERGE_RULES in p
    assert main_agent.GENERAL_REPLY in p
    assert "AskUserQuestion" in p
    for name in ("server_infra", "system_design", "ai", "general_it_trend"):
        assert name in p
    for gone in ("complex_discusion_topic", "general_it_trend_topic", "unknown", "전달 완료", "fact_check", "request_clarification"):
        assert gone not in p


def test_options_wire_subagents_gate_and_hooks():
    agent = MainAgent()
    opts = main_agent.build_options(agent)
    assert opts.model == "claude-opus-5"
    assert opts.system_prompt == main_agent.SYSTEM_PROMPT
    assert opts.tools == ["Agent", "AskUserQuestion"]
    assert set(opts.agents) == {"server_infra", "system_design", "ai", "general_it_trend"}
    assert set(opts.mcp_servers) == {"rag"}
    # 권한은 게이트 한 함수가 전부 결정한다. 자동 승인 목록은 비어 있다.
    assert opts.permission_mode == "default"
    assert opts.allowed_tools == []
    assert opts.can_use_tool == agent.permission_gate
    assert set(opts.hooks) == {"PostToolUse", "Stop"}
    assert opts.hooks["PostToolUse"][0].matcher == "Agent|Task"
    assert opts.hooks["PostToolUse"][0].hooks == [agent.guardrail.record_subagent_answer]
    assert opts.hooks["Stop"][0].hooks == [agent.guardrail.fact_check_on_stop]
    assert opts.max_budget_usd == 3.0
    assert opts.env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"


def test_max_budget_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_BUDGET_USD", "1.5")
    assert main_agent.build_options(MainAgent()).max_budget_usd == 1.5


def test_new_session_uses_main_agent_as_driver():
    class Client:
        def __init__(self, options):
            pass

    session = main_agent.new_session(client_factory=Client)
    assert isinstance(session._driver, MainAgent)
    session.close()


# ---------------------------------------------------------------------------
# 권한 게이트
# ---------------------------------------------------------------------------
def _ctx(agent_id=None):
    return ToolPermissionContext(tool_use_id="t1", agent_id=agent_id)


def _gate(turn=None, ask_user=None):
    agent = MainAgent(ask_user=ask_user)
    if turn is not None:
        agent.turn = turn
    return agent.permission_gate


def test_ask_user_question_waits_for_answer_and_returns_it():
    seen = {}

    async def ask_user(questions):
        seen["questions"] = questions
        return {"예상 트래픽 규모는?": "MAU 10만"}

    questions = [{"question": "예상 트래픽 규모는?", "header": "규모", "options": [], "multiSelect": False}]
    out = run(_gate(ask_user=ask_user)("AskUserQuestion", {"questions": questions}, _ctx()))
    assert isinstance(out, PermissionResultAllow)
    assert out.updated_input == {"questions": questions, "answers": {"예상 트래픽 규모는?": "MAU 10만"}}
    assert seen["questions"] == questions


def test_ask_user_question_denies_and_interrupts_when_session_closed():
    from async_utils import SessionClosed

    async def ask_user(questions):
        raise SessionClosed("대화가 초기화되었습니다")

    out = run(_gate(ask_user=ask_user)("AskUserQuestion", {"questions": [{"question": "q"}]}, _ctx()))
    assert isinstance(out, PermissionResultDeny) and out.interrupt is True


@pytest.mark.parametrize("tool", ["Agent", "Task"])
def test_delegation_is_once_per_domain_but_trend_and_failed_are_free(tool):
    turn = TurnState(domain_answers=[DomainAnswer(domain="ai", question="q", answer="a")])
    gate = _gate(turn)
    assert isinstance(run(gate(tool, {"subagent_type": "ai", "prompt": "보충"}, _ctx())), PermissionResultDeny)
    assert isinstance(run(gate(tool, {"subagent_type": "server_infra", "prompt": "q"}, _ctx())), PermissionResultAllow)
    assert isinstance(run(gate(tool, {"subagent_type": "general_it_trend", "prompt": "q"}, _ctx())), PermissionResultAllow)
    empty = _gate(TurnState())
    assert isinstance(run(empty(tool, {"subagent_type": "ai", "prompt": "q"}, _ctx())), PermissionResultAllow)


@pytest.mark.parametrize("tool", [
    "mcp__rag__vector_search_tool", "mcp__obsidian__vault_read", "WebSearch", "WebFetch",
])
def test_research_tools_only_inside_subagents(tool):
    gate = _gate()
    denied = run(gate(tool, {}, _ctx()))
    assert isinstance(denied, PermissionResultDeny) and "서브에이전트" in denied.message
    assert isinstance(run(gate(tool, {}, _ctx(agent_id="sub-1"))), PermissionResultAllow)


@pytest.mark.parametrize("tool", [
    "mcp__obsidian__vault_write", "mcp__obsidian__vault_delete", "mcp__obsidian__command_execute",
    "mcp__obsidian__brand_new_write_tool",
])
def test_vault_stays_read_only_even_inside_subagents(tool):
    denied = run(_gate()(tool, {}, _ctx(agent_id="sub-1")))
    assert isinstance(denied, PermissionResultDeny) and "읽기 전용" in denied.message


@pytest.mark.parametrize("tool", ["Bash", "Read", "Write", "Edit", "Glob", "mcp__unknown__x", "Skill"])
def test_everything_else_is_denied(tool):
    assert isinstance(run(_gate()(tool, {}, _ctx())), PermissionResultDeny)
    assert isinstance(run(_gate()(tool, {}, _ctx(agent_id="sub-1"))), PermissionResultDeny)


def test_is_research_tool_and_short_name():
    assert main_agent.is_research_tool("mcp__rag__vector_search_tool")
    assert main_agent.is_research_tool("mcp__obsidian__anything")
    assert main_agent.is_research_tool("WebSearch") and main_agent.is_research_tool("WebFetch")
    assert not main_agent.is_research_tool("Agent") and not main_agent.is_research_tool("AskUserQuestion")
    assert main_agent.short_tool_name("mcp__obsidian__vault_read") == "vault_read"
    assert main_agent.short_tool_name("Agent") == "Agent"


# ---------------------------------------------------------------------------
# 메시지 → 이벤트, 결과 → done/error
# ---------------------------------------------------------------------------
def _result(text="최종 답", is_error=False, subtype="success"):
    return ResultMessage(subtype=subtype, duration_ms=1, duration_api_ms=1, is_error=is_error,
                         num_turns=3, session_id="sess-9", result=text, total_cost_usd=0.12)


def test_driver_turns_messages_into_nested_events():
    agent = MainAgent()
    agent.begin_turn()
    script = [
        AssistantMessage(content=[ToolUseBlock(id="a1", name="Agent", input={"subagent_type": "ai", "prompt": "RAG?"})],
                         model="m"),
        AssistantMessage(content=[ToolUseBlock(id="v1", name="mcp__rag__vector_search_tool", input={"query": "RAG"})],
                         model="m", parent_tool_use_id="a1"),
        UserMessage(content=[ToolResultBlock(tool_use_id="v1", content=[{"type": "text", "text": "문서"}])],
                    parent_tool_use_id="a1"),
        UserMessage(content=[ToolResultBlock(tool_use_id="a1", content=[{"type": "text", "text": "AI 답"}])]),
        AssistantMessage(content=[TextBlock(text="최종 답")], model="m"),
        AssistantMessage(content=[TextBlock(text="서브 안 텍스트")], model="m", parent_tool_use_id="a1"),
    ]
    events = [e for m in script for e in agent.events(m)]
    assert [(e.kind, e.name, e.nested) for e in events] == [
        ("tool_call", "Agent", False),
        ("tool_call", "vector_search_tool", True),
        ("tool_result", "vector_search_tool", True),
        ("tool_result", "Agent", False),
        ("answer", "", False),
    ]
    assert events[0].data == {"full_name": "Agent"}
    assert events[2].parent_tool_use_id == "a1"


def test_result_event_carries_answer_and_turn_data():
    agent = MainAgent()
    turn = agent.begin_turn()
    turn.domain_answers.append(DomainAnswer(domain="ai", question="q", answer="a"))
    turn.fact_check = {"verdict": "PASS", "score": 0.9, "comment": "ok"}
    done = agent.result(_result("답"))
    assert (done.kind, done.body) == ("done", "답")
    assert done.data["session_id"] == "sess-9" and done.data["cost_usd"] == 0.12 and done.data["num_turns"] == 3
    assert done.data["domain_answers"][0]["domain"] == "ai"
    assert done.data["fact_check"]["verdict"] == "PASS"


def test_empty_result_falls_back_like_finalize():
    assert MainAgent().result(_result("  ")).body == "답변을 생성하지 못했습니다."


def test_error_results_become_error_events():
    ev = MainAgent().result(_result("Credit balance is too low", is_error=True))
    assert (ev.kind, ev.body, ev.is_error) == ("error", "Credit balance is too low", True)
    budget = MainAgent().result(_result(None, is_error=False, subtype="error_max_budget_usd"))
    assert budget.kind == "error" and "비용 상한" in budget.body
    assert MainAgent().result(None).kind == "error"
```

- [ ] **Step 2: 실패 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_main_agent.py`
Expected: ImportError (`MainAgent` 없음).

- [ ] **Step 3: `main_agent.py` 전체 교체**

```python
"""메인 에이전트.

app/main_agent.py 는 LangGraph 그래프다.

  orchestrator ─┬─ complex ─▶ decompose ─(Send)─▶ domain_worker ×N ─▶ merge ─┐
                ├─ general ─▶ general_it_topic_agent ──────────────────────┤
                └─ unknown ─▶ general_topic_agent ─────────────────────────┴─▶ finalize

Agent SDK 에서는 이 그래프가 없다. 공식 문서: "Claude automatically decides when to invoke
subagents based on the task and each subagent's description." 메인 에이전트(ClaudeSDKClient)가
곧 오케스트레이터이고, 코드는 네 가지만 준다.

  1. 짧은 시스템 프롬프트 (SYSTEM_PROMPT)      — 역할, 재질문 기준, 병합 규칙
  2. 서브에이전트 정의 (agents=)              — 위임 판단의 근거는 각 description
  3. 권한 게이트 한 함수 (can_use_tool)        — 재질문 대기, 화이트리스트, 조사 도구 격리, 도메인당 1회
  4. guardrail 훅 두 개 (judge_agent.Guardrail) — 서브에이전트 답 수집, Stop 에서 팩트체크

  LangGraph (app/)                    Agent SDK (app2/)
  classify + route_by_intent      ->  없음 (서브에이전트 description 으로 자동 위임)
  decompose + Send fan-out        ->  메인이 한 응답에서 Agent 를 여러 번 부름 (병렬)
  merge                           ->  메인의 응답 자체 (ResultMessage.result)
  _guard (1회 재작성)              ->  Stop 훅 + stop_hook_active
  interrupt / Command(resume)     ->  AskUserQuestion + can_use_tool 대기
  InMemorySaver (thread_id)       ->  ClaudeSDKClient 상주 (async_utils.AgentSession)
"""
from __future__ import annotations

import logging
import os
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from async_utils import AgentSession, AskUser, SessionClosed
from domain_agents import DOMAINS, build_domain_agents
from general_it_trend_topic_agent import TREND_AGENT, WEB_TOOLS, build_trend_agent
from judge_agent import MERGE_RULES, Guardrail
from models import AGENT_MAX_TOKENS, base_env, isolated_options, resolve_model_id
from obsidian_tools import OBSIDIAN_SERVER, is_allowed_vault_tool, tls_env
from rag_tools import RAG_SERVER, rag_server
from state import Event, TurnState

logger = logging.getLogger(__name__)

AGENT_TOOL_NAMES = ("Agent", "Task")
# 메인에게 여는 내장 도구. 파일·셸·웹 도구는 없다. 조사는 서브에이전트가 한다.
BUILTIN_TOOLS = ["Agent", "AskUserQuestion"]

# app/ general_node 가 돌려주는 고정 안내문
GENERAL_REPLY = (
    "IT 개념 질의나 서버·인프라·시스템설계·AI 아키텍처 논의를 "
    "도와드릴 수 있습니다. 무엇이 궁금하신가요?"
)
EMPTY_REPLY = "답변을 생성하지 못했습니다."

# ===========================================================================
# 재질문 기준 — app/ 과 같은 문장
# ===========================================================================
AMBIGUITY_RULES = """
> 재질문 기준 (엄격하게 적용)
재질문은 사용자를 멈춰 세우므로 아래 둘 중 하나일 때만 합니다.
- 1. 설계를 요청했는데 규모·트래픽·제약이 **전혀** 없어 답이 완전히 달라지는 경우
- 2. 요구가 서로 **상충**해 한쪽을 고르지 않으면 답할 수 없는 경우

그 외에는 재질문하지 않습니다. 정보가 조금 부족하면 합리적으로 가정하고 진행하되,
가정한 내용을 답변에 명시하면 됩니다.

특히 아래는 재질문 대상이 **아닙니다**.
- 개념을 묻는 질문 (예: JWT 와 세션 인증의 차이)
- 진단 방법을 묻는 질문 (예: p99 지연의 원인을 어떻게 구분하나)
- 선택지가 질문에 이미 제시된 경우 (예: REST 와 Kafka 중 무엇)
"""

SYSTEM_PROMPT = f"""당신은 IT 질문에 답하는 어시스턴트입니다. 조사는 전문 서브에이전트에게 맡기고, 돌아온 답을 하나의 글로 합쳐 사용자에게 답합니다.

> 위임
- 서버·인프라·시스템 설계·AI 의 트레이드오프 판단이 필요한 논의는 해당 도메인 서브에이전트({", ".join(DOMAINS)})에 맡깁니다.
  질문이 여러 도메인에 걸치면 도메인별 서브 질문으로 나누어 **한 번의 응답에서 Agent 도구를 여러 번** 불러 병렬로 조사시킵니다.
  각 서브 질문은 다른 도메인의 답을 몰라도 독립적으로 답할 수 있어야 하고, 원 질문의 제약(규모, 트래픽, 기술 스택)을 그대로 담습니다.
- 단순 개념 질의나 최신 트렌드·시세는 {TREND_AGENT} 서브에이전트에 맡깁니다. 이전 대화 맥락이 필요하면 그 맥락을 prompt 에 덧붙입니다.
- 인사말·잡담처럼 조사할 것이 없는 말에는 도구를 쓰지 말고 아래 문장으로만 답합니다.
  {GENERAL_REPLY}
- 직접 조사하지 마세요. 조사는 서브에이전트의 몫입니다.
{AMBIGUITY_RULES}
- 재질문이 필요하면 조사하기 전에 AskUserQuestion 으로 한 문장을 묻습니다.

> 답변
- 서브에이전트가 하나였다면 그 답을 바탕으로 답합니다. 내용을 덧붙이거나 줄이지 마세요.
- 여러 서브에이전트의 답은 아래 규칙으로 합칩니다.
{MERGE_RULES}
- 답변에는 처리 과정(분류, 위임, 검증)을 쓰지 말고 답변 본문만 씁니다.
"""

_RESEARCH_PREFIXES = (f"mcp__{RAG_SERVER}__", f"mcp__{OBSIDIAN_SERVER}__")
_VAULT_PREFIX = f"mcp__{OBSIDIAN_SERVER}__"


def is_research_tool(name: str) -> bool:
    """조사 도구 — 서브에이전트만 쓴다(app/ 에서 오케스트레이터는 도구가 없다)."""
    return name.startswith(_RESEARCH_PREFIXES) or name in WEB_TOOLS


def short_tool_name(name: str) -> str:
    """mcp__obsidian__vault_read -> vault_read (app/ 의 도구 이름과 같게 보여준다)."""
    return name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name


def max_budget_usd() -> float:
    """턴당 비용 상한. 실측: 프롬프트만으로는 같은 도메인에 보충 위임을 반복해 $8.5 까지 갔다."""
    return float(os.getenv("AGENT_MAX_BUDGET_USD", "3"))


class MainAgent:
    """대화 하나의 메인 에이전트 정책. AgentSession 이 TurnDriver 로 쓴다.

    훅과 권한 게이트는 client 를 만들 때 한 번 묶이지만 턴마다 상태가 바뀌므로,
    둘 다 self.turn(현재 턴)을 본다.
    """

    def __init__(self, ask_user: AskUser | None = None) -> None:
        self.ask_user = ask_user
        self.turn = TurnState()
        self.guardrail = Guardrail(lambda: self.turn)
        self._names: dict[str, str] = {}

    # ---- TurnDriver --------------------------------------------------------
    def options(self, ask_user: AskUser):
        self.ask_user = ask_user
        return build_options(self)

    def begin_turn(self) -> TurnState:
        self.turn = TurnState()
        self._names = {}
        return self.turn

    def events(self, message: Any) -> list[Event]:
        return message_events(message, self._names)

    def result(self, result: ResultMessage | None) -> Event:
        return result_event(result, self.turn)

    # ---- 권한 게이트 --------------------------------------------------------
    async def permission_gate(
        self, tool_name: str, input_data: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """can_use_tool. allowed_tools 가 비어 있으므로 모든 도구 호출이 여기로 온다.

        1. AskUserQuestion  -> UI 의 답을 기다렸다가 answers 로 돌려준다 (공식 user-input 패턴)
        2. Agent/Task       -> 이번 턴에 이미 답한 도메인이면 거부 (app/ 의 분해 1회)
        3. 조사 도구         -> 서브에이전트 안(agent_id)에서만. 볼트는 화이트리스트 5종만 (읽기 전용)
        4. 그 외            -> 거부
        """
        if tool_name == "AskUserQuestion":
            questions = list(input_data.get("questions") or [])
            if self.ask_user is None:
                return PermissionResultDeny(message="재질문을 받을 UI 가 없습니다. 합리적으로 가정하고 진행하세요.")
            try:
                answers = await self.ask_user(questions)
            except SessionClosed as e:
                return PermissionResultDeny(message=str(e), interrupt=True)
            return PermissionResultAllow(updated_input={"questions": questions, "answers": answers})

        if tool_name in AGENT_TOOL_NAMES:
            domain = input_data.get("subagent_type")
            if domain in DOMAINS and domain in {a.domain for a in self.turn.domain_answers}:
                return PermissionResultDeny(message=f"{domain} 서브에이전트의 답은 이미 받았습니다. 받은 답으로 병합하세요.")
            return PermissionResultAllow()

        if is_research_tool(tool_name):
            if not context.agent_id:
                return PermissionResultDeny(message="조사는 서브에이전트의 몫입니다. Agent 도구로 위임하세요.")
            if tool_name.startswith(_VAULT_PREFIX) and not is_allowed_vault_tool(tool_name):
                return PermissionResultDeny(message=f"볼트는 읽기 전용입니다. {tool_name} 은 허용되지 않습니다.")
            return PermissionResultAllow()

        return PermissionResultDeny(message=f"{tool_name} 은 이 앱에서 허용되지 않는 도구입니다.")


def build_options(agent: MainAgent):
    """메인 에이전트 실행 옵션. 대화(client)당 한 번 만든다."""
    return isolated_options(
        model=resolve_model_id("orchestrator"),  # 역할 이름은 app/ 과 같게 둔다(모델 배정표 공유)
        system_prompt=SYSTEM_PROMPT,
        tools=list(BUILTIN_TOOLS),
        agents={**build_domain_agents(), TREND_AGENT: build_trend_agent()},
        mcp_servers={RAG_SERVER: rag_server()},
        # 자동 승인 없음. 모든 호출이 permission_gate 로 온다.
        permission_mode="default",
        allowed_tools=[],
        can_use_tool=agent.permission_gate,
        hooks={
            "PostToolUse": [HookMatcher(matcher="Agent|Task", hooks=[agent.guardrail.record_subagent_answer])],
            "Stop": [HookMatcher(hooks=[agent.guardrail.fact_check_on_stop])],
        },
        max_budget_usd=max_budget_usd(),
        env={**base_env(AGENT_MAX_TOKENS), **tls_env()},
    )


def new_session(client_factory: Any = ClaudeSDKClient) -> AgentSession:
    """채팅 세션 하나. app.py 가 st.session_state 에 둔다."""
    return AgentSession(MainAgent(), client_factory=client_factory)


# ===========================================================================
# 메시지 스트림 -> UI 이벤트
# ===========================================================================
def message_events(message: Any, names: dict[str, str]) -> list[Event]:
    """ClaudeSDKClient 메시지 하나를 UI 이벤트로. names 는 tool_use_id -> 도구 이름(턴마다 새로)."""
    events: list[Event] = []
    if isinstance(message, AssistantMessage):
        nested = message.parent_tool_use_id is not None
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                names[block.id] = block.name
                events.append(
                    Event(
                        "tool_call",
                        name=short_tool_name(block.name),
                        body=block.input,
                        nested=nested,
                        tool_use_id=block.id,
                        parent_tool_use_id=message.parent_tool_use_id,
                        data={"full_name": block.name},
                    )
                )
            elif isinstance(block, TextBlock) and not nested and block.text.strip():
                events.append(Event("answer", body=block.text))
    elif isinstance(message, UserMessage) and isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                events.append(
                    Event(
                        "tool_result",
                        name=short_tool_name(names.get(block.tool_use_id, "")),
                        body=block.content,
                        nested=message.parent_tool_use_id is not None,
                        tool_use_id=block.tool_use_id,
                        parent_tool_use_id=message.parent_tool_use_id,
                        is_error=bool(block.is_error),
                    )
                )
    return events


def result_event(result: ResultMessage | None, turn: TurnState) -> Event:
    """턴의 결과. 최종 답은 메인 에이전트의 응답 그대로다(app/ finalize 처럼 비면 안내문)."""
    if result is None:
        return Event("error", body="Claude Code 가 결과 없이 종료되었습니다", is_error=True)
    if result.is_error or result.subtype != "success":
        body = result.result or result.subtype
        if result.subtype == "error_max_budget_usd":
            body = f"이번 턴의 비용 상한(${max_budget_usd():.2f})에 도달했습니다. 질문을 좁혀 다시 물어보세요."
        return Event("error", body=body, is_error=True, data={"session_id": result.session_id})
    return Event(
        "done",
        body=(result.result or "").strip() or EMPTY_REPLY,
        data={
            "session_id": result.session_id,
            "cost_usd": result.total_cost_usd,
            "num_turns": result.num_turns,
            "usage": result.usage,
            "domain_answers": [a.__dict__ for a in turn.domain_answers],
            "trend_answers": list(turn.trend_answers),
            "fact_check": turn.fact_check,
        },
    )
```

- [ ] **Step 4: 통과 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_main_agent.py tests/test_judge_agent.py tests/test_async_utils.py tests/test_general_it_trend.py`
Expected: 모두 통과.

- [ ] **Step 5: 커밋**

```bash
git add app2/main_agent.py app2/tests/test_main_agent.py
git commit -m "feat(app2): 메인 에이전트 — 오케스트레이터 프롬프트 제거, can_use_tool 게이트, new_session

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `app.py` — 세션 상주와 question → answer → resume

**Files:**
- Modify: `app2/app.py` (import, 세션 상태, 사이드바 캡션, 초기화, 턴 실행 블록, 입력 처리)
- Rewrite: `app2/tests/test_app_ui.py`

**Interfaces:**
- Consumes: `main_agent.new_session() -> AgentSession` (`send`, `answer`, `resume`, `close`, `session_id`, `lost_context`), `state.Event`
- Produces: 세션 상태 키 `agent`, `pending_question`, `pending_answer` (기존 `agent_session_id`, `pending_clarification`, `resume_value` 삭제)

- [ ] **Step 1: 실패하는 테스트 작성**

`app2/tests/test_app_ui.py` 전체:

```python
"""Streamlit 화면 흐름 — AppTest 로 확인한다. new_session 을 가짜 세션으로 바꿔 Claude Code 없이 돌린다."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import main_agent
from state import Event

APP_PY = str(Path(__file__).resolve().parents[1] / "app.py")


class FakeSession:
    """AgentSession 과 같은 API. send/resume 은 미리 정한 이벤트 목록을 차례로 내준다."""

    def __init__(self, box):
        self.box = box
        self.session_id = None
        self.lost_context = False
        self.waiting_question = None
        self.closed = False

    def send(self, prompt):
        self.box["calls"].append(("send", prompt))
        return self._play()

    def answer(self, text):
        self.box["calls"].append(("answer", text))
        self.waiting_question = None

    def resume(self):
        self.box["calls"].append(("resume", None))
        return self._play()

    def close(self):
        self.closed = True
        self.box["calls"].append(("close", None))

    def _play(self):
        for ev in self.box["scripts"].pop(0):
            if ev.kind == "question":
                self.waiting_question = ev.body
            if ev.kind == "done":
                self.session_id = ev.data.get("session_id")
            yield ev


@pytest.fixture
def turns(monkeypatch):
    box = {"calls": [], "scripts": [], "sessions": []}

    def fake_new_session(client_factory=None):
        s = FakeSession(box)
        box["sessions"].append(s)
        return s

    monkeypatch.setattr(main_agent, "new_session", fake_new_session)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    return box


def _done(text, sid="sess-1"):
    return Event("done", body=text, data={"session_id": sid})


def _all_text(node):
    out = []
    if type(node).__name__ in ("Markdown", "Code") and isinstance(getattr(node, "value", None), str):
        out.append(node.value)
    for child in getattr(node, "children", {}).values():
        out.extend(_all_text(child))
    return out


def _run(at):
    at.run(timeout=30)
    assert not at.exception, at.exception
    return at


def test_welcome_screen(turns):
    at = _run(AppTest.from_file(APP_PY))
    assert [b.label for b in at.button if b.key and b.key.startswith("chip_")] == [
        "JWT와 세션 인증의 차이점이 뭔가?",
        "트래픽 급증 시 p99 지연이 튀면 무엇부터 봐야 하나?",
        "MSA에서 동기 REST와 Kafka 이벤트 중 무엇을 골라야 하나?",
    ]
    assert turns["sessions"] == []  # 첫 질문 전에는 세션을 만들지 않는다


def test_question_shows_answer_and_subagent_tool_steps(turns):
    turns["scripts"] = [[
        Event("tool_call", name="Agent", body={"subagent_type": "general_it_trend"}, nested=False),
        Event("tool_call", name="WebSearch", body={"query": "JWT"}, nested=True),
        Event("tool_result", name="WebSearch", body=[{"type": "text", "text": "결과"}], nested=True),
        _done("JWT 는 토큰, 세션은 서버 저장 [Source 1]"),
    ]]
    at = _run(AppTest.from_file(APP_PY))
    at.button(key="chip_JWT와 세션 인증의 차이점이 뭔가?").click()
    _run(at)

    assert turns["calls"] == [("send", "JWT와 세션 인증의 차이점이 뭔가?")]
    texts = _all_text(at.main)
    assert "JWT 는 토큰, 세션은 서버 저장 [Source 1]" in texts
    assert "**WebSearch**" in texts and "↳ WebSearch" in texts
    assert '{"query": "JWT"}' in texts and "결과" in texts
    assert "**Agent**" not in texts
    assert len(turns["sessions"]) == 1


def test_two_questions_share_one_session(turns):
    turns["scripts"] = [[_done("첫 답")], [_done("둘째 답")]]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("첫 질문").run()
    _run(at)
    at.chat_input(key="chat_input").set_value("둘째 질문").run()
    _run(at)
    assert turns["calls"] == [("send", "첫 질문"), ("send", "둘째 질문")]
    assert len(turns["sessions"]) == 1


def test_question_then_answer_resumes_same_turn(turns):
    turns["scripts"] = [
        [Event("question", body="예상 트래픽 규모는?")],
        [_done("MAU 10만 기준 설계")],
    ]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("우리 서비스 아키텍처 좀 잡아줘").run()
    _run(at)
    assert "**확인이 필요합니다** — 예상 트래픽 규모는?" in [m.value for m in at.markdown]

    at.chat_input(key="chat_input").set_value("MAU 10만").run()
    _run(at)
    assert turns["calls"] == [("send", "우리 서비스 아키텍처 좀 잡아줘"), ("answer", "MAU 10만"), ("resume", None)]
    assert "MAU 10만 기준 설계" in [m.value for m in at.markdown]
    assert at.session_state.pending_question is None


def test_error_is_shown_like_app(turns):
    turns["scripts"] = [[Event("error", body="Credit balance is too low", is_error=True)]]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("안녕").run()
    _run(at)
    assert "⚠️ 오류가 발생했습니다: `Credit balance is too low`" in [m.value for m in at.markdown]


def test_reset_closes_session_and_starts_fresh(turns):
    turns["scripts"] = [[_done("답")], [_done("새 답")]]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("안녕").run()
    _run(at)
    first = turns["sessions"][0]
    at.button[[b.label for b in at.button].index("대화 초기화")].click()
    _run(at)
    assert first.closed is True
    assert at.session_state.agent is None
    assert at.session_state.messages == []

    at.chat_input(key="welcome_input").set_value("다시").run()
    _run(at)
    assert len(turns["sessions"]) == 2
```

- [ ] **Step 2: 실패 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_app_ui.py`
Expected: FAIL (app.py 가 `iterate_in_thread`, `stream_turn` 을 import 하지 못함).

- [ ] **Step 3: `app.py` 수정**

(a) import 두 줄을 바꾼다:

```python
from main_agent import new_session  # noqa: E402  (load_dotenv 이후에 읽어야 한다)
```
(`from async_utils import iterate_in_thread` 줄과 `from main_agent import clarification_prompt, stream_turn` 줄 삭제.)

(b) 모듈 docstring 의 "내부 차이" 두 줄을 다음으로:

```
  - graph.stream(...) 대신 AgentSession.send(...) 의 이벤트를 그린다.
  - 대화 이어가기는 체크포인터(thread_id) 대신 상주하는 Claude Code 세션(client 1개)이 맡는다.
  - 재질문은 AskUserQuestion. 세션이 question 에서 멈추면 답을 받아 answer()+resume() 으로 이어간다.
```

(c) 세션 상태 초기화 블록을 다음으로 바꾼다(`agent_session_id`, `pending_clarification` 삭제):

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "agent" not in st.session_state:
    # 채팅 세션당 Claude Code client 하나. 첫 질문에서 만든다.
    st.session_state.agent = None
if "pending_question" not in st.session_state:
    # AskUserQuestion 이 기다리는 질문. 있으면 다음 입력은 새 질문이 아니라 그 답이다.
    st.session_state.pending_question = None


def agent_session():
    if st.session_state.agent is None:
        st.session_state.agent = new_session()
    return st.session_state.agent
```

(d) 사이드바 캡션: `st.caption("메인 에이전트 → 서브에이전트 자동 위임 · 병렬 → 병합 (Claude Agent SDK)")`

(e) "대화 초기화" 버튼 블록:

```python
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("run_prompt", None)
        st.session_state.pop("pending_answer", None)
        st.session_state.pending_question = None
        # client 를 닫아야 이전 대화와 섞이지 않는다.
        if st.session_state.agent is not None:
            st.session_state.agent.close()
            st.session_state.agent = None
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
```

(f) 턴 실행 블록(`if "run_prompt" in st.session_state:` 부터 `st.session_state.messages.append({"role": "assistant", ...})` 까지)을 다음으로 바꾼다:

```python
if "run_prompt" in st.session_state:
    prompt = st.session_state.pop("run_prompt")
    # 재질문에 답한 경우: 새 턴이 아니라 기다리던 같은 턴을 이어간다.
    pending_answer = st.session_state.pop("pending_answer", None)

    with st.chat_message("assistant"):
        status = st.status("생각하는 중...", expanded=show_steps) if show_steps else None
        answer = ""

        # Langfuse: 같은 채팅 세션의 질문들을 하나의 session 으로 묶어서 기록
        langfuse = get_langfuse()
        recorder = None
        root_cm = attrs_cm = None
        if langfuse:
            from langfuse import propagate_attributes

            root_cm = langfuse.start_as_current_observation(name="main_agent", as_type="span", input=prompt)
            root = root_cm.__enter__()
            attrs_cm = propagate_attributes(
                session_id=st.session_state.session_id, tags=["streamlit", "main_agent"], trace_name="main_agent"
            )
            attrs_cm.__enter__()
            recorder = LangfuseTurn(langfuse, root)

        try:
            session = agent_session()
            if pending_answer is not None:
                session.answer(pending_answer)
                events = session.resume()
            else:
                events = session.send(prompt)
            for ev in events:
                if recorder:
                    recorder.on_event(ev)
                # 도구 호출 과정은 서브에이전트 안에서 난 것만 표시(app/ 의 namespace 조건과 같다)
                if ev.kind == "tool_call" and status and ev.nested:
                    status.markdown(f"**{ev.name}**")
                    status.code(truncate(ev.body), language="json")
                elif ev.kind == "tool_result" and status and ev.nested:
                    status.markdown(f"↳ {ev.name}")
                    status.code(truncate(to_text(ev.body)))
                elif ev.kind == "question":
                    # 메인 에이전트가 AskUserQuestion 으로 되물었다. 턴은 답을 기다리며 멈춰 있다.
                    st.session_state.pending_question = ev.body
                elif ev.kind == "done":
                    answer = ev.body
                elif ev.kind == "error":
                    raise RuntimeError(ev.body)

            if status:
                status.update(label="완료", state="complete", expanded=False)
        except Exception as e:
            if status:
                status.update(label="오류 발생", state="error")
            answer = f"⚠️ 오류가 발생했습니다: `{e}`"
        finally:
            if recorder:
                recorder.close()
                attrs_cm.__exit__(None, None, None)
                root_cm.__exit__(None, None, None)
                langfuse.flush()

        if st.session_state.pending_question:
            # 아직 답변이 없다. 되묻고 사용자의 답을 기다린다.
            answer = f"**확인이 필요합니다** — {st.session_state.pending_question}"
        elif st.session_state.agent is not None and st.session_state.agent.lost_context:
            st.session_state.agent.lost_context = False
            st.caption("이전 대화 맥락을 잃었습니다. 새 세션으로 이어갑니다.")

        st.markdown(answer or "_(응답이 비어 있습니다)_")

    st.session_state.messages.append({"role": "assistant", "content": answer})
```

(g) 마지막 입력 처리 블록:

```python
if typed:
    if st.session_state.pending_question:
        # 재질문에 대한 답. 새 질문이 아니라 기다리던 턴을 이어간다.
        st.session_state.pending_question = None
        st.session_state.pending_answer = typed
        st.session_state.messages.append({"role": "user", "content": typed})
        st.session_state.run_prompt = typed
        st.rerun()
    else:
        submit(typed)
```

- [ ] **Step 4: 통과 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_app_ui.py`
Expected: 6 passed.

- [ ] **Step 5: 커밋**

```bash
git add app2/app.py app2/tests/test_app_ui.py
git commit -m "feat(app2): Streamlit 이 상주 세션을 쓰고 AskUserQuestion 을 answer/resume 으로 잇는다

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: 옛 오케스트레이션 제거와 테스트 정리

**Files:**
- Modify: `app2/judge_agent.py` (docstring, `Judge`, `DECOMPOSE_RULES`, `JUDGE_SERVER`, `FACT_CHECK_TOOL`, `AGENT_TOOL_NAMES`, `tool` import 삭제)
- Modify: `app2/state.py` (`fact_checks`, `checked_answer`, `verdict`, `clarifying_question` 필드와 `clarification` 이벤트 종류 삭제, docstring 갱신)
- Modify: `app2/obsidian_tools.py` (`deny_vault_writes` 삭제, docstring 의 "세 겹" 설명 갱신)
- Modify: `app2/models.py` (`ask()` 삭제, docstring 갱신)
- Modify: `app2/domain_agents.py` (`DOMAIN_DESCRIPTIONS` 다듬기, docstring)
- Modify: `app2/tests/test_judge_agent.py` (Judge 테스트 삭제), `app2/tests/test_obsidian_tools.py` (훅 테스트 → `is_allowed_vault_tool`), `app2/tests/test_domain_agents.py` (description 검사 추가), `app2/tests/test_parity.py` (오케스트레이션 규칙 검사 정리)

**Interfaces:**
- Produces: `obsidian_tools.is_allowed_vault_tool(name) -> bool` (이미 있음, 유지), `domain_agents.DOMAIN_DESCRIPTIONS`
- 삭제: `judge_agent.Judge`, `judge_agent.DECOMPOSE_RULES`, `judge_agent.FACT_CHECK_TOOL`, `judge_agent.JUDGE_SERVER`, `judge_agent.AGENT_TOOL_NAMES`, `obsidian_tools.deny_vault_writes`, `models.ask`, `state.TurnState.{fact_checks, checked_answer, verdict, clarifying_question}`

- [ ] **Step 1: 테스트 먼저 고친다**

`app2/tests/test_judge_agent.py`: 상단 import 를 `from judge_agent import GUARDRAIL_THRESHOLD, Guardrail, decide, last_assistant_text, response_text` 로 바꾸고 `from state import DomainAnswer, TurnState` 유지. 다음 함수들을 **삭제**한다: `test_fact_check_tool_counts_and_instructs`, `test_record_domain_answer_collects_only_top_level_domain_agents`, `test_fact_check_is_limited_to_once_per_turn`, `test_stop_hook_requires_fact_check_after_multi_domain_merge`, `test_stop_hook_allows_single_domain_and_clarification`, `_deleg`, `_decision`, `test_decomposition_is_one_round_per_domain`, `test_failed_delegation_can_be_retried`, `test_at_most_three_domains_and_trend_is_not_limited`. Task 3 에서 넣은 `from judge_agent import Guardrail, last_assistant_text, response_text  # noqa: E402` 줄도 지운다(상단 import 로 옮겼으므로). 다음을 추가:

```python
def test_old_orchestration_symbols_are_gone():
    import judge_agent

    for name in ("Judge", "DECOMPOSE_RULES", "FACT_CHECK_TOOL", "JUDGE_SERVER"):
        assert not hasattr(judge_agent, name), name
```

`app2/tests/test_obsidian_tools.py`: `_decide` 와 훅 테스트 3개(`test_hook_denies_everything_outside_whitelist`, `test_hook_passes_whitelisted_read_tools`, `test_hook_ignores_non_obsidian_tools`)를 다음으로 바꾼다:

```python
from obsidian_tools import is_allowed_vault_tool  # noqa: E402


@pytest.mark.parametrize("name", ["mcp__obsidian__vault_write", "mcp__obsidian__vault_delete",
                                  "mcp__obsidian__command_execute", "mcp__obsidian__brand_new_write_tool"])
def test_whitelist_rejects_everything_outside_it(name):
    """화이트리스트라서 플러그인이 새 쓰기 도구를 추가해도 자동으로 막힌다(권한 게이트가 쓴다)."""
    assert not is_allowed_vault_tool(name)


@pytest.mark.parametrize("name", OBSIDIAN_TOOLS)
def test_whitelist_accepts_read_tools(name):
    assert is_allowed_vault_tool(name)


def test_whitelist_is_only_about_obsidian_tools():
    assert not is_allowed_vault_tool("mcp__rag__vector_search_tool")
    assert not hasattr(__import__("obsidian_tools"), "deny_vault_writes")
```
(`import asyncio` 가 더 이상 안 쓰이면 지운다.)

`app2/tests/test_domain_agents.py` 끝에 추가:

```python
def test_descriptions_carry_app_domain_definitions_for_auto_delegation():
    """위임 판단은 Claude Code 가 description 을 보고 한다. app/ DECOMPOSE_PROMPT 의 도메인 정의를 담는다."""
    from domain_agents import DOMAIN_DESCRIPTIONS

    assert "운영 중인 시스템을 돌리는 층" in DOMAIN_DESCRIPTIONS["server_infra"]
    assert "지금 돌아가는 걸 고치는" in DOMAIN_DESCRIPTIONS["server_infra"]
    assert "코드와 데이터의 구조를 정하는 층" in DOMAIN_DESCRIPTIONS["system_design"]
    assert "무엇을 만들지 정하는" in DOMAIN_DESCRIPTIONS["system_design"]
    assert "모델과 그 주변" in DOMAIN_DESCRIPTIONS["ai"]
    assert "Pinecone" in DOMAIN_DESCRIPTIONS["ai"]
```

`app2/tests/test_parity.py`:
- `test_trend_prompts_identical` 를 다음으로 바꾼다:

```python
def test_trend_prompt_keeps_app_role_and_constraints(app):
    """도구 단계만 내장 WebSearch/WebFetch 로 바뀌고 역할·제약 문단은 app/ 과 같다."""
    import general_it_trend_topic_agent as trend

    theirs = app["trend_system_prompt"]
    role = theirs[: theirs.index("> 지시사항")]
    constraints = theirs[theirs.index("> 답변 제약 사항"):]
    assert role in trend.system_prompt
    assert constraints in trend.system_prompt
```
- `test_orchestrator_rules_come_from_app_prompts` 를 다음으로 바꾼다:

```python
def test_rules_the_model_still_reads_come_from_app_prompts(app):
    """재질문·병합·재작성 규칙 문장은 app/ 과 같다. 분류표·분해 규칙은 없어졌다(자동 위임)."""
    import judge_agent
    import main_agent

    assert main_agent.AMBIGUITY_RULES == app["ambiguity_rules"]
    assert judge_agent.MERGE_RULES in app["merge_prompt"]
    assert judge_agent.FIX_RULES in app["fix_prompt"]
    assert judge_agent.GUARDRAIL_THRESHOLD == app["guardrail_threshold"]
    assert judge_agent._DOMAIN_LABELS == app["domain_labels"]
    assert not hasattr(main_agent, "INTENT_RULES")
    assert not hasattr(judge_agent, "DECOMPOSE_RULES")
```
- `test_tools_identical_to_app` 를 다음으로 바꾼다(벡터 검색만 비교):

```python
def test_vector_search_tool_identical_to_app(app):
    """모델이 보는 도구 이름·설명·인자 스키마가 같다(웹 도구는 내장 WebSearch/WebFetch 로 대체)."""
    import rag_tools

    lc = app["tools"][0]
    assert rag_tools.vector_search_tool.name == lc["name"]
    assert rag_tools.vector_search_tool.description == lc["description"]
    assert _norm(rag_tools.vector_search_tool.input_schema) == _norm(lc["parameters"])
```
- `_DUMP` 는 그대로 둔다(app/ 쪽 값 추출).

- [ ] **Step 2: 실패 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q tests/test_judge_agent.py tests/test_obsidian_tools.py tests/test_domain_agents.py`
Expected: `test_old_orchestration_symbols_are_gone`, `test_whitelist_is_only_about_obsidian_tools`, `test_descriptions_carry_app_domain_definitions_for_auto_delegation` 실패.

- [ ] **Step 3: 소스 정리**

`app2/judge_agent.py`:
- 모듈 docstring 을 다음으로:
```python
"""판단 guardrail — 병합 규칙과 팩트체크.

app/ 은 분해(decompose)·병합(merge)을 판단 Agent 의 LLM 호출 두 번으로 하고 _guard 가 1회
검증한다. app2 에서 분해는 Claude Code 의 서브에이전트 위임(각 AgentDefinition.description 을
보고 메인 에이전트가 고른다)이고, 병합은 메인 에이전트의 응답 자체다. 코드에 남는 것은
guardrail 뿐이며 훅 두 개로 구현한다.

  PostToolUse(Agent)  서브에이전트가 돌려준 답을 TurnState 에 모은다 (검증 근거)
  Stop                도메인 답이 2개 이상이면 최종 답을 팩트체크. 미달이면 block + FIX_RULES 로
                      한 번 고쳐 쓰게 한다. 두 번째 Stop 은 stop_hook_active 가 True 라 통과한다
                      (SDK 내장 "한 번만" 의미론 — app/ 의 1회 재작성과 같다).

규칙 문장(MERGE_RULES, FIX_RULES)은 app/ 의 MERGE_PROMPT / FIX_PROMPT 에서 그대로 가져왔다.
"""
```
- `from claude_agent_sdk import tool` 삭제. `JUDGE_SERVER`, `FACT_CHECK_TOOL`, `AGENT_TOOL_NAMES`, `DECOMPOSE_RULES`, `class Judge` 전체 삭제. `_response_text = response_text` 별칭도 삭제.

`app2/state.py`: docstring 을 갱신하고 `TurnState` 를 다음으로:

```python
"""턴 상태와 UI 이벤트.

app/state.py 는 LangGraph 가 노드 사이로 넘기는 그래프 상태(MainQueryState 등)를 정의한다.
Agent SDK 에서는 대화 상태를 Claude Code 세션(상주 client)이 들고 있다. 여기에 남는 건 둘뿐이다.
  - TurnState: 한 턴 동안 훅과 권한 게이트가 함께 보는 값(서브에이전트 답, 팩트체크 결과)
  - Event:     UI 가 그리는 스트림 이벤트
"""
```
```python
@dataclass
class TurnState:
    # 도메인 서브에이전트(server_infra/system_design/ai)가 돌려준 답
    domain_answers: list[DomainAnswer] = field(default_factory=list)
    # 트렌드 서브에이전트가 돌려준 답
    trend_answers: list[str] = field(default_factory=list)
    # Stop 훅의 팩트체크 결과 {"verdict": "PASS"|"REWRITE", "score": float|None, "comment": str}. 안 돌았으면 None
    fact_check: dict[str, Any] | None = None


EventKind = Literal["tool_call", "tool_result", "answer", "question", "done", "error"]
```

`app2/obsidian_tools.py`: `deny_vault_writes` 함수와 `# ---- 읽기 전용 훅 ----` 구분선 주석을 지우고(`_PREFIX`, `is_allowed_vault_tool` 은 남긴다) docstring 의 "세 겹으로 막는다" 목록을 다음으로:
```
볼트는 절대 읽기 전용이다. 플러그인은 쓰기/삭제/커맨드 실행까지 16종을 노출하므로
두 겹으로 막는다. 모두 화이트리스트라서 플러그인이 새 쓰기 도구를 추가해도 자동 차단된다.
  1. AgentDefinition.tools          서브에이전트에게 보이는 도구 = 화이트리스트 5종
  2. can_use_tool (main_agent)       mcp__obsidian__* 중 화이트리스트 밖이면 거부 (is_allowed_vault_tool)
```

`app2/models.py`: `ask()` 함수 삭제. docstring 의 `도구 안의 단발 LLM 호출` 줄을 `팩트체크의 구조화 출력 단발 호출   query(options=one_shot_options("judge", output_format=...))` 로. `# 단발 LLM 호출 — 도구 안에서 쓰는 요약/확장/검증` 주석을 `# 단발 LLM 호출 — Stop 훅의 팩트체크(구조화 출력)` 로.

`app2/domain_agents.py`: docstring 의 "오케스트레이터가 한 턴에 Agent 도구를 여러 번 부르면" 문장은 그대로 두되, 앞에 한 줄 추가: `어느 서브에이전트에 맡길지는 메인 에이전트가 각 description 을 보고 정한다(공식 문서의 자동 위임).` `DOMAIN_DESCRIPTIONS` 를 다음으로:

```python
# 메인 에이전트가 어느 서브에이전트에 맡길지 고를 때 읽는 설명 — 자동 위임의 근거.
# 도메인 정의와 경계 기준은 app/ 판단 Agent 의 DECOMPOSE_PROMPT 와 같다.
DOMAIN_DESCRIPTIONS = {
    "server_infra": (
        "운영 중인 시스템을 돌리는 층의 전문가. 배포, 쿠버네티스/컨테이너, 네트워크, DB 운영·튜닝, "
        "모니터링, 장애 진단, 용량 산정. \"지금 돌아가는 걸 고치는\" 질문이나 그 서브 질문을 맡긴다. "
        "개인 옵시디언 볼트만 근거로 답한다."
    ),
    "system_design": (
        "코드와 데이터의 구조를 정하는 층의 전문가. 서비스 경계, 모듈 분리, API 계약, 데이터 모델링, "
        "일관성/확장성 트레이드오프. \"무엇을 만들지 정하는\" 질문이나 그 서브 질문을 맡긴다. "
        "개인 옵시디언 볼트만 근거로 답한다."
    ),
    "ai": (
        "모델과 그 주변의 전문가. LLM, RAG, 에이전트, 모델 서빙, 임베딩, 벡터 검색, 프롬프트에 관한 "
        "질문이나 그 서브 질문을 맡긴다. 색인된 논문(Pinecone)과 개인 옵시디언 볼트만 근거로 답한다."
    ),
}
```

- [ ] **Step 4: 전체 오프라인 스위트 통과 확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q`
Expected: 전부 통과 (llm/obsidian 제외). `test_parity.py` 는 app/ 을 서브프로세스로 import 하므로 app/ 의존성(langchain 등)이 설치돼 있어야 한다 — 이미 같은 venv 에 있다.

- [ ] **Step 5: 커밋**

```bash
git add app2/judge_agent.py app2/state.py app2/obsidian_tools.py app2/models.py app2/domain_agents.py app2/tests
git commit -m "refactor(app2): 옛 오케스트레이션(Judge, 분해 규칙, 커스텀 재질문/팩트체크 도구, 훅) 제거

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: 라이브 테스트, README, 실측 기록

**Files:**
- Rewrite: `app2/tests/test_live.py` (Task 1 의 스파이크 두 개는 유지)
- Rewrite: `app2/README.md`

**Interfaces:**
- Consumes: `main_agent.new_session`, `main_agent.GENERAL_REPLY`

- [ ] **Step 1: `test_live.py` 앞부분 교체**

파일 상단부터 Task 1 에서 넣은 `# SDK 실측` 구분선 **직전까지**를 다음으로 바꾼다:

```python
"""실제 Claude Code 로 도는 end-to-end 테스트 (pytest -m llm, 비용 발생).

app/ 의 llm 테스트(분류·재질문·도메인 실행)에 대응한다. app2 에서는 분류가
"어느 서브에이전트에 위임했는가"로 드러나므로 그걸 확인한다.
"""
import pytest

from main_agent import GENERAL_REPLY, new_session

pytestmark = pytest.mark.llm


@pytest.fixture
def session():
    s = new_session()
    yield s
    s.close()


def _delegations(events):
    return [e.body.get("subagent_type") for e in events if e.kind == "tool_call" and e.name == "Agent"]


def test_greeting_gets_general_reply_without_tools(session):
    events = list(session.send("안녕하세요"))
    assert not [e for e in events if e.kind == "tool_call"]
    assert events[-1].kind == "done"
    assert "도와드릴 수 있습니다" in events[-1].body or GENERAL_REPLY in events[-1].body


def test_concept_question_is_delegated_to_trend_agent_with_builtin_web_search(session):
    events = list(session.send("JWT와 세션 인증의 차이점이 뭔가?"))
    done = events[-1]
    assert done.kind == "done", done.body
    assert _delegations(events) == ["general_it_trend"]
    nested = {e.name for e in events if e.kind == "tool_call" and e.nested}
    assert "WebSearch" in nested, nested
    assert "Source" in done.body


def test_vague_design_request_asks_user_then_continues_same_turn(session):
    events = list(session.send("우리 서비스 아키텍처 좀 잡아줘"))
    assert events[-1].kind == "question", [e.kind for e in events]
    assert _delegations(events) == []
    assert session.waiting_question

    session.answer("MAU 10만, 피크 200 rps, 스프링부트 + MySQL, 3명 팀")
    rest = list(session.resume())
    assert rest[-1].kind == "done", rest[-1].body
    assert _delegations(events + rest)  # 답을 받은 뒤 위임이 일어난다


def test_multi_domain_question_runs_parallel_subagents_and_one_fact_check(session):
    events = list(session.send(
        "MAU 10만 사내 챗봇을 만든다. RAG 파이프라인 구조와 그걸 돌릴 쿠버네티스 배포·모니터링을 같이 잡아줘."
    ))
    done = events[-1]
    assert done.kind == "done", done.body
    domains = set(_delegations(events)) & {"server_infra", "system_design", "ai"}
    assert len(domains) >= 2, domains
    assert done.data["fact_check"] is not None
    assert done.data["fact_check"]["verdict"] in ("PASS", "REWRITE")


```

- [ ] **Step 2: 라이브 실행**

Run: `cd app2; $env:APP2_TEST_USE_CLAUDE_LOGIN="1"; ..\.venv\Scripts\python.exe -m pytest -q -m llm tests/test_live.py -s`
Expected: 6 passed (Task 1 의 2개 포함). 실행 불가면 그 사실을 보고하고 README 의 "실측" 절에 "미실측" 으로 적는다.

실패 시 확인 순서:
1. 트렌드 테스트에서 `WebSearch` 가 nested 에 없으면 → 서브에이전트가 메인의 `tools` 밖 내장 도구를 못 쓰는 것이다. `main_agent.BUILTIN_TOOLS = ["Agent", "AskUserQuestion", "WebSearch", "WebFetch"]` 로 바꾼다(게이트가 메인의 웹 호출은 여전히 거부한다). `test_main_agent.test_options_wire_subagents_gate_and_hooks` 의 `opts.tools` 기대값도 같이 바꾼다.
2. WebSearch 가 네트워크 오류로 실패하면 → `models.base_env` 에서 `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` 줄을 지우고 `test_models.py` 에 그 키를 기대하는 assert 가 있으면 함께 지운다.
3. Stop 훅이 팩트체크를 안 돌리면(`fact_check is None`) → Task 1 의 출력에서 `last_assistant_message` 형식을 확인하고 `judge_agent.response_text` 가 그 형식을 다루는지 본다.

- [ ] **Step 3: README 교체**

`app2/README.md` 전체:

````markdown
# app2 — 멀티 에이전트 IT 어시스턴트 (Claude Agent SDK)

`app/`(LangGraph)과 **화면·도메인 프롬프트·팩트체크가 같은** 앱을 Claude Agent SDK 방식으로 다시 만든 것.
그래프도, 손으로 짠 오케스트레이터도 없다. 메인 에이전트에게 서브에이전트 4종과 규칙 몇 줄을 주면
공식 문서대로 "Claude automatically decides when to invoke subagents based on the task and each
subagent's description."

## 실행

```powershell
cd app2
..\.venv\Scripts\python.exe -m streamlit run app.py
```

- 키는 `app2/.env` 가 있으면 그것을, 없으면 `app/.env` 를 읽는다.
- `LITELLM_BASE_URL` 이 있으면 게이트웨이를 경유한다(app/ 과 같은 규칙). Claude Code 는
  Anthropic Messages API(`/v1/messages`)로 붙는데, LiteLLM 이 같은 포트에서 제공한다.
- API 키도 게이트웨이도 없으면 Claude Code 는 이 PC 의 Claude 로그인을 쓴다.
- 전제조건은 app/ 과 같다: Obsidian 실행 중, infra 스택(LiteLLM·Langfuse). 웹 검색은 Claude Code
  내장 도구라 Serper 키가 필요 없다.
- `AGENT_MAX_BUDGET_USD`(기본 3): 턴당 비용 상한.

## 구조

```
사용자 ─ Streamlit (app.py)
   │  AgentSession.send(prompt) → Event 스트림     ← 채팅 세션당 client 1개 상주, 턴마다 client.query()
   ▼
메인 에이전트 (Opus 5, main_agent.py)              ← 짧은 시스템 프롬프트. 누구에게 맡길지는 Claude Code 가 description 으로 판단
   │  tools = Agent, AskUserQuestion
   │  can_use_tool = 권한 게이트 한 함수 (재질문 대기 / 도메인당 1회 / 조사 도구는 서브에이전트만 / 볼트 읽기 전용)
   │  hooks: PostToolUse(Agent) 답 수집, Stop 팩트체크
   ├─ server_infra   ──▶ 옵시디언 MCP (이 서브에이전트에만 연결)
   ├─ system_design  ──▶ 옵시디언 MCP
   ├─ ai             ──▶ vector_search_tool (in-process) + 옵시디언 MCP
   └─ general_it_trend ─▶ 내장 WebSearch / WebFetch
```

### LangGraph → Agent SDK 대응

| app/ (LangGraph) | app2/ (Agent SDK) |
|---|---|
| `classify` 노드 + `route_by_intent` | 없음. 각 `AgentDefinition.description` 을 보고 Claude Code 가 위임 |
| `decompose` + `Send` fan-out | 메인 에이전트가 한 응답에서 `Agent` 를 여러 번 호출 → 병렬 서브에이전트 |
| `create_agent` 도메인 에이전트 3종 | `AgentDefinition` 3종 (`domain_agents.py`, 프롬프트는 글자까지 같음) |
| 트렌드 서브그래프 + 웹 도구 3종 | `AgentDefinition` 1개 + 내장 `WebSearch`/`WebFetch` |
| `merge` LLM 호출 | 메인 에이전트의 응답 자체 (병합 규칙 문장은 app/ 그대로) |
| `decompose` 의 `[:3]` | `can_use_tool` — 이미 답한 도메인에 다시 위임하면 거부 |
| `_guard` (fact_check → 1회 재작성) | `Stop` 훅이 최종 답을 검증, 미달이면 block. 두 번째 Stop 은 `stop_hook_active` 로 통과 |
| `interrupt()` + `Command(resume)` | 내장 `AskUserQuestion` → `can_use_tool` 이 UI 의 답을 기다렸다가 돌려줌 |
| `InMemorySaver` (thread_id) | `ClaudeSDKClient` 상주 (`async_utils.AgentSession`) |
| `langchain_mcp_adapters` 변환 | MCP 서버 설정(dict)만 넘김 — Claude Code 가 MCP 클라이언트 |
| `SummarizationMiddleware` | Claude Code 자동 컴팩션 (임계치 120k 로 맞춤) |
| `graph.stream(subgraphs=True)` | 메시지 스트림 (`parent_tool_use_id` 로 중첩 구분) |
| LangChain `CallbackHandler` (Langfuse) | 메시지 스트림으로 trace 구성 (`app.py` 의 `LangfuseTurn`) |
| LangSmith 자동 추적 | `langsmith.integrations.claude_agent_sdk` 공식 연동 |

### 파일

app/ 과 파일 이름이 같다. 역할도 같은 자리에 있다.

| 파일 | 역할 |
|---|---|
| `main_agent.py` | 시스템 프롬프트, `MainAgent`(권한 게이트, 메시지→이벤트), `build_options`, `new_session` |
| `judge_agent.py` | 병합/재작성 규칙, `Guardrail`(PostToolUse 수집 + Stop 팩트체크) |
| `domain_agents.py` | 도메인 서브에이전트 3종 (프롬프트는 app/ 과 글자까지 같음, description 은 위임 근거) |
| `general_it_trend_topic_agent.py` | 트렌드 서브에이전트 (내장 WebSearch/WebFetch) |
| `rag_tools.py` | `vector_search_tool`, 팩트체크 (Pinecone/OpenAI/Cohere 공식 클라이언트) |
| `obsidian_tools.py` | 옵시디언 MCP 설정, 읽기 전용 화이트리스트 |
| `models.py` | 역할별 모델, 게이트웨이/실행 환경, 구조화 출력 단발 `query()` |
| `state.py` | 턴 상태(TurnState), UI 이벤트 |
| `async_utils.py` | `AgentSession` — Streamlit(동기)에서 상주하는 client, question/answer/resume |
| `app.py` | Streamlit UI (CSS·테마·문구 app/ 과 동일) |
| `pre_process_save_document.py` | PDF → Pinecone 색인 (app/ 과 동일 파일) |

## Agent SDK 로 바꾸며 얻은 것

- **오케스트레이터가 없다.** 의도 분류표도, 분해 프롬프트도, "받은 답을 그대로 전달하라"는 프로토콜도
  없다. 서브에이전트 description 이 곧 라우팅 규칙이고, 병합은 메인 에이전트가 답을 쓰는 것 자체다.
- **강제 규칙은 함수 하나.** 재질문 대기·도메인당 1회·조사 도구 격리·볼트 읽기 전용이 전부
  `can_use_tool` 한 함수에 있다. `allowed_tools` 가 비어 있어 빠져나가는 호출이 없다.
- **"1회 재작성"이 SDK 의미론이다.** Stop 훅이 block 하면 모델이 고쳐 쓰고 다시 Stop 하는데, 그때는
  `stop_hook_active` 가 True 라 통과한다. 횟수를 세는 코드가 없다.
- **재질문이 진짜 interrupt 다.** 턴이 끝나지 않고 멈춘 채 사용자 답을 기다린다. 커스텀 도구도,
  "[사용자 추가 정보]" 접두어도, resume 도 없다.
- **도구 안에서 LLM 을 안 부른다.** 질의 확장·페이지 요약은 서브에이전트가 직접 한다. Claude Code
  프로세스를 띄우는 단발 호출은 팩트체크(턴당 최대 1회)뿐이다.
- **컨텍스트 격리·도구 스코핑·TLS** 는 이전과 같다(옵시디언 MCP 는 서브에이전트에만, 인증서는
  `NODE_EXTRA_CA_CERTS`).

## 실측으로 알게 된 것 (함정)

- **`tools=[]` 는 `Agent` 도구까지 끈다.** `tools=["Agent", "AskUserQuestion"]` 로 연다.
- **서브에이전트가 백그라운드로 뜬다.** 기본값이면 "Async agent launched" 만 받고 턴이 끝난다.
  `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` + `AgentDefinition(background=False)`.
- **출력 한도 초과는 에러다.** 4096 이면 서브에이전트가 종료된다. 에이전트 세션은 16000.
- **`can_use_tool` 은 `permission_mode="default"` 에서만 불린다.** `dontAsk` 면 SDK 가 경고하고
  콜백을 건너뛴다(AskUserQuestion 도 못 받는다).
- **in-process MCP 서버는 서브에이전트에 인라인으로 못 넣는다.** SDK 가 최상위 `mcp_servers` 이름으로만
  브리지를 만든다. 그래서 `mcp__rag__*` 는 메인에게도 보이고, 게이트가 `agent_id` 없는 호출을 거부한다.
- **위임 폭주.** 프롬프트만으로는 같은 도메인에 보충 조사를 다시 맡겨 5회까지 갔다($8.5). 게이트의
  도메인당 1회 + `max_budget_usd` 로 막는다.
- **Stop 훅 입력의 최종 답.** (Task 1 실측 결과를 여기에 적는다: `last_assistant_message` 의 형식,
  없을 때 `transcript_path` 폴백 사용 여부.)
- **`st.session_state` 는 스크립트 스레드에서만.** 세션 스레드는 `AgentSession` 안의 큐로만 말한다.
- **사용자 PC 설정이 섞인다.** `setting_sources=[]`, `skills=[]`, `strict_mcp_config=True` 로 격리한다.

## app/ 과 의도적으로 다른 점

- **단일 서브에이전트 답을 글자 그대로 전달하지 않는다.** 메인 에이전트가 "그 답을 바탕으로" 답한다.
- **"unknown" 안내문·빈 답 안내문은 프롬프트 지시다.** 코드가 문장을 고정하지 않는다.
- **웹 검색 엔진이 다르다.** Serper(구글 kr) 대신 Claude Code 내장 WebSearch.
- **리듀서 누적 동작을 재현하지 않았다.** app/ 은 같은 thread 에서 `domain_answers` 가 턴마다 누적된다(버그).
- **모델이 보는 도구 이름**은 `mcp__<서버>__<도구>` 형태다. UI 에는 app/ 처럼 `vault_read` 로 보인다.

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
| `test_parity.py` | app/ 을 별도 프로세스로 import 해 도메인 프롬프트·팩트체크 스키마·재질문/병합 규칙·UI 문자열이 **바이트 단위로 같은지**, 벡터 검색 결과가 같은지(llm) |
| `test_main_agent.py` | 옵션 배선, 시스템 프롬프트에 분류표가 없음, 권한 게이트 판정, 메시지 스트림 → 이벤트, 결과 → done/error |
| `test_judge_agent.py` | PASS/REWRITE/검증불가, Stop 훅(2개 미만 통과·미달 block·`stop_hook_active` 통과), PostToolUse 수집 |
| `test_async_utils.py` | 가짜 client 로 `AgentSession`: 턴 재사용, question → answer → resume, 오류 복구, close |
| `test_app_ui.py` | Streamlit AppTest — 질문·도구 과정 표시·재질문→답→같은 턴·오류·초기화 |
| `test_live.py` | (llm) 인사말 / 개념 질문→트렌드+WebSearch / 모호한 설계→AskUserQuestion→이어짐 / 다중 도메인→병렬+팩트체크 / SDK 실측 2종 |
````

Task 1 의 실측 결과를 "Stop 훅 입력의 최종 답" 항목에 실제 문장으로 적는다(예: "`last_assistant_message` 는 `{"role": "assistant", "content": [...]}` 형식으로 온다. `response_text` 가 처리한다.").

- [ ] **Step 4: 오프라인 스위트 재확인**

Run: `cd app2; ..\.venv\Scripts\python.exe -m pytest -q`
Expected: 전부 통과.

- [ ] **Step 5: 커밋**

```bash
git add app2/tests/test_live.py app2/README.md app2/main_agent.py app2/models.py app2/tests/test_main_agent.py app2/tests/test_models.py
git commit -m "docs(app2): SDK 네이티브 구조 README, 라이브 테스트 4종

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
(`main_agent.py`/`models.py`/`test_models.py` 는 Step 2 의 실패 시 조치를 했을 때만 바뀐다.)

---

## Self-review 결과

- **스펙 커버리지:** 메인 에이전트·프롬프트(Task 5), 권한 게이트 5단계(Task 5), Stop 팩트체크·PostToolUse 수집(Task 3), 서브에이전트 description·트렌드 내장 도구(Task 4, 7), `AgentSession`·UI(Task 2, 6), 오류 처리(Task 2 `_main` except, Task 5 `result_event`, Task 6 `lost_context` 캡션), 테스트 표(Task 2~8), README 대응표(Task 8), 실측 스파이크(Task 1). 스펙의 "Stop 훅 `last_assistant_message` 형식 실측" 은 Task 1 + Task 8 Step 3.
- **타입 일관성:** `TurnDriver.options(ask_user)` ↔ `MainAgent.options(ask_user)`; `AgentSession.ask_user(questions) -> dict[str, str]` ↔ `permission_gate` 의 `answers`; `Guardrail(turn_of)` ↔ `MainAgent.guardrail = Guardrail(lambda: self.turn)`; `Event` 종류 `question` ↔ app.py/test_app_ui; `result_event` 의 `data["fact_check"]` ↔ `test_live` / `TurnState.fact_check`.
- **임시 상태:** Task 2 뒤 `test_app_ui.py`, Task 5 뒤 `test_app_ui.py`/`test_live.py` 가 일시적으로 실패한다. 각 태스크의 검증 명령은 그 태스크의 테스트 파일로 한정했고, Task 7 Step 4 와 Task 8 Step 4 에서 전체 스위트를 돌린다.
