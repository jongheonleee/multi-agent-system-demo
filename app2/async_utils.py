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
        """client 를 닫고 스레드를 끝낸다. 재질문 대기와 진행 중인 턴은 취소된다."""
        if self._loop is None:
            return
        loop, thread = self._loop, self._thread
        assert thread is not None and self._requests is not None
        self._closing = True
        if self._reply is not None:
            fut = self._reply
            loop.call_soon_threadsafe(lambda: fut.done() or fut.cancel())
        if self._client is not None:
            asyncio.run_coroutine_threadsafe(self._interrupt(), loop)
        loop.call_soon_threadsafe(self._requests.put_nowait, _CLOSE)
        thread.join(timeout=15)
        if thread.is_alive():
            logger.warning("에이전트 세션 스레드가 15초 안에 끝나지 않았습니다")
        self._loop = self._thread = self._requests = None
        self._reply = None
        self._client = None
        self.waiting_question = None
        self._closing = False
        self._events = queue.Queue()  # 타임아웃난 스레드의 잔여 이벤트가 다음 세션으로 새지 않게 한다.

    async def _interrupt(self) -> None:
        """진행 중인 턴이 있으면 중단시킨다. close() 가 루프 쪽으로 스케줄한다."""
        if self._client is None:
            return
        try:
            await self._client.interrupt()
        except Exception:  # noqa: BLE001 — 이미 닫는 중이니 실패해도 무시한다.
            logger.warning("client.interrupt() 호출 실패", exc_info=True)

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
            except Exception as e:  # noqa: BLE001 — client 가 죽었거나 드라이버가 예외를 냈다. 어느 쪽이든 UI 에 error 를 보내고 다음 턴에 client 를 새로 만든다.
                self._client = None
                self._reply = None
                self.waiting_question = None
                if self._closing:
                    return
                logger.exception("에이전트 세션 오류")
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
