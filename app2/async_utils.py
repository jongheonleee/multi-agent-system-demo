"""동기 코드(Streamlit)에서 비동기 코드를 쓰는 유틸.

Agent SDK 는 전부 async 다. Streamlit 스크립트는 동기라서, 별도 스레드에서 이벤트
루프를 돌리고 결과를 큐로 넘겨 받는다. st.* 호출은 소비하는 쪽(스크립트 스레드)에서
하므로 안전하다.
"""
from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Callable, Iterator, TypeVar

T = TypeVar("T")
_DONE = object()


def run_coro(coro):
    """실행 중인 루프가 있든 없든 코루틴을 끝까지 돌리고 결과를 반환한다."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def iterate_in_thread(make_stream: Callable[[], AsyncIterator[T]]) -> Iterator[T]:
    """async 제너레이터를 별도 스레드의 루프에서 돌리며 항목을 하나씩 동기로 내준다."""
    items: queue.Queue = queue.Queue()

    async def pump() -> None:
        async for item in make_stream():
            items.put(item)

    def run() -> None:
        try:
            asyncio.run(pump())
        except BaseException as e:  # noqa: BLE001 - 소비하는 쪽에서 다시 raise
            items.put(e)
        finally:
            items.put(_DONE)

    # 호출한 쪽의 컨텍스트(LangSmith 부모 run 등)를 스레드로 넘긴다.
    thread = threading.Thread(target=contextvars.copy_context().run, args=(run,), daemon=True)
    thread.start()
    while True:
        item = items.get()
        if item is _DONE:
            break
        if isinstance(item, BaseException):
            thread.join()
            raise item
        yield item
    thread.join()
