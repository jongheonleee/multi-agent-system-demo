"""동기 코드에서 코루틴을 안전하게 실행하는 유틸.

MCP 어댑터가 만든 도구는 coroutine 만 있고 sync func 이 없다. 그래프는 동기로
유지하면서 이 도구들을 쓰려면 곳곳에서 코루틴을 돌려야 하는데, 이미 이벤트 루프가
도는 상황(비동기 노드 안, Streamlit 등)에서 asyncio.run 을 부르면 RuntimeError 가
난다. 그 경우 별도 스레드에서 새 루프를 연다.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


def run_coro(coro):
    """실행 중인 루프가 있든 없든 코루틴을 끝까지 돌리고 결과를 반환한다."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 실행 중인 루프가 없다 -> 그냥 새로 연다.
        return asyncio.run(coro)

    # 이미 루프가 돌고 있다 -> 별도 스레드에서 새 루프로 돌린다.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
