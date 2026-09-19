"""
Agent SDK 실행 루프를 Streamlit(동기 코드)에서 쓸 수 있게 감싼 어댑터.

claude_agent_sdk.query() 는 async 이터레이터라 그대로는 Streamlit 스크립트에서
쓸 수 없다. asyncio.run() 을 메인 스레드에서 돌리고, 이벤트가 나올 때마다
동기 콜백을 호출한다. 콜백이 메인 스레드에서 실행되므로 그 안에서 st.* 를
호출해도 안전하다.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from agents import build_options

EventKind = Literal["text", "tool_use", "tool_result", "done", "error"]


@dataclass
class Event:
    kind: EventKind
    # tool_use / tool_result 일 때 도구 이름
    name: str = ""
    # 사람이 읽을 본문
    body: str = ""
    # 서브에이전트가 낸 이벤트인지 (parent_tool_use_id 가 있으면 서브에이전트)
    nested: bool = False
    is_error: bool = False


@dataclass
class RunOutcome:
    answer: str = ""
    session_id: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    events: list[Event] = field(default_factory=list)
    error: str | None = None


def _short_tool_name(name: str) -> str:
    """mcp__research__vector_search -> vector_search"""
    return name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name


def _stringify(content: Any, limit: int = 800) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or json.dumps(block, ensure_ascii=False, default=str))
            else:
                parts.append(str(block))
        text = "\n".join(parts)
    else:
        text = json.dumps(content, ensure_ascii=False, default=str)
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ..."


async def _run_async(
    prompt: str,
    on_event: Callable[[Event], None],
    resume: str | None,
    cwd: str | None,
) -> RunOutcome:
    outcome = RunOutcome()
    options = build_options(cwd=cwd)
    if resume:
        options.resume = resume

    def emit(event: Event) -> None:
        outcome.events.append(event)
        on_event(event)

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            nested = message.parent_tool_use_id is not None
            for block in message.content:
                if isinstance(block, TextBlock):
                    if not block.text.strip():
                        continue
                    if nested:
                        # 서브에이전트의 중간 서술. 진행 상황으로만 표시한다.
                        emit(Event("text", body=block.text, nested=True))
                    else:
                        # 라우터의 최종 답변. 마지막 것이 답이 된다.
                        outcome.answer = block.text
                        emit(Event("text", body=block.text))
                elif isinstance(block, ToolUseBlock):
                    emit(
                        Event(
                            "tool_use",
                            name=_short_tool_name(block.name),
                            body=_stringify(block.input, limit=400),
                            nested=nested,
                        )
                    )

        elif isinstance(message, UserMessage):
            if isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        emit(
                            Event(
                                "tool_result",
                                body=_stringify(block.content),
                                nested=message.parent_tool_use_id is not None,
                                is_error=bool(block.is_error),
                            )
                        )

        elif isinstance(message, ResultMessage):
            outcome.session_id = message.session_id
            outcome.cost_usd = message.total_cost_usd
            outcome.num_turns = message.num_turns
            if message.is_error:
                outcome.error = message.result or f"실행이 {message.subtype} 로 종료되었습니다"
                emit(Event("error", body=outcome.error, is_error=True))
            elif message.result and not outcome.answer:
                # 라우터가 텍스트 블록 없이 끝난 경우의 폴백
                outcome.answer = message.result
            emit(Event("done", body=outcome.answer))

    return outcome


def run(
    prompt: str,
    on_event: Callable[[Event], None] | None = None,
    resume: str | None = None,
    cwd: str | None = None,
) -> RunOutcome:
    """동기 진입점. 메인 스레드에서 호출할 것."""
    callback = on_event or (lambda _event: None)
    try:
        return asyncio.run(_run_async(prompt, callback, resume, cwd))
    except Exception as e:  # CLI 미설치, 인증 실패, 네트워크 등
        outcome = RunOutcome(error=f"{type(e).__name__}: {e}")
        callback(Event("error", body=outcome.error, is_error=True))
        return outcome
