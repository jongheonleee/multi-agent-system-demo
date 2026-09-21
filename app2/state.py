"""턴 상태와 UI 이벤트.

app/state.py 는 LangGraph 가 노드 사이로 넘기는 그래프 상태(MainQueryState 등)를 정의한다.
Agent SDK 에서는 대화 상태를 Claude Code **세션**이 들고 있다. 다음 턴에는
session_id 로 resume 하면 이전 질문·답·도구 결과가 그대로 이어진다(체크포인터 불필요).

그래서 여기에 남는 건 두 가지뿐이다.
  - TurnState: 한 턴 동안 훅과 도구가 함께 보는 값(도메인 답변, 검증 횟수, 재질문)
  - Event:     UI 가 그리는 스트림 이벤트
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Domain = Literal["server_infra", "system_design", "ai"]


@dataclass
class DomainAnswer:
    """도메인 서브에이전트 한 개가 낸 답변 (app/ 의 DomainAnswer 와 같은 필드)."""

    domain: Domain
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)


@dataclass
class TurnState:
    # 도메인 서브에이전트(server_infra/system_design/ai)가 돌려준 답
    domain_answers: list[DomainAnswer] = field(default_factory=list)
    # 트렌드 서브에이전트가 돌려준 답
    trend_answers: list[str] = field(default_factory=list)
    fact_checks: int = 0
    # fact_check 에 넘어온 병합 답과 판정(PASS / REWRITE)
    checked_answer: str = ""
    verdict: str = ""
    clarifying_question: str = ""


EventKind = Literal["tool_call", "tool_result", "answer", "clarification", "done", "error"]


@dataclass
class Event:
    kind: EventKind
    name: str = ""
    body: Any = None
    # 서브에이전트 안에서 난 이벤트인지. app/ 의 "namespace 가 있는 업데이트" 와 같다.
    nested: bool = False
    tool_use_id: str | None = None
    parent_tool_use_id: str | None = None
    is_error: bool = False
    data: dict[str, Any] = field(default_factory=dict)
