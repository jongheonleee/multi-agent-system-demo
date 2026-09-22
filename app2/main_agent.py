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
