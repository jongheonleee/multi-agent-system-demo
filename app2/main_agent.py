"""메인 에이전트(오케스트레이터).

app/main_agent.py 는 LangGraph 그래프다.

  orchestrator ─┬─ complex ─▶ decompose ─(Send)─▶ domain_worker ×N ─▶ merge ─┐
                ├─ general ─▶ general_it_topic_agent ──────────────────────┤
                └─ unknown ─▶ general_topic_agent ─────────────────────────┴─▶ finalize

Agent SDK 에서는 이 그래프를 코드로 짜지 않는다. 오케스트레이터 하나에게 규칙을 주고
서브에이전트와 도구를 쥐여주면, 라우팅·분해·병렬 위임·병합을 모델이 한 턴 안에서 한다.

  LangGraph (app/)                       Agent SDK (app2/)
  classify + route_by_intent         ->  오케스트레이터 프롬프트의 분류 규칙
  decompose + Send fan-out           ->  한 응답에서 Agent 도구를 여러 번 호출(병렬 실행)
  domain_worker / general_it 노드    ->  AgentDefinition 서브에이전트
  merge                              ->  오케스트레이터가 직접 병합
  _guard (fact_check, 1회 재작성)     ->  fact_check 도구 + PreToolUse/Stop 훅 (judge_agent.py)
  interrupt / Command(resume)        ->  request_clarification 도구 + 세션 resume
  InMemorySaver (thread_id)          ->  Claude Code 세션 (session_id 로 resume)
  graph.stream(subgraphs=True)       ->  ClaudeSDKClient 메시지 스트림 (parent_tool_use_id 로 중첩 구분)
  SummarizationMiddleware            ->  Claude Code 자동 컴팩션

코드가 강제해야 하는 규칙(읽기 전용, 검증 1회, 오케스트레이터는 직접 조사 금지)만 훅으로 건다.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)

from domain_agents import DOMAINS, build_domain_agents
from general_it_trend_topic_agent import TREND_AGENT, WEB_TOOLS, build_trend_agent, web_server
from judge_agent import (
    AGENT_TOOL_NAMES,
    DECOMPOSE_RULES,
    FACT_CHECK_TOOL,
    JUDGE_SERVER,
    MERGE_RULES,
    Judge,
)
from models import AGENT_MAX_TOKENS, base_env, isolated_options, resolve_model_id
from obsidian_tools import OBSIDIAN_TOOLS, deny_vault_writes, tls_env
from rag_tools import RAG_SERVER, VECTOR_SEARCH_TOOL, rag_server, text_result
from state import Event, TurnState

logger = logging.getLogger(__name__)

ORCHESTRATOR_SERVER = "orchestrator"
CLARIFY_TOOL = f"mcp__{ORCHESTRATOR_SERVER}__request_clarification"

# app/ general_node 가 돌려주는 고정 안내문
GENERAL_REPLY = (
    "IT 개념 질의나 서버·인프라·시스템설계·AI 아키텍처 논의를 "
    "도와드릴 수 있습니다. 무엇이 궁금하신가요?"
)
EMPTY_REPLY = "답변을 생성하지 못했습니다."
CLARIFICATION_PREFIX = "[사용자 추가 정보]"

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

# app/ CLASSIFY_PROMPT 의 의도 정의
INTENT_RULES = """> 의도
- complex_discusion_topic: 서버·인프라·시스템설계·AI 의 트레이드오프 판단이 필요한 논의
  예) 트래픽 급증 시 p99 지연 원인 구분, MSA 통신 방식 선택, 사내 챗봇 전체 아키텍처
- general_it_trend_topic: 단순 개념 질의 또는 최신 트렌드·시세
  예) JWT 와 세션 인증의 차이, 트랜잭션 격리 수준, 현재 맥북 가격
- unknown: 인사말·잡담 등 위에 해당하지 않는 것
"""

ORCHESTRATOR_PROMPT = f"""당신은 IT 질문을 분류해 전문 서브에이전트에게 맡기고, 돌아온 답을 하나로 합치는 오케스트레이터입니다.

{INTENT_RULES}
{AMBIGUITY_RULES}
> 의도별 처리

[complex_discusion_topic]
- 아래 도메인 정의와 규칙대로 질문을 서브 질문으로 분해하고, 서브 질문마다 그 도메인의
  서브에이전트({", ".join(DOMAINS)})를 Agent 도구로 부릅니다. 서브 질문을 prompt 로 넘깁니다.
- 서브 질문이 여러 개면 **한 번의 응답에서 Agent 도구를 여러 번 동시에** 호출해 병렬로 조사시킵니다.
- 도메인을 가를 수 없으면 원 질문을 그대로 system_design 서브에이전트에 맡깁니다.

{DECOMPOSE_RULES}
- 서브에이전트가 1개였다면 그 답은 시스템이 그대로 사용자에게 전달합니다. 병합·검증하지 말고 "전달 완료" 라고만 쓰고 끝냅니다.
- 2개 이상이면 아래 병합 규칙으로 하나의 답을 쓰되, **텍스트로 먼저 쓰지 말고** 그 답 전문을 곧바로 fact_check 의
  answer 로 넘깁니다. 결과가 PASS 면 그 답이 그대로 전달되므로 "전달 완료" 라고만 쓰고, REWRITE 면 지시대로
  한 번 고쳐 쓴 답을 최종 답변으로 씁니다.

{MERGE_RULES}
[general_it_trend_topic]
- {TREND_AGENT} 서브에이전트에 질문을 맡깁니다. 이전 대화 맥락이 필요한 질문이면 그 맥락을 prompt 에 덧붙입니다.
- 돌려받은 답은 시스템이 그대로 사용자에게 전달합니다. 고쳐 쓰지 말고 "전달 완료" 라고만 쓰고 끝냅니다.

[unknown]
- 도구를 쓰지 말고 아래 문장만 그대로 답합니다.
  {GENERAL_REPLY}

> 재질문
- 재질문 기준에 해당하면 조사하지 말고 request_clarification 으로 사용자에게 물을 한 문장을 전달한 뒤,
  다른 말 없이 턴을 끝냅니다.
- 사용자가 "{CLARIFICATION_PREFIX}" 로 답하면 원 질문에 그 정보를 합쳐 이어서 처리합니다.

> 공통
- 직접 조사하지 마세요. 조사는 서브에이전트의 몫입니다.
- 최종 답변에는 분류 결과나 처리 과정(위임, 검증)을 쓰지 말고 답변 본문만 씁니다.
"""

_RESEARCH_TOOLS = {VECTOR_SEARCH_TOOL, *WEB_TOOLS, *OBSIDIAN_TOOLS}


async def research_only_in_subagents(input_data: dict, tool_use_id: str | None, context: Any) -> dict:
    """PreToolUse. 조사 도구는 서브에이전트만 쓴다(app/ 에서 오케스트레이터는 도구가 없다)."""
    if input_data.get("tool_name") in _RESEARCH_TOOLS and not input_data.get("agent_id"):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "조사는 서브에이전트의 몫입니다. Agent 도구로 위임하세요.",
            }
        }
    return {}


def _clarify_tool(turn: TurnState):
    @tool(
        "request_clarification",
        "재질문 기준에 해당할 때만 사용자에게 한 문장으로 되묻는다. 호출한 뒤에는 턴을 끝낸다.",
        {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    )
    async def request_clarification(args: dict[str, Any]) -> dict[str, Any]:
        turn.clarifying_question = args["question"]
        logger.info("재질문: %s", turn.clarifying_question)
        return text_result("질문을 사용자에게 전달했습니다. 사용자의 답을 기다리도록 더 말하지 말고 턴을 끝내세요.")

    return request_clarification


def build_options(turn: TurnState, resume: str | None = None):
    """오케스트레이터 실행 옵션. 턴마다 새로 만든다(훅이 이번 턴의 TurnState 를 본다)."""
    judge = Judge(turn)
    agents = {**build_domain_agents(), TREND_AGENT: build_trend_agent()}

    return isolated_options(
        model=resolve_model_id("orchestrator"),
        system_prompt=ORCHESTRATOR_PROMPT,
        # 내장 도구는 서브에이전트를 부르는 Agent 하나만 연다(파일·셸·웹 도구 없음).
        tools=["Agent"],
        agents=agents,
        mcp_servers={
            RAG_SERVER: rag_server(),
            "web": web_server(),
            JUDGE_SERVER: create_sdk_mcp_server(JUDGE_SERVER, tools=[judge.fact_check_tool()]),
            ORCHESTRATOR_SERVER: create_sdk_mcp_server(ORCHESTRATOR_SERVER, tools=[_clarify_tool(turn)]),
        },
        allowed_tools=[*AGENT_TOOL_NAMES, FACT_CHECK_TOOL, CLARIFY_TOOL, *_RESEARCH_TOOLS],
        # 허용 목록 밖은 묻지 않고 거부한다.
        permission_mode="dontAsk",
        hooks={
            "PreToolUse": [
                HookMatcher(
                    hooks=[deny_vault_writes, research_only_in_subagents, judge.limit_decomposition, judge.limit_fact_check]
                )
            ],
            "PostToolUse": [HookMatcher(matcher="Agent|Task", hooks=[judge.record_domain_answer])],
            "Stop": [HookMatcher(hooks=[judge.require_fact_check])],
        },
        resume=resume,
        env={**base_env(AGENT_MAX_TOKENS), **tls_env()},
    )


def short_tool_name(name: str) -> str:
    """mcp__obsidian__vault_read -> vault_read (app/ 의 도구 이름과 같게 보여준다)."""
    return name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name


async def stream_turn(prompt: str, session_id: str | None = None) -> AsyncIterator[Event]:
    """한 턴을 실행하며 이벤트를 흘려보낸다. 마지막 이벤트는 항상 done 이다."""
    turn = TurnState()
    names: dict[str, str] = {}
    result: ResultMessage | None = None

    async with ClaudeSDKClient(options=build_options(turn, resume=session_id)) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                nested = message.parent_tool_use_id is not None
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        names[block.id] = block.name
                        yield Event(
                            "tool_call",
                            name=short_tool_name(block.name),
                            body=block.input,
                            nested=nested,
                            tool_use_id=block.id,
                            parent_tool_use_id=message.parent_tool_use_id,
                            data={"full_name": block.name},
                        )
                    elif isinstance(block, TextBlock) and not nested and block.text.strip():
                        yield Event("answer", body=block.text)
            elif isinstance(message, UserMessage) and isinstance(message.content, list):
                for block in message.content:
                    if isinstance(block, ToolResultBlock):
                        yield Event(
                            "tool_result",
                            name=short_tool_name(names.get(block.tool_use_id, "")),
                            body=block.content,
                            nested=message.parent_tool_use_id is not None,
                            tool_use_id=block.tool_use_id,
                            parent_tool_use_id=message.parent_tool_use_id,
                            is_error=bool(block.is_error),
                        )
            elif isinstance(message, ResultMessage):
                result = message

    if result is None:
        yield Event("error", body="Claude Code 가 결과 없이 종료되었습니다", is_error=True)
        return
    if result.is_error:
        yield Event("error", body=result.result or result.subtype, is_error=True, data={"session_id": result.session_id})
        return

    if turn.clarifying_question:
        yield Event("clarification", body=turn.clarifying_question)
    answer = final_answer(turn, result.result or "")
    yield Event(
        "done",
        body=answer,
        data={
            "session_id": result.session_id,
            "cost_usd": result.total_cost_usd,
            "num_turns": result.num_turns,
            "usage": result.usage,
            "clarifying_question": turn.clarifying_question,
            "domain_answers": [a.__dict__ for a in turn.domain_answers],
            "fact_checks": turn.fact_checks,
        },
    )


def final_answer(turn: TurnState, last_text: str) -> str:
    """사용자에게 보여줄 답을 고른다. app/ 의 merge / general_it_node / finalize 와 같은 규칙.

      재질문                          -> (UI 가 재질문을 보여준다)
      서브에이전트 답이 1개            -> 그 답 그대로       (merge 의 단일 도메인 통과, general_it_node)
      병합 + 검증 PASS                -> 검증한 병합 답 그대로 (_guard 통과)
      그 외(REWRITE, unknown 안내 등)  -> 오케스트레이터의 마지막 답
      비어 있으면                      -> "답변을 생성하지 못했습니다."  (finalize)
    """
    if turn.clarifying_question:
        return last_text.strip()
    subagent_answers = [a.answer for a in turn.domain_answers] + turn.trend_answers
    if len(subagent_answers) == 1 and not turn.fact_checks:
        answer = subagent_answers[0]
    elif turn.verdict == "PASS" and turn.checked_answer:
        answer = turn.checked_answer
    else:
        answer = last_text
    return answer.strip() or EMPTY_REPLY


def clarification_prompt(answer: str) -> str:
    """재질문에 대한 사용자 답을 다음 턴 입력으로 만든다(app/ decompose_node 와 같은 표기)."""
    return f"{CLARIFICATION_PREFIX} {answer}"
