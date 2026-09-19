"""메인 그래프.

orchestrator ─┬─ complex ─▶ decompose ─(Send fan-out)─▶ domain_worker ×N ─▶ merge ─┐
              ├─ general ─▶ general_it_topic_agent ───────────────────────────────┤
              └─ unknown ─▶ general_topic_agent ─────────────────────────────────  ┴─▶ finalize

핵심은 decompose 뒤의 fan-out 이다. 판단 Agent 가 쪼갠 서브질문 하나당
domain_worker 를 Send 로 하나씩 띄워 병렬 실행하고, domain_answers 필드의
operator.add 리듀서가 결과를 모은다. merge 는 모든 분기가 끝나야 실행된다.

Orchestrator(의도 분류 + 재질문)는 이 그래프의 진입 노드이고 다른 곳에서
쓰이지 않으므로 같은 파일에 둔다. 도메인/판단 에이전트는 각자 모듈에 있다.

주의: 모든 노드는 **바뀐 필드만 담은 dict** 를 반환해야 한다. state 객체를
통째로 반환하면 LangGraph 가 모든 필드에 리듀서를 다시 적용해서
domain_answers 가 노드를 지날 때마다 중복 누적된다(3개 -> 12개).
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, filter_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send, interrupt
from pydantic import BaseModel, Field

from domain_agents import run_domain
from general_it_trend_topic_agent import general_it_trend_graph
from judge_agent import decompose, merge
from models import get_model
from state import DomainTaskState, MainQueryState

logger = logging.getLogger(__name__)


def _question_of(state: MainQueryState) -> str:
    return state.messages[-1].text if state.messages else ""


# ===========================================================================
# Orchestrator — 의도 분류와 재질문 (Human in the Loop)
#
# 재질문은 사용자를 멈춰 세우는 행위이므로 기준을 엄격히 잡는다. 정보가 조금
# 부족한 정도는 가정하고 진행하며, 가정한 내용을 답변에 밝히는 쪽이 낫다.
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

CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
    """당신은 IT 질문을 분류하는 오케스트레이터입니다.

> 의도
- complex_discusion_topic: 서버·인프라·시스템설계·AI 의 트레이드오프 판단이 필요한 논의
  예) 트래픽 급증 시 p99 지연 원인 구분, MSA 통신 방식 선택, 사내 챗봇 전체 아키텍처
- general_it_trend_topic: 단순 개념 질의 또는 최신 트렌드·시세
  예) JWT 와 세션 인증의 차이, 트랜잭션 격리 수준, 현재 맥북 가격
- unknown: 인사말·잡담 등 위에 해당하지 않는 것

{ambiguity_rules}

> 질문
{question}
"""
)


class IntentOutput(BaseModel):
    intent: Literal["complex_discusion_topic", "general_it_trend_topic", "unknown"] = Field(
        ..., description="질문 의도"
    )
    needs_clarification: bool = Field(
        default=False,
        description="재질문이 필요한지. 기준을 엄격히 적용해 웬만하면 False 로 둘 것.",
    )
    clarifying_question: str = Field(
        default="", description="재질문이 필요할 때 사용자에게 물을 한 문장"
    )
    reason: str = Field(..., description="판단 이유 (50자 이내)")


def classify(question: str) -> IntentOutput:
    """질문의 의도와 재질문 필요 여부를 판단한다."""
    chain = CLASSIFY_PROMPT | get_model("orchestrator").with_structured_output(IntentOutput)
    try:
        result = chain.invoke({"question": question, "ambiguity_rules": AMBIGUITY_RULES})
        if isinstance(result, IntentOutput):
            return result
        logger.error("분류 결과가 비정상입니다: %r", result)
    except Exception as e:
        logger.error("의도 분류 실패: %s", e)
    return IntentOutput(intent="unknown", reason="분류 실패 폴백")


def orchestrate(state: MainQueryState, config: RunnableConfig) -> dict:
    """의도를 분류하고 필요하면 재질문한다.

    반환은 반드시 **바뀐 필드만 담은 dict** 여야 한다. state 객체를 통째로
    돌려주면 LangGraph 가 모든 필드에 리듀서를 다시 적용해서, operator.add 를
    쓰는 domain_answers 가 노드를 지날 때마다 중복 누적된다.
    """
    question = state.messages[-1].text if state.messages else ""

    decision = classify(question)
    logger.info("의도=%s 이유=%s", decision.intent, decision.reason)

    update: dict = {"query_type": decision.intent}

    if decision.needs_clarification and decision.clarifying_question:
        # 그래프를 멈추고 사용자 답을 받는다. checkpointer 가 있어야 동작한다.
        logger.info("재질문: %s", decision.clarifying_question)
        answer = interrupt({"clarifying_question": decision.clarifying_question})
        update["clarification"] = str(answer)

    return update


# ===========================================================================
# 그래프 노드
# ===========================================================================


def decompose_node(state: MainQueryState, config: RunnableConfig) -> dict:
    question = _question_of(state)
    if state.clarification:
        # 재질문으로 받은 정보를 분해 입력에 합친다.
        question = f"{question}\n\n[사용자 추가 정보] {state.clarification}"

    sub_questions = decompose(question)
    logger.info("서브질문 %d개: %s", len(sub_questions), [s.domain for s in sub_questions])
    return {"sub_questions": sub_questions}


def fan_out_domains(state: MainQueryState):
    """서브질문 1개당 domain_worker 를 하나씩 병렬로 띄운다."""
    if not state.sub_questions:
        return "merge"

    original = _question_of(state)
    return [
        Send("domain_worker", DomainTaskState(sub_question=sq, original_question=original))
        for sq in state.sub_questions
    ]


def merge_node(state: MainQueryState, config: RunnableConfig) -> dict:
    merged = merge(_question_of(state), state.domain_answers)
    return {"merged_answer": merged, "messages": [AIMessage(content=merged)]}


def general_it_node(state: MainQueryState, config: RunnableConfig) -> dict:
    response = general_it_trend_graph.invoke(
        input={"messages": state.messages},
        config=config,
        output_keys=["messages"],
    )
    ai = filter_messages(response.get("messages", []), include_types=[AIMessage])
    return {"messages": [ai[-1]]} if ai else {}


def general_node(state: MainQueryState, config: RunnableConfig) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "IT 개념 질의나 서버·인프라·시스템설계·AI 아키텍처 논의를 "
                    "도와드릴 수 있습니다. 무엇이 궁금하신가요?"
                )
            )
        ]
    }


def finalize_node(state: MainQueryState, config: RunnableConfig) -> dict:
    if not state.messages or not state.messages[-1].text:
        return {"messages": [AIMessage(content="답변을 생성하지 못했습니다.")]}
    return {}


def route_by_intent(state: MainQueryState) -> str:
    if state.query_type == "complex_discusion_topic":
        return "decompose"
    if state.query_type == "general_it_trend_topic":
        return "general_it_topic_agent"
    return "general_topic_agent"


def build_main_graph(checkpointer=None):
    b = StateGraph(MainQueryState)

    b.add_node("orchestrator", orchestrate, retry_policy=RetryPolicy(max_interval=3))
    b.add_node("decompose", decompose_node)
    b.add_node("domain_worker", run_domain, input_schema=DomainTaskState)
    b.add_node("merge", merge_node)
    b.add_node("general_it_topic_agent", general_it_node)
    b.add_node("general_topic_agent", general_node)
    b.add_node("finalize", finalize_node)

    b.add_edge(START, "orchestrator")
    b.add_conditional_edges(
        "orchestrator",
        route_by_intent,
        ["decompose", "general_it_topic_agent", "general_topic_agent"],
    )
    b.add_conditional_edges("decompose", fan_out_domains, ["domain_worker", "merge"])
    b.add_edge("domain_worker", "merge")
    b.add_edge("merge", "finalize")
    b.add_edge("general_it_topic_agent", "finalize")
    b.add_edge("general_topic_agent", "finalize")
    b.add_edge("finalize", END)

    return b.compile(name="main_agent", checkpointer=checkpointer)


main_graph = build_main_graph()
