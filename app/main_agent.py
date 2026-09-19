"""메인 그래프.

orchestrator ─┬─ complex ─▶ decompose ─(Send fan-out)─▶ domain_worker ×N ─▶ merge ─┐
              ├─ general ─▶ general_it_topic_agent ───────────────────────────────┤
              └─ unknown ─▶ general_topic_agent ─────────────────────────────────  ┴─▶ finalize

핵심은 decompose 뒤의 fan-out 이다. 판단 Agent 가 쪼갠 서브질문 하나당
domain_worker 를 Send 로 하나씩 띄워 병렬 실행하고, domain_answers 필드의
operator.add 리듀서가 결과를 모은다. merge 는 모든 분기가 끝나야 실행된다.

주의: 모든 노드는 **바뀐 필드만 담은 dict** 를 반환해야 한다. state 객체를
통째로 반환하면 LangGraph 가 모든 필드에 리듀서를 다시 적용해서
domain_answers 가 노드를 지날 때마다 중복 누적된다(3개 -> 12개).
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from domain_agents import run_domain
from general_it_trend_topic_agent import general_it_trend_graph
from judge_agent import decompose, merge
from orchestrator import orchestrate
from state import DomainTaskState, MainQueryState

logger = logging.getLogger(__name__)


def _question_of(state: MainQueryState) -> str:
    return state.messages[-1].text if state.messages else ""


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
