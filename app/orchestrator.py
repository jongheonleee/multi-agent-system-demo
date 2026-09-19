"""Orchestrator — 질문 유형 판단과 필요 시 재질문.

그래프의 진입 노드. 의도를 분류해 하위 경로를 정하고, 질문이 정말로 모호하면
interrupt 로 그래프를 멈춰 사용자에게 되묻는다(Human in the Loop).

재질문은 사용자를 멈춰 세우는 행위이므로 기준을 엄격히 잡는다. 정보가 조금
부족한 정도는 가정하고 진행하며, 가정한 내용을 답변에 밝히는 쪽이 낫다.
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from models import get_model
from state import MainQueryState

logger = logging.getLogger(__name__)

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


def orchestrate(state: MainQueryState, config: RunnableConfig) -> MainQueryState:
    question = state.messages[-1].text if state.messages else ""

    decision = classify(question)
    state.query_type = decision.intent
    logger.info("의도=%s 이유=%s", decision.intent, decision.reason)

    if decision.needs_clarification and decision.clarifying_question:
        # 그래프를 멈추고 사용자 답을 받는다. checkpointer 가 있어야 동작한다.
        logger.info("재질문: %s", decision.clarifying_question)
        answer = interrupt({"clarifying_question": decision.clarifying_question})
        state.clarification = str(answer)

    return state
