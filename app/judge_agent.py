"""판단 Agent — 질문 분해와 결과 병합.

원 질문을 도메인별 서브 질문으로 쪼개 병렬 에이전트에 나눠주고(decompose),
돌아온 답변들을 하나의 글로 합친다(merge).
"""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from models import get_model
from state import DomainAnswer, SubQuestion

logger = logging.getLogger(__name__)


class DecomposeOutput(BaseModel):
    sub_questions: list[SubQuestion] = Field(
        ...,
        description="도메인별로 분해한 서브 질문. 단일 도메인 질문이면 1개만.",
    )


DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """당신은 IT 질문을 도메인별로 분해하는 판단 에이전트입니다.

> 도메인 정의 (서로 배타적으로 적용할 것)
- server_infra: **운영 중인 시스템을 돌리는 층**.
  배포, 쿠버네티스/컨테이너, 네트워크, DB 운영·튜닝, 모니터링, 장애 진단, 용량 산정
- system_design: **코드와 데이터의 구조를 정하는 층**.
  서비스 경계, 모듈 분리, API 계약, 데이터 모델링, 일관성/확장성 트레이드오프
- ai: **모델과 그 주변**.
  LLM, RAG, 에이전트, 모델 서빙, 임베딩, 벡터 검색, 프롬프트

경계가 애매하면 이렇게 가르세요.
"지금 돌아가는 걸 고치는가" -> server_infra
"무엇을 만들지 정하는가"   -> system_design

> 규칙
- 1. 질문에 실제로 포함된 도메인만 만듭니다. 억지로 3개를 채우지 마세요.
- 2. 단일 도메인 질문이면 서브 질문 1개만 반환합니다.
- 3. 각 서브 질문은 다른 도메인의 답을 몰라도 **독립적으로 답할 수 있어야** 합니다.
     "위 결과를 바탕으로" 같은 의존 표현을 쓰지 마세요.
- 4. 서브 질문은 최대 3개입니다.
- 5. 원 질문의 제약 조건(규모, 트래픽, 기술 스택)은 각 서브 질문에 그대로 복사해 넣습니다.

> 원 질문
{question}
"""
)


def decompose(question: str) -> list[SubQuestion]:
    """원 질문을 도메인별 서브 질문으로 분해한다."""
    chain = DECOMPOSE_PROMPT | get_model("judge").with_structured_output(DecomposeOutput)
    try:
        result = chain.invoke({"question": question})
        if isinstance(result, DecomposeOutput) and result.sub_questions:
            return result.sub_questions[:3]
        logger.error("분해 결과가 비정상입니다: %r", result)
    except Exception as e:
        logger.error("질문 분해 실패: %s", e)

    # 분해 실패 시 원 질문을 system_design 단일 서브질문으로 떨군다.
    return [SubQuestion(domain="system_design", question=question, reason="분해 실패 폴백")]


MERGE_PROMPT = ChatPromptTemplate.from_template(
    """당신은 도메인 전문가들의 답변을 하나로 합치는 판단 에이전트입니다.

> 원 질문
{question}

> 도메인별 답변
{answers}

> 병합 규칙
- 1. 원 질문에 답하는 **하나의 글**로 씁니다. 도메인별로 단순 나열하지 마세요.
- 2. 여러 도메인이 같은 말을 하면 한 번만 씁니다.
- 3. 도메인 간 권고가 **상충하면** 숨기지 말고 드러낸 뒤, 어떤 조건에서 무엇을 택할지 씁니다.
- 4. 전문가가 "찾지 못했다"고 한 부분은 그대로 한계로 밝힙니다. 지어내지 마세요.
- 5. 전문가 답변에 없는 내용을 새로 추가하지 마세요.
- 6. 각 전문가의 Sources 를 모아 말미에 "Sources" 섹션으로 통합합니다.
- 7. 한국어로 씁니다.
"""
)

_DOMAIN_LABELS = {
    "server_infra": "서버·인프라",
    "system_design": "시스템 설계",
    "ai": "AI",
}


def _format_answers(answers: list[DomainAnswer]) -> str:
    return "\n\n".join(
        f"### [{_DOMAIN_LABELS.get(a.domain, a.domain)}] {a.question}\n{a.answer}"
        for a in answers
    )


def merge(question: str, answers: list[DomainAnswer]) -> str:
    """도메인 답변들을 하나의 답변으로 합친다."""
    if not answers:
        return "도메인 에이전트가 답변을 생성하지 못했습니다."

    # 단일 도메인이면 합칠 것이 없다. LLM 호출을 아낀다.
    if len(answers) == 1:
        return answers[0].answer

    chain = MERGE_PROMPT | get_model("judge")
    try:
        merged = chain.invoke({"question": question, "answers": _format_answers(answers)}).text
    except Exception as e:
        logger.error("병합 실패, 원본을 이어붙입니다: %s", e)
        return _format_answers(answers)

    return _guard(merged, answers)


GUARDRAIL_THRESHOLD = 0.8

FIX_PROMPT = ChatPromptTemplate.from_template(
    """아래 답변에서 출처가 뒷받침하지 않는 문장을 제거하거나 완화해 다시 쓰세요.
내용을 새로 추가하지 말고, 근거가 약한 부분만 손보세요.

> 검증 결과
{report}

> 답변
{merged}
"""
)


def _guard(merged: str, answers: list[DomainAnswer]) -> str:
    """병합 답변이 도메인 근거를 벗어나지 않았는지 1회 검증한다.

    기준 미달이면 1회만 재작성한다. 재작성을 반복하면 비용만 늘고 품질은
    수렴하지 않는다.
    """
    from rag_tools import VERIFY_FAILED_COMMENT_PREFIX, fact_check_tool

    context = "\n\n".join(a.answer for a in answers)
    try:
        report = fact_check_tool.invoke({"text": merged, "context": context})
    except Exception as e:
        logger.warning("guardrail 검증 실패, 원본을 반환합니다: %s", e)
        return merged

    comment = report.get("overall_accuracy_comment", "")
    if comment.startswith(VERIFY_FAILED_COMMENT_PREFIX):
        # 검증 자체가 안 된 것이지 답변이 틀린 게 아니다. 재작성하면 안 된다.
        logger.warning("guardrail 검증 불가, 원본을 유지합니다: %s", comment)
        return merged

    score = report.get("overall_accuracy", 1.0)
    if score >= GUARDRAIL_THRESHOLD:
        logger.info("guardrail 통과 (%.2f)", score)
        return merged

    logger.info("guardrail 미달(%.2f), 1회 재작성합니다", score)
    try:
        return (
            (FIX_PROMPT | get_model("judge"))
            .invoke(
                {
                    "report": report.get("overall_accuracy_comment", ""),
                    "merged": merged,
                }
            )
            .text
        )
    except Exception as e:
        logger.warning("재작성 실패, 원본을 반환합니다: %s", e)
        return merged
