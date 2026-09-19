import operator

from pydantic import BaseModel, Field
from typing import Annotated, Literal 

from langchain_core.messages import AnyMessage
from langchain_core.documents import Document
from langgraph.graph import add_messages

class BaseState(BaseModel):

    # 이 부분은 상태 전이 과정에서 대화 메시지를 누적한 내용은 아닌가? 
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory = list ,
        description = "대화 메시지 누적, add_message 리듀서로 병합"
    )

Domain = Literal["server_infra", "system_design", "ai"]


class SubQuestion(BaseModel):
    """판단 Agent 가 원 질문에서 분해해낸 도메인별 서브 질문."""

    domain: Domain = Field(..., description="이 서브 질문을 담당할 도메인")
    question: str = Field(..., description="해당 도메인에 던질 독립적인 질문")
    reason: str = Field(..., description="이 도메인으로 분류한 이유 (50자 이내)")


class DomainAnswer(BaseModel):
    """도메인 에이전트 한 개가 낸 답변."""

    domain: Domain
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)


class DomainTaskState(BaseModel):
    """Send 로 병렬 실행되는 도메인 노드 하나가 받는 입력."""

    sub_question: SubQuestion
    original_question: str = ""


class MainQueryState(BaseState):
    query_type: Literal["general_it_trend_topic", "complex_discusion_topic", "unknown"] | None = Field(
        default = None,
        description = "사용자 질문의 의도(Intent) 분류 결과",
    )

    sub_questions: list[SubQuestion] = Field(
        default_factory = list,
        description = "판단 Agent 가 분해한 서브 질문 목록",
    )

    # 병렬 도메인 노드들이 동시에 쓴다. 리듀서 없으면 InvalidUpdateError.
    domain_answers: Annotated[list[DomainAnswer], operator.add] = Field(
        default_factory = list,
        description = "도메인 에이전트들이 낸 답변 누적",
    )

    merged_answer: str = Field(
        default = "",
        description = "판단 Agent 가 병합한 최종 답변",
    )

    clarification: str = Field(
        default = "",
        description = "재질문(Human in the Loop)에 대한 사용자 답변",
    )

class GeneralITTopicState(BaseState):

    """
    - 단순 질문 상태 정의 
    - 웹 서칭 기반 일반 IT 트렌드 및 간단한 개념 상태 
    """

    # 사용자 질문 내용 
    question: str = Field(
        default = "",
        description = "사용자 원본 질문",
    )

    # 웹 서칭에 사용된 질의 목록 
    search_queries: list[str] = Field(
        default_factory = list,
        description = "웹 서칭에 사용할 질의 목록",
    )

    # 검색 결과 목록 
    search_results: list[dict] = Field(
        default_factory = list,
        description = "Serper 검색 결과 원본 (title, link, snippet)",
    )

    # ?? 
    loaded_documents: list[Document] = Field(
        default_factory = list,
        description = "WebBaseLoader로 본문을 적재한 문서",
    )

    # 리랭커를 통해 우선순위 별로 재정렬한 문서 
    reranked_documents: list[Document] = Field(
        default_factory = list,
        description = "리랭커가 재정렬한 상위 문서",
    )

    # 답변 내용 
    answer: str = Field(
        default = "",
        description = "생성된 답변",
    )

    # 웹 서칭에서 참고한 URL 출처 
    sources: list[str] = Field(
        default_factory = list,
        description = "인용할 출처 URL 목록",
    )

class ComplexTopicState(BaseState):
    """
    - 벡터데이터베이스 검색 기반 RAG 상태
    - 옵시디언 검색 기반 결과 상태
    """

    question: str = Field(
        default = "",
        description = "사용자 원본 질문",
    )

    queries: list[str] = Field(
        default_factory = list,
        description = "멀티 쿼리로 확장된 검색 질의 목록",       
    )

    documents: list[Document] = Field(
        default_factory = list,
        description = "벡터 검색으로 회수한 문서",
    )

    reranked_documents: list[Document] = Field(
        default_factory = list,
        description = "리랭커가 재정렬한 상위 문서",  
    )

    ### 추가로 옵시디언 조회 결과 담을 필드 추가해놓기 

    answer: str = Field(
        default = "",
        description = "생성된 답변",
    )

    fact_check_score: float = Field(
        default = 0.0,
        ge = 0.0,
        le = 1.0,
        description = "팩트체크 종합 정확도 (0.0 ~ 1.0)",
    )

    fact_check_comment: str = Field(
        default = "",
        description = "팩트체크 요약 코멘트",
    )

    retry_count: int = Field(
        default = 0,
        description = "답변 재작성 횟수",
    )


