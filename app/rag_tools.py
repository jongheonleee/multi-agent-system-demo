"""

> - Pinecone 논문 벡터 검색과 팩트체크
> - 여러 에이전트들이 사용(공유)

"""
from __future__ import annotations

import logging
import os

from typing import Literal, Sequence
from pinecone import Pinecone
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore

from models import get_model, get_embeddings, get_original_reranker

logger = logging.getLogger(__name__)


# 검증 자체가 불가능했음을 나타내는 플래그. 
# - 모델 응답이 잘렸거나 출처가 없을 때 사용 
VERIFY_FAILED_COMMENT_PREFIX = "[검증불가]"

# 팩트 체크용 모델이 사용할 프롬프트 
FACT_CHECK_PROMPT = ChatPromptTemplate.from_template(
    """> 
### 역할: 
- 1. 당신은 꼼꼼한 팩트 체커 AI입니다.
- 2. 제공된 출처 문서만을 근거로 주어진 답변의 사실인지 검증합니다.

### 검증할 답변:
- {answer}

### 출처 문서:
- {context}

### 지시사항: 
- 1. 답변을 개별 사실 주장 또는 문장 단위로 분해할 것 
- 2. 각 문장을 출처 문서와 대조해 다음 중 하나로 판정합니다.
     - `accurate`: 출처 문서가 직접적이고 완전하게 뒷받침함
     - `partially accurate`: 부분적으로 뒷받침되거나 뉘앙스 차이·누락이 있음
     - `inaccurate`: 출처 문서와 모순됨
     - `unverifiable`: 제공된 출처만으로는 검증할 수 없음
- 3. 각 문장에 신뢰도 점수(0.0~1.0)를 매깁니다.
- 4. accurate / partially accurate 이면 뒷받침하는 출처를 명시하고,unverifiable / inaccurate 이면 그 이유를 서술합니다.
- 5. 전체 정확도 점수(문장별 평균)와 요약 코멘트를 작성합니다.
"""
)


# 한 문장에 대한 팩트 체크용 모델(DTO)
class SentenceFactCheckResult(BaseModel):
    sentence: str = Field(
        ..., 
        description="사실 확인 중인 문장"
    )

    verdict: Literal["accurate", "partially accurate", "inaccurate", "unverifiable"] = Field(
        ..., 
        description="사실 확인 판정"
    )

    source: str | None = Field(
        None,
        description="해당 문장을 뒷받침하거나 반박하는 출처"
    )

    score: float = Field(
        ..., 
        description="진술의 신뢰도 점수 (0.0 ~ 1.0)"
    )

    reason: str | None = Field(
        None, 
        description="정확하지 않은 경우 그 판단의 이유"
    )

# 여러 문장에 대한 팩트 체크용 모델(DTO)
class FactCheckResult(BaseModel):
    sentences: list[SentenceFactCheckResult] = Field(
        ..., 
        description="각 문장에 대한 사실 확인 결과"
    )

    overall_accuracy: float = Field(
        ..., 
        description="전체 정확도 점수 (0.0 ~ 1.0)"
    )

    overall_accuracy_comment: str = Field(
        ..., 
        description="전체 정확도에 대한 설명"
    )


def get_doc_compress_reranker(docs: list, query: str) -> Sequence[Document]:

    """문서를 질문과의 관련도 순으로 재정렬한다."""

    return get_original_reranker().compress_documents(documents=docs, query=query)


@tool(
    description=(
        "Pinecone VectorDB에서 관련 논문 문서를 검색하고 리랭킹해 반환한다. "
        "질의를 바꿔가며 2~3회 호출하면 회수율이 올라간다."
    )
)
def vector_search_tool(query: str) -> Sequence[Document]:
    try:
        host = os.getenv("PINECONE_HOST")
        if not host:
            logger.error("PINECONE_HOST 환경변수가 설정되지 않았습니다")
            return []

        # host 를 위치 인자로 주면 인덱스 '이름' 으로 해석되어 404 가 난다.
        # 반드시 host= 키워드로 넘길 것.
        index = Pinecone().Index(host=host)
        vector_store = PineconeVectorStore(index=index, embedding=get_embeddings())

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.7},
        )
        docs = retriever.invoke(query)
        if not docs:
            return []

        cleaned = [
            Document(
                page_content=d.page_content,
                metadata={k: v for k, v in d.metadata.items() if k not in ("score", "file_path")},
            )
            for d in docs
        ]
        return get_doc_compress_reranker(cleaned, query)

    except Exception as e:
        logger.error("벡터 검색 실패: %s", e)
        return []



# 검증 자체가 불가능했음을 나타내는 플래그. 모델 응답이 잘렸거나 출처가 없을 때다.
# 이걸 정확도 0.0 과 구분하지 않으면, 검증 실패가 "부정확한 답변"으로 오인되어
# 멀쩡한 답변을 재작성시킨다.



@tool(description="생성된 텍스트를 출처 문서와 대조해 문장 단위로 사실 검증한다")
def fact_check_tool(text: str, context: str | None = None) -> dict:

    """주어진 텍스트의 사실 여부를 출처 문서 기반으로 확인한다.

    context 가 없으면 검증할 근거가 없다는 뜻이다. 예전에는 여기서 질문과
    무관한 Gemini 논문 스니펫을 하드코딩해 대신 썼는데, 그러면 엉뚱한 문서로
    검증이 통과해버려 guardrail 이 무의미해진다. 이제는 검증 불가로 반환한다.
    """

    if not context:
        return FactCheckResult(
            sentences=[],
            overall_accuracy=0.0,
            overall_accuracy_comment=f"{VERIFY_FAILED_COMMENT_PREFIX} 출처 문서가 제공되지 않았습니다.",
        ).model_dump()

    try:

        model = get_model("judge", max_tokens=16000)
        chain = FACT_CHECK_PROMPT | model.with_structured_output(FactCheckResult)
        result = chain.invoke({"answer": text, "context": context})
        return result.model_dump()

    except Exception as e:
        logger.error("팩트체크 실패: %s", e)
        return FactCheckResult(
            sentences=[
                SentenceFactCheckResult(
                    sentence=text[:200],
                    source=None,
                    verdict="unverifiable",
                    score=0.0,
                    reason=f"오류 발생: {e!s}",
                )
            ],
            overall_accuracy=0.0,
            overall_accuracy_comment=f"{VERIFY_FAILED_COMMENT_PREFIX} 사실 확인 중 오류: {e!s}",
        ).model_dump()
