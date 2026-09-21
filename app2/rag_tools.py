"""공용 RAG 도구 — Agent SDK 커스텀 도구(in-process MCP 서버).

  vector_search_tool -> ai 서브에이전트가 부른다
  fact_check         -> 판단 guardrail 이 쓴다 (judge_agent.py)

app/rag_tools.py 는 LangChain(PineconeVectorStore, CohereRerank)으로 감싸 쓰지만,
여기서는 각 서비스의 공식 클라이언트를 직접 쓴다. 검색 파라미터는 app/ 과 같다.
  - 임베딩: OpenAI text-embedding-3-small, 512 차원 (색인과 같아야 한다)
  - Pinecone top_k=10, 관련도 (cosine+1)/2 >= 0.7   (LangChain 의 similarity_score_threshold)
  - Cohere rerank-multilingual-v3.0, 상위 3개        (CohereRerank 기본 top_n)

Agent SDK 에서 도구는 @tool 로 정의한 async 함수이고, create_sdk_mcp_server 로 묶으면
Claude Code 안에서 mcp__rag__vector_search_tool 로 보인다. 프로세스를 따로 띄우지
않는 in-process MCP 서버라서 파이썬 객체·환경변수를 그대로 쓴다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Literal

from claude_agent_sdk import create_sdk_mcp_server, tool
from pydantic import BaseModel, Field

from models import LLMCallError, ask_structured

logger = logging.getLogger(__name__)

RAG_SERVER = "rag"
VECTOR_SEARCH_TOOL = f"mcp__{RAG_SERVER}__vector_search_tool"

_TOP_K = 10
_SCORE_THRESHOLD = 0.7
_RERANK_TOP_N = 3


def text_result(payload: Any, is_error: bool = False) -> dict[str, Any]:
    """Agent SDK 도구 반환 형식. 문자열이 아니면 JSON 으로 직렬화한다."""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
    result: dict[str, Any] = {"content": [{"type": "text", "text": body}]}
    if is_error:
        result["is_error"] = True
    return result


# --------------------------------------------------------------------------
# 벡터 검색
# --------------------------------------------------------------------------
def _embed(query: str) -> list[float]:
    from openai import OpenAI

    return OpenAI().embeddings.create(model="text-embedding-3-small", input=query, dimensions=512).data[0].embedding


def _rerank(query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import cohere

    client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    res = client.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=[d["page_content"] for d in docs],
        top_n=_RERANK_TOP_N,
    )
    return [
        {**docs[r.index], "metadata": {**docs[r.index]["metadata"], "relevance_score": r.relevance_score}}
        for r in res.results
    ]


def search_papers(query: str) -> list[dict[str, Any]]:
    """Pinecone 에서 논문 청크를 찾아 리랭킹한다. 실패하면 빈 리스트(app/ 과 같다)."""
    try:
        from pinecone import Pinecone

        host = os.getenv("PINECONE_HOST")
        if not host:
            logger.error("PINECONE_HOST 환경변수가 설정되지 않았습니다")
            return []

        # host 를 위치 인자로 주면 인덱스 '이름' 으로 해석되어 404 가 난다. 반드시 host= 로.
        index = Pinecone().Index(host=host)
        matches = index.query(vector=_embed(query), top_k=_TOP_K, include_metadata=True).matches

        docs = []
        for m in matches:
            # Pinecone 코사인 유사도 [-1, 1] 을 [0, 1] 관련도로 바꿔 임계치를 적용한다.
            if (m.score + 1) / 2 < _SCORE_THRESHOLD:
                continue
            metadata = dict(m.metadata or {})
            text = metadata.pop("text", None)
            if text is None:
                continue
            metadata = {k: v for k, v in metadata.items() if k not in ("score", "file_path")}
            docs.append({"page_content": text, "metadata": metadata})
        if not docs:
            return []
        return _rerank(query, docs)

    except Exception as e:
        logger.error("벡터 검색 실패: %s", e)
        return []


@tool(
    "vector_search_tool",
    (
        "Pinecone VectorDB에서 관련 논문 문서를 검색하고 리랭킹해 반환한다. "
        "질의를 바꿔가며 2~3회 호출하면 회수율이 올라간다."
    ),
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
async def vector_search_tool(args: dict[str, Any]) -> dict[str, Any]:
    # Pinecone/OpenAI/Cohere 는 동기 클라이언트라 스레드로 보내 이벤트 루프를 막지 않는다.
    return text_result(await asyncio.to_thread(search_papers, args["query"]))


def rag_server():
    return create_sdk_mcp_server(name=RAG_SERVER, version="1.0.0", tools=[vector_search_tool])


# --------------------------------------------------------------------------
# 팩트체크
# --------------------------------------------------------------------------
class SentenceFactCheckResult(BaseModel):
    sentence: str = Field(..., description="사실 확인 중인 문장")
    verdict: Literal["accurate", "partially accurate", "inaccurate", "unverifiable"] = Field(
        ..., description="사실 확인 판정"
    )
    source: str | None = Field(None, description="해당 문장을 뒷받침하거나 반박하는 출처")
    score: float = Field(..., description="진술의 신뢰도 점수 (0.0 ~ 1.0)")
    reason: str | None = Field(None, description="정확하지 않은 경우 그 판단의 이유")


class FactCheckResult(BaseModel):
    sentences: list[SentenceFactCheckResult] = Field(
        ..., description="각 문장에 대한 사실 확인 결과"
    )
    overall_accuracy: float = Field(..., description="전체 정확도 점수 (0.0 ~ 1.0)")
    overall_accuracy_comment: str = Field(..., description="전체 정확도에 대한 설명")


# 검증 자체가 불가능했음을 나타내는 플래그. 모델 응답이 잘렸거나 출처가 없을 때다.
# 이걸 정확도 0.0 과 구분하지 않으면, 검증 실패가 "부정확한 답변"으로 오인되어
# 멀쩡한 답변을 재작성시킨다.
VERIFY_FAILED_COMMENT_PREFIX = "[검증불가]"


FACT_CHECK_PROMPT = """> 역할
- 1. 당신은 꼼꼼한 팩트 체커 AI입니다.
- 2. 제공된 출처 문서만을 근거로 주어진 답변의 사실적 정확성을 검증합니다.

**검증할 답변:**
{answer}

**출처 문서:**
{context}

**지시사항**
- 1. 답변을 개별 사실 주장 또는 문장 단위로 분해합니다.
- 2. 각 문장을 출처 문서와 대조해 다음 중 하나로 판정합니다.
     - `accurate`: 출처 문서가 직접적이고 완전하게 뒷받침함
     - `partially accurate`: 부분적으로 뒷받침되거나 뉘앙스 차이·누락이 있음
     - `inaccurate`: 출처 문서와 모순됨
     - `unverifiable`: 제공된 출처만으로는 검증할 수 없음
- 3. 각 문장에 신뢰도 점수(0.0~1.0)를 매깁니다.
- 4. accurate / partially accurate 이면 뒷받침하는 출처를 명시하고,
     unverifiable / inaccurate 이면 그 이유를 서술합니다.
- 5. 전체 정확도 점수(문장별 평균)와 요약 코멘트를 작성합니다.
"""


async def fact_check(text: str, context: str | None = None) -> dict:
    """주어진 텍스트의 사실 여부를 출처 문서 기반으로 확인한다.

    context 가 없으면 검증할 근거가 없다는 뜻이므로 검증 불가로 반환한다.
    """
    if not context:
        return FactCheckResult(
            sentences=[],
            overall_accuracy=0.0,
            overall_accuracy_comment=f"{VERIFY_FAILED_COMMENT_PREFIX} 출처 문서가 제공되지 않았습니다.",
        ).model_dump()

    try:
        # 문장마다 판정을 내므로 출력이 길다. 기본 4096 이면 잘려서 파싱이 실패한다.
        result = await ask_structured(
            "judge",
            FACT_CHECK_PROMPT.format(answer=text, context=context),
            FactCheckResult,
            max_tokens=16000,
        )
        return result.model_dump()

    except (LLMCallError, Exception) as e:
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
