"""
서브 에이전트가 사용하는 커스텀 도구.

Claude Agent SDK 의 @tool 은 in-process MCP 서버로 노출된다.
Claude 에게 보이는 이름은 mcp__<서버명>__<도구명> 형태가 된다.

도구 핸들러는 async 여야 하므로, 동기 라이브러리(Pinecone/langchain/requests)
호출은 asyncio.to_thread 로 감싸 이벤트 루프를 막지 않는다.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Any

from claude_agent_sdk import create_sdk_mcp_server, tool

logger = logging.getLogger(__name__)

# app/ 의 CohereRerank 기본값(top_n=3)과 동일하게 맞춘다.
RERANK_TOP_N = 3


def _text(payload: Any, is_error: bool = False) -> dict[str, Any]:
    """도구 반환 형식 헬퍼. Agent SDK 는 {"content": [...]} 를 기대한다."""
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
    result: dict[str, Any] = {"content": [{"type": "text", "text": body}]}
    if is_error:
        result["is_error"] = True
    return result


# ────────────────────────────── 공통 모델 ──────────────────────────────
def _embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)


def _reranker():
    from langchain_cohere import CohereRerank

    return CohereRerank(model="rerank-multilingual-v3.0", top_n=RERANK_TOP_N)


# ───────────────────── 복잡한 논의 주제: Pinecone RAG ─────────────────────
def _vector_search_sync(query: str) -> list[dict[str, Any]]:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

    host = os.getenv("PINECONE_HOST")
    if not host:
        raise RuntimeError("PINECONE_HOST 환경변수가 설정되지 않았습니다")

    # host 를 위치 인자로 주면 인덱스 '이름' 으로 해석되어 404 가 난다. 반드시 host= 로 넘길 것.
    index = Pinecone().Index(host=host)
    store = PineconeVectorStore(index=index, embedding=_embeddings())
    retriever = store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.7},
    )

    docs = retriever.invoke(query)
    if not docs:
        return []

    reranked = _reranker().compress_documents(documents=docs, query=query)
    return [
        {
            "index": i + 1,
            "content": d.page_content,
            "source": d.metadata.get("source") or d.metadata.get("file_path") or "",
            "page": d.metadata.get("page"),
        }
        for i, d in enumerate(reranked)
    ]


@tool(
    "vector_search",
    "Pinecone 벡터 DB에서 논문 문서를 검색하고 Cohere 리랭커로 재정렬해 반환한다. "
    "질문을 여러 관점으로 바꿔가며 2~3회 호출하면 회수율이 올라간다.",
    {"query": Annotated[str, "검색할 질의. 한 번에 하나씩."]},
)
async def vector_search(args: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    try:
        docs = await asyncio.to_thread(_vector_search_sync, args["query"])
    except Exception as e:
        logger.exception("vector_search 실패")
        return _text(f"검색 실패: {e}", is_error=True)

    if not docs:
        return _text("검색 결과가 없습니다. 질의를 바꿔서 다시 시도하세요.")
    return _text(docs)


def _fact_check_sync(answer: str, context: str) -> dict[str, Any]:
    import anthropic

    schema = {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence": {"type": "string"},
                        "verdict": {
                            "type": "string",
                            "enum": ["accurate", "partially accurate", "inaccurate", "unverifiable"],
                        },
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["sentence", "verdict", "score", "reason"],
                    "additionalProperties": False,
                },
            },
            "overall_accuracy": {"type": "number"},
            "overall_accuracy_comment": {"type": "string"},
        },
        "required": ["sentences", "overall_accuracy", "overall_accuracy_comment"],
        "additionalProperties": False,
    }

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=os.getenv("FACT_CHECK_MODEL", "claude-opus-5"),
        max_tokens=16000,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        system=(
            "당신은 꼼꼼한 팩트 체커입니다. 제공된 출처 문서만을 근거로 답변의 사실적 정확성을 "
            "문장 단위로 검증하세요. 출처에 없는 내용은 unverifiable 로 판정합니다."
        ),
        messages=[
            {
                "role": "user",
                "content": f"**검증할 답변**\n{answer}\n\n**출처 문서**\n{context}",
            }
        ],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return json.loads(text)


@tool(
    "fact_check",
    "작성한 답변을 검색된 출처 문서와 대조해 문장 단위로 사실 검증한다. "
    "최종 답변을 확정하기 전에 반드시 한 번 호출한다.",
    {
        "answer": Annotated[str, "검증할 답변 전문"],
        "context": Annotated[str, "vector_search 로 회수한 출처 문서 본문을 이어붙인 문자열"],
    },
)
async def fact_check(args: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    try:
        result = await asyncio.to_thread(_fact_check_sync, args["answer"], args["context"])
    except Exception as e:
        logger.exception("fact_check 실패")
        return _text(f"팩트체크 실패: {e}", is_error=True)
    return _text(result)


# ───────────────────── 일반 IT 트렌드: 웹 검색 ─────────────────────
def _web_search_sync(query: str, num_results: int) -> list[dict[str, str]]:
    from langchain_community.utilities import GoogleSerperAPIWrapper

    search = GoogleSerperAPIWrapper(k=num_results, gl="kr", hl="ko")
    organic = search.results(query).get("organic", [])
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", ""),
        }
        for r in organic
    ]


@tool(
    "web_search",
    "Serper(Google) 로 웹을 검색해 제목·URL·스니펫을 반환한다. "
    "질문을 여러 각도로 바꿔가며 2~3회 호출하면 좋다.",
    {
        "query": Annotated[str, "검색어"],
        "num_results": Annotated[int, "가져올 결과 수 (기본 2, app/ 과 동일)"],
    },
)
async def web_search(args: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    try:
        results = await asyncio.to_thread(
            _web_search_sync, args["query"], int(args.get("num_results") or 2)
        )
    except Exception as e:
        logger.exception("web_search 실패")
        return _text(f"웹 검색 실패: {e}", is_error=True)

    if not results:
        return _text("검색 결과가 없습니다.")
    return _text(results)


def _fetch_webpage_sync(urls: list[str]) -> list[dict[str, str]]:
    from langchain_community.document_loaders import WebBaseLoader

    docs = []
    for doc in WebBaseLoader(urls).lazy_load():
        body = " ".join(doc.page_content.split())
        docs.append(
            {
                "url": doc.metadata.get("source", ""),
                "title": doc.metadata.get("title", ""),
                # 본문은 에이전트가 직접 요약하므로 과도한 토큰만 잘라낸다.
                "content": body[:4000],
            }
        )
    return docs


@tool(
    "fetch_webpage",
    "web_search 로 찾은 URL의 본문을 가져온다. 스니펫만으로 답하기 부족할 때 사용한다.",
    {"urls": Annotated[list[str], "본문을 가져올 URL 목록 (최대 3개 권장)"]},
)
async def fetch_webpage(args: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    urls = args.get("urls") or []
    if not urls:
        return _text("urls 가 비어 있습니다.", is_error=True)

    try:
        docs = await asyncio.to_thread(_fetch_webpage_sync, list(urls)[:3])
    except Exception as e:
        logger.exception("fetch_webpage 실패")
        return _text(f"본문 수집 실패: {e}", is_error=True)
    return _text(docs)


# ────────────────────────────── MCP 서버 ──────────────────────────────
# 서버명이 도구 이름의 접두사가 된다: mcp__research__vector_search 등
research_server = create_sdk_mcp_server(
    name="research",
    version="1.0.0",
    tools=[vector_search, fact_check, web_search, fetch_webpage],
)

TOOL_VECTOR_SEARCH = "mcp__research__vector_search"
TOOL_FACT_CHECK = "mcp__research__fact_check"
TOOL_WEB_SEARCH = "mcp__research__web_search"
TOOL_FETCH_WEBPAGE = "mcp__research__fetch_webpage"
