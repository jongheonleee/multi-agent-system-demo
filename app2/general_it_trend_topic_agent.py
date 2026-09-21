"""웹 검색 기반 IT 트렌드 서브에이전트.

app/ 은 도구 3종 + create_agent 로 만든 ReAct 에이전트를 노드 하나짜리 서브그래프로
감싼다. Agent SDK 에서는 AgentDefinition 하나로 끝난다. 도구 3종(이름·설명·인자)과
시스템 프롬프트는 app/ 과 같다.

도구 안의 LLM 호출(질의 확장, 본문 요약)은 models.ask() — 단발 query() 로 한다.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from claude_agent_sdk import AgentDefinition, create_sdk_mcp_server, tool

from models import ask, resolve_model_id
from rag_tools import text_result

logger = logging.getLogger(__name__)

WEB_SERVER = "web"
WEB_TOOLS = [f"mcp__{WEB_SERVER}__{n}" for n in ("expand_query", "web_search_tool", "fetch_webpage_content")]

TREND_AGENT = "general_it_trend"

QUERY_EXPANSION_TEMPLATE = """
원본 질문이 주어집니다: {question}

> 역할
- 1. 당신의 임무는 질문의 의미를 효과적으로 확장하는 것 
- 2. 원본 질문을 다양한 각도나 하위 주제에서 더 깊이 파고드는 추가 질문들을 생성할 것 
- 3. 각 질문은 한 줄에 하나씩 제시할 것
- 4. 원본 질문의 서로 다른 측면이나 세부사항에 초점을 맞추고 다른 질문과 내용이 겹치지 않아야함 

> 출력 제약
- 1. 새로 생성한 질문만 제공할 것 
- 2. 각 질문은 별도의 줄에 작성할 것 
- 3. 설명, 예시, 그 외 어떤 부가 텍스트도 포함하지 말것 


"""


async def expand(query: str) -> list[str]:
    """원 질문 + 확장 질문 2개. app/ 의 expand_query 와 같은 후처리."""
    text = await ask("trend", QUERY_EXPANSION_TEMPLATE.format(question=query))
    similar = [q.strip() for q in text.split("\n") if q.strip()][:2]
    return [query] + similar


@tool(
    "expand_query",
    "입력된 질문을 여러 개의 유사한 질문으로 확장하는 도구",
    {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
)
async def expand_query(args: dict[str, Any]) -> dict[str, Any]:
    return text_result(await expand(args["query"]))


def search_web(query: str, num_results: int = 2) -> list[dict[str, str]]:
    """Serper(구글 검색). app/ 의 GoogleSerperAPIWrapper(k, gl=kr, hl=ko) 와 같은 요청."""
    import requests

    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": os.getenv("SERPER_API_KEY", ""), "Content-Type": "application/json"},
        params={"q": query, "gl": "kr", "hl": "ko", "num": num_results},
        timeout=30,
    )
    resp.raise_for_status()
    return [
        {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
        for r in resp.json().get("organic", [])
    ]


@tool(
    "web_search_tool",
    "검색 쿼리를 사용하여 웹에서 정보를 서칭하는 도구",
    {
        "type": "object",
        "properties": {"query": {"type": "string"}, "num_results": {"type": "integer", "default": 2}},
        "required": ["query"],
    },
)
async def web_search_tool(args: dict[str, Any]) -> dict[str, Any]:
    results = await asyncio.to_thread(search_web, args["query"], int(args.get("num_results") or 2))
    return text_result(results)


def load_page(url: str) -> dict[str, Any]:
    """페이지 본문과 메타데이터. app/ 의 WebBaseLoader 와 같은 방식(html.parser, get_text)."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, headers={"User-Agent": os.getenv("USER_AGENT", "toyproject-agent/1.0")}, timeout=30)
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")
    metadata: dict[str, Any] = {"source": url}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "No description found.")
    if html := soup.find("html"):
        metadata["language"] = html.get("lang", "No language found.")
    return {"page_content": soup.get_text(), "metadata": metadata}


async def fetch_pages(urls: list[str]) -> list[dict[str, Any]]:
    """본문을 받아 페이지마다 200자 요약으로 줄인다(app/ 과 같은 요약 프롬프트)."""
    docs = []
    for url in urls:
        page = await asyncio.to_thread(load_page, url)
        summary = await ask(
            "trend",
            f"## 임 문서를 핵심만 남겨서 200자 내외로 요약해줘, \n\n ##Document: \n{page['page_content']}",
        )
        docs.append({"page_content": summary, "metadata": page["metadata"]})
    return docs


@tool(
    "fetch_webpage_content",
    "웹 서칭에서 URL의 내용을 가져오는 도구",
    {
        "type": "object",
        "properties": {"urls": {"type": "array", "items": {"type": "string"}}},
        "required": ["urls"],
    },
)
async def fetch_webpage_content(args: dict[str, Any]) -> dict[str, Any]:
    return text_result(await fetch_pages(list(args.get("urls") or [])))


def web_server():
    return create_sdk_mcp_server(
        name=WEB_SERVER, version="1.0.0", tools=[expand_query, web_search_tool, fetch_webpage_content]
    )


system_prompt = """
> 역할
- 1. 당신은 웹에서 정보를 찾아 제공하는 데 특화된 유능한 AI 어시스턴트임
- 2. 당신의 목표는 오직 도구를 통해 검색한 정보만을 근거로 간결하고 정확하며 잘 구조화된 답변을 작성하는 것 

> 지시사항
- 1. 질문 이해: 사용자가 어떤 정보를 요구하는지 분석함
- 2. 다중 쿼리 생성: expand_query 도구로 관련된 여러 개의 쿼리를 생성함
- 3. 정보 검색: web_search_tool로 웹에서 관련 정보를 조회함
- 4. 본문 수집: fetch_webpage_content로 해당 페이지의 전체 내용을 조회함
- 5. 정보 종합: 검색된 문서들의 정보를 종합하여 질문에 직접 닫힘
- 6. 출처 인용: 답변 말미에 "Sources" 섹션을 두고 사용한 URL을 나열함 

> 답변 제약 사항
- 1. 찾아낸 정보를 근거로 항상 상세한 답변을 제공할 것 
- 2. 특정 정보를 인용할 때 [Source 1], [Source 2] 형식의 인라인 인용을 포함할 것
- 3. 주어진 정보를 답할 수 없는 질문이라면 그 사실을 명확히 밝힐 것 
- 4. 별도 지시가 없는 한 한국어로 답변할 것 
"""


def build_trend_agent() -> AgentDefinition:
    return AgentDefinition(
        description=(
            "단순 개념 질의 또는 최신 IT 트렌드·시세를 웹 검색으로 답하는 에이전트. "
            "예) JWT 와 세션 인증의 차이, 트랜잭션 격리 수준, 현재 맥북 가격"
        ),
        prompt=system_prompt,
        tools=list(WEB_TOOLS),
        mcpServers=[WEB_SERVER],
        model=resolve_model_id("trend"),
        background=False,
    )
