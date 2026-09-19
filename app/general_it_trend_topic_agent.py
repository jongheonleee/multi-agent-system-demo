
import logging 

from typing import List, Sequence
from langgraph.graph import StateGraph, START, END 
from langchain.agents import create_agent
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import filter_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import WebBaseLoader

from state import GeneralITTopicState 
from models import build_summarizer, get_model, get_original_reranker 

logger = logging.getLogger(__name__)

@tool(description = "입력된 질문을 여러 개의 유사한 질문으로 확장하는 도구")
def expand_query(query: str) -> List[str]:

    """
    - 주어진 질문에서 여러 개의 유사한 질문을 생성함
    """

    query_expansion_template = """
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
    prompt = ChatPromptTemplate.from_template(query_expansion_template)
    chain = prompt | get_model("trend") | (lambda x: x.text.split("\n"))

    # 사용자 입력 질문에 대해 유사한 질문으로 확장 
    similar_queries = chain.invoke({"question": query})
    similar_queries = [q.strip() for q in similar_queries if q.strip()][:2]

    # 확장된 질문을 반환 
    all_queries = [query] + similar_queries
    return all_queries

@tool(description = "검색 쿼리를 사용하여 웹에서 정보를 서칭하는 도구")
def web_search_tool(query: str, num_results: int = 2) -> List[dict]:

    """
    - 웹에서 관련 정보를 검색함 
    """

    search = GoogleSerperAPIWrapper(
        k = num_results,
        gl = "kr",
        hl = "ko",
    )
    results = search.results(query)

    organic_results = results.get("organic", [])
    filtered_results = [
        {
            "title": result.get("title", ""),
            "url": result.get("link", ""),
            "snippet": result.get("snippet", ""),
        }
        for result in organic_results
    ]

    return filtered_results 


@tool(description = "웹 서칭에서 URL의 내용을 가져오는 도구")
def fetch_webpage_content(urls: List[str]) -> List[Document]:

    """
    - 웹 페이지의 내용을 가져옴
    """

    loader = WebBaseLoader(urls)
    docs = []

    for doc in loader.lazy_load():
        result = get_model("trend").invoke(
            f"## 임 문서를 핵심만 남겨서 200자 내외로 요약해줘, \n\n ##Document: \n{doc.page_content}"
        )

        docs.append(
            Document(
                page_content = result.text,
                metadata = doc.metadata,
            )
        )

    return docs 

def rerank_results(query: str, documents: List[dict]) -> List[dict]:

    """
    - 검색 결과를 리랭킹하여 질문과 가장 관련성 높은 결과를 반환함 
    """

    reranker = get_original_reranker()

    docs_for_reranking = [
        Document(
            page_content = doc.get("title", "") + " " + doc.get("snippet", ""),
            metadata = {"url": doc.get("url", "")}
        )
        for doc in documents 
    ]

    reranked_docs = reranker.compress_documents(
        documents = docs_for_reranking,
        query = query,
    )

    return [
        {
            "url": doc.metadata.get("url"),
            "title": doc.metadata.get("title"),
            "sinppet": doc.metadata.get("snippet")
        }


        for doc in reranked_docs
    ]

def rerank_content(query: str, documents: Sequence[Document]) -> Sequence[Document]:

    """
    - 웹 페이지 내용을 리랭킹하여 질문과 가장 관련성 높은 내용을 반환함 
    """

    reranker = get_original_reranker()

    rerank_docs = reranker.compress_documents(
        documents = documents,
        query = query,
    )

    return rerank_docs 

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

tools = [
    expand_query,           # 입력된 질문을 여러 개의 유사한 질문으로 확장하는 도구
    web_search_tool,        # 검색 쿼리를 사용하여 웹에서 정보를 서칭하는 도구
    fetch_webpage_content,  # 웹 서칭에서 URL의 내용을 가져오는 도구
]

agent_executor = create_agent(
    # "Sonnet" 은 role 이름이 아니라 unknown 으로 떨어진다. 명시적으로 trend role 사용.
    model = get_model("trend"),
    tools = tools,
    system_prompt = system_prompt,
    middleware = [build_summarizer()],
    name = "general_it_trend_react_agent",
)

general_it_trend_graph_builder = StateGraph(GeneralITTopicState)
general_it_trend_graph_builder.add_node("T1. general_it_trend_agent", agent_executor)
general_it_trend_graph_builder.add_edge(START, "T1. general_it_trend_agent")
general_it_trend_graph_builder.add_edge("T1. general_it_trend_agent", END)

general_it_trend_graph = general_it_trend_graph_builder.compile(
    name = "general_IT_Trend_topic_agent",
    # debug=True 는 노드별 상태를 통째로 stdout 에 덤프해 로그를 가린다.
    # 필요할 때만 켤 것.
    debug = False,
)