
import logging, os 

from typing import Any, Dict, List, Literal, Sequence
from pydantic import BaseModel, Field 

from langgraph.graph import StateGraph, START, END 
from langchain.agents import create_agent 
from langchain.messages import ToolMessage
from langchain.tools import tool 
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from state import ComplexTopicState 
from models import get_model, get_original_reranker, get_embeddings


logger = logging.getLogger(__name__)

@tool(description = "입력된 질문을 3개의 유사한 질문으로 분리하는 도구(원본 질문을 포함해서 4개 리턴)")
def get_multiple_queries(query: str) -> list[str]:

    """
    - 주어진 질문에서 유사한 질문을 추가로 생성함
    """

    query_prompt = ChatPromptTemplate.from_template(
        """
        > 역할
        - 당신은 AI 언어 모델 어시스턴트임
        - 당신의 임무는 주어진 사용자 질문에 대해 서로 다른 3개의 버전을 생성하여 벡터 데이터베이스에서 관련 문서를 검색하는 것 
        - 사용자 질문에 대해 여러 관련 문서를 검색하는 것 
        - 사용자 질문에 대해 여러 관점을 생성함으로써, 거리 기반 유사도 검색이 가진 한계를 사용자가 극복할 수 있도록 돕는 것이 목표
        - 대안 질문들은 줄바꿈으로 구분하여 제시할 것 


        원본 질문: {question}
        """
    )

    query_chain = query_prompt | get_model("domain") | (lambda x: x.text.split("\n"))
    similar_queries = query_chain.invoke({ "question": query})
    similar_queries = [q.strip() for q in similar_queries if q.strip()][:2]
    all_queries = [query] + similar_queries
    return all_queries

@tool(description = "Pinecone VectorDB에서 관련 문서를 검색하고, 문서의 리랭킹을 통해 정렬한 뒤에 반환함")
def vector_search_tool(query: str) -> Sequence[Document]:
    try:
        embed_model = get_embeddings()
        host = os.getenv("PINECONE_HOST") 

        if not host:
            logger.error("PINECONE_HOST enviroment variable is not set")
            return []

        pc = Pinecone()
        # host 를 위치 인자로 주면 인덱스 '이름' 으로 해석되어 404 가 난다. 반드시 host= 로 넘길 것.
        index = pc.Index(host = host)

        vector_store = PineconeVectorStore(
            index = index,
            embedding = embed_model,
        )

        retriever  = vector_store.as_retriever(
            search_type = "similarity_score_threshold",
            search_kwargs = {"k": 10, "score_threshold": 0.7},
        )
        docs = retriever.invoke(query)
        docss = []

        for doc in docs:
            doc.metadata["score"] = ""
            doc.metadata["file_path"] = ""
            docss.append(Document(
                page_content = doc.page_content,
                metadata = doc.metadata,
            ))

        return get_doc_compress_reranker(docss, query)

    except Exception as e:
        logger.error(f"Error during vector search: {str(e)}")
        return []

@tool(description = "생성된 텍스트의 사실 여부를 검증하는 도구")
def fact_check_tool(text: str, context: Sequence[Document] | None = None) -> dict:

    """
    - 주어진 텍스트의 사실 여부를 문서를 기반으로 확인합니다.
    """

    fact_check_template = """
> 역할

- 1. 당신은 꼼꼼한 팩트 체커 AI입니다.
- 2. 당신의 임무는 제공된 출처 문서만을 근거로 주어진 답변의 사실적 정확성을 검증하는 것입니다.

**검증할 답변:**
{answer}

**출처 문서:**
{context}

**지시사항**
- 1. **문장 단위 분석:** 답변을 개별 사실 주장 또는 문장 단위로 분해함
- 2. **검증:** 각 문장 / 주장을 출처 문서와 대조하여 다음 중 하나로 판정함
     - * `accurate`: 출처 문서에 의해 직접적이고 완전하게 뒷받침됨 
     - * `parially accurate`: 부분적으로 뒷받침되거나 뉘앙스 차이나 누락이 있을 수 있음
     - * `inaccurate`: 출처 문서와 무순됨
     - * `ubverifiable`: 제공된 출처 문서만으로는 검증할 수 없음
- 3. **점수 부여:** 각 문장에 신뢰도 점수(0.0~1.0)을 매깁니다.
- 4. **출처 식별:** accurate 또는 parially accurate인 경우 이를 뒷받침하는 구체적인 출처 문서를 명시합니다. (예: "[Document 1]"). unverifiable 또는 inaccurate인 경우 그 이유를 서술함
- 5. **종합 평가:** 전체 정확도 점수(문장별 점수의 평균)을 산출하고, 검증 결과를 요약하는 짧은 코멘트를 작성함

**출력 형식:** 아래 JSON 스키마에 부합하는 JSON 객체만으로 응답할 것

```json 
{output_schema}
```
"""
    fact_check_prompt = ChatPromptTemplate.from_template(fact_check_template).partial(
        output_schema = FactCheckResult.model_json_schema(),
    )

    try:
        context_str = "[Document 1]\n본 리포트에서 소개하는 Gemini 1.5 Pro는 이전 2월 버전 Gemini 1.5 Pro의 업데이트이며, 대부분의 역량과 벤치마크에서 이전 버전을 능가합니다. 종합하면 Gemini 1.5 시리즈는 모델 성능과 학습 효율성 면에서 세대적 도약을 보여줍니다. Gemini 1.5 Pro는 광범위한 벤치마크에서 Gemini 1.0 Pro와 1.0 Ultra를 뛰어넘으면서도 상당히 적은\n\n[Document 2]\nGemini 1.5 시리즈가 이전 세대인 Gemini 1.0 시리즈(Gemini 1.0 Pro, Gemini 1.0 Ultra) 대비 얼마나 개선되었는지 측정하기 위함입니다. 우리의 목표는 긴 컨텍스트 역량에서 뛰어난 1.5 세대 Gemini 모델과 긴 컨텍스트가 아닌 작업에서의 성능 사이에 트레이드오프가 존재하는지, 존재한다면 그 정도가 어떠한지를 밝히는 것입니다. 특히 1.5 시리즈를 개발하면서 우리는\n\n[Document 3]\n해당 diff가 보안 관련인지 여부입니다. 결과는 표 39를 참고하십시오. Gemini 1.5 Pro는 Gemini 1.0 Ultra 대비 성능 개선을 보이지 않았습니다."

        if context:
            context_str = context 

        llm_with_tool = get_model("judge").with_structured_output(FactCheckResult)

        _fact_check_chain = fact_check_prompt | llm_with_tool
        result = _fact_check_chain.invoke({
            "answer": text,
            "context": context_str
        })

        return result.model_dump()

    except Exception as e:
        logger.error(f"Error during fact-checking with function calling: {e!s}")
        return FactCheckResult(
            sentences = [
                SentenceFactCheckResult(
                    setence = text,
                    source = None,
                    verdict = "unverifiable",
                    score = 0.0,
                    reason = f"오류 발생: {e!s}",
                )
            ],
            overall_accuracy = 0.0, 
            overall_accuracy_comment = "사실 확인 중 오류가 발생했습니다.",
        ).model_dump()

def convert_docs_to_tool_messages(docs: Sequence[Document]) -> ToolMessage:

    """
    - 문서 리스트를 ToolMessage 형태로 반환함
    """ 

    try:
        doc_content = ""

        for i, doc in enumerate(docs):
            metadata_str = ",".join([
                f"{k}: {v}"
                for k, v in doc.metadata.items()
                if k not in ["source", "file_path"]
            ])

            doc_content += (
                f"[Document {i+1}] (Metadata: {metadata_str})\n{doc.page_content}\n\n"
            )

        return ToolMessage(
            content = doc_content
        )
    except Exception as e:
        logger.error(f"Error Converting documents to ToolMessage => {e}")
        return ToolMessage(content = "Error converting documents to ToolMessage")

def get_doc_compress_reranker(docs: list, query: str) -> Sequence[Document]:
    
    """
    - 문서 압축 및 리랭커, 문서의 질문과의 유사도를 높여줌 
    """

    reranker = get_original_reranker()

    reranking_docs = reranker.compress_documents(
        documents = docs,
        query = query,
    )

    return reranking_docs  

class SentenceFactCheckResult(BaseModel):
    setence: str = Field(
        ...,
        description = "사실 확인인 중인 문장",
    )

    verdict: Literal["accurate", "partially accurate", "inaccurate", "unverifiable"] = (
        Field(
            ...,
            description = "사실 확인 판정",
        )
    )

    source: str | None = Field(
        None,
        description = "해당 문장의 진술을 뒷받침하거나 반박하는 출처 문서",
    )

    score: float = Field(
        ...,
        description = "진술의 신뢰도 점수 (0.0 ~ 1.0)",
    )

    reason: str | None = Field(
        None,
        description = "정확하지 않은 경우, 그 판단의 이유",
    )

class FactCheckResult(BaseModel):
    
    sentences: list[SentenceFactCheckResult] = Field(
        ...,
        description = "각 문장에 대한 사실 확인 결과",
    )

    overall_accuracy: float = Field(
        ..., 
        description = "전체 정확도 점수(0.0 ~ 1.0)",
    )

    overall_accuracy_comment: str = Field(
        ...,
        description = "콘텐츠의 전체적인 정확도에 대한 디테일한 설명(이유)",
    )

system_prompt = """

> 역할 
- 1. 당신은 검색 증강 생성(RAG)에 특화된 유능한 AI 어시스턴트임
- 2. 당신의 목표는 오직 도구를 통해 검색한 정보만을 근거로 간결하고 정확하며 잘 구조화되 답변을 작성하는 것 

> 지시사항
- 1. 질문 이해: 사용자가 어떤 정보를 요구하는지 분석함
- 2. 다중 쿼리 생성: `get_multiple_queries` 도구로 관련 문서를 검색하기 위한 여러 개의 쿼리를 생성함
- 3. 정보 검색: `vector_search_tool`로 관련 문서를 검색함
- 4. 정보 종합: 검색된 문서들의 정보를 종합하여 질문에 직접 답함 
- 5. 사실 검증: 답변을 확정하기 전에 `fact_check_tool`로 정확성을 검증함
- 6. 출처 인용: 문맥의 특정 정보를 인용할 때마다 [Source 1], [Source 2] 형식으로 표기할 것 
- 7. 출처 목록: 답변 말미에 "source" 섹션을 두고 인용한 모든 출처

> 반드시 지킬 것!
- 1. 사실 검증 결과 정확도가 낮으면(0.8 미만) 답변을 수정하고 다시 시도함
- 2. 검색된 정보만을 사용하여 질문에 상세히 답변함. 인라인 인용과 Sources 섹션을 표현함
- 3. 주어진 정보로 답할 수 없는 질문이라면 그 사실을 명확히 밝히므
"""

tools = [get_multiple_queries, vector_search_tool, fact_check_tool]
agent_executor = create_agent(
    model = get_model("domain"),
    tools = tools, 
    system_prompt = system_prompt,
    name = "complex_discusion_topic_react_agent",
)

complex_topic_graph_builder = StateGraph(ComplexTopicState)
complex_topic_graph_builder.add_node("G1. Complex Discusion Topic Agent", agent_executor)
complex_topic_graph_builder.set_entry_point("G1. Complex Discusion Topic Agent")
complex_topic_graph_builder.set_finish_point("G1. Complex Discusion Topic Agent")

complex_topic_graph = complex_topic_graph_builder.compile(
    name = "complex_discussion_topic",
    # debug=True 는 노드별 상태를 통째로 stdout 에 덤프해 로그를 가린다.
    # 필요할 때만 켤 것.
    debug = False,
)