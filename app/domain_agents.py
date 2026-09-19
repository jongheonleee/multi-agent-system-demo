"""도메인 전문 에이전트 3종.

server_infra  -> 옵시디언 볼트 (운영/인프라 MOC)
system_design -> 옵시디언 볼트 (구조/설계 MOC)
ai            -> Pinecone 논문 벡터스토어 + 옵시디언 (RAG/AI MOC)

세 에이전트는 Send 로 병렬 실행되며, 각자 자기 서브 질문에만 답한다.
볼트는 읽기 전용이다 — obsidian_tools 가 읽기 도구만 넘기고, 프롬프트에도 명시한다.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, filter_messages
from langchain_core.runnables import RunnableConfig

from async_utils import run_coro
from rag_tools import vector_search_tool
from models import build_summarizer, get_model
from obsidian_tools import MOC_ENTRYPOINTS, MOC_HUB, get_obsidian_tools
from state import DomainAnswer, DomainTaskState

logger = logging.getLogger(__name__)

_COMMON_RULES = f"""
> 볼트 취급 원칙
- 볼트는 **읽기 전용**입니다. 노트를 만들거나 고치거나 지우려 하지 마세요.
  읽기 도구만 주어져 있으니 조회에만 쓰면 됩니다.

> 공통 규칙
- 1. 오직 도구로 조회한 내용만을 근거로 답합니다. 사전 지식으로 채우지 마세요.
- 2. 근거를 못 찾으면 "참고 자료에서 찾지 못했다"고 명시합니다. 추측하지 마세요.
- 3. 도구 호출이 실패하면 "내부 문서를 조회할 수 없었다"고 밝히고 그 한계를 답변에 포함합니다.
- 4. 당신에게 주어진 서브 질문에만 답합니다. 다른 도메인 영역으로 넘어가지 마세요.
- 5. 답변 말미에 "Sources" 섹션을 두고 참고한 노트 경로를 나열합니다.
- 6. 한국어로 답합니다.

> 막혔을 때
- 진입점 MOC 에서 답을 못 찾으면 최상위 허브를 읽어 다른 영역으로 넓히세요.
  {MOC_HUB}
"""


def _vault_prompt(role: str, scope: str, domain: str, extra: str = "") -> str:
    mocs = "\n".join(f"  - {p}" for p in MOC_ENTRYPOINTS[domain])
    return f"""당신은 {role} 전문 에이전트입니다.
{scope}

> 조사 방법
- 볼트에는 기술 태그가 없으므로 태그 검색은 쓰지 마세요.
- 먼저 아래 MOC(Map of Content) 노트를 vault_read 로 읽어 관련 노트를 찾으세요.
  MOC 는 허브라서 하위 노트로 가는 링크가 정리되어 있습니다.
{mocs}
- 그다음 search_simple 로 핵심 용어를 검색해 개별 노트를 읽습니다.
- 긴 노트는 vault_get_document_map 으로 구조를 먼저 보고 필요한 부분만 읽으세요.
{extra}{_COMMON_RULES}"""


DOMAIN_PROMPTS = {
    "server_infra": _vault_prompt(
        "서버·인프라",
        "운영 중인 시스템을 돌리는 층을 다룹니다: 배포, 쿠버네티스/컨테이너, 네트워크,\n"
        "DB 운영·튜닝, 모니터링, 장애 진단, 용량 산정.",
        "server_infra",
    ),
    "system_design": _vault_prompt(
        "시스템 설계",
        "코드와 데이터의 구조를 정하는 층을 다룹니다: 서비스 경계, 모듈 분리, API 계약,\n"
        "데이터 모델링, 일관성/확장성 트레이드오프.",
        "system_design",
    ),
    "ai": _vault_prompt(
        "AI",
        "모델과 그 주변을 다룹니다: LLM, RAG, 에이전트, 모델 서빙, 임베딩,\n벡터 검색, 프롬프트.",
        "ai",
        extra=(
            "- 논문 근거가 필요하면 vector_search_tool 로 색인된 논문을 검색합니다.\n"
            "  질의를 바꿔가며 2~3회 호출하면 회수율이 올라갑니다.\n"
        ),
    ),
}


@lru_cache(maxsize=3)
def build_domain_agent(domain: str):
    """도메인별 ReAct 에이전트.

    MCP 연결 비용이 있으므로 캐시한다.
    """
    obsidian = get_obsidian_tools()
    tools = [vector_search_tool, *obsidian] if domain == "ai" else list(obsidian)

    if not tools:
        logger.warning(
            "%s 도메인에 사용 가능한 도구가 없습니다 (Obsidian 실행 여부 확인)", domain
        )

    return create_agent(
        model=get_model("domain"),
        tools=tools,
        system_prompt=DOMAIN_PROMPTS[domain],
        middleware=[build_summarizer()],
        name=f"{domain}_agent",
    )


async def _ainvoke_agent(agent, domain: str, question: str, config: RunnableConfig) -> str:
    response = await agent.ainvoke(
        {"messages": [HumanMessage(question)]},
        config={**(config or {}), "run_name": f"domain:{domain}"},
    )
    ai_messages = filter_messages(response.get("messages", []), include_types=[AIMessage])
    return ai_messages[-1].text if ai_messages else ""


def run_domain(state: DomainTaskState, config: RunnableConfig) -> dict:
    """Send 로 병렬 실행되는 노드. 리듀서에 맞춰 리스트를 반환한다.

    MCP 어댑터가 만든 도구는 coroutine 만 있고 sync func 이 없어서
    (`StructuredTool does not support sync invocation`) 에이전트를 반드시
    ainvoke 로 돌려야 한다. 그래프 자체는 동기로 유지하기 위해 여기서
    이벤트 루프를 연다. LangGraph 의 sync 실행은 Send 분기를 스레드로
    돌리므로 스레드마다 루프가 없어 asyncio.run 이 안전하다.
    """
    domain = state.sub_question.domain
    question = state.sub_question.question

    try:
        # 에이전트 생성은 루프 밖(동기)에서 해야 MCP 도구 로딩이 정상 동작한다.
        agent = build_domain_agent(domain)
        answer = run_coro(_ainvoke_agent(agent, domain, question, config))
    except Exception as e:
        logger.error("%s 도메인 실행 실패: %s", domain, e)
        answer = f"({domain} 영역 조사 실패: {e})"

    return {
        "domain_answers": [
            DomainAnswer(domain=domain, question=question, answer=answer, sources=[])
        ]
    }


async def arun_domain(state: DomainTaskState, config: RunnableConfig) -> dict:
    """비동기 그래프 실행용. 루프를 새로 열지 않고 그대로 await 한다."""
    domain = state.sub_question.domain
    question = state.sub_question.question

    try:
        agent = build_domain_agent(domain)
        answer = await _ainvoke_agent(agent, domain, question, config)
    except Exception as e:
        logger.error("%s 도메인 실행 실패: %s", domain, e)
        answer = f"({domain} 영역 조사 실패: {e})"

    return {
        "domain_answers": [
            DomainAnswer(domain=domain, question=question, answer=answer, sources=[])
        ]
    }
