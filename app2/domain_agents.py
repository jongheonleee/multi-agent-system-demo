"""도메인 전문 서브에이전트 3종.

server_infra  -> 옵시디언 볼트 (운영/인프라 MOC)
system_design -> 옵시디언 볼트 (구조/설계 MOC)
ai            -> Pinecone 논문 벡터스토어 + 옵시디언 (RAG/AI MOC)

app/ 은 create_agent 로 ReAct 에이전트를 만들고, 그래프의 Send 로 병렬 실행한다.
Agent SDK 에서는 AgentDefinition 하나가 곧 에이전트다. 오케스트레이터가 한 턴에
Agent 도구를 여러 번 부르면 Claude Code 가 서브에이전트들을 **병렬로** 돌린다.
각 서브에이전트는 자기 컨텍스트에서 조사하고 최종 답만 오케스트레이터에 돌려준다.

프롬프트(DOMAIN_PROMPTS)는 app/ 과 글자 하나까지 같다.
"""
from __future__ import annotations

from claude_agent_sdk import AgentDefinition

from models import resolve_model_id
from obsidian_tools import (
    MOC_ENTRYPOINTS,
    MOC_HUB,
    OBSIDIAN_SERVER,
    OBSIDIAN_TOOLS,
    obsidian_mcp_server,
)
from rag_tools import RAG_SERVER, VECTOR_SEARCH_TOOL

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

# 오케스트레이터가 어느 서브에이전트에 맡길지 고를 때 읽는 설명.
# app/ 판단 Agent 의 도메인 정의와 같다.
DOMAIN_DESCRIPTIONS = {
    "server_infra": (
        "운영 중인 시스템을 돌리는 층의 전문가. 배포, 쿠버네티스/컨테이너, 네트워크, "
        "DB 운영·튜닝, 모니터링, 장애 진단, 용량 산정. 개인 옵시디언 볼트를 근거로 답한다."
    ),
    "system_design": (
        "코드와 데이터의 구조를 정하는 층의 전문가. 서비스 경계, 모듈 분리, API 계약, "
        "데이터 모델링, 일관성/확장성 트레이드오프. 개인 옵시디언 볼트를 근거로 답한다."
    ),
    "ai": (
        "모델과 그 주변의 전문가. LLM, RAG, 에이전트, 모델 서빙, 임베딩, 벡터 검색, 프롬프트. "
        "색인된 논문(Pinecone)과 개인 옵시디언 볼트를 근거로 답한다."
    ),
}

DOMAINS = tuple(DOMAIN_PROMPTS)


def build_domain_agents() -> dict[str, AgentDefinition]:
    """도메인 서브에이전트 정의. 호출할 때마다 옵시디언 연결 가능 여부를 다시 본다."""
    obsidian = obsidian_mcp_server()
    agents = {}
    for domain in DOMAINS:
        tools = list(OBSIDIAN_TOOLS) if obsidian else []
        # 옵시디언 서버는 이 서브에이전트에만 붙는다(인라인 설정). 메인에는 보이지 않는다.
        servers: list[str | dict] = [{OBSIDIAN_SERVER: obsidian}] if obsidian else []
        if domain == "ai":
            tools = [VECTOR_SEARCH_TOOL, *tools]
            servers.append(RAG_SERVER)
        agents[domain] = AgentDefinition(
            description=DOMAIN_DESCRIPTIONS[domain],
            prompt=DOMAIN_PROMPTS[domain],
            tools=tools,
            mcpServers=servers,
            model=resolve_model_id("domain"),
            background=False,
        )
    return agents
