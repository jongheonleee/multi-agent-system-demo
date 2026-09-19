"""옵시디언 볼트를 Local REST API with MCP 플러그인으로 읽는다.

플러그인(Adam Coddington, obsidian-local-rest-api v5.1.0)이 MCP 서버를 내장하므로
별도 브릿지 패키지가 필요 없다.

전제조건: Obsidian 앱이 실행 중이어야 한다.

--------------------------------------------------------------------------
볼트는 절대 읽기 전용이다.
  플러그인은 쓰기/삭제/커맨드 실행 도구까지 16종을 노출하지만, 이 앱은
  OBSIDIAN_TOOL_NAMES 화이트리스트에 있는 읽기 도구만 에이전트에 넘긴다.
  블랙리스트가 아니라 화이트리스트여야 한다 — 플러그인이 새 쓰기 도구를
  추가해도 자동으로 차단되도록.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

OBSIDIAN_MCP_URL = os.getenv("OBSIDIAN_MCP_URL", "https://127.0.0.1:27124/mcp/")
OBSIDIAN_API_KEY = os.getenv("OBSIDIAN_API_KEY", "")

# 에이전트에 노출할 읽기 전용 도구. 이 목록에 없는 도구는 전달되지 않는다.
# vault_write / vault_append / vault_patch / vault_delete / vault_move /
# vault_copy / command_execute / open_file 은 의도적으로 제외한다.
OBSIDIAN_TOOL_NAMES = (
    "search_simple",
    "search_query",
    "vault_read",
    "vault_list",
    "vault_get_document_map",
)

# 볼트에는 기술 태그가 없다(태그 27개가 전부 농업/경제/공고문 계열).
# 대신 #MOC 태그가 붙은 Map of Content 노트 33개가 도메인 허브 역할을 하므로
# 이것을 진입점으로 쓴다. 아래 경로는 search_query 로 실제 조회해 확인한 값이다.
_CS = "04. 나의 성장기/02. 나의 CS 학습"

# 전체 기술 영역을 잇는 최상위 허브. 어느 도메인에서 출발해도 여기서 넓힐 수 있다.
MOC_HUB = f"{_CS}/기술 아키텍처 지도.md"

MOC_ENTRYPOINTS = {
    # 운영 중인 시스템을 돌리는 층
    "server_infra": [
        f"{_CS}/00. 네트워크/네트워크 MOC.md",
        f"{_CS}/01. 운영체제(Linux)/운영체제 MOC.md",
        f"{_CS}/03. 컨테이너(Docker)/컨테이너 MOC.md",
        f"{_CS}/05. 컨테이너 오케스트레이션(k8s)/쿠버네티스 MOC.md",
        f"{_CS}/04. 모니터링 시스템(Prometheus, Grafana)/모니터링 MOC.md",
        f"{_CS}/07. 로깅 파이프라인(ELK, Grafana Loki)/로깅 파이프라인 MOC.md",
        f"{_CS}/10. CICD 파이프라인(젠킨스)/CICD MOC.md",
        f"{_CS}/11. 클라우드(AWS)/클라우드 MOC.md",
        f"{_CS}/12. IaC(Terraform)/IaC MOC.md",
        f"{_CS}/17. 리버스 프록시(Nginx)/리버스 프록시 MOC.md",
    ],
    # 코드와 데이터의 구조를 정하는 층
    "system_design": [
        f"{_CS}/19. 시스템 아키텍처/시스템 아키텍처 MOC.md",
        f"{_CS}/13. 소프트웨어 아키텍처(Clean Architecture)/소프트웨어 아키텍처 MOC.md",
        f"{_CS}/06. 관계형 데이터베이스(MySQL)/관계형 데이터베이스 MOC.md",
        f"{_CS}/22. NoSQL(MongoDB)/NoSQL MOC.md",
        f"{_CS}/08. 메시징(Kafka, Redis)/메시징 MOC.md",
        f"{_CS}/09. 캐시(Redis)/캐시 MOC.md",
        f"{_CS}/14. 디자인패턴/디자인패턴 MOC.md",
        f"{_CS}/23. 검색엔진(Elasticsearch)/검색엔진 MOC.md",
        f"{_CS}/25. 그래프 데이터베이스(Neo4j)/그래프 데이터베이스 MOC.md",
        f"{_CS}/18. 스프링부트 & JPA/스프링부트 MOC.md",
    ],
    # 모델과 그 주변. Pinecone 논문 검색과 함께 쓴다.
    "ai": [
        f"{_CS}/26. RAG&AI Agent&Ontology/RAG & AI Agent MOC.md",
        f"{_CS}/20. AI Fundamental/AI MOC.md",
        f"{_CS}/24. 벡터 데이터베이스(pgvector, Milvus)/벡터 데이터베이스 MOC.md",
    ],
}


def _insecure_client(headers=None, timeout=None, auth=None):
    """플러그인이 자체서명 인증서를 쓰므로 TLS 검증을 끈다.

    127.0.0.1 로컬 연결에만 쓰이므로 허용 가능하다.
    """
    import httpx

    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout,
        auth=auth,
        verify=False,
        follow_redirects=True,
    )


def _connection() -> dict:
    return {
        "obsidian": {
            "transport": "streamable_http",
            "url": OBSIDIAN_MCP_URL,
            "headers": {"Authorization": f"Bearer {OBSIDIAN_API_KEY}"},
            "httpx_client_factory": _insecure_client,
        }
    }


async def _load() -> list:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return await MultiServerMCPClient(_connection()).get_tools()


def get_obsidian_tools() -> list:
    """읽기 전용 옵시디언 도구를 반환한다.

    Obsidian 이 꺼져 있으면 연결이 실패한다. 그래프 전체를 죽이지 않고
    빈 리스트로 degrade 하며, 에이전트 프롬프트가 "조회 실패" 를 명시하도록 한다.
    """
    if not OBSIDIAN_API_KEY:
        logger.warning("OBSIDIAN_API_KEY 가 비어 있습니다. 옵시디언 도구를 건너뜁니다.")
        return []

    try:
        tools = asyncio.run(_load())
    except Exception as e:
        logger.warning("옵시디언 MCP 연결 실패(Obsidian 이 실행 중인지 확인): %s", e)
        return []

    allowed = [t for t in tools if t.name in OBSIDIAN_TOOL_NAMES]
    dropped = [t.name for t in tools if t.name not in OBSIDIAN_TOOL_NAMES]
    if dropped:
        logger.info("읽기 전용 정책으로 제외한 도구: %s", ", ".join(sorted(dropped)))
    return allowed
