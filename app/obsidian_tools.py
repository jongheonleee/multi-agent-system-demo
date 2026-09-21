"""
> - 옵시디언 MCP 접속 및 툴 로딩
"""
from __future__ import annotations
import logging
import os
from async_utils import run_coro

logger = logging.getLogger(__name__)

# 로컬 환경 옵시디언 URL, API KEY 
OBSIDIAN_MCP_URL = os.getenv("OBSIDIAN_MCP_URL", "https://127.0.0.1:27124/mcp/")
OBSIDIAN_API_KEY = os.getenv("OBSIDIAN_API_KEY", "")

# 에이전트에 노출할 읽기 전용 도구. 이 목록에 없는 도구는 전달되지 않는다.
OBSIDIAN_TOOL_NAMES = (
    "search_simple",            # Obsidian 내장 검색 엔진을 이용한 전문(full-text) 검색. 쿼리 문자열로 볼트 전체를 훑어 매칭된 파일명과 스코어링된 컨텍스트 스니펫(주변 텍스트)을 반환함
    "search_query",             # JsonLogic 구조화 쿼리를 사용한 검색. frontmatter, 태그, 경로, 본문 등 노트 메타데이터를 조건으로 정교하게 필터링할 수 있음
    "vault_read",               # 지정한 파일의 내용, frontmatter, 태그, stat(수정일 등 파일 정보)를 조회함 
    "vault_list",               # 특정 디렉토리 안의 파일 / 하위 디렉토리 목록을 나열함 
    "vault_get_document_map",   # 파일 안의 헤딩(heading) 구조, 블록 참조(block reference), frontmatter 필드 목록을 반환함. 즉, 파일 내부를 통째로 읽지 않고 "이 노트가 어떤 섹션들로 구성돼 있는지" 개략적인 틀에 대한 정보를 조회함 
)

# 나의 옵시디언 경로중 "04. 나의 성장기/02. 나의 CS 학습"에 CS 학습 내용이 문서로 정리되어 있음 
_CS = "04. 나의 성장기/02. 나의 CS 학습"

# 전체 기술 영역을 잇는 최상위 허브. 여기가 최초 진입점 
MOC_HUB = f"{_CS}/기술 아키텍처 지도.md"

# 서버 및 인프라, 시스템 설계, AI 영역에 대한 각각의 MOC 경로 정의 
# MOC(Map of Content, 콘텐츠 지도): 내가 직접 정의한 각 노트 간 관계 정보. 지식 그래프 형태로 활용 가능함 
MOC_ENTRYPOINTS = {

    # 1. 서버 및 인프라 관련 MOC 모음 
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

    # 2. 시스템 설계 관련 MOC 모음
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


    # 3. AI 관련 MOC 모음
    "ai": [
        f"{_CS}/26. RAG&AI Agent&Ontology/RAG & AI Agent MOC.md",
        f"{_CS}/20. AI Fundamental/AI MOC.md",
        f"{_CS}/24. 벡터 데이터베이스(pgvector, Milvus)/벡터 데이터베이스 MOC.md",
    ],
}


# TLS 인증 비활성화 
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


# 옵시디언 MCP 연동 
# - MCP 클라이언트가 서버에 연결하기 위한 설정 정보(connection config)
def _connection() -> dict:
    return {
        "obsidian": {
            "transport": "streamable_http",
            "url": OBSIDIAN_MCP_URL,
            "headers": {"Authorization": f"Bearer {OBSIDIAN_API_KEY}"},
            "httpx_client_factory": _insecure_client,
        }
    }


# 옵시디언 툴 로드
# - _connection()으로 옵시디언 MCP 서버에 접속
# - 그 서버가 제공하는 툴을 LangChain에서 쓸 수 있는 형태로 로딩함 
async def _load() -> list:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    obsidianConnection = _connection()
    obsidianMCPClient = MultiServerMCPClient(obsidianConnection)
    return await obsidianMCPClient.get_tools()


# 옵시디언에서 허용된 툴을 리스트 형식으로 반환 
def get_obsidian_tools() -> list:

    """읽기 전용 옵시디언 도구를 반환한다.

    Obsidian 이 꺼져 있으면 연결이 실패한다. 그래프 전체를 죽이지 않고
    빈 리스트로 degrade 하며, 에이전트 프롬프트가 "조회 실패" 를 명시하도록 한다.
    """

    if not OBSIDIAN_API_KEY:
        logger.warning("OBSIDIAN_API_KEY 가 비어 있습니다. 옵시디언 도구를 건너뜁니다.")
        return []

    try:
        obsidianTools = _load()
        tools = run_coro(obsidianTools)
    except Exception as e:
        logger.warning("옵시디언 MCP 연결 실패(Obsidian 이 실행 중인지 확인): %s", e)
        return []

    return [
        tool for tool in tools 
        if tool.name in OBSIDIAN_TOOL_NAMES
    ]
