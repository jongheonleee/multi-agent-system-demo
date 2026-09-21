"""옵시디언 볼트 연결 — 읽기 전용.

app/ 은 langchain_mcp_adapters 로 MCP 도구를 LangChain 도구로 변환해 에이전트에 넘긴다.
Agent SDK 에서는 **Claude Code 가 MCP 클라이언트**다. 서버 설정(dict)만 넘기면 연결·
도구 목록·호출을 Claude Code 가 한다. 변환 코드가 필요 없다.

그리고 이 서버는 도메인 서브에이전트의 AgentDefinition.mcpServers 에만 넣는다.
오케스트레이터(메인 에이전트)의 컨텍스트에는 옵시디언 도구가 아예 나타나지 않는다.

볼트는 절대 읽기 전용이다. 플러그인은 쓰기/삭제/커맨드 실행까지 16종을 노출하므로
세 겹으로 막는다. 모두 화이트리스트라서 플러그인이 새 쓰기 도구를 추가해도 자동 차단된다.
  1. AgentDefinition.tools       서브에이전트에게 보이는 도구 = 화이트리스트 5종
  2. allowed_tools + dontAsk     허용 목록에 없는 도구 호출은 권한 단계에서 거부
  3. PreToolUse 훅               mcp__obsidian__* 중 화이트리스트 밖이면 deny
"""
from __future__ import annotations

import logging
import os
import ssl
import tempfile
import urllib.request
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# 로컬 환경 옵시디언 URL, API KEY
OBSIDIAN_MCP_URL = os.getenv("OBSIDIAN_MCP_URL", "https://127.0.0.1:27124/mcp/")
OBSIDIAN_API_KEY = os.getenv("OBSIDIAN_API_KEY", "")

OBSIDIAN_SERVER = "obsidian"

# 에이전트에 노출할 읽기 전용 도구. 이 목록에 없는 도구는 전달되지 않는다.
OBSIDIAN_TOOL_NAMES = (
    "search_simple",            # Obsidian 내장 검색 엔진을 이용한 전문(full-text) 검색
    "search_query",             # JsonLogic 구조화 쿼리를 사용한 검색
    "vault_read",               # 지정한 파일의 내용, frontmatter, 태그, stat 조회
    "vault_list",               # 특정 디렉토리 안의 파일 / 하위 디렉토리 목록
    "vault_get_document_map",   # 파일 안의 헤딩 구조, 블록 참조, frontmatter 필드 목록
)

# Claude Code 가 붙이는 이름: mcp__<서버>__<도구>
OBSIDIAN_TOOLS = [f"mcp__{OBSIDIAN_SERVER}__{name}" for name in OBSIDIAN_TOOL_NAMES]

# 나의 옵시디언 경로중 "04. 나의 성장기/02. 나의 CS 학습"에 CS 학습 내용이 문서로 정리되어 있음
_CS = "04. 나의 성장기/02. 나의 CS 학습"

# 전체 기술 영역을 잇는 최상위 허브. 여기가 최초 진입점
MOC_HUB = f"{_CS}/기술 아키텍처 지도.md"

# 서버 및 인프라, 시스템 설계, AI 영역에 대한 각각의 MOC 경로 정의
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


# --------------------------------------------------------------------------
# 연결 설정
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def plugin_certificate() -> str | None:
    """플러그인의 자체서명 인증서를 받아 파일로 저장하고 경로를 돌려준다.

    app/ 은 파이썬 httpx 연결의 TLS 검증을 꺼서 해결했다. Claude Code(Node)는
    NODE_EXTRA_CA_CERTS 로 이 인증서 **하나만** 추가로 신뢰시키면 되므로
    검증을 끄지 않아도 된다. 인증서를 받는 이 한 번의 요청만 검증 없이 한다
    (127.0.0.1 로컬, 공개 인증서 다운로드).
    """
    base = OBSIDIAN_MCP_URL.split("/mcp", 1)[0]
    try:
        ctx = ssl._create_unverified_context()
        pem = urllib.request.urlopen(f"{base}/obsidian-local-rest-api.crt", context=ctx, timeout=3).read()
    except Exception as e:
        logger.warning("옵시디언 인증서를 받지 못했습니다(Obsidian 이 실행 중인지 확인): %s", e)
        return None
    path = os.path.join(tempfile.gettempdir(), "app2-obsidian-local-rest-api.crt")
    with open(path, "wb") as f:
        f.write(pem)
    return path


def obsidian_mcp_server() -> dict[str, Any] | None:
    """Claude Code 에 넘길 옵시디언 MCP 서버 설정. 쓸 수 없으면 None.

    None 이면 도메인 서브에이전트는 볼트 도구 없이 돌고, 프롬프트 규칙대로
    "내부 문서를 조회할 수 없었다"고 답한다(app/ 의 degrade 와 같다).
    """
    if not OBSIDIAN_API_KEY:
        logger.warning("OBSIDIAN_API_KEY 가 비어 있습니다. 옵시디언 도구를 건너뜁니다.")
        return None
    if not plugin_certificate():
        return None
    return {
        "type": "http",
        "url": OBSIDIAN_MCP_URL,
        "headers": {"Authorization": f"Bearer {OBSIDIAN_API_KEY}"},
    }


def tls_env() -> dict[str, str]:
    cert = plugin_certificate() if OBSIDIAN_API_KEY else None
    return {"NODE_EXTRA_CA_CERTS": cert} if cert else {}


# --------------------------------------------------------------------------
# 읽기 전용 훅
# --------------------------------------------------------------------------
_PREFIX = f"mcp__{OBSIDIAN_SERVER}__"


def is_allowed_vault_tool(tool_name: str) -> bool:
    return tool_name.startswith(_PREFIX) and tool_name[len(_PREFIX):] in OBSIDIAN_TOOL_NAMES


async def deny_vault_writes(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
    """PreToolUse 훅. 옵시디언 도구 중 화이트리스트 밖은 실행 전에 거부한다."""
    name = input_data.get("tool_name", "")
    if name.startswith(_PREFIX) and not is_allowed_vault_tool(name):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"볼트는 읽기 전용입니다. {name} 은 허용되지 않습니다.",
            }
        }
    return {}
