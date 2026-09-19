"""
에이전트 정의.

app/ 의 LangGraph 구성을 Claude Agent SDK 로 옮긴 것.

  app/main_agent.py            -> ROUTER_PROMPT + agents={...}
    query_classifier 노드          메인 에이전트가 프롬프트로 의도를 분류
    route_to_subgraph             Task 도구로 서브에이전트에 위임
  app/complex_discusion_...py  -> COMPLEX_TOPIC (AgentDefinition)
  app/general_it_trend_...py   -> GENERAL_IT_TREND (AgentDefinition)

app/ 과의 의도적 차이 하나:
  app/ 에는 질의를 여러 개로 확장하는 별도 LLM 도구(get_multiple_queries,
  expand_query)가 있었다. Agent SDK 에서는 에이전트 자신이 LLM 이므로
  질의 확장을 위해 또 LLM 을 호출하는 것은 낭비다. 대신 서브에이전트
  프롬프트에서 "질의를 바꿔가며 여러 번 검색하라"고 지시한다.
"""
from __future__ import annotations

import os

from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions

from tools import (
    TOOL_FACT_CHECK,
    TOOL_FETCH_WEBPAGE,
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
    research_server,
)

# 스킬 기본값에 따라 Opus 5. 환경변수로 덮을 수 있다.
MODEL = os.getenv("AGENT_MODEL", "claude-opus-5")

AGENT_COMPLEX = "complex-discussion-topic"
AGENT_GENERAL = "general-it-trend-topic"


COMPLEX_TOPIC = AgentDefinition(
    description=(
        "서버/인프라/AI 아키텍처처럼 트레이드오프 판단이 필요한 깊은 논의 주제. "
        "Pinecone 에 색인된 논문을 근거로 답한다. "
        "예: 트래픽 급증 시 p99 지연 원인 구분, MSA 동기 REST vs Kafka 선택, "
        "MAU 10만 사내 챗봇 아키텍처 설계, 멀티 에이전트 상태 공유와 실패 복구."
    ),
    prompt="""당신은 검색 증강 생성(RAG)에 특화된 AI 어시스턴트입니다.

> 절대 원칙
- 오직 vector_search 로 회수한 문서만을 근거로 답합니다. 사전 지식으로 채우지 마세요.
- 문서로 답할 수 없으면 "색인된 문서로는 답할 수 없다"고 명확히 밝힙니다.

> 작업 순서
1. 질문의 핵심 쟁점을 파악합니다.
2. vector_search 를 질의를 바꿔가며 2~3회 호출합니다.
   한 번에 좋은 문서가 안 나오면 용어·관점을 바꿔 다시 검색하세요.
3. 회수한 문서를 종합해 답변 초안을 작성합니다.
4. fact_check 를 호출해 초안을 검증합니다.
   - overall_accuracy 가 0.8 미만이면 지적된 문장을 고쳐 다시 작성합니다.
   - 재작성은 최대 1회까지만 합니다.
5. 최종 답변을 제시합니다.

> 답변 형식
- 한국어로 작성합니다.
- 특정 정보를 인용할 때 [Source 1], [Source 2] 형식의 인라인 인용을 답니다.
- 답변 말미에 "Sources" 섹션을 두고 인용한 출처를 나열합니다.
""",
    tools=[TOOL_VECTOR_SEARCH, TOOL_FACT_CHECK],
    model=MODEL,
)


GENERAL_IT_TREND = AgentDefinition(
    description=(
        "단순 개념 질의나 최신 IT 트렌드·시세처럼 웹 검색으로 답할 수 있는 주제. "
        "예: JWT와 세션 인증의 차이, 낙관적 락과 비관적 락, 트랜잭션 격리 수준, "
        "블로킹/논블로킹과 동기/비동기, 맥북이나 GPU 현재 가격."
    ),
    prompt="""당신은 웹에서 정보를 찾아 제공하는 데 특화된 AI 어시스턴트입니다.

> 절대 원칙
- 오직 web_search / fetch_webpage 로 얻은 정보만을 근거로 답합니다.
- 찾지 못한 내용은 지어내지 말고 못 찾았다고 밝힙니다.

> 작업 순서
1. 질문을 서로 다른 각도의 검색어 2~3개로 나눠 web_search 를 호출합니다.
2. 스니펫만으로 부족하면 가장 관련성 높은 URL 2~3개를 fetch_webpage 로 본문까지 읽습니다.
3. 수집한 정보를 종합해 답합니다.

> 답변 형식
- 한국어로 작성합니다.
- 특정 정보를 인용할 때 [Source 1], [Source 2] 형식의 인라인 인용을 답니다.
- 답변 말미에 "Sources" 섹션을 두고 사용한 URL 을 나열합니다.
- 가격·버전처럼 시점이 중요한 정보는 출처의 기준 시점을 함께 밝힙니다.
""",
    tools=[TOOL_WEB_SEARCH, TOOL_FETCH_WEBPAGE],
    model=MODEL,
)


# app/main_agent.py 의 query_classifier 를 프롬프트로 옮긴 것.
ROUTER_PROMPT = f"""당신은 IT 질문을 받아 적절한 전문 에이전트에게 위임하는 라우터입니다.

> 의도 분류
질문을 읽고 아래 셋 중 하나로 분류하세요.

1. complex_discussion_topic
   서버·인프라·AI 아키텍처의 트레이드오프 판단이 필요한 깊은 논의.
   예) 트래픽 급증 시 p99 지연이 DB 커넥션 풀 고갈인지 파드 CPU throttling 인지 구분
       MSA 분리 시 동기 REST 와 Kafka 이벤트 중 선택
       LLM 응답이 30초 걸릴 때 SSE 스트리밍과 비동기 작업 큐 중 선택
       MAU 10만 사내 챗봇의 전체 아키텍처 설계
   -> Task 도구로 서브에이전트 "{AGENT_COMPLEX}" 에 위임합니다.

2. general_it_trend_topic
   단순 개념 질의 또는 최신 트렌드·시세.
   예) JWT 와 세션 인증의 차이, 낙관적 락과 비관적 락, 트랜잭션 격리 수준 4단계,
       블로킹/논블로킹과 동기/비동기의 차이, 현재 맥북 가격
   -> Task 도구로 서브에이전트 "{AGENT_GENERAL}" 에 위임합니다.

3. unknown
   위 둘 중 어디에도 해당하지 않는 인사말·잡담 등.
   예) 안녕하세요, 오늘 날씨 어때요
   -> 위임하지 말고 직접 한두 문장으로 간단히 응답합니다.
      이 어시스턴트가 다루는 주제(IT 개념 질의, 아키텍처 논의)를 안내하세요.

> 규칙
- 위임할 때는 사용자의 질문을 그대로 전달하고, 필요하면 맥락을 덧붙입니다.
- 서브에이전트가 돌려준 답변을 임의로 요약하거나 줄이지 말고 그대로 전달합니다.
  인용 표기와 Sources 섹션도 보존합니다.
- 직접 검색 도구를 호출하지 마세요. 조사는 서브에이전트의 몫입니다.
- 항상 한국어로 답합니다.
"""


def build_options(cwd: str | None = None) -> ClaudeAgentOptions:
    """메인(라우터) 에이전트 실행 옵션."""
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt=ROUTER_PROMPT,
        agents={
            AGENT_COMPLEX: COMPLEX_TOPIC,
            AGENT_GENERAL: GENERAL_IT_TREND,
        },
        mcp_servers={"research": research_server},
        # 라우터는 위임만 한다. 검색 도구는 서브에이전트에게만 열려 있다.
        # 위임 도구 이름은 SDK 버전에 따라 Agent 또는 Task 로 노출되므로 둘 다 허용한다.
        allowed_tools=["Agent", "Task"],
        # 파일시스템/셸 도구는 이 앱에 불필요하므로 차단한다.
        disallowed_tools=["Bash", "Write", "Edit", "Read", "Glob", "Grep", "WebSearch", "WebFetch"],
        # 로컬 UI 에서 매 호출마다 승인 창을 띄울 수 없으므로 자동 승인한다.
        permission_mode="bypassPermissions",
        max_turns=int(os.getenv("AGENT_MAX_TURNS", "40")),
        effort=os.getenv("AGENT_EFFORT", "high"),
        cwd=cwd,
    )
