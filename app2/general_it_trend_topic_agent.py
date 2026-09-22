"""웹 검색 기반 IT 트렌드 서브에이전트.

app/ 은 도구 3종(expand_query, web_search_tool, fetch_webpage_content)을 만들어 ReAct 에이전트에
쥐여 주고, 도구 안에서 다시 LLM 을 불러 질의를 확장하고 페이지를 요약했다. Claude Code 에는
WebSearch / WebFetch 가 내장돼 있으므로 AgentDefinition 에 그 둘만 열어 준다. 질의 확장과
본문 읽기는 도구가 아니라 서브에이전트 자신이 한다. 역할·제약·Sources 규칙은 app/ 과 같다.
"""
from __future__ import annotations

from claude_agent_sdk import AgentDefinition

from models import resolve_model_id

TREND_AGENT = "general_it_trend"
WEB_TOOLS = ["WebSearch", "WebFetch"]

system_prompt = """
> 역할
- 1. 당신은 웹에서 정보를 찾아 제공하는 데 특화된 유능한 AI 어시스턴트임
- 2. 당신의 목표는 오직 도구를 통해 검색한 정보만을 근거로 간결하고 정확하며 잘 구조화된 답변을 작성하는 것 

> 지시사항
- 1. 질문 이해: 사용자가 어떤 정보를 요구하는지 분석함
- 2. 다중 쿼리 생성: 질문의 서로 다른 측면이나 세부사항에 초점을 맞춘 검색어를 2~3개 만듦
- 3. 정보 검색: 검색어마다 WebSearch 로 웹에서 관련 정보를 조회함
- 4. 본문 수집: 답에 필요한 페이지는 WebFetch 로 전체 내용을 조회함
- 5. 정보 종합: 검색된 문서들의 정보를 종합하여 질문에 직접 답함
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
            "트레이드오프 판단이 아니라 '무엇인가', '차이가 뭔가', '지금 얼마인가' 류의 질문을 맡긴다. "
            "예) JWT 와 세션 인증의 차이, 트랜잭션 격리 수준, 현재 맥북 가격"
        ),
        prompt=system_prompt,
        tools=list(WEB_TOOLS),
        model=resolve_model_id("trend"),
        background=False,
    )
