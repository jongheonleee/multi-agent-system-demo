"""트렌드 서브에이전트 — 내장 WebSearch/WebFetch 만 쓴다."""
import general_it_trend_topic_agent as trend


def test_trend_agent_uses_builtin_web_tools_only():
    agent = trend.build_trend_agent()
    assert agent.prompt == trend.system_prompt
    assert agent.model == "claude-sonnet-5"
    assert agent.tools == ["WebSearch", "WebFetch"]
    assert agent.mcpServers is None
    assert agent.background is False
    assert trend.WEB_TOOLS == ["WebSearch", "WebFetch"]


def test_description_says_when_to_delegate():
    d = trend.build_trend_agent().description
    assert "개념" in d and "트렌드" in d and "웹 검색" in d


def test_prompt_keeps_app_role_and_constraints_and_names_builtin_tools():
    p = trend.system_prompt
    assert "당신은 웹에서 정보를 찾아 제공하는 데 특화된 유능한 AI 어시스턴트임" in p
    assert "오직 도구를 통해 검색한 정보만을 근거로" in p
    assert "[Source 1], [Source 2] 형식의 인라인 인용" in p
    assert '"Sources" 섹션' in p
    assert "WebSearch" in p and "WebFetch" in p
    for gone in ("expand_query", "web_search_tool", "fetch_webpage_content"):
        assert gone not in p


def test_custom_web_tools_are_gone():
    for name in ("expand_query", "web_search_tool", "fetch_webpage_content", "web_server", "search_web", "load_page"):
        assert not hasattr(trend, name), name
