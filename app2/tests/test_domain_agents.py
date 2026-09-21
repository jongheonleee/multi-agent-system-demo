import pytest

from domain_agents import DOMAIN_PROMPTS


def test_all_three_domains_have_prompts():
    assert set(DOMAIN_PROMPTS) == {"server_infra", "system_design", "ai"}


def test_vault_domains_mention_their_moc_entrypoints():
    assert "쿠버네티스 MOC" in DOMAIN_PROMPTS["server_infra"]
    assert "시스템 아키텍처 MOC" in DOMAIN_PROMPTS["system_design"]
    assert "RAG & AI Agent MOC" in DOMAIN_PROMPTS["ai"]


def test_ai_prompt_mentions_paper_search():
    assert "vector_search_tool" in DOMAIN_PROMPTS["ai"]


def test_prompts_forbid_writing_to_vault():
    for domain, prompt in DOMAIN_PROMPTS.items():
        assert "읽기" in prompt, domain


OBS = {"type": "http", "url": "https://127.0.0.1:27124/mcp/", "headers": {"Authorization": "Bearer k"}}


@pytest.fixture
def with_obsidian(monkeypatch):
    import domain_agents

    monkeypatch.setattr(domain_agents, "obsidian_mcp_server", lambda: OBS)


@pytest.fixture
def without_obsidian(monkeypatch):
    import domain_agents

    monkeypatch.setattr(domain_agents, "obsidian_mcp_server", lambda: None)


def test_subagents_get_whitelisted_vault_tools_only(with_obsidian):
    from domain_agents import build_domain_agents
    from obsidian_tools import OBSIDIAN_TOOLS

    agents = build_domain_agents()
    assert set(agents) == {"server_infra", "system_design", "ai"}
    assert agents["server_infra"].tools == OBSIDIAN_TOOLS
    assert agents["system_design"].tools == OBSIDIAN_TOOLS
    assert agents["ai"].tools == ["mcp__rag__vector_search_tool", *OBSIDIAN_TOOLS]


def test_obsidian_server_is_scoped_to_each_subagent(with_obsidian):
    """옵시디언 MCP 는 서브에이전트에 인라인으로만 붙는다(오케스트레이터에는 안 보인다)."""
    from domain_agents import build_domain_agents

    agents = build_domain_agents()
    assert agents["server_infra"].mcpServers == [{"obsidian": OBS}]
    assert agents["ai"].mcpServers == [{"obsidian": OBS}, "rag"]


def test_subagents_use_domain_model_prompt_and_run_synchronously(with_obsidian):
    from domain_agents import DOMAIN_DESCRIPTIONS, build_domain_agents

    for domain, agent in build_domain_agents().items():
        assert agent.model == "claude-opus-5"
        assert agent.prompt == DOMAIN_PROMPTS[domain]
        assert agent.description == DOMAIN_DESCRIPTIONS[domain]
        assert agent.background is False


def test_without_obsidian_vault_agents_have_no_tools(without_obsidian):
    """app/ 처럼 도구 없이 돌고, 프롬프트 규칙대로 '조회할 수 없었다'고 답하게 둔다."""
    from domain_agents import build_domain_agents

    agents = build_domain_agents()
    assert agents["server_infra"].tools == []
    assert agents["server_infra"].mcpServers == []
    assert agents["ai"].tools == ["mcp__rag__vector_search_tool"]
