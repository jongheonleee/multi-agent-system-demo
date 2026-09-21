import asyncio

import pytest

from obsidian_tools import MOC_ENTRYPOINTS, MOC_HUB, OBSIDIAN_TOOL_NAMES, OBSIDIAN_TOOLS


def test_moc_entrypoints_cover_all_three_domains():
    assert set(MOC_ENTRYPOINTS) == {"server_infra", "system_design", "ai"}
    assert any("쿠버네티스" in n for n in MOC_ENTRYPOINTS["server_infra"])
    assert any("시스템 아키텍처" in n for n in MOC_ENTRYPOINTS["system_design"])
    assert any("RAG" in n for n in MOC_ENTRYPOINTS["ai"])


def test_moc_paths_are_markdown_files():
    assert MOC_HUB.endswith(".md")
    for paths in MOC_ENTRYPOINTS.values():
        for p in paths:
            assert p.endswith(".md"), p
            assert "/" in p, f"전체 경로여야 합니다: {p}"


def test_exposed_tools_are_read_only():
    banned = (
        "vault_write", "vault_append", "vault_patch", "vault_delete",
        "vault_move", "vault_copy", "command_execute", "open_file",
    )
    for name in banned:
        assert name not in OBSIDIAN_TOOL_NAMES, f"{name} 이 노출되면 안 됩니다"


def test_whitelist_contains_only_known_read_tools():
    assert set(OBSIDIAN_TOOL_NAMES) == {
        "search_simple", "search_query", "vault_read", "vault_list", "vault_get_document_map",
    }
    assert OBSIDIAN_TOOLS == [f"mcp__obsidian__{n}" for n in OBSIDIAN_TOOL_NAMES]


def _decide(tool_name):
    from obsidian_tools import deny_vault_writes

    out = asyncio.run(deny_vault_writes({"tool_name": tool_name, "tool_input": {}}, "t", None))
    return out.get("hookSpecificOutput", {}).get("permissionDecision", "pass")


@pytest.mark.parametrize("name", ["mcp__obsidian__vault_write", "mcp__obsidian__vault_delete",
                                  "mcp__obsidian__command_execute", "mcp__obsidian__brand_new_write_tool"])
def test_hook_denies_everything_outside_whitelist(name):
    """화이트리스트라서 플러그인이 새 쓰기 도구를 추가해도 자동으로 막힌다."""
    assert _decide(name) == "deny"


@pytest.mark.parametrize("name", OBSIDIAN_TOOLS)
def test_hook_passes_whitelisted_read_tools(name):
    assert _decide(name) == "pass"


def test_hook_ignores_non_obsidian_tools():
    assert _decide("mcp__rag__vector_search_tool") == "pass"
    assert _decide("Agent") == "pass"


def test_server_config_requires_api_key(monkeypatch):
    import obsidian_tools

    monkeypatch.setattr(obsidian_tools, "OBSIDIAN_API_KEY", "")
    assert obsidian_tools.obsidian_mcp_server() is None
    assert obsidian_tools.tls_env() == {}


def test_server_config_degrades_without_certificate(monkeypatch):
    import obsidian_tools

    monkeypatch.setattr(obsidian_tools, "OBSIDIAN_API_KEY", "k")
    monkeypatch.setattr(obsidian_tools, "plugin_certificate", lambda: None)
    assert obsidian_tools.obsidian_mcp_server() is None


def test_server_config_is_http_with_bearer_and_trusts_plugin_cert(monkeypatch):
    """TLS 검증을 끄지 않고 플러그인 인증서만 추가로 신뢰한다."""
    import obsidian_tools

    monkeypatch.setattr(obsidian_tools, "OBSIDIAN_API_KEY", "k")
    monkeypatch.setattr(obsidian_tools, "plugin_certificate", lambda: "C:/tmp/obsidian.crt")
    cfg = obsidian_tools.obsidian_mcp_server()
    assert cfg == {"type": "http", "url": obsidian_tools.OBSIDIAN_MCP_URL, "headers": {"Authorization": "Bearer k"}}
    assert obsidian_tools.tls_env() == {"NODE_EXTRA_CA_CERTS": "C:/tmp/obsidian.crt"}


@pytest.mark.obsidian
def test_plugin_certificate_downloads():
    from obsidian_tools import plugin_certificate

    path = plugin_certificate()
    assert path and open(path, encoding="ascii").read().startswith("-----BEGIN CERTIFICATE-----")
