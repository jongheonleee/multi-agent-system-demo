import pytest

from obsidian_tools import MOC_ENTRYPOINTS, MOC_HUB, OBSIDIAN_TOOL_NAMES


def test_moc_entrypoints_cover_all_three_domains():
    assert set(MOC_ENTRYPOINTS) == {"server_infra", "system_design", "ai"}
    assert any("쿠버네티스" in n for n in MOC_ENTRYPOINTS["server_infra"])
    assert any("시스템 아키텍처" in n for n in MOC_ENTRYPOINTS["system_design"])
    assert any("RAG" in n for n in MOC_ENTRYPOINTS["ai"])


def test_moc_paths_are_markdown_files():
    # vault_read 는 경로를 받으므로 .md 로 끝나는 전체 경로여야 한다.
    assert MOC_HUB.endswith(".md")
    for paths in MOC_ENTRYPOINTS.values():
        for p in paths:
            assert p.endswith(".md"), p
            assert "/" in p, f"전체 경로여야 합니다: {p}"


def test_exposed_tools_are_read_only():
    """볼트는 절대 읽기 전용이다. 쓰기 계열 도구가 새어나가면 안 된다."""
    banned = (
        "vault_write",
        "vault_append",
        "vault_patch",
        "vault_delete",
        "vault_move",
        "vault_copy",
        "command_execute",
        "open_file",
    )
    for name in banned:
        assert name not in OBSIDIAN_TOOL_NAMES, f"{name} 이 노출되면 안 됩니다"


def test_whitelist_contains_only_known_read_tools():
    """블랙리스트가 아니라 화이트리스트여야 한다.

    플러그인이 새 쓰기 도구를 추가해도 자동으로 차단되도록.
    """
    assert set(OBSIDIAN_TOOL_NAMES) == {
        "search_simple",
        "search_query",
        "vault_read",
        "vault_list",
        "vault_get_document_map",
    }


@pytest.mark.llm
def test_tools_load_from_plugin():
    from obsidian_tools import get_obsidian_tools

    tools = get_obsidian_tools()
    names = {t.name for t in tools}
    assert "search_simple" in names, names
    assert "vault_read" in names, names
    # 필터링 후에도 쓰기 도구가 없어야 한다.
    assert not (names - set(OBSIDIAN_TOOL_NAMES))


@pytest.mark.llm
def test_every_moc_entrypoint_exists_in_vault():
    """진입점 경로가 실제 노트와 일치하는지 확인한다.

    경로가 틀리면 도메인 에이전트가 근거를 못 찾고 조용히 빈손으로 끝난다.
    """
    import asyncio
    import json

    from obsidian_tools import MOC_HUB, get_obsidian_tools

    tools = {t.name: t for t in get_obsidian_tools()}
    read = tools["vault_read"]

    targets = [MOC_HUB] + [p for paths in MOC_ENTRYPOINTS.values() for p in paths]
    missing = []
    for path in targets:
        try:
            result = asyncio.run(read.arun({"path": path}))
        except Exception as e:  # noqa: BLE001
            missing.append(f"{path} ({type(e).__name__})")
            continue
        text = result[0]["text"] if isinstance(result, list) and result else str(result)
        if not text or "not found" in text.lower():
            missing.append(path)

    assert not missing, "볼트에 없는 진입점:\n" + "\n".join(missing)
