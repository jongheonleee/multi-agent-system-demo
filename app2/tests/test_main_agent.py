import asyncio

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

import main_agent
from state import TurnState


@pytest.fixture(autouse=True)
def no_obsidian(monkeypatch):
    import domain_agents

    monkeypatch.setattr(domain_agents, "obsidian_mcp_server", lambda: None)
    monkeypatch.setattr(main_agent, "tls_env", lambda: {})


def test_orchestrator_prompt_carries_routing_rules():
    p = main_agent.ORCHESTRATOR_PROMPT
    assert main_agent.AMBIGUITY_RULES in p
    assert main_agent.INTENT_RULES in p
    assert main_agent.DECOMPOSE_RULES in p and main_agent.MERGE_RULES in p
    assert main_agent.GENERAL_REPLY in p
    for name in ("server_infra", "system_design", "ai", "general_it_trend", "fact_check", "request_clarification"):
        assert name in p


def test_options_wire_subagents_tools_and_hooks():
    opts = main_agent.build_options(TurnState())
    assert opts.model == "claude-opus-5"
    assert opts.system_prompt == main_agent.ORCHESTRATOR_PROMPT
    # 내장 도구는 서브에이전트 호출용 Agent 하나만
    assert opts.tools == ["Agent"]
    assert set(opts.agents) == {"server_infra", "system_design", "ai", "general_it_trend"}
    assert set(opts.mcp_servers) == {"rag", "web", "judge", "orchestrator"}
    assert opts.permission_mode == "dontAsk"
    assert "mcp__judge__fact_check" in opts.allowed_tools
    assert "mcp__orchestrator__request_clarification" in opts.allowed_tools
    assert not any(t.startswith("mcp__obsidian__") and not t.split("__")[-1] in
                   ("search_simple", "search_query", "vault_read", "vault_list", "vault_get_document_map")
                   for t in opts.allowed_tools)
    assert set(opts.hooks) == {"PreToolUse", "PostToolUse", "Stop"}
    assert opts.hooks["PostToolUse"][0].matcher == "Agent|Task"
    pre = [h.__name__ for h in opts.hooks["PreToolUse"][0].hooks]
    assert pre == ["deny_vault_writes", "research_only_in_subagents", "limit_decomposition", "limit_fact_check"]
    assert opts.env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"


def test_options_resume_session():
    assert main_agent.build_options(TurnState(), resume="sess-1").resume == "sess-1"


@pytest.mark.parametrize("tool,agent_id,expected", [
    ("mcp__rag__vector_search_tool", None, "deny"),
    ("mcp__web__web_search_tool", None, "deny"),
    ("mcp__obsidian__vault_read", None, "deny"),
    ("mcp__rag__vector_search_tool", "sub-1", "pass"),
    ("Agent", None, "pass"),
    ("mcp__judge__fact_check", None, "pass"),
])
def test_research_tools_only_inside_subagents(tool, agent_id, expected):
    data = {"tool_name": tool, "tool_input": {}}
    if agent_id:
        data["agent_id"] = agent_id
    out = asyncio.run(main_agent.research_only_in_subagents(data, "t", None))
    assert out.get("hookSpecificOutput", {}).get("permissionDecision", "pass") == expected


def test_short_tool_name():
    assert main_agent.short_tool_name("mcp__obsidian__vault_read") == "vault_read"
    assert main_agent.short_tool_name("Agent") == "Agent"


def test_clarification_prompt_uses_app_marker():
    assert main_agent.clarification_prompt("MAU 10만") == "[사용자 추가 정보] MAU 10만"


# ---------------------------------------------------------------------------
# 스트림 -> 이벤트
# ---------------------------------------------------------------------------
class FakeClient:
    """ClaudeSDKClient 자리에 들어가 정해진 메시지 스트림을 내보낸다."""

    script: list = []
    last_options = None
    last_prompt = None

    def __init__(self, options=None):
        FakeClient.last_options = options

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        FakeClient.last_prompt = prompt

    async def receive_response(self):
        for item in FakeClient.script:
            if callable(item):
                await item(FakeClient.last_options)
                continue
            yield item


def _result(text="최종 답", is_error=False):
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=is_error,
                         num_turns=3, session_id="sess-9", result=text, total_cost_usd=0.12)


def _events(monkeypatch, script, prompt="질문", session_id=None):
    FakeClient.script = script
    monkeypatch.setattr(main_agent, "ClaudeSDKClient", FakeClient)

    async def collect():
        return [e async for e in main_agent.stream_turn(prompt, session_id)]

    return asyncio.run(collect())


def test_stream_marks_subagent_events_as_nested(monkeypatch):
    script = [
        AssistantMessage(content=[ToolUseBlock(id="a1", name="Agent", input={"subagent_type": "ai", "prompt": "RAG?"})],
                         model="m"),
        AssistantMessage(content=[ToolUseBlock(id="v1", name="mcp__rag__vector_search_tool", input={"query": "RAG"})],
                         model="m", parent_tool_use_id="a1"),
        UserMessage(content=[ToolResultBlock(tool_use_id="v1", content=[{"type": "text", "text": "문서"}])],
                    parent_tool_use_id="a1"),
        UserMessage(content=[ToolResultBlock(tool_use_id="a1", content=[{"type": "text", "text": "AI 답"}])]),
        AssistantMessage(content=[TextBlock(text="최종 답")], model="m"),
        _result(),
    ]
    events = _events(monkeypatch, script, session_id="sess-1")
    calls = [(e.kind, e.name, e.nested) for e in events if e.kind in ("tool_call", "tool_result")]
    assert calls == [
        ("tool_call", "Agent", False),
        ("tool_call", "vector_search_tool", True),
        ("tool_result", "vector_search_tool", True),
        ("tool_result", "Agent", False),
    ]
    done = events[-1]
    assert (done.kind, done.body, done.data["session_id"]) == ("done", "최종 답", "sess-9")
    assert FakeClient.last_options.resume == "sess-1"
    assert FakeClient.last_prompt == "질문"


def test_stream_reports_clarification(monkeypatch):
    async def ask_user(options):
        tool = options.mcp_servers["orchestrator"]["instance"]
        # in-process MCP 서버의 도구 핸들러를 Claude Code 가 부른 것처럼 직접 부른다
        handler = main_agent._clarify_tool  # noqa: F841 - 존재 확인
        from claude_agent_sdk._internal.sdk_mcp_bridge import SdkMcpBridge

        bridge = SdkMcpBridge("orchestrator", tool)
        await bridge.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
        await bridge.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        await bridge.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "request_clarification", "arguments": {"question": "예상 트래픽 규모는?"}}})
        await bridge.aclose()

    events = _events(monkeypatch, [ask_user, _result("")])
    kinds = [e.kind for e in events]
    assert kinds == ["clarification", "done"]
    assert events[0].body == "예상 트래픽 규모는?"
    assert events[-1].data["clarifying_question"] == "예상 트래픽 규모는?"


def test_empty_result_falls_back_like_finalize(monkeypatch):
    events = _events(monkeypatch, [_result("  ")])
    assert events[-1].body == "답변을 생성하지 못했습니다."


def test_error_result_becomes_error_event(monkeypatch):
    events = _events(monkeypatch, [_result("Credit balance is too low", is_error=True)])
    assert (events[-1].kind, events[-1].body) == ("error", "Credit balance is too low")


# ---------------------------------------------------------------------------
# 최종 답 선택 — app/ 의 merge / general_it_node / finalize 규칙
# ---------------------------------------------------------------------------
from state import DomainAnswer  # noqa: E402


def _da(domain, answer):
    return DomainAnswer(domain=domain, question="q", answer=answer)


def test_single_domain_answer_passes_through_verbatim():
    turn = TurnState(domain_answers=[_da("ai", "AI 전문가 답 원문")])
    assert main_agent.final_answer(turn, "전달 완료") == "AI 전문가 답 원문"


def test_trend_answer_passes_through_verbatim():
    turn = TurnState(trend_answers=["웹 검색 답 [Source 1]"])
    assert main_agent.final_answer(turn, "전달 완료") == "웹 검색 답 [Source 1]"


def test_checked_merge_is_delivered_when_pass():
    turn = TurnState(domain_answers=[_da("ai", "a"), _da("server_infra", "b")], fact_checks=1,
                     checked_answer="병합본", verdict="PASS")
    assert main_agent.final_answer(turn, "전달 완료") == "병합본"


def test_rewrite_uses_orchestrator_rewritten_text():
    turn = TurnState(domain_answers=[_da("ai", "a"), _da("server_infra", "b")], fact_checks=1,
                     checked_answer="병합본", verdict="REWRITE")
    assert main_agent.final_answer(turn, "재작성본") == "재작성본"


def test_unknown_path_uses_orchestrator_text_and_empty_falls_back():
    assert main_agent.final_answer(TurnState(), main_agent.GENERAL_REPLY) == main_agent.GENERAL_REPLY
    assert main_agent.final_answer(TurnState(), "") == "답변을 생성하지 못했습니다."
