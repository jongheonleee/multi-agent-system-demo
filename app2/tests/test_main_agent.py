"""메인 에이전트 — 옵션 배선, 권한 게이트, 메시지 → 이벤트."""
import asyncio

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

import main_agent
from main_agent import MainAgent
from state import DomainAnswer, TurnState


@pytest.fixture(autouse=True)
def no_obsidian(monkeypatch):
    import domain_agents

    monkeypatch.setattr(domain_agents, "obsidian_mcp_server", lambda: None)
    monkeypatch.setattr(main_agent, "tls_env", lambda: {})


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 프롬프트와 옵션
# ---------------------------------------------------------------------------
def test_system_prompt_has_no_hand_written_orchestration():
    p = main_agent.SYSTEM_PROMPT
    assert main_agent.AMBIGUITY_RULES in p
    assert main_agent.MERGE_RULES in p
    assert main_agent.GENERAL_REPLY in p
    assert "AskUserQuestion" in p
    for name in ("server_infra", "system_design", "ai", "general_it_trend"):
        assert name in p
    for gone in ("complex_discusion_topic", "general_it_trend_topic", "unknown", "전달 완료", "fact_check", "request_clarification"):
        assert gone not in p


def test_options_wire_subagents_gate_and_hooks():
    agent = MainAgent()
    opts = main_agent.build_options(agent)
    assert opts.model == "claude-opus-5"
    assert opts.system_prompt == main_agent.SYSTEM_PROMPT
    assert opts.tools == ["Agent", "AskUserQuestion"]
    assert set(opts.agents) == {"server_infra", "system_design", "ai", "general_it_trend"}
    assert set(opts.mcp_servers) == {"rag"}
    # 권한은 게이트 한 함수가 전부 결정한다. 자동 승인 목록은 비어 있다.
    assert opts.permission_mode == "default"
    assert opts.allowed_tools == []
    assert opts.can_use_tool == agent.permission_gate
    assert set(opts.hooks) == {"PostToolUse", "Stop"}
    assert opts.hooks["PostToolUse"][0].matcher == "Agent|Task"
    assert opts.hooks["PostToolUse"][0].hooks == [agent.guardrail.record_subagent_answer]
    assert opts.hooks["Stop"][0].hooks == [agent.guardrail.fact_check_on_stop]
    assert opts.max_budget_usd == 3.0
    assert opts.env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"


def test_max_budget_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_BUDGET_USD", "1.5")
    assert main_agent.build_options(MainAgent()).max_budget_usd == 1.5


def test_new_session_uses_main_agent_as_driver():
    class Client:
        def __init__(self, options):
            pass

    session = main_agent.new_session(client_factory=Client)
    assert isinstance(session._driver, MainAgent)
    session.close()


# ---------------------------------------------------------------------------
# 권한 게이트
# ---------------------------------------------------------------------------
def _ctx(agent_id=None):
    return ToolPermissionContext(tool_use_id="t1", agent_id=agent_id)


def _gate(turn=None, ask_user=None):
    agent = MainAgent(ask_user=ask_user)
    if turn is not None:
        agent.turn = turn
    return agent.permission_gate


def test_ask_user_question_waits_for_answer_and_returns_it():
    seen = {}

    async def ask_user(questions):
        seen["questions"] = questions
        return {"예상 트래픽 규모는?": "MAU 10만"}

    questions = [{"question": "예상 트래픽 규모는?", "header": "규모", "options": [], "multiSelect": False}]
    out = run(_gate(ask_user=ask_user)("AskUserQuestion", {"questions": questions}, _ctx()))
    assert isinstance(out, PermissionResultAllow)
    assert out.updated_input == {"questions": questions, "answers": {"예상 트래픽 규모는?": "MAU 10만"}}
    assert seen["questions"] == questions


def test_ask_user_question_denies_and_interrupts_when_session_closed():
    from async_utils import SessionClosed

    async def ask_user(questions):
        raise SessionClosed("대화가 초기화되었습니다")

    out = run(_gate(ask_user=ask_user)("AskUserQuestion", {"questions": [{"question": "q"}]}, _ctx()))
    assert isinstance(out, PermissionResultDeny) and out.interrupt is True


@pytest.mark.parametrize("tool", ["Agent", "Task"])
def test_delegation_is_once_per_domain_but_trend_and_failed_are_free(tool):
    turn = TurnState(domain_answers=[DomainAnswer(domain="ai", question="q", answer="a")])
    gate = _gate(turn)
    assert isinstance(run(gate(tool, {"subagent_type": "ai", "prompt": "보충"}, _ctx())), PermissionResultDeny)
    assert isinstance(run(gate(tool, {"subagent_type": "server_infra", "prompt": "q"}, _ctx())), PermissionResultAllow)
    assert isinstance(run(gate(tool, {"subagent_type": "general_it_trend", "prompt": "q"}, _ctx())), PermissionResultAllow)
    empty = _gate(TurnState())
    assert isinstance(run(empty(tool, {"subagent_type": "ai", "prompt": "q"}, _ctx())), PermissionResultAllow)


@pytest.mark.parametrize("tool", [
    "mcp__rag__vector_search_tool", "mcp__obsidian__vault_read", "WebSearch", "WebFetch",
])
def test_research_tools_only_inside_subagents(tool):
    gate = _gate()
    denied = run(gate(tool, {}, _ctx()))
    assert isinstance(denied, PermissionResultDeny) and "서브에이전트" in denied.message
    assert isinstance(run(gate(tool, {}, _ctx(agent_id="sub-1"))), PermissionResultAllow)


@pytest.mark.parametrize("tool", [
    "mcp__obsidian__vault_write", "mcp__obsidian__vault_delete", "mcp__obsidian__command_execute",
    "mcp__obsidian__brand_new_write_tool",
])
def test_vault_stays_read_only_even_inside_subagents(tool):
    denied = run(_gate()(tool, {}, _ctx(agent_id="sub-1")))
    assert isinstance(denied, PermissionResultDeny) and "읽기 전용" in denied.message


@pytest.mark.parametrize("tool", ["Bash", "Read", "Write", "Edit", "Glob", "mcp__unknown__x", "Skill"])
def test_everything_else_is_denied(tool):
    assert isinstance(run(_gate()(tool, {}, _ctx())), PermissionResultDeny)
    assert isinstance(run(_gate()(tool, {}, _ctx(agent_id="sub-1"))), PermissionResultDeny)


def test_is_research_tool_and_short_name():
    assert main_agent.is_research_tool("mcp__rag__vector_search_tool")
    assert main_agent.is_research_tool("mcp__obsidian__anything")
    assert main_agent.is_research_tool("WebSearch") and main_agent.is_research_tool("WebFetch")
    assert not main_agent.is_research_tool("Agent") and not main_agent.is_research_tool("AskUserQuestion")
    assert main_agent.short_tool_name("mcp__obsidian__vault_read") == "vault_read"
    assert main_agent.short_tool_name("Agent") == "Agent"


# ---------------------------------------------------------------------------
# 메시지 → 이벤트, 결과 → done/error
# ---------------------------------------------------------------------------
def _result(text="최종 답", is_error=False, subtype="success"):
    return ResultMessage(subtype=subtype, duration_ms=1, duration_api_ms=1, is_error=is_error,
                         num_turns=3, session_id="sess-9", result=text, total_cost_usd=0.12)


def test_driver_turns_messages_into_nested_events():
    agent = MainAgent()
    agent.begin_turn()
    script = [
        AssistantMessage(content=[ToolUseBlock(id="a1", name="Agent", input={"subagent_type": "ai", "prompt": "RAG?"})],
                         model="m"),
        AssistantMessage(content=[ToolUseBlock(id="v1", name="mcp__rag__vector_search_tool", input={"query": "RAG"})],
                         model="m", parent_tool_use_id="a1"),
        UserMessage(content=[ToolResultBlock(tool_use_id="v1", content=[{"type": "text", "text": "문서"}])],
                    parent_tool_use_id="a1"),
        UserMessage(content=[ToolResultBlock(tool_use_id="a1", content=[{"type": "text", "text": "AI 답"}])]),
        AssistantMessage(content=[TextBlock(text="최종 답")], model="m"),
        AssistantMessage(content=[TextBlock(text="서브 안 텍스트")], model="m", parent_tool_use_id="a1"),
    ]
    events = [e for m in script for e in agent.events(m)]
    assert [(e.kind, e.name, e.nested) for e in events] == [
        ("tool_call", "Agent", False),
        ("tool_call", "vector_search_tool", True),
        ("tool_result", "vector_search_tool", True),
        ("tool_result", "Agent", False),
        ("answer", "", False),
    ]
    assert events[0].data == {"full_name": "Agent"}
    assert events[2].parent_tool_use_id == "a1"


def test_result_event_carries_answer_and_turn_data():
    agent = MainAgent()
    turn = agent.begin_turn()
    turn.domain_answers.append(DomainAnswer(domain="ai", question="q", answer="a"))
    turn.fact_check = {"verdict": "PASS", "score": 0.9, "comment": "ok"}
    done = agent.result(_result("답"))
    assert (done.kind, done.body) == ("done", "답")
    assert done.data["session_id"] == "sess-9" and done.data["cost_usd"] == 0.12 and done.data["num_turns"] == 3
    assert done.data["domain_answers"][0]["domain"] == "ai"
    assert done.data["fact_check"]["verdict"] == "PASS"


def test_empty_result_falls_back_like_finalize():
    assert MainAgent().result(_result("  ")).body == "답변을 생성하지 못했습니다."


def test_error_results_become_error_events():
    ev = MainAgent().result(_result("Credit balance is too low", is_error=True))
    assert (ev.kind, ev.body, ev.is_error) == ("error", "Credit balance is too low", True)
    budget = MainAgent().result(_result(None, is_error=False, subtype="error_max_budget_usd"))
    assert budget.kind == "error" and "비용 상한" in budget.body
    assert MainAgent().result(None).kind == "error"
