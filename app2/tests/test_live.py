"""실제 Claude Code 로 도는 end-to-end 테스트 (pytest -m llm, 비용 발생).

app/ 의 llm 테스트(분류·재질문·도메인 실행)에 대응한다. app2 에서는 분류가
"어느 서브에이전트에 위임했는가"로 드러나므로 그걸 확인한다.
"""
import pytest

from async_utils import iterate_in_thread
from main_agent import GENERAL_REPLY, stream_turn

pytestmark = pytest.mark.llm


def _turn(prompt, session_id=None):
    return list(iterate_in_thread(lambda: stream_turn(prompt, session_id)))


def _delegations(events):
    return [e.body.get("subagent_type") for e in events if e.kind == "tool_call" and e.name == "Agent"]


def test_greeting_gets_fixed_reply_without_tools():
    events = _turn("안녕하세요")
    assert _delegations(events) == []
    assert not [e for e in events if e.kind == "tool_call"]
    assert events[-1].kind == "done" and events[-1].body == GENERAL_REPLY


def test_concept_question_goes_to_trend_agent_and_is_delivered_verbatim():
    events = _turn("JWT와 세션 인증의 차이점이 뭔가?")
    done = events[-1]
    assert done.kind == "done", done.body
    assert _delegations(events) == ["general_it_trend"]
    nested = {e.name for e in events if e.kind == "tool_call" and e.nested}
    assert "web_search_tool" in nested
    assert not done.data["clarifying_question"]
    # 트렌드 에이전트의 답이 그대로 전달된다(app/ 의 general_it_node 와 같다)
    assert "Sources" in done.body or "Source" in done.body


def test_vague_design_request_triggers_clarification():
    events = _turn("우리 서비스 아키텍처 좀 잡아줘")
    kinds = [e.kind for e in events]
    assert "clarification" in kinds
    assert _delegations(events) == []
    assert events[-1].data["session_id"]


# ---------------------------------------------------------------------------
# SDK 실측 — 설계가 기대는 사실. 실패하면 설계를 고쳐야 한다.
# ---------------------------------------------------------------------------
def test_ask_user_question_reaches_can_use_tool_in_default_mode():
    """tools=["AskUserQuestion"] + permission_mode="default" 면 AskUserQuestion 이 can_use_tool 로 온다."""
    import asyncio

    from claude_agent_sdk import ClaudeSDKClient, PermissionResultAllow, ResultMessage

    from models import AGENT_MAX_TOKENS, base_env, isolated_options

    seen = {}

    async def gate(tool_name, input_data, context):
        seen["tool"] = tool_name
        seen["input"] = input_data
        questions = input_data.get("questions") or []
        return PermissionResultAllow(
            updated_input={"questions": questions, "answers": {q["question"]: "파란색" for q in questions}}
        )

    async def run():
        options = isolated_options(
            model="claude-sonnet-5",
            system_prompt="사용자에게 좋아하는 색을 AskUserQuestion 으로 물은 뒤, 그 색을 한 단어로 답하세요.",
            tools=["AskUserQuestion"],
            permission_mode="default",
            can_use_tool=gate,
            env=base_env(AGENT_MAX_TOKENS),
        )
        async with ClaudeSDKClient(options=options) as client:
            await client.query("시작")
            async for m in client.receive_response():
                if isinstance(m, ResultMessage):
                    return m

    result = asyncio.run(run())
    assert seen.get("tool") == "AskUserQuestion", seen
    assert seen["input"]["questions"][0]["question"]
    assert any(w in (result.result or "") for w in ("파란", "파랑")), result.result


def test_stop_hook_input_carries_last_assistant_message():
    """Stop 훅 입력에 last_assistant_message(또는 transcript_path)가 있다."""
    import asyncio

    from claude_agent_sdk import ClaudeSDKClient, HookMatcher, ResultMessage

    from models import AGENT_MAX_TOKENS, base_env, isolated_options

    seen = {}

    async def on_stop(input_data, tool_use_id, context):
        seen.update(input_data)
        return {}

    async def run():
        options = isolated_options(
            model="claude-sonnet-5",
            system_prompt="숫자만 답하세요.",
            tools=[],
            hooks={"Stop": [HookMatcher(hooks=[on_stop])]},
            env=base_env(AGENT_MAX_TOKENS),
        )
        async with ClaudeSDKClient(options=options) as client:
            await client.query("1+1은?")
            async for m in client.receive_response():
                if isinstance(m, ResultMessage):
                    return m

    asyncio.run(run())
    assert "stop_hook_active" in seen
    assert "last_assistant_message" in seen or "transcript_path" in seen, sorted(seen)
    print("STOP HOOK INPUT KEYS:", sorted(seen), "last_assistant_message=", repr(seen.get("last_assistant_message"))[:300])
