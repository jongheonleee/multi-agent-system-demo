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
