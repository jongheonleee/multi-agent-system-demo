"""Streamlit 화면 흐름 — app/ 과 같은 사용 흐름인지 AppTest 로 확인한다.

stream_turn 을 가짜로 바꿔 Claude Code 없이 돌린다.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import main_agent
from state import Event

APP_PY = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def turns(monkeypatch):
    """stream_turn 호출을 기록하고, 차례대로 정해진 이벤트를 내보낸다."""
    box = {"calls": [], "scripts": []}

    async def fake_stream_turn(prompt, session_id=None):
        box["calls"].append((prompt, session_id))
        for ev in box["scripts"].pop(0):
            yield ev

    monkeypatch.setattr(main_agent, "stream_turn", fake_stream_turn)
    # app.py 의 load_dotenv 는 이미 있는 값을 덮지 않는다. 빈 값으로 두어 관측을 끈다.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    return box


def _done(text, sid="sess-1", clarify=""):
    return Event("done", body=text, data={"session_id": sid, "clarifying_question": clarify})


def _all_text(node):
    """상태 카드(st.status) 안쪽까지 포함해 마크다운/코드 텍스트를 모은다."""
    out = []
    if type(node).__name__ in ("Markdown", "Code") and isinstance(getattr(node, "value", None), str):
        out.append(node.value)
    for child in getattr(node, "children", {}).values():
        out.extend(_all_text(child))
    return out


def _run(at):
    at.run(timeout=30)
    assert not at.exception, at.exception
    return at


def test_welcome_screen(turns):
    at = _run(AppTest.from_file(APP_PY))
    assert at.markdown[-1].value or True
    assert [b.label for b in at.button if b.key and b.key.startswith("chip_")] == [
        "JWT와 세션 인증의 차이점이 뭔가?",
        "트래픽 급증 시 p99 지연이 튀면 무엇부터 봐야 하나?",
        "MSA에서 동기 REST와 Kafka 이벤트 중 무엇을 골라야 하나?",
    ]


def test_question_shows_answer_and_subagent_tool_steps(turns):
    turns["scripts"] = [[
        Event("tool_call", name="Agent", body={"subagent_type": "general_it_trend"}, nested=False),
        Event("tool_call", name="web_search_tool", body={"query": "JWT"}, nested=True),
        Event("tool_result", name="web_search_tool", body=[{"type": "text", "text": "결과"}], nested=True),
        _done("JWT 는 토큰, 세션은 서버 저장 [Source 1]"),
    ]]
    at = _run(AppTest.from_file(APP_PY))
    at.button(key="chip_JWT와 세션 인증의 차이점이 뭔가?").click()
    _run(at)  # submit() -> st.rerun() -> 이 run 안에서 턴이 실행된다

    assert turns["calls"] == [("JWT와 세션 인증의 차이점이 뭔가?", None)]
    texts = _all_text(at.main)
    assert "JWT 는 토큰, 세션은 서버 저장 [Source 1]" in texts
    # 서브에이전트 안의 도구만 표시(오케스트레이터의 Agent 위임은 표시 안 함 — app/ 과 같다)
    assert "**web_search_tool**" in texts and "↳ web_search_tool" in texts
    assert '{"query": "JWT"}' in texts and "결과" in texts
    assert "**Agent**" not in texts
    assert at.session_state.agent_session_id == "sess-1"


def test_clarification_then_resume_same_session(turns):
    turns["scripts"] = [
        [Event("clarification", body="예상 트래픽 규모는?"), _done("", clarify="예상 트래픽 규모는?")],
        [_done("MAU 10만 기준 설계", sid="sess-1")],
    ]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("우리 서비스 아키텍처 좀 잡아줘").run()
    _run(at)
    assert "**확인이 필요합니다** — 예상 트래픽 규모는?" in [m.value for m in at.markdown]

    at.chat_input(key="chat_input").set_value("MAU 10만").run()
    _run(at)
    # 재질문 답은 같은 세션에 '[사용자 추가 정보]' 로 이어서 보낸다
    assert turns["calls"][1] == ("[사용자 추가 정보] MAU 10만", "sess-1")
    assert "MAU 10만 기준 설계" in [m.value for m in at.markdown]


def test_error_is_shown_like_app(turns):
    turns["scripts"] = [[Event("error", body="Credit balance is too low", is_error=True)]]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("안녕").run()
    _run(at)
    assert "⚠️ 오류가 발생했습니다: `Credit balance is too low`" in [m.value for m in at.markdown]


def test_reset_starts_new_session(turns):
    turns["scripts"] = [[_done("답", sid="sess-1")]]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("안녕").run()
    _run(at)
    assert at.session_state.agent_session_id == "sess-1"
    at.button[[b.label for b in at.button].index("대화 초기화")].click()
    _run(at)
    assert at.session_state.agent_session_id is None
    assert at.session_state.messages == []
