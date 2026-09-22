"""Streamlit 화면 흐름 — AppTest 로 확인한다. new_session 을 가짜 세션으로 바꿔 Claude Code 없이 돌린다."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import main_agent
from state import Event

APP_PY = str(Path(__file__).resolve().parents[1] / "app.py")


class FakeSession:
    """AgentSession 과 같은 API. send/resume 은 미리 정한 이벤트 목록을 차례로 내준다."""

    def __init__(self, box):
        self.box = box
        self.session_id = None
        self.lost_context = False
        self.waiting_question = None
        self.closed = False

    def send(self, prompt):
        self.box["calls"].append(("send", prompt))
        return self._play()

    def answer(self, text):
        self.box["calls"].append(("answer", text))
        self.waiting_question = None

    def resume(self):
        self.box["calls"].append(("resume", None))
        return self._play()

    def close(self):
        self.closed = True
        self.box["calls"].append(("close", None))

    def _play(self):
        for ev in self.box["scripts"].pop(0):
            if ev.kind == "question":
                self.waiting_question = ev.body
            if ev.kind == "done":
                self.session_id = ev.data.get("session_id")
            yield ev


@pytest.fixture
def turns(monkeypatch):
    box = {"calls": [], "scripts": [], "sessions": []}

    def fake_new_session(client_factory=None):
        s = FakeSession(box)
        box["sessions"].append(s)
        return s

    monkeypatch.setattr(main_agent, "new_session", fake_new_session)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    return box


def _done(text, sid="sess-1"):
    return Event("done", body=text, data={"session_id": sid})


def _all_text(node):
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
    assert [b.label for b in at.button if b.key and b.key.startswith("chip_")] == [
        "JWT와 세션 인증의 차이점이 뭔가?",
        "트래픽 급증 시 p99 지연이 튀면 무엇부터 봐야 하나?",
        "MSA에서 동기 REST와 Kafka 이벤트 중 무엇을 골라야 하나?",
    ]
    assert turns["sessions"] == []  # 첫 질문 전에는 세션을 만들지 않는다


def test_question_shows_answer_and_subagent_tool_steps(turns):
    turns["scripts"] = [[
        Event("tool_call", name="Agent", body={"subagent_type": "general_it_trend"}, nested=False),
        Event("tool_call", name="WebSearch", body={"query": "JWT"}, nested=True),
        Event("tool_result", name="WebSearch", body=[{"type": "text", "text": "결과"}], nested=True),
        _done("JWT 는 토큰, 세션은 서버 저장 [Source 1]"),
    ]]
    at = _run(AppTest.from_file(APP_PY))
    at.button(key="chip_JWT와 세션 인증의 차이점이 뭔가?").click()
    _run(at)

    assert turns["calls"] == [("send", "JWT와 세션 인증의 차이점이 뭔가?")]
    texts = _all_text(at.main)
    assert "JWT 는 토큰, 세션은 서버 저장 [Source 1]" in texts
    assert "**WebSearch**" in texts and "↳ WebSearch" in texts
    assert '{"query": "JWT"}' in texts and "결과" in texts
    assert "**Agent**" not in texts
    assert len(turns["sessions"]) == 1


def test_two_questions_share_one_session(turns):
    turns["scripts"] = [[_done("첫 답")], [_done("둘째 답")]]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("첫 질문").run()
    _run(at)
    at.chat_input(key="chat_input").set_value("둘째 질문").run()
    _run(at)
    assert turns["calls"] == [("send", "첫 질문"), ("send", "둘째 질문")]
    assert len(turns["sessions"]) == 1


def test_question_then_answer_resumes_same_turn(turns):
    turns["scripts"] = [
        [Event("question", body="예상 트래픽 규모는?")],
        [_done("MAU 10만 기준 설계")],
    ]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("우리 서비스 아키텍처 좀 잡아줘").run()
    _run(at)
    assert "**확인이 필요합니다** — 예상 트래픽 규모는?" in [m.value for m in at.markdown]

    at.chat_input(key="chat_input").set_value("MAU 10만").run()
    _run(at)
    assert turns["calls"] == [("send", "우리 서비스 아키텍처 좀 잡아줘"), ("answer", "MAU 10만"), ("resume", None)]
    assert "MAU 10만 기준 설계" in [m.value for m in at.markdown]
    assert at.session_state.pending_question is None


def test_error_is_shown_like_app(turns):
    turns["scripts"] = [[Event("error", body="Credit balance is too low", is_error=True)]]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("안녕").run()
    _run(at)
    assert "⚠️ 오류가 발생했습니다: `Credit balance is too low`" in [m.value for m in at.markdown]


def test_reset_closes_session_and_starts_fresh(turns):
    turns["scripts"] = [[_done("답")], [_done("새 답")]]
    at = _run(AppTest.from_file(APP_PY))
    at.chat_input(key="welcome_input").set_value("안녕").run()
    _run(at)
    first = turns["sessions"][0]
    at.button[[b.label for b in at.button].index("대화 초기화")].click()
    _run(at)
    assert first.closed is True
    assert at.session_state.agent is None
    assert at.session_state.messages == []

    at.chat_input(key="welcome_input").set_value("다시").run()
    _run(at)
    assert len(turns["sessions"]) == 2
