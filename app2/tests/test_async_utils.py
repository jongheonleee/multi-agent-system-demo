"""AgentSession — 상주 client 위에서 send / question → answer → resume / close 가 맞게 도는지.

Claude Code 없이 가짜 client 로 돈다. 가짜 client 의 스크립트 항목이 callable 이면
"모델이 AskUserQuestion 을 불렀다"는 뜻으로 options 를 넘겨 await 한다.
"""
import asyncio
import threading

import pytest
from claude_agent_sdk import ResultMessage

from async_utils import AgentSession, SessionClosed
from state import Event


class Driver:
    """TurnDriver 최소 구현. 문자열 메시지는 answer 이벤트로 바꾼다."""

    def __init__(self):
        self.turns = 0
        self.ask_user = None

    def options(self, ask_user):
        self.ask_user = ask_user
        return {"ask_user": ask_user}

    def begin_turn(self):
        self.turns += 1

    def events(self, message):
        return [Event("answer", body=message)] if isinstance(message, str) else []

    def result(self, result):
        if result is None:
            return Event("error", body="결과 없음", is_error=True)
        return Event("done", body=result.result, data={"session_id": result.session_id})


class FakeClient:
    instances: list = []
    scripts: list = []

    def __init__(self, options):
        self.options = options
        self.prompts = []
        FakeClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self.prompts.append(prompt)

    async def receive_response(self):
        for item in FakeClient.scripts.pop(0):
            if callable(item):
                await item(self.options)
            else:
                yield item

    async def interrupt(self):
        pass


def _result(text, sid="sess-1"):
    return ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                         num_turns=1, session_id=sid, result=text, total_cost_usd=0.01)


@pytest.fixture
def session():
    FakeClient.instances = []
    FakeClient.scripts = []
    s = AgentSession(Driver(), client_factory=FakeClient)
    yield s
    s.close()


def test_two_turns_reuse_one_client(session):
    FakeClient.scripts = [["첫 답", _result("첫 답")], ["둘째 답", _result("둘째 답", sid="sess-1")]]
    first = list(session.send("q1"))
    second = list(session.send("q2"))
    assert [e.kind for e in first] == ["answer", "done"]
    assert [e.kind for e in second] == ["answer", "done"]
    assert len(FakeClient.instances) == 1
    assert FakeClient.instances[0].prompts == ["q1", "q2"]
    assert session.session_id == "sess-1"
    assert session._driver.turns == 2


def test_question_pauses_turn_until_answer(session):
    box = {}

    async def ask(options):
        box["answers"] = await options["ask_user"]([{"question": "예상 트래픽 규모는?", "header": "규모"}])

    FakeClient.scripts = [["시작", ask, "이어서", _result("MAU 10만 기준 설계")]]
    first = list(session.send("아키텍처 잡아줘"))
    assert [(e.kind, e.body) for e in first] == [("answer", "시작"), ("question", "예상 트래픽 규모는?")]
    assert session.waiting_question == "예상 트래픽 규모는?"

    session.answer("MAU 10만")
    rest = list(session.resume())
    assert [(e.kind, e.body) for e in rest] == [("answer", "이어서"), ("done", "MAU 10만 기준 설계")]
    assert box["answers"] == {"예상 트래픽 규모는?": "MAU 10만"}
    assert session.waiting_question is None


def test_send_while_waiting_is_an_error(session):
    async def ask(options):
        await options["ask_user"]([{"question": "규모?"}])

    FakeClient.scripts = [[ask, _result("x")]]
    list(session.send("q"))
    with pytest.raises(RuntimeError):
        list(session.send("또 질문"))
    session.answer("답")
    list(session.resume())


def test_client_failure_yields_error_and_next_turn_gets_new_client(session):
    async def boom(options):
        raise RuntimeError("Claude Code 프로세스 종료")

    FakeClient.scripts = [[boom], ["복구", _result("복구")]]
    first = list(session.send("q1"))
    assert [e.kind for e in first] == ["error"]
    assert "프로세스 종료" in first[0].body
    second = list(session.send("q2"))
    assert [e.kind for e in second] == ["answer", "done"]
    assert len(FakeClient.instances) == 2
    assert session.lost_context is True


def test_close_cancels_waiting_question_and_stops_thread(session):
    box = {}

    async def ask(options):
        try:
            await options["ask_user"]([{"question": "규모?"}])
        except SessionClosed as e:
            box["closed"] = str(e)
            raise

    FakeClient.scripts = [[ask, _result("x")]]
    list(session.send("q"))
    thread = session._thread
    session.close()
    assert box["closed"]
    assert not thread.is_alive()
    assert session.waiting_question is None
    assert session._loop is None


def test_close_interrupts_turn_in_flight():
    """close() 는 재질문 대기뿐 아니라 진행 중인 턴도 interrupt() 로 끊어야 한다."""
    resume = asyncio.Event()
    interrupted = {"called": False}

    class StuckClient(FakeClient):
        async def interrupt(self):
            interrupted["called"] = True
            resume.set()

    async def wait_for_interrupt(options):
        await resume.wait()

    FakeClient.instances = []
    FakeClient.scripts = [["시작", wait_for_interrupt, _result("중단")]]
    session = AgentSession(Driver(), client_factory=StuckClient)

    it = session.send("q")
    assert next(it).body == "시작"
    thread = session._thread

    session.close()

    assert interrupted["called"]
    assert not thread.is_alive()
    assert session._loop is None

    FakeClient.scripts = [["다음", _result("다음")]]
    second = list(session.send("q2"))
    assert [e.kind for e in second] == ["answer", "done"]
    assert len(FakeClient.instances) == 2

    session.close()


def test_close_before_start_is_noop():
    AgentSession(Driver(), client_factory=FakeClient).close()
