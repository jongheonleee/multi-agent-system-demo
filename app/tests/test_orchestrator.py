import pytest

from orchestrator import IntentOutput


def test_intent_output_requires_clarifying_question_when_flagged():
    out = IntentOutput(
        intent="complex_discusion_topic",
        needs_clarification=True,
        clarifying_question="예상 트래픽 규모가 어느 정도인가요?",
        reason="규모 정보 없음",
    )
    assert out.needs_clarification
    assert out.clarifying_question


def test_intent_output_defaults_no_clarification():
    out = IntentOutput(intent="unknown", reason="인사말")
    assert out.needs_clarification is False
    assert out.clarifying_question == ""


@pytest.mark.llm
def test_specific_question_does_not_trigger_clarification():
    from orchestrator import classify

    out = classify("JWT와 세션 인증의 차이점이 뭔가?")
    assert out.intent == "general_it_trend_topic"
    assert out.needs_clarification is False


@pytest.mark.llm
def test_complex_specific_question_no_clarification():
    from orchestrator import classify

    out = classify(
        "트래픽 급증시 p99 지연이 튀는데, DB 커넥션 풀 고갈인지 파드 CPU throttling 인지 어떻게 구분하는가?"
    )
    assert out.intent == "complex_discusion_topic"
    assert out.needs_clarification is False


@pytest.mark.llm
def test_vague_design_request_triggers_clarification():
    from orchestrator import classify

    out = classify("우리 서비스 아키텍처 좀 잡아줘")
    assert out.needs_clarification is True
