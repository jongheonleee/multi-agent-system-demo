import pytest


def test_fact_check_without_context_returns_unverifiable():
    """출처가 없으면 검증 불가여야 한다.

    예전에는 질문과 무관한 Gemini 논문 스니펫을 하드코딩해 대신 썼는데,
    그러면 엉뚱한 문서로 검증이 통과해 guardrail 이 무의미해진다.
    """
    from rag_tools import fact_check_tool

    out = fact_check_tool.invoke({"text": "아무 주장", "context": None})
    assert out["overall_accuracy"] == 0.0
    assert out["sentences"] == []


def test_no_hardcoded_gemini_snippet_in_source():
    import inspect

    import rag_tools

    src = inspect.getsource(rag_tools)
    assert "본 리포트에서 소개하는" not in src


def test_guard_is_wired_into_merge():
    import inspect

    import judge_agent

    assert "_guard(merged, answers)" in inspect.getsource(judge_agent.merge)


@pytest.mark.llm
def test_guard_passes_grounded_answer():
    from judge_agent import _guard
    from state import DomainAnswer

    answers = [
        DomainAnswer(domain="ai", question="q", answer="Pinecone 에 문서를 색인한다."),
        DomainAnswer(domain="server_infra", question="q2", answer="Nginx 를 앞단에 둔다."),
    ]
    merged = "Pinecone 에 문서를 색인하고 Nginx 를 앞단에 둔다."
    out = _guard(merged, answers)
    assert out  # 통과하든 재작성되든 문자열이 나와야 한다
