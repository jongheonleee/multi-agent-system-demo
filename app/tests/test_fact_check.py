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


def test_verify_failure_does_not_trigger_rewrite(monkeypatch):
    """검증 실패(모델 응답 잘림 등)는 재작성 사유가 아니다.

    fact_check 가 max_tokens 에 걸려 파싱 실패하면 0.0 을 반환하는데, 이걸
    "부정확한 답변"으로 오인하면 멀쩡한 답변을 재작성시킨다.
    """
    import judge_agent
    import rag_tools
    from state import DomainAnswer

    class FakeTool:
        """StructuredTool 은 pydantic 모델이라 메서드를 못 바꾼다. 통째로 교체한다."""

        @staticmethod
        def invoke(_args):
            return {
                "sentences": [],
                "overall_accuracy": 0.0,
                "overall_accuracy_comment": (
                    f"{rag_tools.VERIFY_FAILED_COMMENT_PREFIX} 응답이 잘림"
                ),
            }

    monkeypatch.setattr(rag_tools, "fact_check_tool", FakeTool)

    original = "원본 병합 답변"
    out = judge_agent._guard(original, [DomainAnswer(domain="ai", question="q", answer="a")])
    assert out == original, "검증 불가일 때는 원본을 그대로 유지해야 합니다"


def test_fact_check_uses_larger_max_tokens():
    import inspect

    import rag_tools

    # @tool 로 감싸져 있으므로 원본 함수(.func)에서 소스를 읽는다.
    src = inspect.getsource(rag_tools.fact_check_tool.func)
    assert "max_tokens=16000" in src
