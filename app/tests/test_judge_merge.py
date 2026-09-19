import pytest

from state import DomainAnswer


def test_merge_single_answer_passes_through():
    from judge_agent import merge

    answers = [DomainAnswer(domain="ai", question="q", answer="원본 답변 본문")]
    # 단일 도메인이면 합칠 것이 없으므로 LLM 을 태우지 않는다.
    assert merge("q", answers) == "원본 답변 본문"


def test_merge_empty_returns_message():
    from judge_agent import merge

    assert "답변" in merge("q", [])


@pytest.mark.llm
def test_merge_combines_two_domains():
    from judge_agent import merge

    answers = [
        DomainAnswer(
            domain="server_infra", question="게이트웨이는?", answer="Nginx 를 앞단에 둔다."
        ),
        DomainAnswer(domain="ai", question="RAG 는?", answer="Pinecone 에 문서를 색인한다."),
    ]
    merged = merge("사내 챗봇 아키텍처", answers)
    assert "Nginx" in merged or "게이트웨이" in merged
    assert "Pinecone" in merged or "RAG" in merged
