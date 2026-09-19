import pytest

from domain_agents import DOMAIN_PROMPTS


def test_all_three_domains_have_prompts():
    assert set(DOMAIN_PROMPTS) == {"server_infra", "system_design", "ai"}


def test_vault_domains_mention_their_moc_entrypoints():
    assert "쿠버네티스 MOC" in DOMAIN_PROMPTS["server_infra"]
    assert "시스템 아키텍처 MOC" in DOMAIN_PROMPTS["system_design"]
    assert "RAG & AI Agent MOC" in DOMAIN_PROMPTS["ai"]


def test_ai_prompt_mentions_paper_search():
    assert "vector_search_tool" in DOMAIN_PROMPTS["ai"]


def test_prompts_forbid_writing_to_vault():
    """읽기 전용 정책을 프롬프트에도 명시해 이중으로 막는다."""
    for domain, prompt in DOMAIN_PROMPTS.items():
        assert "읽기" in prompt, domain


@pytest.mark.llm
def test_run_domain_returns_reducer_shaped_dict():
    from domain_agents import run_domain
    from state import DomainTaskState, SubQuestion

    out = run_domain(
        DomainTaskState(
            sub_question=SubQuestion(
                domain="ai", question="RAG 와 롱컨텍스트의 차이는?", reason="AI"
            ),
            original_question="RAG 와 롱컨텍스트의 차이는?",
        ),
        {},
    )
    # operator.add 리듀서에 맞춰 리스트로 감싸 반환해야 한다.
    assert isinstance(out["domain_answers"], list)
    assert out["domain_answers"][0].domain == "ai"
    assert out["domain_answers"][0].answer
