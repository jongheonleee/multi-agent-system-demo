def test_summarizer_is_configured():
    from langchain.agents.middleware import SummarizationMiddleware

    from models import build_summarizer

    assert isinstance(build_summarizer(), SummarizationMiddleware)


def test_domain_agent_wires_summarization():
    import inspect

    import domain_agents

    src = inspect.getsource(domain_agents.build_domain_agent)
    assert "middleware" in src
    assert "build_summarizer" in src
