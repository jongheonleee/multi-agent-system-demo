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


def test_general_it_agent_wires_summarization():
    import inspect

    import general_it_trend_topic_agent as m

    src = inspect.getsource(m)
    assert "middleware = [build_summarizer()]" in src


def test_summary_trigger_is_not_too_low():
    """낮은 임계치는 조사 도중 도구 결과를 날려버린다.

    옵시디언 MOC 노트 1개가 ~852 토큰이라, 8000 이면 9~10개만 읽어도 발동한다.
    """
    from models import SUMMARY_TRIGGER_TOKENS

    assert SUMMARY_TRIGGER_TOKENS >= 100_000
