from state import MainQueryState, SubQuestion


def test_graph_has_expected_nodes():
    from main_agent import main_graph

    nodes = set(main_graph.get_graph().nodes)
    for name in [
        "orchestrator",
        "decompose",
        "domain_worker",
        "merge",
        "general_it_topic_agent",
        "general_topic_agent",
        "finalize",
    ]:
        assert name in nodes, f"{name} 누락: {nodes}"


def test_fanout_emits_one_send_per_sub_question():
    from main_agent import fan_out_domains

    state = MainQueryState(
        sub_questions=[
            SubQuestion(domain="server_infra", question="a", reason="r"),
            SubQuestion(domain="ai", question="b", reason="r"),
        ]
    )
    sends = fan_out_domains(state)
    assert len(sends) == 2
    assert {s.node for s in sends} == {"domain_worker"}


def test_fanout_with_no_sub_questions_goes_to_merge():
    from main_agent import fan_out_domains

    assert fan_out_domains(MainQueryState()) == "merge"


def test_route_by_intent_covers_all_three():
    from main_agent import route_by_intent

    assert route_by_intent(MainQueryState(query_type="complex_discusion_topic")) == "decompose"
    assert (
        route_by_intent(MainQueryState(query_type="general_it_trend_topic"))
        == "general_it_topic_agent"
    )
    assert route_by_intent(MainQueryState(query_type="unknown")) == "general_topic_agent"


def test_nodes_return_partial_dicts_not_full_state():
    """노드가 state 객체를 통째로 반환하면 안 된다.

    pydantic state 를 그대로 돌려주면 LangGraph 가 모든 필드에 리듀서를 다시
    적용해서, operator.add 를 쓰는 domain_answers 가 노드를 지날 때마다
    중복 누적된다(실제로 3개가 12개가 됐다).
    """
    import inspect

    import main_agent
    import orchestrator

    nodes = [
        main_agent.decompose_node,
        main_agent.merge_node,
        main_agent.general_it_node,
        main_agent.general_node,
        main_agent.finalize_node,
        orchestrator.orchestrate,
    ]
    for fn in nodes:
        src = inspect.getsource(fn)
        assert "return state\n" not in src, f"{fn.__name__} 이 state 를 통째로 반환합니다"
        # from __future__ import annotations 때문에 어노테이션이 문자열로 남는다.
        ann = inspect.signature(fn).return_annotation
        assert ann in (dict, "dict"), f"{fn.__name__} 의 반환 타입은 dict 여야 합니다 (got {ann!r})"


def test_domain_answers_accumulate_once_per_worker():
    """리듀서가 도메인당 정확히 1개씩만 쌓는지 확인한다."""
    import operator

    from state import DomainAnswer

    existing = [DomainAnswer(domain="ai", question="q", answer="a")]
    incoming = [DomainAnswer(domain="server_infra", question="q2", answer="a2")]
    merged = operator.add(existing, incoming)
    assert len(merged) == 2
