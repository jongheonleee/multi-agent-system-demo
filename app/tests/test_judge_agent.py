import pytest

from judge_agent import DecomposeOutput
from state import SubQuestion


def test_decompose_output_accepts_multiple_domains():
    out = DecomposeOutput(
        sub_questions=[
            SubQuestion(domain="server_infra", question="a", reason="r"),
            SubQuestion(domain="ai", question="b", reason="r"),
        ]
    )
    assert {s.domain for s in out.sub_questions} == {"server_infra", "ai"}


@pytest.mark.llm
def test_decompose_splits_mixed_question():
    from judge_agent import decompose

    subs = decompose(
        "MAU 10만 사내 챗봇의 전체 아키텍처(게이트웨이, 대화 이력 저장, RAG, 모델 서빙, 모니터링)를 잡는다면"
    )
    domains = {s.domain for s in subs}
    assert len(subs) >= 2
    assert "ai" in domains
    assert domains & {"server_infra", "system_design"}


@pytest.mark.llm
def test_decompose_single_domain_returns_one():
    from judge_agent import decompose

    subs = decompose("DB 커넥션 풀 고갈을 어떻게 진단하나?")
    assert len(subs) == 1
    assert subs[0].domain in {"server_infra", "system_design"}
