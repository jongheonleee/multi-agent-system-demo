import operator

from state import DomainAnswer, DomainTaskState, MainQueryState, SubQuestion


def test_sub_question_shape():
    sq = SubQuestion(domain="server_infra", question="커넥션 풀은?", reason="인프라 영역")
    assert sq.domain == "server_infra"


def test_domain_answers_uses_add_reducer():
    # 병렬 노드가 동시에 append 하므로 리듀서가 operator.add 여야 한다.
    field = MainQueryState.model_fields["domain_answers"]
    assert field.metadata and field.metadata[0] is operator.add


def test_domain_task_state_carries_one_sub_question():
    st = DomainTaskState(
        sub_question=SubQuestion(domain="ai", question="RAG?", reason="AI 영역"),
        original_question="전체 질문",
    )
    assert st.sub_question.domain == "ai"


def test_main_state_defaults():
    s = MainQueryState()
    assert s.sub_questions == []
    assert s.domain_answers == []
    assert s.merged_answer == ""
    assert s.clarification == ""
