import asyncio

import pytest

import judge_agent
from judge_agent import FACT_CHECK_TOOL, GUARDRAIL_THRESHOLD, Judge, decide
from state import DomainAnswer, TurnState


def run(coro):
    return asyncio.run(coro)


def _answers():
    return [
        DomainAnswer(domain="server_infra", question="게이트웨이는?", answer="Nginx 를 앞단에 둔다."),
        DomainAnswer(domain="ai", question="RAG 는?", answer="Pinecone 에 문서를 색인한다."),
    ]


@pytest.fixture
def report(monkeypatch):
    """fact_check 결과를 정해두고, 넘어온 근거(context)를 기록한다."""
    box = {"report": None, "context": None}

    async def fake(text, context=None):
        box["context"] = context
        return box["report"]

    monkeypatch.setattr(judge_agent, "fact_check", fake)
    return box


def test_threshold_matches_app():
    assert GUARDRAIL_THRESHOLD == 0.8


def test_decide_passes_above_threshold(report):
    report["report"] = {"overall_accuracy": 0.8, "overall_accuracy_comment": "ok"}
    assert run(decide("병합", _answers()))[0] == "PASS"
    # 근거는 도메인 답변들을 이어붙인 것(app/ 의 _guard 와 같다)
    assert report["context"] == "Nginx 를 앞단에 둔다.\n\nPinecone 에 문서를 색인한다."


def test_decide_rewrites_below_threshold(report):
    report["report"] = {"overall_accuracy": 0.5, "overall_accuracy_comment": "근거 부족"}
    assert run(decide("병합", _answers()))[0] == "REWRITE"


def test_verify_failure_does_not_trigger_rewrite(report):
    """검증 실패(모델 응답 잘림 등)는 재작성 사유가 아니다."""
    report["report"] = {"overall_accuracy": 0.0, "overall_accuracy_comment": "[검증불가] 응답이 잘림"}
    assert run(decide("병합", _answers()))[0] == "PASS"


def test_fact_check_tool_counts_and_instructs(report):
    turn = TurnState(domain_answers=_answers())
    tool = Judge(turn).fact_check_tool()
    assert tool.name == "fact_check"

    report["report"] = {"overall_accuracy": 0.3, "overall_accuracy_comment": "3번 문장 근거 없음"}
    out = run(tool.handler({"answer": "병합 답"}))["content"][0]["text"]
    assert out.startswith("REWRITE")
    assert "3번 문장 근거 없음" in out and judge_agent.FIX_RULES in out
    assert turn.fact_checks == 1


def test_record_domain_answer_collects_only_top_level_domain_agents():
    turn = TurnState()
    judge = Judge(turn)
    ok = {"tool_name": "Agent", "tool_input": {"subagent_type": "ai", "prompt": "RAG?"},
          "tool_response": {"content": [{"type": "text", "text": "RAG 답"}]}}
    trend = {"tool_name": "Agent", "tool_input": {"subagent_type": "general_it_trend", "prompt": "q"}, "tool_response": "x"}
    nested = {**ok, "agent_id": "sub-1"}
    for data in (ok, trend, nested):
        run(judge.record_domain_answer(data, "t", None))
    assert [(a.domain, a.question, a.answer) for a in turn.domain_answers] == [("ai", "RAG?", "RAG 답")]


def test_fact_check_is_limited_to_once_per_turn():
    turn = TurnState()
    judge = Judge(turn)
    data = {"tool_name": FACT_CHECK_TOOL, "tool_input": {}}
    assert run(judge.limit_fact_check(data, "t", None)) == {}
    turn.fact_checks = 1
    out = run(judge.limit_fact_check(data, "t", None))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_stop_hook_requires_fact_check_after_multi_domain_merge():
    turn = TurnState(domain_answers=_answers())
    judge = Judge(turn)
    assert run(judge.require_fact_check({"stop_hook_active": False}, None, None))["decision"] == "block"
    # 이미 한 번 막혔으면(무한 루프 방지) 통과
    assert run(judge.require_fact_check({"stop_hook_active": True}, None, None)) == {}
    turn.fact_checks = 1
    assert run(judge.require_fact_check({"stop_hook_active": False}, None, None)) == {}


def test_stop_hook_allows_single_domain_and_clarification():
    """단일 도메인은 합칠 것이 없으므로 검증도 없다(app/ merge 와 같다)."""
    single = Judge(TurnState(domain_answers=_answers()[:1]))
    assert run(single.require_fact_check({"stop_hook_active": False}, None, None)) == {}
    asking = Judge(TurnState(domain_answers=_answers(), clarifying_question="규모?"))
    assert run(asking.require_fact_check({"stop_hook_active": False}, None, None)) == {}


def test_format_answers_matches_app_fallback_format():
    assert judge_agent.format_answers(_answers()) == (
        "### [서버·인프라] 게이트웨이는?\nNginx 를 앞단에 둔다.\n\n### [AI] RAG 는?\nPinecone 에 문서를 색인한다."
    )


def _deleg(domain, agent_id=None):
    data = {"tool_name": "Agent", "tool_input": {"subagent_type": domain, "prompt": "q"}}
    if agent_id:
        data["agent_id"] = agent_id
    return data


def _decision(out):
    return out.get("hookSpecificOutput", {}).get("permissionDecision", "pass")


def test_decomposition_is_one_round_per_domain():
    """실측: 규칙을 프롬프트로만 주면 같은 도메인에 보충 위임을 반복했다(5회)."""
    turn = TurnState(domain_answers=[_answers()[0]])  # server_infra 답 있음
    judge = Judge(turn)
    assert _decision(run(judge.limit_decomposition(_deleg("server_infra"), "t", None))) == "deny"
    assert _decision(run(judge.limit_decomposition(_deleg("ai"), "t", None))) == "pass"


def test_failed_delegation_can_be_retried():
    """위임이 실패하면 답이 없으므로 같은 도메인에 다시 맡길 수 있다(4096 초과 복구 실측)."""
    judge = Judge(TurnState())
    assert _decision(run(judge.limit_decomposition(_deleg("ai"), "t", None))) == "pass"


def test_at_most_three_domains_and_trend_is_not_limited():
    turn = TurnState(domain_answers=[
        DomainAnswer(domain="server_infra", question="q", answer="a"),
        DomainAnswer(domain="system_design", question="q", answer="b"),
        DomainAnswer(domain="ai", question="q", answer="c"),
    ])
    judge = Judge(turn)
    assert _decision(run(judge.limit_decomposition(_deleg("ai"), "t", None))) == "deny"
    assert _decision(run(judge.limit_decomposition(_deleg("general_it_trend"), "t", None))) == "pass"
    # 서브에이전트 안에서의 호출은 대상이 아니다
    assert _decision(run(judge.limit_decomposition(_deleg("ai", agent_id="sub"), "t", None))) == "pass"


# ---------------------------------------------------------------------------
# Guardrail — Stop 훅 하나로 "1회 재작성"
# ---------------------------------------------------------------------------
from judge_agent import Guardrail, last_assistant_text, response_text  # noqa: E402


def _guard(turn):
    return Guardrail(lambda: turn)


def test_response_text_handles_str_blocks_and_message_dict():
    assert response_text("x") == "x"
    assert response_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert response_text({"content": [{"type": "text", "text": "c"}]}) == "c"
    assert response_text({"role": "assistant", "content": [{"type": "text", "text": "d"}]}) == "d"
    assert response_text({"message": {"role": "assistant", "content": [{"type": "text", "text": "e"}]}}) == "e"
    assert response_text(None) == ""


def test_last_assistant_text_prefers_hook_field_then_transcript(tmp_path):
    assert last_assistant_text({"last_assistant_message": "최종 답"}) == "최종 답"
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        '{"type":"user","message":{"role":"user","content":"q"}}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"첫 답"}]}}\n'
        'not json\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"마지막 답"}]}}\n',
        encoding="utf-8",
    )
    assert last_assistant_text({"transcript_path": str(transcript)}) == "마지막 답"
    assert last_assistant_text({}) == ""


def test_guardrail_collects_only_top_level_domain_and_trend_answers():
    turn = TurnState()
    g = _guard(turn)
    ok = {"tool_name": "Agent", "tool_input": {"subagent_type": "ai", "prompt": "RAG?"},
          "tool_response": {"content": [{"type": "text", "text": "RAG 답"}]}}
    trend = {"tool_name": "Agent", "tool_input": {"subagent_type": "general_it_trend", "prompt": "q"}, "tool_response": "웹 답"}
    nested = {**ok, "agent_id": "sub-1"}
    for data in (ok, trend, nested):
        run(g.record_subagent_answer(data, "t", None))
    assert [(a.domain, a.question, a.answer) for a in turn.domain_answers] == [("ai", "RAG?", "RAG 답")]
    assert turn.trend_answers == ["웹 답"]


def test_stop_hook_passes_with_fewer_than_two_domain_answers(report):
    turn = TurnState(domain_answers=_answers()[:1])
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": False, "last_assistant_message": "답"}, None, None)) == {}
    assert report["context"] is None  # 팩트체크를 부르지 않았다
    assert turn.fact_check is None


def test_stop_hook_blocks_once_with_fix_rules_when_below_threshold(report):
    report["report"] = {"overall_accuracy": 0.3, "overall_accuracy_comment": "3번 문장 근거 없음"}
    turn = TurnState(domain_answers=_answers())
    out = run(_guard(turn).fact_check_on_stop({"stop_hook_active": False, "last_assistant_message": "병합 답"}, None, None))
    assert out["decision"] == "block"
    assert judge_agent.FIX_RULES in out["reason"] and "3번 문장 근거 없음" in out["reason"]
    assert report["context"] == "Nginx 를 앞단에 둔다.\n\nPinecone 에 문서를 색인한다."
    assert turn.fact_check == {"verdict": "REWRITE", "score": 0.3, "comment": "3번 문장 근거 없음"}
    # 재작성 뒤의 두 번째 Stop 은 stop_hook_active 라서 통과 (SDK 내장 1회 의미론)
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": True, "last_assistant_message": "재작성"}, None, None)) == {}


def test_stop_hook_passes_when_accurate_or_unverifiable(report):
    turn = TurnState(domain_answers=_answers())
    report["report"] = {"overall_accuracy": 0.9, "overall_accuracy_comment": "ok"}
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": False, "last_assistant_message": "답"}, None, None)) == {}
    assert turn.fact_check["verdict"] == "PASS"
    report["report"] = {"overall_accuracy": 0.0, "overall_accuracy_comment": "[검증불가] 응답이 잘림"}
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": False, "last_assistant_message": "답"}, None, None)) == {}


def test_stop_hook_passes_when_final_text_is_empty(report):
    turn = TurnState(domain_answers=_answers())
    assert run(_guard(turn).fact_check_on_stop({"stop_hook_active": False}, None, None)) == {}
    assert report["context"] is None
