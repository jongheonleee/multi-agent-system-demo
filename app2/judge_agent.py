"""판단 Agent — 질문 분해, 결과 병합, guardrail.

app/ 은 분해(decompose)와 병합(merge)을 판단 Agent 의 LLM 호출 두 번으로 하고,
그 사이를 LangGraph 가 이어준다. Agent SDK 에서는 오케스트레이터(Opus)가 직접
판단한다. 분해 = "어느 서브에이전트에 어떤 서브 질문을 줄지" 정해 Agent 도구를
부르는 것이고, 병합 = 돌아온 답들로 최종 답을 쓰는 것이다. 규칙 문장은 app/ 의
DECOMPOSE_PROMPT / MERGE_PROMPT 에서 그대로 가져왔다.

guardrail 은 코드가 강제한다(모델이 잊어도 빠지지 않게).
  PostToolUse 훅  서브에이전트가 돌려준 답을 TurnState 에 모은다 (fact_check 의 근거)
  fact_check 도구  병합 답을 도메인 답과 대조 -> 통과 / 1회 재작성 지시
  PreToolUse 훅   도메인 위임은 도메인당 1회, 최대 3개 (app/ decompose 의 [:3] 과 같은 규칙)
  PreToolUse 훅   fact_check 는 턴당 1회만 (재작성 반복 금지 — app/ 과 같은 규칙)
  Stop 훅         도메인 답이 2개 이상인데 검증 없이 끝내려 하면 막는다
"""
from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import tool

from domain_agents import DOMAINS
from general_it_trend_topic_agent import TREND_AGENT
from rag_tools import VERIFY_FAILED_COMMENT_PREFIX, fact_check, text_result
from state import DomainAnswer, TurnState

logger = logging.getLogger(__name__)

JUDGE_SERVER = "judge"
FACT_CHECK_TOOL = f"mcp__{JUDGE_SERVER}__fact_check"
AGENT_TOOL_NAMES = ("Agent", "Task")

GUARDRAIL_THRESHOLD = 0.8

# app/ DECOMPOSE_PROMPT 의 도메인 정의와 규칙
DECOMPOSE_RULES = """> 도메인 정의 (서로 배타적으로 적용할 것)
- server_infra: **운영 중인 시스템을 돌리는 층**.
  배포, 쿠버네티스/컨테이너, 네트워크, DB 운영·튜닝, 모니터링, 장애 진단, 용량 산정
- system_design: **코드와 데이터의 구조를 정하는 층**.
  서비스 경계, 모듈 분리, API 계약, 데이터 모델링, 일관성/확장성 트레이드오프
- ai: **모델과 그 주변**.
  LLM, RAG, 에이전트, 모델 서빙, 임베딩, 벡터 검색, 프롬프트

경계가 애매하면 이렇게 가르세요.
"지금 돌아가는 걸 고치는가" -> server_infra
"무엇을 만들지 정하는가"   -> system_design

> 규칙
- 1. 질문에 실제로 포함된 도메인만 만듭니다. 억지로 3개를 채우지 마세요.
- 2. 단일 도메인 질문이면 서브 질문 1개만 반환합니다.
- 3. 각 서브 질문은 다른 도메인의 답을 몰라도 **독립적으로 답할 수 있어야** 합니다.
     "위 결과를 바탕으로" 같은 의존 표현을 쓰지 마세요.
- 4. 서브 질문은 최대 3개입니다.
- 5. 원 질문의 제약 조건(규모, 트래픽, 기술 스택)은 각 서브 질문에 그대로 복사해 넣습니다.
"""

# app/ MERGE_PROMPT 의 병합 규칙
MERGE_RULES = """> 병합 규칙
- 1. 원 질문에 답하는 **하나의 글**로 씁니다. 도메인별로 단순 나열하지 마세요.
- 2. 여러 도메인이 같은 말을 하면 한 번만 씁니다.
- 3. 도메인 간 권고가 **상충하면** 숨기지 말고 드러낸 뒤, 어떤 조건에서 무엇을 택할지 씁니다.
- 4. 전문가가 "찾지 못했다"고 한 부분은 그대로 한계로 밝힙니다. 지어내지 마세요.
- 5. 전문가 답변에 없는 내용을 새로 추가하지 마세요.
- 6. 각 전문가의 Sources 를 모아 말미에 "Sources" 섹션으로 통합합니다.
- 7. 한국어로 씁니다.
"""

# app/ FIX_PROMPT
FIX_RULES = """아래 답변에서 출처가 뒷받침하지 않는 문장을 제거하거나 완화해 다시 쓰세요.
내용을 새로 추가하지 말고, 근거가 약한 부분만 손보세요."""

_DOMAIN_LABELS = {
    "server_infra": "서버·인프라",
    "system_design": "시스템 설계",
    "ai": "AI",
}


def format_answers(answers: list[DomainAnswer]) -> str:
    """병합 실패 시 원본을 이어붙이는 형식(app/ 과 같다)."""
    return "\n\n".join(
        f"### [{_DOMAIN_LABELS.get(a.domain, a.domain)}] {a.question}\n{a.answer}" for a in answers
    )


def _response_text(response: Any) -> str:
    """Agent 도구 결과(서브에이전트의 최종 답)에서 텍스트만 뽑는다."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "content" in response:
            return _response_text(response["content"])
        if "result" in response:
            return _response_text(response["result"])
        return response.get("text", "")
    if isinstance(response, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in response
            if not isinstance(b, dict) or b.get("type", "text") == "text"
        )
    return str(response or "")


async def decide(answer: str, answers: list[DomainAnswer]) -> tuple[str, dict]:
    """병합 답을 검증해 PASS / REWRITE 를 정한다. app/ 의 _guard 와 같은 분기."""
    context = "\n\n".join(a.answer for a in answers)
    report = await fact_check(answer, context)

    comment = report.get("overall_accuracy_comment", "")
    if comment.startswith(VERIFY_FAILED_COMMENT_PREFIX):
        # 검증 자체가 안 된 것이지 답변이 틀린 게 아니다. 재작성하면 안 된다.
        logger.warning("guardrail 검증 불가, 원본을 유지합니다: %s", comment)
        return "PASS", report

    score = report.get("overall_accuracy", 1.0)
    if score >= GUARDRAIL_THRESHOLD:
        logger.info("guardrail 통과 (%.2f)", score)
        return "PASS", report

    logger.info("guardrail 미달(%.2f), 1회 재작성합니다", score)
    return "REWRITE", report


class Judge:
    """한 턴의 판단 guardrail. 훅과 도구가 같은 TurnState 를 공유한다."""

    def __init__(self, turn: TurnState) -> None:
        self.turn = turn

    # ---- 훅 --------------------------------------------------------------
    async def record_domain_answer(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        """PostToolUse(Agent). 서브에이전트가 돌려준 답을 모은다(실패한 위임은 PostToolUse 가 안 온다)."""
        if input_data.get("agent_id"):
            return {}
        tool_input = input_data.get("tool_input") or {}
        agent = tool_input.get("subagent_type")
        text = _response_text(input_data.get("tool_response"))
        if agent in DOMAINS:
            self.turn.domain_answers.append(
                DomainAnswer(domain=agent, question=tool_input.get("prompt", ""), answer=text)
            )
        elif agent == TREND_AGENT:
            self.turn.trend_answers.append(text)
        return {}

    async def limit_decomposition(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        """PreToolUse(Agent). 분해는 한 번 — 도메인당 답 1개, 도메인 최대 3개.

        실측: 프롬프트 규칙만 주면 답을 받은 뒤 같은 도메인에 보충 조사를 다시 맡겨
        위임이 5회까지 늘었다. app/ 은 분해를 한 번만 하고 [:3] 으로 자른다.
        위임이 실패한 도메인(답 없음)은 다시 맡길 수 있다.
        """
        if input_data.get("tool_name") not in AGENT_TOOL_NAMES or input_data.get("agent_id"):
            return {}
        domain = (input_data.get("tool_input") or {}).get("subagent_type")
        if domain not in DOMAINS:
            return {}
        answered = {a.domain for a in self.turn.domain_answers}
        reason = None
        if domain in answered:
            reason = f"{domain} 서브에이전트의 답은 이미 받았습니다. 받은 답으로 병합하세요."
        elif len(answered) >= 3:
            reason = "서브 질문은 최대 3개입니다. 받은 답으로 병합하세요."
        if reason:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        return {}

    async def limit_fact_check(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        """PreToolUse(fact_check). 검증과 재작성은 턴당 1회만."""
        if input_data.get("tool_name") == FACT_CHECK_TOOL and self.turn.fact_checks >= 1:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "검증은 턴당 1회만 합니다. 지금 답변을 최종 답변으로 내세요.",
                }
            }
        return {}

    async def require_fact_check(self, input_data: dict, tool_use_id: str | None, context: Any) -> dict:
        """Stop. 도메인 답이 2개 이상이면 병합 답을 검증하기 전에는 끝내지 못한다."""
        if input_data.get("stop_hook_active"):
            return {}
        if len(self.turn.domain_answers) >= 2 and self.turn.fact_checks == 0 and not self.turn.clarifying_question:
            return {
                "decision": "block",
                "reason": (
                    "여러 도메인의 답을 병합했습니다. 최종 답변을 내기 전에 병합한 답변 전문으로 "
                    "fact_check 를 호출하세요."
                ),
            }
        return {}

    # ---- 도구 ------------------------------------------------------------
    def fact_check_tool(self):
        judge = self

        @tool(
            "fact_check",
            (
                "병합한 답변을 도메인 전문가들의 답변과 대조해 문장 단위로 사실 검증한다. "
                "여러 도메인의 답을 병합했다면 최종 답변을 내기 전에 반드시 한 번 호출한다."
            ),
            {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        )
        async def fact_check_merged(args: dict[str, Any]) -> dict[str, Any]:
            judge.turn.fact_checks += 1
            verdict, report = await decide(args["answer"], judge.turn.domain_answers)
            judge.turn.checked_answer, judge.turn.verdict = args["answer"], verdict
            if verdict == "PASS":
                return text_result("PASS — 이 답변이 그대로 사용자에게 전달됩니다. \"전달 완료\" 라고만 쓰고 끝내세요.")
            return text_result(
                "REWRITE — 검증 결과 근거가 약한 문장이 있습니다. 한 번만 다시 쓴 뒤 그것을 최종 답변으로 내세요.\n\n"
                f"{FIX_RULES}\n\n> 검증 결과\n{report.get('overall_accuracy_comment', '')}"
            )

        return fact_check_merged
