"""대화 요약 메모리.

LangChain 1.x 에서 ConversationSummaryBufferMemory 계열은 제거됐다
(langchain.memory 모듈 자체가 없음). 현행 대체제인 SummarizationMiddleware 를 쓴다.

토큰이 임계치에 닿으면 오래된 메시지를 요약으로 치환하고 최근 메시지는 원문으로
남긴다. AI/Tool 메시지 쌍이 끊기지 않게 처리하며, 요약 호출이 실패하면 3회 재시도
후 에러를 올린다(요약을 날조하지 않는다).

외부 저장소는 쓰지 않는다. 세션 상태는 LangGraph checkpointer 가 들고 있다.
"""
from __future__ import annotations

import os

from langchain.agents.middleware import SummarizationMiddleware

from models import get_model

# 임계치를 낮게 잡으면 조사 도중에 요약이 끼어들어 도구 결과가 날아간다.
#
# 실측: 옵시디언 MOC 노트 1개 vault_read 가 ~852 토큰. 처음에 8000 으로 뒀더니
# 노트 9~10개만 읽어도 트리거되어, 도메인 에이전트가 "대화 맥락 유실로 조사
# 결과가 반환되지 않았다"고 답하는 일이 생겼다.
#
# 모델 컨텍스트가 1M 이므로 요약은 "폭주 방지용 안전장치"로만 쓴다.
# 정상적인 조사 루프에서는 발동하지 않아야 한다.
DEFAULT_TRIGGER_TOKENS = 120_000
DEFAULT_KEEP_MESSAGES = 40


def build_summarizer() -> SummarizationMiddleware:
    """도구를 많이 부르는 에이전트에 붙일 요약 미들웨어."""
    return SummarizationMiddleware(
        model=get_model("judge"),
        trigger=("tokens", int(os.getenv("SUMMARY_TRIGGER_TOKENS", DEFAULT_TRIGGER_TOKENS))),
        keep=("messages", int(os.getenv("SUMMARY_KEEP_MESSAGES", DEFAULT_KEEP_MESSAGES))),
    )
