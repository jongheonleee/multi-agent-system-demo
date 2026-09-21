import os

# 각 에이전트 이름별 할당되는 모델명 기록
_ROLE_DEFAULTS = {
    "orchestrator": "claude-opus-5",  ### 오케스트레이터: OPUS5
    "judge": "claude-opus-5",         ### 판단 에이전트: OPUS5
    "domain": "claude-opus-5",        ### 도메인 에이전트(서버 및 인프라, 시스템 설계, AI): OPUS5
    "trend": "claude-sonnet-5",       ### 트렌드 서치용 에이전트: Sonnet5 
    "default": "claude-sonnet-5",     ### 그외의 Sonnet5
}

# 각 에이전트(이름)에 맞는 모델 ID 문자열 반환
# 설정 파일에 모델 ID 문자열에 맞는 모델명이 정의되어 있음 
def resolve_model_id(role: str = "default") -> str:

    """
    - 역할 이름을 실제 모델 ID 로 해석한다.
    """

    override = os.getenv(f"AGENT_{role.upper()}_MODEL")    
    return override or _ROLE_DEFAULTS.get(role, _ROLE_DEFAULTS["default"])


def get_model(role: str = "default", **overrides):
    """
    - 역할에 맞는 LLM 인스턴스를 반환한다.
    - LITELLM_BASE_URL 이 설정되어 있으면 게이트웨이를 경유한다.
    - 게이트웨이는 OpenAI 호환 API 를 노출하므로 ChatOpenAI 로 붙는다.
    """
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic

    kwargs = {
        "model": resolve_model_id(role), 
        "max_tokens": 4096, 
        **overrides
    }

    base_url = os.getenv("LITELLM_BASE_URL")
    if base_url:
        return ChatOpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=os.getenv("LITELLM_API_KEY", "sk-noop"),
            **kwargs,
        )
    
    return ChatAnthropic(**kwargs)


def get_embeddings():
    """임베딩 모델.

    dimensions 는 Pinecone testdb 인덱스와 같은 512 여야 한다.
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)


def get_original_reranker():
    """Cohere Rerank 모델 인스턴스를 반환."""
    from langchain_cohere import CohereRerank
    return CohereRerank(
        model="rerank-multilingual-v3.0"
    )


# ---------------------------------------------------------------------------
# 대화 요약 메모리
#
# LangChain 1.x 에서 ConversationSummaryBufferMemory 계열은 제거됐다
# (langchain.memory 모듈 자체가 없음). 현행 대체제가 SummarizationMiddleware 다.
# 토큰이 임계치에 닿으면 오래된 메시지를 요약으로 치환하고 최근 메시지는 원문으로
# 남긴다. 외부 저장소는 쓰지 않는다.
#
# 임계치를 낮게 잡으면 조사 도중에 요약이 끼어들어 도구 결과가 날아간다.
# 실측: 옵시디언 MOC 노트 1개 vault_read 가 ~852 토큰. 처음에 8000 으로 뒀더니
# 노트 9~10개만 읽어도 트리거되어, 도메인 에이전트가 "대화 맥락 유실로 조사
# 결과가 반환되지 않았다"고 답하는 일이 생겼다. 모델 컨텍스트가 1M 이므로
# 요약은 "폭주 방지용 안전장치"로만 쓰고 정상 조사 루프에서는 발동하지 않게 한다.
# ---------------------------------------------------------------------------
SUMMARY_TRIGGER_TOKENS = 120_000
SUMMARY_KEEP_MESSAGES = 40


def build_summarizer():
    """도구를 많이 부르는 에이전트에 붙일 요약 미들웨어."""
    from langchain.agents.middleware import SummarizationMiddleware

    return SummarizationMiddleware(
        model=get_model("judge"),
        trigger=("tokens", int(os.getenv("SUMMARY_TRIGGER_TOKENS", SUMMARY_TRIGGER_TOKENS))),
        keep=("messages", int(os.getenv("SUMMARY_KEEP_MESSAGES", SUMMARY_KEEP_MESSAGES))),
    )
