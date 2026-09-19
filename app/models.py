import os

# 역할별 기본 모델. 환경변수 AGENT_<ROLE>_MODEL 로 덮어쓸 수 있다.
_ROLE_DEFAULTS = {
    "orchestrator": "claude-opus-5",
    "judge": "claude-opus-5",
    "domain": "claude-sonnet-5",
    "trend": "claude-sonnet-5",
    "default": "claude-sonnet-5",
}


def resolve_model_id(role: str = "default") -> str:
    """역할 이름을 실제 모델 ID 로 해석한다."""
    override = os.getenv(f"AGENT_{role.upper()}_MODEL")
    if override:
        return override
    return _ROLE_DEFAULTS.get(role, _ROLE_DEFAULTS["default"])


def get_model(role: str = "default", **overrides):
    """역할에 맞는 LLM 인스턴스를 반환한다.

    LITELLM_BASE_URL 이 설정되어 있으면 게이트웨이를 경유한다.
    게이트웨이는 OpenAI 호환 API 를 노출하므로 ChatOpenAI 로 붙는다.
    """
    kwargs = {"model": resolve_model_id(role), "max_tokens": 4096, **overrides}

    base_url = os.getenv("LITELLM_BASE_URL")
    if base_url:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=f"{base_url.rstrip('/')}/v1",
            api_key=os.getenv("LITELLM_API_KEY", "sk-noop"),
            **kwargs,
        )

    from langchain_anthropic import ChatAnthropic

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

    return CohereRerank(model="rerank-multilingual-v3.0")
