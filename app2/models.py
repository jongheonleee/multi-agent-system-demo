"""역할별 모델 배정과 Claude Code 실행 환경.

app/models.py 는 역할마다 LangChain 모델 인스턴스(ChatAnthropic/ChatOpenAI)를 만든다.
Agent SDK 에서는 모델이 "인스턴스"가 아니라 에이전트 정의의 한 속성이다.

  메인 에이전트(오케스트레이터)   ClaudeAgentOptions(model=resolve_model_id("orchestrator"))
  서브에이전트(도메인/트렌드)     AgentDefinition(model=resolve_model_id("domain"))
  도구 안의 단발 LLM 호출         query(options=one_shot_options("judge"))

역할 이름, 기본 모델, AGENT_<ROLE>_MODEL 환경변수 덮어쓰기 규칙은 app/ 과 같다.
"""
import os
import tempfile
from typing import Any

# 각 에이전트 이름별 할당되는 모델명 기록
_ROLE_DEFAULTS = {
    "orchestrator": "claude-opus-5",  ### 오케스트레이터: OPUS5
    "judge": "claude-opus-5",         ### 판단 에이전트: OPUS5
    "domain": "claude-opus-5",        ### 도메인 에이전트(서버 및 인프라, 시스템 설계, AI): OPUS5
    "trend": "claude-sonnet-5",       ### 트렌드 서치용 에이전트: Sonnet5
    "default": "claude-sonnet-5",     ### 그외의 Sonnet5
}

# app/ 의 get_model() 기본 max_tokens 와 같다. 도구 안의 단발 호출에 쓴다.
DEFAULT_MAX_TOKENS = 4096

# 에이전트 세션(오케스트레이터·서브에이전트)의 출력 한도.
# app/ 은 4096 에서 조용히 잘린 답을 돌려주지만, Claude Code 는 한도 초과를 에러로 보고
# 서브에이전트를 종료시킨다(실측: "response exceeded the 4096 output token maximum").
# adaptive thinking 이 한도를 함께 쓰므로 fact_check 와 같은 16000 으로 둔다.
AGENT_MAX_TOKENS = 16000


def resolve_model_id(role: str = "default") -> str:
    """역할 이름을 실제 모델 ID 로 해석한다."""
    override = os.getenv(f"AGENT_{role.upper()}_MODEL")
    return override or _ROLE_DEFAULTS.get(role, _ROLE_DEFAULTS["default"])


# ---------------------------------------------------------------------------
# 대화 요약(컨텍스트 관리)
#
# app/ 은 SummarizationMiddleware 를 에이전트마다 붙인다. Claude Code 는 자동
# 컴팩션(auto-compact)을 내장하고 있어서 따로 붙일 것이 없다 — 임계치만 맞춘다.
#
# 임계치를 낮게 잡으면 조사 도중에 요약이 끼어들어 도구 결과가 날아간다.
# 실측: 옵시디언 MOC 노트 1개 vault_read 가 ~852 토큰. 8000 으로 두면 노트 9~10개만
# 읽어도 발동한다. 요약은 "폭주 방지용 안전장치"로만 쓴다.
# ---------------------------------------------------------------------------
SUMMARY_TRIGGER_TOKENS = 120_000


def summary_trigger_tokens() -> int:
    return int(os.getenv("SUMMARY_TRIGGER_TOKENS", SUMMARY_TRIGGER_TOKENS))


# Claude Code 가 도는 작업 디렉터리. 파일 도구는 꺼져 있지만, 세션 기록이
# 이 디렉터리 기준으로 저장되므로 앱 전용 디렉터리를 쓴다.
AGENT_CWD = os.path.join(tempfile.gettempdir(), "app2-agent")


def gateway_env() -> dict[str, str]:
    """LITELLM_BASE_URL 이 있으면 모든 호출이 게이트웨이를 경유하게 한다(app/ 과 같은 규칙).

    app/ 은 게이트웨이의 OpenAI 호환 API 로 붙지만, Claude Code 는 Anthropic
    Messages API(/v1/messages)를 쓴다. LiteLLM 은 같은 포트에서 둘 다 노출한다.
    """
    base_url = os.getenv("LITELLM_BASE_URL")
    if not base_url:
        return {}
    key = os.getenv("LITELLM_API_KEY", "sk-noop")
    return {"ANTHROPIC_BASE_URL": base_url.rstrip("/"), "ANTHROPIC_API_KEY": key, "ANTHROPIC_AUTH_TOKEN": key}


def base_env(max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, str]:
    """모든 Claude Code 프로세스에 공통으로 넘기는 환경변수."""
    env = {
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_tokens),
        # 서브에이전트를 백그라운드로 돌리면 결과가 오기 전에 턴이 끝난다. 동기 실행으로 고정.
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        # general-purpose 같은 내장 서브에이전트는 이 앱에 없다.
        "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(summary_trigger_tokens()),
        "CLAUDE_AGENT_SDK_CLIENT_APP": "toyproject-app2/1.0",
    }
    env.update(gateway_env())
    return env


def isolated_options(**kwargs: Any):
    """사용자 PC 의 Claude Code 설정(CLAUDE.md, 훅, 플러그인, 스킬)을 끌어오지 않는 옵션."""
    from claude_agent_sdk import ClaudeAgentOptions

    os.makedirs(AGENT_CWD, exist_ok=True)
    kwargs.setdefault("env", base_env())
    return ClaudeAgentOptions(
        setting_sources=[],
        skills=[],
        strict_mcp_config=True,
        cwd=AGENT_CWD,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 단발 LLM 호출 — 도구 안에서 쓰는 요약/확장/검증
# ---------------------------------------------------------------------------
class LLMCallError(RuntimeError):
    pass


def one_shot_options(role: str, *, max_tokens: int = DEFAULT_MAX_TOKENS, output_format: dict | None = None):
    """도구 없이 한 번 묻고 답을 받는 query() 옵션."""
    return isolated_options(
        model=resolve_model_id(role),
        tools=[],
        output_format=output_format,
        env=base_env(max_tokens),
        extra_args={"no-session-persistence": None},
    )


async def ask(role: str, prompt: str, *, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """app/ 의 (PROMPT | get_model(role)).invoke(...).text 에 해당."""
    from claude_agent_sdk import ResultMessage, query

    result = None
    async for message in query(prompt=prompt, options=one_shot_options(role, max_tokens=max_tokens)):
        if isinstance(message, ResultMessage):
            result = message
    if result is None or result.is_error:
        raise LLMCallError(getattr(result, "result", None) or "LLM 호출 실패")
    return result.result or ""


async def ask_structured(role: str, prompt: str, schema: type, *, max_tokens: int = DEFAULT_MAX_TOKENS):
    """app/ 의 PROMPT | model.with_structured_output(schema) 에 해당. pydantic 모델을 돌려준다."""
    from claude_agent_sdk import ResultMessage, query

    options = one_shot_options(
        role,
        max_tokens=max_tokens,
        output_format={"type": "json_schema", "schema": schema.model_json_schema()},
    )
    result = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    if result is None or result.is_error or result.structured_output is None:
        raise LLMCallError(getattr(result, "result", None) or "구조화 출력이 비어 있습니다")
    return schema.model_validate(result.structured_output)


# ---------------------------------------------------------------------------
# 임베딩 — pre_process_save_document.py(색인 스크립트, app/ 과 동일 파일)가 쓴다
# ---------------------------------------------------------------------------
def get_embeddings():
    """임베딩 모델. dimensions 는 Pinecone 인덱스와 같은 512 여야 한다."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)

