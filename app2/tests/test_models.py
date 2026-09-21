from models import resolve_model_id


def test_role_maps_to_models():
    assert resolve_model_id("orchestrator") == "claude-opus-5"
    assert resolve_model_id("judge") == "claude-opus-5"
    assert resolve_model_id("domain") == "claude-opus-5"
    assert resolve_model_id("trend") == "claude-sonnet-5"


def test_unknown_role_falls_back_to_default():
    assert resolve_model_id("does-not-exist") == "claude-sonnet-5"


def test_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_DOMAIN_MODEL", "claude-sonnet-5")
    assert resolve_model_id("domain") == "claude-sonnet-5"


def test_gateway_env_follows_litellm_base_url(monkeypatch):
    from models import gateway_env

    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    assert gateway_env() == {}

    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-gw")
    env = gateway_env()
    assert env["ANTHROPIC_BASE_URL"] == "http://localhost:4000"
    assert env["ANTHROPIC_API_KEY"] == env["ANTHROPIC_AUTH_TOKEN"] == "sk-gw"


def test_base_env_runs_subagents_synchronously_and_caps_output():
    """백그라운드 서브에이전트면 결과가 오기 전에 턴이 끝난다(실측). 반드시 꺼야 한다."""
    from models import base_env

    env = base_env()
    assert env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"
    assert base_env(16000)["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16000"


def test_summary_trigger_is_not_too_low():
    """낮은 임계치는 조사 도중 도구 결과를 날려버린다(MOC 노트 1개 ~852 토큰)."""
    from models import SUMMARY_TRIGGER_TOKENS, base_env

    assert SUMMARY_TRIGGER_TOKENS >= 100_000
    assert base_env()["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == str(SUMMARY_TRIGGER_TOKENS)


def test_isolated_options_do_not_load_user_claude_settings():
    """사용자 PC 의 CLAUDE.md/훅/플러그인/스킬이 에이전트에 섞이면 안 된다."""
    from models import isolated_options

    opts = isolated_options(model="m")
    assert opts.setting_sources == []
    assert opts.skills == []
    assert opts.strict_mcp_config is True


def test_one_shot_options_have_no_tools():
    from models import one_shot_options

    opts = one_shot_options("judge", max_tokens=16000)
    assert opts.tools == []
    assert opts.model == "claude-opus-5"
    assert opts.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16000"


def test_no_invalid_role_names_in_codebase():
    """정의되지 않은 role 은 조용히 default 모델로 떨어진다. 코드에 없어야 한다."""
    import pathlib
    import re

    from models import _ROLE_DEFAULTS

    app_dir = pathlib.Path(__file__).resolve().parent.parent
    bad = []
    pattern = r'(?:resolve_model_id|ask_structured|ask|one_shot_options)\(\s*"([^"]+)"'
    for path in app_dir.glob("*.py"):
        for m in re.finditer(pattern, path.read_text(encoding="utf-8")):
            if m.group(1) not in _ROLE_DEFAULTS:
                bad.append(f"{path.name}: {m.group(1)}")
    assert not bad, "정의되지 않은 role: " + ", ".join(bad)
