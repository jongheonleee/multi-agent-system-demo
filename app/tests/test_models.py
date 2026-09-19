from models import resolve_model_id


def test_role_maps_to_distinct_models():
    assert resolve_model_id("orchestrator") == "claude-opus-5"
    assert resolve_model_id("judge") == "claude-opus-5"
    assert resolve_model_id("domain") == "claude-sonnet-5"
    assert resolve_model_id("trend") == "claude-sonnet-5"


def test_unknown_role_falls_back_to_default():
    assert resolve_model_id("does-not-exist") == "claude-sonnet-5"


def test_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_DOMAIN_MODEL", "claude-opus-5")
    assert resolve_model_id("domain") == "claude-opus-5"
