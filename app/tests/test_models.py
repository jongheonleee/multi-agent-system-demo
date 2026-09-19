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


def test_no_invalid_role_names_in_codebase():
    """get_model 은 정의된 role 만 받아야 한다.

    "Opus"/"Sonnet" 같은 이름은 unknown 으로 떨어져 조용히 default 모델이
    쓰인다. fact_check 가 Opus 를 요청했지만 실제로는 Sonnet 이 돌고 있었다.
    """
    import pathlib
    import re

    from models import _ROLE_DEFAULTS

    app_dir = pathlib.Path(__file__).resolve().parent.parent
    bad = []
    for path in app_dir.glob("*.py"):
        for m in re.finditer(r'get_model\("([^"]+)"\)', path.read_text(encoding="utf-8")):
            if m.group(1) not in _ROLE_DEFAULTS:
                bad.append(f"{path.name}: {m.group(1)}")
    assert not bad, "정의되지 않은 role: " + ", ".join(bad)
