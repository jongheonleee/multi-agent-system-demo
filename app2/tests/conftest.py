"""테스트 공통 설정.

모듈 최상단에서 os.getenv 를 읽는 모듈(obsidian_tools 등)이 있으므로 import 보다
먼저 .env 를 로드한다. app2/.env 가 있으면 그것을, 없으면 app/.env 를 쓴다
(app.py 와 같은 규칙).
"""
from pathlib import Path

from dotenv import load_dotenv

_APP2 = Path(__file__).resolve().parent.parent
_ENV = _APP2 / ".env" if (_APP2 / ".env").exists() else _APP2.parent / "app" / ".env"
load_dotenv(_ENV)

# 라이브 테스트(-m llm)를 API 키 대신 이 PC 의 Claude Code 로그인으로 돌리고 싶을 때.
#   APP2_TEST_USE_CLAUDE_LOGIN=1 pytest -m llm
# API 키/게이트웨이 설정을 지우면 Claude Code 가 자기 로그인 인증을 쓴다.
import os  # noqa: E402

if os.getenv("APP2_TEST_USE_CLAUDE_LOGIN") == "1":
    for _key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "LITELLM_BASE_URL"):
        os.environ.pop(_key, None)
