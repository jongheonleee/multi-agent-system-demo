"""테스트 공통 설정.

모듈 최상단에서 os.getenv 를 읽는 모듈(obsidian_tools 등)이 있으므로,
그 import 보다 먼저 app/.env 를 로드해야 한다. conftest 는 테스트 수집
전에 실행되므로 여기가 적절한 위치다.
"""
from pathlib import Path

from dotenv import load_dotenv

# app/tests/conftest.py -> app/.env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
