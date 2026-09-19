# app/ 멀티 에이전트 설계 개선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `app/` 의 complex 경로를 단일 에이전트에서 "판단 Agent 가 질문을 도메인별 서브질문으로 분해 → 서버·인프라/시스템설계/AI 에이전트 3종이 병렬 답변 → 판단 Agent 가 병합" 구조로 바꾸고, Orchestrator·요약 메모리·HITL 재질문을 추가한다.

**Architecture:** LangGraph `Send` API 로 서브질문을 fan-out 해 도메인 에이전트를 병렬 실행하고, `Annotated[list, operator.add]` 리듀서로 결과를 모아 병합 노드에서 단일 답변을 만든다. 메모리는 `SummarizationMiddleware`(요약 치환) + `InMemorySaver`(세션 상태)로 처리하며 외부 저장소를 쓰지 않는다. 옵시디언 볼트는 Local REST API with MCP 플러그인의 Streamable HTTP 엔드포인트로 읽는다.

**Tech Stack:** LangGraph 1.2.11, langchain 1.4.0, langchain-core 1.6.3, langchain-mcp-adapters 0.3.2(신규), httpx 0.28.1, Pinecone, Streamlit 1.63

**Spec:** `docs/superpowers/specs/2026-09-19-app-multi-agent-design.md`

## Global Constraints

- 모든 모듈은 `app/` 안에서 **평탄 import** 한다 (`from state import ...`). 실행 디렉터리는 항상 `app/`.
- 파이썬 실행은 `..\.venv\Scripts\python.exe` 를 쓴다. 전역 python 금지.
- 환경변수는 `app/.env` 에서 읽는다.
- 모델 ID 는 `claude-opus-5`, `claude-sonnet-5` 만 사용한다. 날짜 접미사를 붙이지 않는다.
- 임베딩은 `text-embedding-3-small`, **dimensions=512** 로 고정한다 (Pinecone `testdb` 인덱스 차원과 일치해야 함).
- Pinecone 인덱스 핸들은 반드시 `pc.Index(host=host)` 로 만든다. 위치 인자로 넘기면 404.
- 테스트는 `app/tests/` 에 두고 `pytest` 로 돌린다. LLM/외부 API 를 호출하는 테스트에는 `@pytest.mark.llm` 을 달아 기본 실행에서 제외한다.
- **외부 저장소(Redis/Postgres)를 새로 도입하지 않는다.** 메모리는 요약 + 인메모리 체크포인터로 끝낸다.
- **옵시디언 볼트는 절대 읽기 전용이다.** 사용자 지시. 어떤 경우에도 볼트의 노트를
  생성·수정·삭제·이동하지 않는다. 구체적으로:
  - 에이전트에 노출할 MCP 도구는 `OBSIDIAN_TOOL_NAMES` 화이트리스트로만 제한한다.
    `vault_write`, `vault_append`, `vault_patch`, `vault_delete`, `vault_move`,
    `vault_copy`, `command_execute`, `open_file` 은 **절대 포함하지 않는다.**
  - 화이트리스트 방식을 유지한다(블랙리스트 금지). 플러그인이 새 쓰기 도구를 추가해도
    자동으로 차단되도록.
  - 사람이 직접 하는 확인 작업도 읽기 도구만 쓴다.
- 커밋은 태스크 단위로 한다. 현재 저장소는 git 이 아니므로 Task 0 에서 초기화한다.

---

### Task 0: git 초기화와 테스트 골격

**Files:**
- Create: `.gitignore`
- Create: `app/tests/__init__.py`
- Create: `app/tests/test_smoke.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: 없음
- Produces: `pytest` 실행 환경. 이후 모든 태스크가 `..\.venv\Scripts\python.exe -m pytest` 로 검증한다.

- [ ] **Step 1: .gitignore 작성**

```gitignore
.venv/
__pycache__/
*.pyc
.env
infra/.env
data/*.json
.streamlit/secrets.toml
```

- [ ] **Step 2: pytest 설정 작성**

`pytest.ini` (저장소 루트):

```ini
[pytest]
testpaths = app/tests
pythonpath = app
markers =
    llm: 실제 LLM/외부 API 를 호출하는 테스트 (기본 실행에서 제외)
addopts = -m "not llm"
```

- [ ] **Step 3: 스모크 테스트 작성**

`app/tests/__init__.py` 는 빈 파일. `app/tests/test_smoke.py`:

```python
def test_state_imports():
    from state import MainQueryState

    s = MainQueryState()
    assert s.messages == []
    assert s.query_type is None
```

- [ ] **Step 4: 테스트 실행**

Run: `..\.venv\Scripts\python.exe -m pytest -v`
Expected: PASS (1 passed)

- [ ] **Step 5: git 초기화 후 커밋**

```bash
git init
git add .gitignore pytest.ini app/tests/
git commit -m "chore: git 초기화 및 pytest 골격 추가"
```

---

### Task 1: models.py 역할별 모델 배정과 LiteLLM 경유

`get_model(model_name)` 이 인자를 무시하고 항상 `claude-sonnet-5` 를 반환한다. 역할별 모델 배정이 이 계획의 전제이므로 먼저 고친다.

**Files:**
- Modify: `app/models.py` (전체 재작성)
- Test: `app/tests/test_models.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `resolve_model_id(role: str = "default") -> str`
  - `get_model(role: str = "default", **overrides) -> BaseChatModel`
  - 유효 role: `"orchestrator"`, `"judge"`, `"domain"`, `"trend"`, `"default"`
  - `get_embeddings() -> OpenAIEmbeddings` (dimensions=512)
  - `get_original_reranker() -> CohereRerank`

- [ ] **Step 1: 실패하는 테스트 작성**

`app/tests/test_models.py`:

```python
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
```

- [ ] **Step 2: 테스트 실행해 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_model_id'`

- [ ] **Step 3: models.py 재작성**

```python
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
    """임베딩 모델. dimensions 는 Pinecone testdb 인덱스와 같은 512 여야 한다."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512)


def get_original_reranker():
    from langchain_cohere import CohereRerank

    return CohereRerank(model="rerank-multilingual-v3.0")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_models.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 기존 호출부가 깨지지 않는지 확인**

`get_model("Opus")` 같은 기존 호출은 unknown role 로 떨어져 `claude-sonnet-5` 를 받는다.

Run: `cd app && ..\.venv\Scripts\python.exe -c "import main_agent; print('ok')"`
Expected: `ok` 출력

- [ ] **Step 6: 커밋**

```bash
git add app/models.py app/tests/test_models.py
git commit -m "feat: 역할별 모델 배정과 LiteLLM 게이트웨이 경유 지원"
```

---

### Task 2: langchain-mcp-adapters 설치

**Files:**
- Modify: `requirements_win.txt`

**Interfaces:**
- Consumes: 없음
- Produces: `langchain_mcp_adapters.client.MultiServerMCPClient` 사용 가능

- [ ] **Step 1: 설치**

```bash
..\.venv\Scripts\python.exe -m pip install langchain-mcp-adapters
```

- [ ] **Step 2: import 확인**

Run:
```bash
..\.venv\Scripts\python.exe -c "from langchain_mcp_adapters.client import MultiServerMCPClient; import httpx; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: requirements 갱신**

```bash
..\.venv\Scripts\python.exe -m pip freeze > requirements_win.txt
```

- [ ] **Step 4: 커밋**

```bash
git add requirements_win.txt
git commit -m "chore: langchain-mcp-adapters 의존성 추가"
```

---

### Task 3: state.py 확장

분해된 서브질문과 도메인별 답변을 담을 상태를 정의한다. 병렬 노드가 같은 필드에 동시에 쓰므로 **리듀서가 필수**다. 리듀서 없이 병렬로 쓰면 LangGraph 가 `InvalidUpdateError` 를 던진다.

**Files:**
- Modify: `app/state.py`
- Test: `app/tests/test_state.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `Domain = Literal["server_infra", "system_design", "ai"]`
  - `SubQuestion(BaseModel)` — `domain: Domain`, `question: str`, `reason: str`
  - `DomainAnswer(BaseModel)` — `domain: Domain`, `question: str`, `answer: str`, `sources: list[str]`
  - `DomainTaskState(BaseModel)` — `sub_question: SubQuestion`, `original_question: str`
  - `MainQueryState` 추가 필드: `sub_questions`, `domain_answers`(리듀서 `operator.add`), `merged_answer`, `clarification`

- [ ] **Step 1: 실패하는 테스트 작성**

`app/tests/test_state.py`:

```python
import operator

from state import DomainAnswer, DomainTaskState, MainQueryState, SubQuestion


def test_sub_question_shape():
    sq = SubQuestion(domain="server_infra", question="커넥션 풀은?", reason="인프라 영역")
    assert sq.domain == "server_infra"


def test_domain_answers_uses_add_reducer():
    # 병렬 노드가 동시에 append 하므로 리듀서가 operator.add 여야 한다.
    field = MainQueryState.model_fields["domain_answers"]
    assert field.metadata and field.metadata[0] is operator.add


def test_domain_task_state_carries_one_sub_question():
    st = DomainTaskState(
        sub_question=SubQuestion(domain="ai", question="RAG?", reason="AI 영역"),
        original_question="전체 질문",
    )
    assert st.sub_question.domain == "ai"


def test_main_state_defaults():
    s = MainQueryState()
    assert s.sub_questions == []
    assert s.domain_answers == []
    assert s.merged_answer == ""
    assert s.clarification == ""
```

- [ ] **Step 2: 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'SubQuestion'`

- [ ] **Step 3: state.py 수정**

파일 상단 import 에 `import operator` 를 추가한다. `MainQueryState` 정의 **앞에** 아래를 넣는다.

```python
Domain = Literal["server_infra", "system_design", "ai"]


class SubQuestion(BaseModel):
    """판단 Agent 가 원 질문에서 분해해낸 도메인별 서브 질문."""

    domain: Domain = Field(..., description="이 서브 질문을 담당할 도메인")
    question: str = Field(..., description="해당 도메인에 던질 독립적인 질문")
    reason: str = Field(..., description="이 도메인으로 분류한 이유 (50자 이내)")


class DomainAnswer(BaseModel):
    """도메인 에이전트 한 개가 낸 답변."""

    domain: Domain
    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)


class DomainTaskState(BaseModel):
    """Send 로 병렬 실행되는 도메인 노드 하나가 받는 입력."""

    sub_question: SubQuestion
    original_question: str = ""
```

`MainQueryState` 를 아래로 교체한다.

```python
class MainQueryState(BaseState):
    query_type: Literal["general_it_trend_topic", "complex_discusion_topic", "unknown"] | None = Field(
        default=None,
        description="사용자 질문의 의도(Intent) 분류 결과",
    )

    sub_questions: list[SubQuestion] = Field(
        default_factory=list,
        description="판단 Agent 가 분해한 서브 질문 목록",
    )

    # 병렬 도메인 노드들이 동시에 쓴다. 리듀서 없으면 InvalidUpdateError.
    domain_answers: Annotated[list[DomainAnswer], operator.add] = Field(
        default_factory=list,
        description="도메인 에이전트들이 낸 답변 누적",
    )

    merged_answer: str = Field(default="", description="판단 Agent 가 병합한 최종 답변")

    clarification: str = Field(default="", description="재질문에 대한 사용자 답변")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_state.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 커밋**

```bash
git add app/state.py app/tests/test_state.py
git commit -m "feat: 서브질문/도메인답변 상태와 병렬 리듀서 추가"
```

---

### Task 4: obsidian_tools.py — Local REST API MCP 연결

플러그인은 설치·기동 확인 완료 (`obsidian-local-rest-api` v5.1.0, 도구 16종). 자체서명 인증서라 TLS 검증을 꺼야 한다.

**Files:**
- Create: `app/obsidian_tools.py`
- Modify: `app/.env` (빈 값 채우기)
- Test: `app/tests/test_obsidian_tools.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `OBSIDIAN_MCP_URL: str`, `OBSIDIAN_API_KEY: str`
  - `MOC_ENTRYPOINTS: dict[str, list[str]]` — 도메인별 MOC 노트 이름
  - `get_obsidian_tools() -> list[BaseTool]` — 실패 시 빈 리스트 + 경고 로그
  - `OBSIDIAN_TOOL_NAMES: tuple[str, ...]` — 도메인 에이전트에 노출할 읽기 전용 도구 이름

- [ ] **Step 1: app/.env 채우기**

`app/.env` 의 빈 두 줄을 채운다. API 키는 Obsidian → 설정 → Local REST API with MCP 에서 복사한다.

```
OBSIDIAN_MCP_URL=https://127.0.0.1:27124/mcp/
OBSIDIAN_API_KEY=<플러그인 설정의 API 키>
```

- [ ] **Step 2: 실패하는 테스트 작성**

`app/tests/test_obsidian_tools.py`:

```python
import pytest

from obsidian_tools import MOC_ENTRYPOINTS, OBSIDIAN_TOOL_NAMES


def test_moc_entrypoints_cover_two_vault_domains():
    assert set(MOC_ENTRYPOINTS) == {"server_infra", "system_design"}
    assert any("컨테이너" in n for n in MOC_ENTRYPOINTS["server_infra"])
    assert any("아키텍처" in n for n in MOC_ENTRYPOINTS["system_design"])


def test_exposed_tools_are_read_only():
    # 에이전트가 볼트를 수정하면 안 된다.
    for banned in ("vault_write", "vault_delete", "vault_patch", "vault_move", "command_execute"):
        assert banned not in OBSIDIAN_TOOL_NAMES


@pytest.mark.llm
def test_tools_load_from_plugin():
    from obsidian_tools import get_obsidian_tools

    tools = get_obsidian_tools()
    names = {t.name for t in tools}
    assert "search_simple" in names, names
    assert "vault_read" in names, names
```

- [ ] **Step 3: 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_obsidian_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'obsidian_tools'`

- [ ] **Step 4: obsidian_tools.py 작성**

```python
"""옵시디언 볼트를 Local REST API with MCP 플러그인으로 읽는다.

플러그인이 MCP 서버를 내장하므로 별도 브릿지가 필요 없다.
전제조건: Obsidian 앱이 실행 중이어야 한다.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

OBSIDIAN_MCP_URL = os.getenv("OBSIDIAN_MCP_URL", "https://127.0.0.1:27124/mcp/")
OBSIDIAN_API_KEY = os.getenv("OBSIDIAN_API_KEY", "")

# 에이전트에 노출할 읽기 전용 도구만 고른다.
# 플러그인은 16종을 노출하지만 쓰기/삭제/커맨드실행은 이 앱에 불필요하고 위험하다.
OBSIDIAN_TOOL_NAMES = (
    "search_simple",
    "search_query",
    "vault_read",
    "vault_list",
    "vault_get_document_map",
)

# 볼트에 기술 태그가 없으므로 MOC(Map of Content) 노트를 도메인 진입점으로 쓴다.
MOC_ENTRYPOINTS = {
    "server_infra": [
        "컨테이너 MOC",
        "네트워크 MOC",
        "모니터링 MOC",
        "99. 도커 아키텍처 구조",
    ],
    "system_design": [
        "시스템 아키텍처 MOC",
        "관계형 데이터베이스 MOC",
        "NoSQL MOC",
        "메시징 MOC",
    ],
}


def _insecure_client(headers=None, timeout=None, auth=None):
    """플러그인이 자체서명 인증서를 쓰므로 TLS 검증을 끈다.

    127.0.0.1 로컬 연결에만 쓰이므로 허용 가능하다.
    """
    import httpx

    return httpx.AsyncClient(
        headers=headers, timeout=timeout, auth=auth, verify=False, follow_redirects=True
    )


def _connection() -> dict:
    return {
        "obsidian": {
            "transport": "streamable_http",
            "url": OBSIDIAN_MCP_URL,
            "headers": {"Authorization": f"Bearer {OBSIDIAN_API_KEY}"},
            "httpx_client_factory": _insecure_client,
        }
    }


async def _load() -> list:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return await MultiServerMCPClient(_connection()).get_tools()


def get_obsidian_tools() -> list:
    """읽기 전용 옵시디언 도구를 반환한다.

    Obsidian 이 꺼져 있으면 연결이 실패한다. 그래프 전체를 죽이지 않고
    빈 리스트로 degrade 하며, 에이전트 프롬프트가 "조회 실패" 를 명시하도록 한다.
    """
    if not OBSIDIAN_API_KEY:
        logger.warning("OBSIDIAN_API_KEY 가 비어 있습니다. 옵시디언 도구를 건너뜁니다.")
        return []

    try:
        tools = asyncio.run(_load())
    except Exception as e:
        logger.warning("옵시디언 MCP 연결 실패(Obsidian 이 실행 중인지 확인): %s", e)
        return []

    return [t for t in tools if t.name in OBSIDIAN_TOOL_NAMES]
```

- [ ] **Step 5: 순수 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_obsidian_tools.py -v`
Expected: PASS (2 passed, 1 deselected)

- [ ] **Step 6: 실제 연결 확인**

Obsidian 을 실행한 상태에서:

Run: `cd app && ..\.venv\Scripts\python.exe -m pytest ../app/tests/test_obsidian_tools.py -v -m llm`
Expected: PASS

실패하면 확인 순서: Obsidian 실행 여부 → `curl -sk https://127.0.0.1:27124/` 가 200 인지 → `app/.env` 의 API 키가 플러그인 설정값과 같은지.

- [ ] **Step 7: 커밋**

```bash
git add app/obsidian_tools.py app/tests/test_obsidian_tools.py
git commit -m "feat: 옵시디언 Local REST API MCP 도구 로더"
```

---

### Task 5: 판단 Agent — 질문 분해

**Files:**
- Create: `app/judge_agent.py`
- Test: `app/tests/test_judge_agent.py`

**Interfaces:**
- Consumes: `state.SubQuestion`, `models.get_model`
- Produces:
  - `DecomposeOutput(BaseModel)` — `sub_questions: list[SubQuestion]`
  - `DECOMPOSE_PROMPT: ChatPromptTemplate`
  - `decompose(question: str) -> list[SubQuestion]`

- [ ] **Step 1: 실패하는 테스트 작성**

`app/tests/test_judge_agent.py`:

```python
import pytest

from judge_agent import DecomposeOutput
from state import SubQuestion


def test_decompose_output_accepts_multiple_domains():
    out = DecomposeOutput(
        sub_questions=[
            SubQuestion(domain="server_infra", question="a", reason="r"),
            SubQuestion(domain="ai", question="b", reason="r"),
        ]
    )
    assert {s.domain for s in out.sub_questions} == {"server_infra", "ai"}


@pytest.mark.llm
def test_decompose_splits_mixed_question():
    from judge_agent import decompose

    subs = decompose(
        "MAU 10만 사내 챗봇의 전체 아키텍처(게이트웨이, 대화 이력 저장, RAG, 모델 서빙, 모니터링)를 잡는다면"
    )
    domains = {s.domain for s in subs}
    assert len(subs) >= 2
    assert "ai" in domains
    assert domains & {"server_infra", "system_design"}


@pytest.mark.llm
def test_decompose_single_domain_returns_one():
    from judge_agent import decompose

    subs = decompose("DB 커넥션 풀 고갈을 어떻게 진단하나?")
    assert len(subs) == 1
    assert subs[0].domain in {"server_infra", "system_design"}
```

- [ ] **Step 2: 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_judge_agent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'judge_agent'`

- [ ] **Step 3: judge_agent.py 작성 (분해 부분)**

```python
"""판단 Agent — 질문 분해와 결과 병합."""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from models import get_model
from state import DomainAnswer, SubQuestion

logger = logging.getLogger(__name__)


class DecomposeOutput(BaseModel):
    sub_questions: list[SubQuestion] = Field(
        ...,
        description="도메인별로 분해한 서브 질문. 단일 도메인 질문이면 1개만.",
    )


DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """당신은 IT 질문을 도메인별로 분해하는 판단 에이전트입니다.

> 도메인 정의 (서로 배타적으로 적용할 것)
- server_infra: **운영 중인 시스템을 돌리는 층**.
  배포, 쿠버네티스/컨테이너, 네트워크, DB 운영·튜닝, 모니터링, 장애 진단, 용량 산정
- system_design: **코드와 데이터의 구조를 정하는 층**.
  서비스 경계, 모듈 분리, API 계약, 데이터 모델링, 일관성/확장성 트레이드오프
- ai: **모델과 그 주변**.
  LLM, RAG, 에이전트, 모델 서빙, 임베딩, 벡터 검색, 프롬프트

경계가 애매하면: "지금 돌아가는 걸 고치는가"(server_infra) vs "무엇을 만들지 정하는가"(system_design).

> 규칙
- 1. 질문에 실제로 포함된 도메인만 만듭니다. 억지로 3개를 채우지 마세요.
- 2. 단일 도메인 질문이면 서브 질문 1개만 반환합니다.
- 3. 각 서브 질문은 다른 도메인의 답을 몰라도 **독립적으로 답할 수 있어야** 합니다.
     "위 결과를 바탕으로" 같은 의존 표현을 쓰지 마세요.
- 4. 서브 질문은 최대 3개입니다.
- 5. 원 질문의 제약 조건(규모, 트래픽, 기술 스택)은 각 서브 질문에 그대로 복사해 넣습니다.

> 원 질문
{question}
"""
)


def decompose(question: str) -> list[SubQuestion]:
    """원 질문을 도메인별 서브 질문으로 분해한다."""
    chain = DECOMPOSE_PROMPT | get_model("judge").with_structured_output(DecomposeOutput)
    try:
        result = chain.invoke({"question": question})
        if isinstance(result, DecomposeOutput) and result.sub_questions:
            return result.sub_questions[:3]
        logger.error("분해 결과가 비정상입니다: %r", result)
    except Exception as e:
        logger.error("질문 분해 실패: %s", e)

    # 분해 실패 시 원 질문을 system_design 단일 서브질문으로 떨군다.
    return [SubQuestion(domain="system_design", question=question, reason="분해 실패 폴백")]
```

- [ ] **Step 4: 순수 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_judge_agent.py -v`
Expected: PASS (1 passed, 2 deselected)

- [ ] **Step 5: 분해 품질 확인**

Run: `cd app && ..\.venv\Scripts\python.exe -m pytest ../app/tests/test_judge_agent.py -v -m llm`
Expected: PASS (2 passed)

`server_infra` 와 `system_design` 이 같은 내용으로 두 번 나오면 도메인 정의 문구를
더 배타적으로 다듬는다.

- [ ] **Step 6: 커밋**

```bash
git add app/judge_agent.py app/tests/test_judge_agent.py
git commit -m "feat: 판단 Agent 질문 분해"
```

---

### Task 6: 도메인 에이전트 3종

**Files:**
- Create: `app/domain_agents.py`
- Test: `app/tests/test_domain_agents.py`

**Interfaces:**
- Consumes: `obsidian_tools.get_obsidian_tools`, `obsidian_tools.MOC_ENTRYPOINTS`, `complex_discusion_topic_agent.vector_search_tool`, `models.get_model`, `state.DomainTaskState`, `state.DomainAnswer`
- Produces:
  - `DOMAIN_PROMPTS: dict[str, str]`
  - `build_domain_agent(domain: str)` — `create_agent` 로 만든 ReAct 에이전트 (`lru_cache`)
  - `run_domain(state: DomainTaskState, config) -> dict` — `{"domain_answers": [DomainAnswer]}`

- [ ] **Step 1: 실패하는 테스트 작성**

`app/tests/test_domain_agents.py`:

```python
import pytest

from domain_agents import DOMAIN_PROMPTS


def test_all_three_domains_have_prompts():
    assert set(DOMAIN_PROMPTS) == {"server_infra", "system_design", "ai"}


def test_vault_domains_mention_their_moc_entrypoints():
    assert "컨테이너 MOC" in DOMAIN_PROMPTS["server_infra"]
    assert "시스템 아키텍처 MOC" in DOMAIN_PROMPTS["system_design"]


def test_ai_prompt_mentions_paper_search():
    assert "vector_search_tool" in DOMAIN_PROMPTS["ai"]


@pytest.mark.llm
def test_run_domain_returns_reducer_shaped_dict():
    from domain_agents import run_domain
    from state import DomainTaskState, SubQuestion

    out = run_domain(
        DomainTaskState(
            sub_question=SubQuestion(domain="ai", question="RAG 와 롱컨텍스트 차이는?", reason="AI"),
            original_question="RAG 와 롱컨텍스트 차이는?",
        ),
        {},
    )
    # operator.add 리듀서에 맞춰 리스트로 감싸 반환해야 한다.
    assert isinstance(out["domain_answers"], list)
    assert out["domain_answers"][0].domain == "ai"
    assert out["domain_answers"][0].answer
```

- [ ] **Step 2: 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_domain_agents.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'domain_agents'`

- [ ] **Step 3: domain_agents.py 작성**

```python
"""도메인 전문 에이전트 3종.

server_infra / system_design -> 옵시디언 볼트 (MCP)
ai                           -> Pinecone 논문 벡터스토어 + 옵시디언
"""
from __future__ import annotations

import logging
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, filter_messages
from langchain_core.runnables import RunnableConfig

from complex_discusion_topic_agent import vector_search_tool
from models import get_model
from obsidian_tools import MOC_ENTRYPOINTS, get_obsidian_tools
from state import DomainAnswer, DomainTaskState

logger = logging.getLogger(__name__)

_COMMON_RULES = """
> 공통 규칙
- 1. 오직 도구로 조회한 내용만을 근거로 답합니다. 사전 지식으로 채우지 마세요.
- 2. 근거를 못 찾으면 "참고 자료에서 찾지 못했다"고 명시합니다. 추측하지 마세요.
- 3. 도구 호출이 실패하면 "내부 문서를 조회할 수 없었다"고 밝히고, 그 한계를 답변에 포함합니다.
- 4. 당신에게 주어진 서브 질문에만 답합니다. 다른 도메인 영역으로 넘어가지 마세요.
- 5. 답변 말미에 "Sources" 섹션을 두고 참고한 노트/문서 이름을 나열합니다.
- 6. 한국어로 답합니다.
"""


def _vault_prompt(role: str, scope: str, domain: str) -> str:
    mocs = "\n".join(f"  - {n}" for n in MOC_ENTRYPOINTS[domain])
    return f"""당신은 {role} 전문 에이전트입니다.
{scope}

> 조사 방법
- 볼트에는 기술 태그가 없으므로 태그 검색은 쓰지 마세요.
- 먼저 아래 MOC(Map of Content) 노트를 vault_read 로 읽어 관련 노트를 찾으세요.
{mocs}
- 그다음 search_simple 로 핵심 용어를 검색해 개별 노트를 읽습니다.
- 긴 노트는 vault_get_document_map 으로 구조를 먼저 보고 필요한 부분만 읽으세요.
{_COMMON_RULES}"""


DOMAIN_PROMPTS = {
    "server_infra": _vault_prompt(
        "서버·인프라",
        "운영 중인 시스템을 돌리는 층을 다룹니다: 배포, 쿠버네티스/컨테이너, 네트워크,\nDB 운영·튜닝, 모니터링, 장애 진단, 용량 산정.",
        "server_infra",
    ),
    "system_design": _vault_prompt(
        "시스템 설계",
        "코드와 데이터의 구조를 정하는 층을 다룹니다: 서비스 경계, 모듈 분리, API 계약,\n데이터 모델링, 일관성/확장성 트레이드오프.",
        "system_design",
    ),
    "ai": """당신은 AI 전문 에이전트입니다.
LLM, RAG, 에이전트, 모델 서빙, 임베딩, 벡터 검색, 프롬프트를 다룹니다.

> 조사 방법
- vector_search_tool 로 색인된 논문을 검색합니다. 질의를 바꿔가며 2~3회 호출하면
  회수율이 올라갑니다.
- 볼트에도 RAG/LLM 노트가 있으므로 search_simple 로 함께 찾아봅니다.
"""
    + _COMMON_RULES,
}


@lru_cache(maxsize=3)
def build_domain_agent(domain: str):
    """도메인별 ReAct 에이전트. MCP 연결 비용 때문에 캐시한다."""
    obsidian = get_obsidian_tools()
    tools = [vector_search_tool, *obsidian] if domain == "ai" else list(obsidian)

    if not tools:
        logger.warning("%s 도메인에 사용 가능한 도구가 없습니다", domain)

    return create_agent(
        model=get_model("domain"),
        tools=tools,
        system_prompt=DOMAIN_PROMPTS[domain],
        name=f"{domain}_agent",
    )


def run_domain(state: DomainTaskState, config: RunnableConfig) -> dict:
    """Send 로 병렬 실행되는 노드. 리듀서에 맞춰 리스트를 반환한다."""
    domain = state.sub_question.domain
    question = state.sub_question.question

    try:
        agent = build_domain_agent(domain)
        response = agent.invoke(
            {"messages": [HumanMessage(question)]},
            config={**(config or {}), "run_name": f"domain:{domain}"},
        )
        ai_messages = filter_messages(response.get("messages", []), include_types=[AIMessage])
        answer = ai_messages[-1].text if ai_messages else ""
    except Exception as e:
        logger.error("%s 도메인 실행 실패: %s", domain, e)
        answer = f"({domain} 영역 조사 실패: {e})"

    return {
        "domain_answers": [
            DomainAnswer(domain=domain, question=question, answer=answer, sources=[])
        ]
    }
```

- [ ] **Step 4: 순수 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_domain_agents.py -v`
Expected: PASS (3 passed, 1 deselected)

- [ ] **Step 5: 옵시디언 도메인 실제 실행 확인**

Obsidian 실행 상태에서:

```bash
cd app
..\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); import logging; logging.basicConfig(level=logging.INFO); from domain_agents import run_domain; from state import DomainTaskState, SubQuestion; r=run_domain(DomainTaskState(sub_question=SubQuestion(domain='server_infra', question='쿠버네티스에서 파드 CPU throttling 을 어떻게 확인하나?', reason='인프라'), original_question='x'), {}); print(r['domain_answers'][0].answer[:400])"
```
Expected: 볼트 노트를 인용한 답변 출력. "찾지 못했다" 만 나오면 MOC 진입점 이름이
실제 노트명과 일치하는지 `search_simple` 로 확인한다.

- [ ] **Step 6: 커밋**

```bash
git add app/domain_agents.py app/tests/test_domain_agents.py
git commit -m "feat: 서버인프라/시스템설계/AI 도메인 에이전트"
```

---

### Task 7: 판단 Agent — 결과 병합

**Files:**
- Modify: `app/judge_agent.py`
- Test: `app/tests/test_judge_merge.py`

**Interfaces:**
- Consumes: `state.DomainAnswer`
- Produces:
  - `MERGE_PROMPT: ChatPromptTemplate`
  - `merge(question: str, answers: list[DomainAnswer]) -> str`

- [ ] **Step 1: 실패하는 테스트 작성**

`app/tests/test_judge_merge.py`:

```python
import pytest

from state import DomainAnswer


def test_merge_single_answer_passes_through():
    from judge_agent import merge

    answers = [DomainAnswer(domain="ai", question="q", answer="원본 답변 본문")]
    # 단일 도메인이면 합칠 것이 없으므로 LLM 을 태우지 않는다.
    assert merge("q", answers) == "원본 답변 본문"


def test_merge_empty_returns_message():
    from judge_agent import merge

    assert "답변" in merge("q", [])


@pytest.mark.llm
def test_merge_combines_two_domains():
    from judge_agent import merge

    answers = [
        DomainAnswer(domain="server_infra", question="게이트웨이는?", answer="Nginx 를 앞단에 둔다."),
        DomainAnswer(domain="ai", question="RAG 는?", answer="Pinecone 에 문서를 색인한다."),
    ]
    merged = merge("사내 챗봇 아키텍처", answers)
    assert "Nginx" in merged or "게이트웨이" in merged
    assert "Pinecone" in merged or "RAG" in merged
```

- [ ] **Step 2: 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_judge_merge.py -v`
Expected: FAIL — `ImportError: cannot import name 'merge'`

- [ ] **Step 3: judge_agent.py 에 병합 추가**

```python
MERGE_PROMPT = ChatPromptTemplate.from_template(
    """당신은 도메인 전문가들의 답변을 하나로 합치는 판단 에이전트입니다.

> 원 질문
{question}

> 도메인별 답변
{answers}

> 병합 규칙
- 1. 원 질문에 답하는 **하나의 글**로 씁니다. 도메인별로 단순 나열하지 마세요.
- 2. 여러 도메인이 같은 말을 하면 한 번만 씁니다.
- 3. 도메인 간 권고가 **상충하면** 숨기지 말고 드러낸 뒤, 어떤 조건에서 무엇을 택할지 씁니다.
- 4. 전문가가 "찾지 못했다"고 한 부분은 그대로 한계로 밝힙니다. 지어내지 마세요.
- 5. 전문가 답변에 없는 내용을 새로 추가하지 마세요.
- 6. 각 전문가의 Sources 를 모아 말미에 "Sources" 섹션으로 통합합니다.
- 7. 한국어로 씁니다.
"""
)

_DOMAIN_LABELS = {
    "server_infra": "서버·인프라",
    "system_design": "시스템 설계",
    "ai": "AI",
}


def _format_answers(answers: list[DomainAnswer]) -> str:
    return "\n\n".join(
        f"### [{_DOMAIN_LABELS.get(a.domain, a.domain)}] {a.question}\n{a.answer}"
        for a in answers
    )


def merge(question: str, answers: list[DomainAnswer]) -> str:
    """도메인 답변들을 하나의 답변으로 합친다."""
    if not answers:
        return "도메인 에이전트가 답변을 생성하지 못했습니다."

    # 단일 도메인이면 합칠 것이 없다. LLM 호출을 아낀다.
    if len(answers) == 1:
        return answers[0].answer

    chain = MERGE_PROMPT | get_model("judge")
    try:
        return chain.invoke({"question": question, "answers": _format_answers(answers)}).text
    except Exception as e:
        logger.error("병합 실패, 원본을 이어붙입니다: %s", e)
        return _format_answers(answers)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_judge_merge.py -v`
Expected: PASS (2 passed, 1 deselected)

- [ ] **Step 5: 커밋**

```bash
git add app/judge_agent.py app/tests/test_judge_merge.py
git commit -m "feat: 판단 Agent 도메인 답변 병합"
```

---

### Task 8: Orchestrator — 분류와 재질문

**Files:**
- Create: `app/orchestrator.py`
- Test: `app/tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `models.get_model`, `state.MainQueryState`
- Produces:
  - `AMBIGUITY_RULES: str`
  - `IntentOutput(BaseModel)` — `intent`, `needs_clarification`, `clarifying_question`, `reason`
  - `classify(question: str) -> IntentOutput`
  - `orchestrate(state: MainQueryState, config) -> MainQueryState`

- [ ] **Step 1: 실패하는 테스트 작성**

`app/tests/test_orchestrator.py`:

```python
import pytest

from orchestrator import IntentOutput


def test_intent_output_requires_clarifying_question_when_flagged():
    out = IntentOutput(
        intent="complex_discusion_topic",
        needs_clarification=True,
        clarifying_question="예상 트래픽 규모가 어느 정도인가요?",
        reason="규모 정보 없음",
    )
    assert out.needs_clarification
    assert out.clarifying_question


def test_intent_output_defaults_no_clarification():
    out = IntentOutput(intent="unknown", reason="인사말")
    assert out.needs_clarification is False
    assert out.clarifying_question == ""


@pytest.mark.llm
def test_specific_question_does_not_trigger_clarification():
    from orchestrator import classify

    out = classify("JWT와 세션 인증의 차이점이 뭔가?")
    assert out.intent == "general_it_trend_topic"
    assert out.needs_clarification is False


@pytest.mark.llm
def test_vague_design_request_triggers_clarification():
    from orchestrator import classify

    out = classify("우리 서비스 아키텍처 좀 잡아줘")
    assert out.needs_clarification is True
```

- [ ] **Step 2: 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'orchestrator'`

- [ ] **Step 3: orchestrator.py 작성**

```python
"""Orchestrator — 질문 유형 판단과 필요 시 재질문."""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from models import get_model
from state import MainQueryState

logger = logging.getLogger(__name__)

AMBIGUITY_RULES = """
> 재질문 기준 (엄격하게 적용)
재질문은 사용자를 멈춰 세우므로 아래 둘 중 하나일 때만 합니다.
- 1. 설계를 요청했는데 규모·트래픽·제약이 **전혀** 없어 답이 완전히 달라지는 경우
- 2. 요구가 서로 **상충**해 한쪽을 고르지 않으면 답할 수 없는 경우

그 외에는 재질문하지 않습니다. 정보가 조금 부족하면 합리적으로 가정하고 진행하되,
가정한 내용을 답변에 명시하면 됩니다.
"""

CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
    """당신은 IT 질문을 분류하는 오케스트레이터입니다.

> 의도
- complex_discusion_topic: 서버·인프라·시스템설계·AI 의 트레이드오프 판단이 필요한 논의
  예) 트래픽 급증 시 p99 지연 원인 구분, MSA 통신 방식 선택, 사내 챗봇 전체 아키텍처
- general_it_trend_topic: 단순 개념 질의 또는 최신 트렌드·시세
  예) JWT 와 세션 인증의 차이, 트랜잭션 격리 수준, 현재 맥북 가격
- unknown: 인사말·잡담 등 위에 해당하지 않는 것

{ambiguity_rules}

> 질문
{question}
"""
)


class IntentOutput(BaseModel):
    intent: Literal["complex_discusion_topic", "general_it_trend_topic", "unknown"] = Field(
        ..., description="질문 의도"
    )
    needs_clarification: bool = Field(
        default=False, description="재질문이 필요한지. 기준을 엄격히 적용할 것."
    )
    clarifying_question: str = Field(
        default="", description="재질문이 필요할 때 사용자에게 물을 한 문장"
    )
    reason: str = Field(..., description="판단 이유 (50자 이내)")


def classify(question: str) -> IntentOutput:
    chain = CLASSIFY_PROMPT | get_model("orchestrator").with_structured_output(IntentOutput)
    try:
        result = chain.invoke({"question": question, "ambiguity_rules": AMBIGUITY_RULES})
        if isinstance(result, IntentOutput):
            return result
        logger.error("분류 결과가 비정상입니다: %r", result)
    except Exception as e:
        logger.error("의도 분류 실패: %s", e)
    return IntentOutput(intent="unknown", reason="분류 실패 폴백")


def orchestrate(state: MainQueryState, config: RunnableConfig) -> MainQueryState:
    question = state.messages[-1].text if state.messages else ""

    decision = classify(question)
    state.query_type = decision.intent
    logger.info("의도=%s 이유=%s", decision.intent, decision.reason)

    if decision.needs_clarification and decision.clarifying_question:
        # 그래프를 멈추고 사용자 답을 받는다. checkpointer 가 있어야 동작한다.
        answer = interrupt({"clarifying_question": decision.clarifying_question})
        state.clarification = str(answer)

    return state
```

- [ ] **Step 4: 순수 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_orchestrator.py -v`
Expected: PASS (2 passed, 2 deselected)

- [ ] **Step 5: 재질문 판단 품질 확인**

Run: `cd app && ..\.venv\Scripts\python.exe -m pytest ../app/tests/test_orchestrator.py -v -m llm`
Expected: PASS (2 passed)

`test_specific_question_does_not_trigger_clarification` 이 실패하면(=과도한 재질문)
`AMBIGUITY_RULES` 를 더 강하게 제한한다.

- [ ] **Step 6: 커밋**

```bash
git add app/orchestrator.py app/tests/test_orchestrator.py
git commit -m "feat: Orchestrator 분류와 재질문 판단"
```

---

### Task 9: main_agent.py 재구성 — 분해·병렬·병합 배선

**Files:**
- Modify: `app/main_agent.py` (전체 재작성)
- Test: `app/tests/test_main_graph.py`

**Interfaces:**
- Consumes: 앞선 모든 모듈
- Produces:
  - `fan_out_domains(state) -> list[Send] | str`
  - `build_main_graph(checkpointer=None)`
  - `main_graph`
  - 노드 이름: `orchestrator`, `decompose`, `domain_worker`, `merge`, `general_it_topic_agent`, `general_topic_agent`, `finalize`

- [ ] **Step 1: 실패하는 테스트 작성**

`app/tests/test_main_graph.py`:

```python
from state import MainQueryState, SubQuestion


def test_graph_has_expected_nodes():
    from main_agent import main_graph

    nodes = set(main_graph.get_graph().nodes)
    for name in [
        "orchestrator",
        "decompose",
        "domain_worker",
        "merge",
        "general_it_topic_agent",
        "general_topic_agent",
        "finalize",
    ]:
        assert name in nodes, f"{name} 누락: {nodes}"


def test_fanout_emits_one_send_per_sub_question():
    from main_agent import fan_out_domains

    state = MainQueryState(
        sub_questions=[
            SubQuestion(domain="server_infra", question="a", reason="r"),
            SubQuestion(domain="ai", question="b", reason="r"),
        ]
    )
    sends = fan_out_domains(state)
    assert len(sends) == 2
    assert {s.node for s in sends} == {"domain_worker"}


def test_fanout_with_no_sub_questions_goes_to_merge():
    from main_agent import fan_out_domains

    assert fan_out_domains(MainQueryState()) == "merge"
```

- [ ] **Step 2: 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_main_graph.py -v`
Expected: FAIL — `ImportError: cannot import name 'fan_out_domains'`

- [ ] **Step 3: main_agent.py 재작성**

```python
"""메인 그래프.

orchestrator ─┬─ complex ─▶ decompose ─(Send fan-out)─▶ domain_worker ×N ─▶ merge ─┐
              ├─ general ─▶ general_it_topic_agent ───────────────────────────────┤
              └─ unknown ─▶ general_topic_agent ─────────────────────────────────  ┴─▶ finalize
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send

from domain_agents import run_domain
from general_it_trend_topic_agent import general_it_trend_graph
from judge_agent import decompose, merge
from orchestrator import orchestrate
from state import DomainTaskState, MainQueryState

logger = logging.getLogger(__name__)


def _question_of(state: MainQueryState) -> str:
    return state.messages[-1].text if state.messages else ""


def decompose_node(state: MainQueryState, config: RunnableConfig) -> MainQueryState:
    question = _question_of(state)
    if state.clarification:
        question = f"{question}\n\n[사용자 추가 정보] {state.clarification}"
    state.sub_questions = decompose(question)
    logger.info(
        "서브질문 %d개: %s", len(state.sub_questions), [s.domain for s in state.sub_questions]
    )
    return state


def fan_out_domains(state: MainQueryState):
    """서브질문 1개당 domain_worker 를 하나씩 병렬로 띄운다."""
    if not state.sub_questions:
        return "merge"
    original = _question_of(state)
    return [
        Send("domain_worker", DomainTaskState(sub_question=sq, original_question=original))
        for sq in state.sub_questions
    ]


def merge_node(state: MainQueryState, config: RunnableConfig) -> MainQueryState:
    state.merged_answer = merge(_question_of(state), state.domain_answers)
    state.messages = [AIMessage(content=state.merged_answer)]
    return state


def general_it_node(state: MainQueryState, config: RunnableConfig) -> MainQueryState:
    response = general_it_trend_graph.invoke(
        input={"messages": state.messages}, config=config, output_keys=["messages"]
    )
    ai = filter_messages(response.get("messages", []), include_types=[AIMessage])
    if ai:
        state.messages = [ai[-1]]
    return state


def general_node(state: MainQueryState, config: RunnableConfig) -> MainQueryState:
    state.messages = [
        AIMessage(
            content="IT 개념 질의나 아키텍처 논의를 도와드릴 수 있습니다. 무엇이 궁금하신가요?"
        )
    ]
    return state


def finalize_node(state: MainQueryState, config: RunnableConfig) -> MainQueryState:
    if not state.messages or not state.messages[-1].text:
        state.messages = [AIMessage(content="답변을 생성하지 못했습니다.")]
    return state


def route_by_intent(state: MainQueryState) -> str:
    if state.query_type == "complex_discusion_topic":
        return "decompose"
    if state.query_type == "general_it_trend_topic":
        return "general_it_topic_agent"
    return "general_topic_agent"


def build_main_graph(checkpointer=None):
    b = StateGraph(MainQueryState)

    b.add_node("orchestrator", orchestrate, retry_policy=RetryPolicy(max_interval=3))
    b.add_node("decompose", decompose_node)
    b.add_node("domain_worker", run_domain, input_schema=DomainTaskState)
    b.add_node("merge", merge_node)
    b.add_node("general_it_topic_agent", general_it_node)
    b.add_node("general_topic_agent", general_node)
    b.add_node("finalize", finalize_node)

    b.add_edge(START, "orchestrator")
    b.add_conditional_edges(
        "orchestrator",
        route_by_intent,
        ["decompose", "general_it_topic_agent", "general_topic_agent"],
    )
    b.add_conditional_edges("decompose", fan_out_domains, ["domain_worker", "merge"])
    b.add_edge("domain_worker", "merge")
    b.add_edge("merge", "finalize")
    b.add_edge("general_it_topic_agent", "finalize")
    b.add_edge("general_topic_agent", "finalize")
    b.add_edge("finalize", END)

    return b.compile(name="main_agent", checkpointer=checkpointer)


main_graph = build_main_graph()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_main_graph.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 단일 도메인 질문 실행 확인**

```bash
cd app
..\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); import logging; logging.disable(logging.CRITICAL); from main_agent import main_graph; r=main_graph.invoke({'messages':[('user','JWT와 세션 인증의 차이점이 뭔가?')]}); print(r['messages'][-1].text[:200])"
```
Expected: 웹 검색 기반 답변 출력, 예외 없음

- [ ] **Step 6: 다중 도메인 병렬 실행 확인**

```bash
cd app
..\.venv\Scripts\python.exe -c "from dotenv import load_dotenv; load_dotenv(); import logging; logging.basicConfig(level=logging.INFO); from main_agent import main_graph; r=main_graph.invoke({'messages':[('user','MAU 10만 사내 챗봇의 전체 아키텍처(게이트웨이, 대화 이력 저장, RAG, 모델 서빙, 모니터링)를 잡는다면')]}); print(r['messages'][-1].text[:300])"
```
Expected: 로그에 `서브질문 2~3개` 가 찍히고, 병합된 단일 답변이 출력됨

- [ ] **Step 7: 커밋**

```bash
git add app/main_agent.py app/tests/test_main_graph.py
git commit -m "feat: 분해-병렬-병합 메인 그래프 재구성"
```

---

### Task 10: 요약 메모리 배선 (SummarizationMiddleware)

긴 대화에서 토큰을 절약하고 문맥을 유지한다. 외부 저장소는 쓰지 않는다.

**Files:**
- Modify: `app/domain_agents.py` (`build_domain_agent`)
- Modify: `app/general_it_trend_topic_agent.py` (`agent_executor`)
- Test: `app/tests/test_summarization.py`

**Interfaces:**
- Consumes: `langchain.agents.middleware.SummarizationMiddleware`, `models.get_model`
- Produces: `app/summarization.py` 의 `build_summarizer() -> SummarizationMiddleware`

- [ ] **Step 1: 실패하는 테스트 작성**

`app/tests/test_summarization.py`:

```python
def test_summarizer_is_configured_with_trigger_and_keep():
    from langchain.agents.middleware import SummarizationMiddleware

    from summarization import build_summarizer

    mw = build_summarizer()
    assert isinstance(mw, SummarizationMiddleware)


def test_domain_agent_has_summarization_middleware():
    import inspect

    import domain_agents

    src = inspect.getsource(domain_agents.build_domain_agent)
    assert "middleware" in src
    assert "build_summarizer" in src
```

- [ ] **Step 2: 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_summarization.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'summarization'`

- [ ] **Step 3: summarization.py 작성**

```python
"""대화 요약 메모리.

LangChain 1.x 에서 ConversationSummaryBufferMemory 계열은 제거됐다.
현행 대체제인 SummarizationMiddleware 를 쓴다. 토큰이 임계치에 닿으면
오래된 메시지를 요약으로 치환하고 최근 메시지는 원문으로 남긴다.
"""
from __future__ import annotations

import os

from langchain.agents.middleware import SummarizationMiddleware

from models import get_model


def build_summarizer() -> SummarizationMiddleware:
    return SummarizationMiddleware(
        model=get_model("judge"),
        trigger=("tokens", int(os.getenv("SUMMARY_TRIGGER_TOKENS", "8000"))),
        keep=("messages", int(os.getenv("SUMMARY_KEEP_MESSAGES", "20"))),
    )
```

- [ ] **Step 4: domain_agents.py 에 배선**

`build_domain_agent` 의 `create_agent(...)` 호출에 `middleware` 를 추가한다.

```python
    from summarization import build_summarizer

    return create_agent(
        model=get_model("domain"),
        tools=tools,
        system_prompt=DOMAIN_PROMPTS[domain],
        middleware=[build_summarizer()],
        name=f"{domain}_agent",
    )
```

- [ ] **Step 5: general_it_trend_topic_agent.py 에 배선**

`agent_executor = create_agent(...)` 에 같은 방식으로 `middleware=[build_summarizer()]` 를
추가하고 파일 상단에 `from summarization import build_summarizer` 를 넣는다.

- [ ] **Step 6: 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_summarization.py -v`
Expected: PASS (2 passed)

`create_agent` 가 `middleware` 인자를 받지 않으면 시그니처를 확인한다:
`..\.venv\Scripts\python.exe -c "import inspect; from langchain.agents import create_agent; print(inspect.signature(create_agent))"`

- [ ] **Step 7: 회귀 확인**

Run: `..\.venv\Scripts\python.exe -m pytest -v`
Expected: 전부 PASS

- [ ] **Step 8: 커밋**

```bash
git add app/summarization.py app/domain_agents.py app/general_it_trend_topic_agent.py app/tests/test_summarization.py
git commit -m "feat: SummarizationMiddleware 기반 요약 메모리"
```

---

### Task 11: Streamlit 에 재질문(HITL) 배선

**Files:**
- Modify: `app/app.py`
- Test: 수동 확인 (UI)

**Interfaces:**
- Consumes: `main_agent.build_main_graph`, `langgraph.checkpoint.memory.InMemorySaver`, `langgraph.types.Command`
- Produces: interrupt 발생 시 재질문 UI, `Command(resume=...)` 재개

- [ ] **Step 1: 체크포인터를 붙인 그래프 로더로 교체**

`app/app.py` 의 `load_graphs()` 를 수정한다.

```python
@st.cache_resource
def load_graphs():
    from complex_discusion_topic_agent import complex_topic_graph
    from general_it_trend_topic_agent import general_it_trend_graph
    from langgraph.checkpoint.memory import InMemorySaver
    from main_agent import build_main_graph

    # interrupt 는 checkpointer 없이는 동작하지 않는다.
    # @st.cache_resource 가 프로세스 수명 동안 유지하므로 rerun 사이에도 상태가 남는다.
    checkpointer = InMemorySaver()

    return {
        "메인 에이전트 (자동 라우팅)": build_main_graph(checkpointer=checkpointer),
        "복잡한 논의 주제 RAG (Pinecone)": complex_topic_graph,
        "일반 IT 웹 검색 (Serper)": general_it_trend_graph,
    }
```

- [ ] **Step 2: thread_id 를 config 에 넣기**

checkpointer 가 붙으면 `thread_id` 가 필수다.

```python
        config = {
            "run_name": agent_name,
            "configurable": {"thread_id": st.session_state.session_id},
            "metadata": {
                "langfuse_session_id": st.session_state.session_id,
                "langfuse_tags": ["streamlit", graphs[agent_name].name],
            },
        }
```

- [ ] **Step 3: 재개 입력 준비**

스트리밍 호출 직전에 입력을 결정한다. 파일 상단에 `from langgraph.types import Command` 를 추가한다.

```python
        resume_value = st.session_state.pop("resume_value", None)
        graph_input = (
            Command(resume=resume_value)
            if resume_value
            else {"messages": history, "question": prompt}
        )
```

그리고 `graphs[agent_name].stream(...)` 의 첫 인자를 `graph_input` 으로 바꾼다.

- [ ] **Step 4: interrupt 감지**

스트리밍 루프 안에서 `__interrupt__` 키를 확인한다.

```python
            for namespace, update in graphs[agent_name].stream(
                graph_input, config=config, stream_mode="updates", subgraphs=True
            ):
                if "__interrupt__" in update:
                    st.session_state.pending_clarification = (
                        update["__interrupt__"][0].value["clarifying_question"]
                    )
                    break
                ...
```

- [ ] **Step 5: 재질문 UI 추가**

히스토리 렌더 뒤에 넣는다.

```python
if st.session_state.get("pending_clarification"):
    with st.chat_message("assistant"):
        st.markdown(f"**확인이 필요합니다** — {st.session_state.pending_clarification}")
    reply = st.chat_input("답변을 입력하세요", key="clarify_input")
    if reply:
        st.session_state.resume_value = reply
        st.session_state.pop("pending_clarification")
        st.session_state.run_prompt = reply
        st.rerun()
```

- [ ] **Step 6: 기동 확인**

```bash
cd app
..\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8603
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8603
```
Expected: `200`

- [ ] **Step 7: 재질문 수동 확인**

브라우저에서 "우리 서비스 아키텍처 좀 잡아줘" 입력.
Expected: "확인이 필요합니다 — …" 가 뜨고, 답을 넣으면 이어서 진행된다.

"JWT와 세션 인증의 차이점이 뭔가?" 입력.
Expected: 재질문 없이 바로 답변한다.

- [ ] **Step 8: 커밋**

```bash
git add app/app.py
git commit -m "feat: Streamlit 재질문(HITL) 배선"
```

---

### Task 12: fact_check 하드코딩 제거 + Guardrail 배선

**Files:**
- Modify: `app/complex_discusion_topic_agent.py` (`fact_check_tool` 내부)
- Modify: `app/judge_agent.py` (`merge` 뒤 검증)
- Test: `app/tests/test_fact_check.py`

**Interfaces:**
- Consumes: `complex_discusion_topic_agent.fact_check_tool`
- Produces: `judge_agent._guard(merged, answers) -> str`

- [ ] **Step 1: 실패하는 테스트 작성**

`app/tests/test_fact_check.py`:

```python
def test_fact_check_without_context_does_not_use_hardcoded_docs():
    import inspect

    import complex_discusion_topic_agent as m

    src = inspect.getsource(m.fact_check_tool)
    # 질문과 무관한 Gemini 스니펫이 컨텍스트로 대체되면 검증이 무의미해진다.
    assert "본 리포트에서 소개하는" not in src
```

- [ ] **Step 2: 실패 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_fact_check.py -v`
Expected: FAIL

- [ ] **Step 3: 하드코딩 제거**

`fact_check_tool` 의 `try:` 블록 첫머리를 아래로 교체한다. 기존 `context_str = "[Document 1]\n본 리포트에서..."` 줄과 `if context: context_str = context` 를 모두 지운다.

```python
    try:
        if not context:
            return FactCheckResult(
                sentences=[],
                overall_accuracy=0.0,
                overall_accuracy_comment="출처 문서가 제공되지 않아 검증할 수 없습니다.",
            ).model_dump()

        context_str = context
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `..\.venv\Scripts\python.exe -m pytest app/tests/test_fact_check.py -v`
Expected: PASS

- [ ] **Step 5: merge 에 guardrail 추가**

`judge_agent.py` 의 `merge` 반환부를 `return _guard(merged, answers)` 로 바꾸고 아래를 추가한다.

```python
def _guard(merged: str, answers: list[DomainAnswer]) -> str:
    """병합 답변이 근거를 벗어나지 않았는지 1회 검증하고, 낮으면 1회 재작성한다."""
    from complex_discusion_topic_agent import fact_check_tool

    context = "\n\n".join(a.answer for a in answers)
    try:
        report = fact_check_tool.invoke({"text": merged, "context": context})
    except Exception as e:
        logger.warning("guardrail 검증 실패, 원본을 반환합니다: %s", e)
        return merged

    if report.get("overall_accuracy", 1.0) >= 0.8:
        return merged

    logger.info("guardrail 미달(%s), 1회 재작성합니다", report.get("overall_accuracy"))
    fix_prompt = ChatPromptTemplate.from_template(
        """아래 답변에서 출처가 뒷받침하지 않는 문장을 제거하거나 완화해 다시 쓰세요.

> 검증 결과
{report}

> 답변
{merged}
"""
    )
    try:
        return (
            (fix_prompt | get_model("judge"))
            .invoke({"report": report.get("overall_accuracy_comment", ""), "merged": merged})
            .text
        )
    except Exception as e:
        logger.warning("재작성 실패, 원본을 반환합니다: %s", e)
        return merged
```

`merge` 안의 반환도 바꾼다.

```python
    try:
        merged = chain.invoke({"question": question, "answers": _format_answers(answers)}).text
    except Exception as e:
        logger.error("병합 실패, 원본을 이어붙입니다: %s", e)
        return _format_answers(answers)

    return _guard(merged, answers)
```

- [ ] **Step 6: 전체 테스트 실행**

Run: `..\.venv\Scripts\python.exe -m pytest -v`
Expected: 전부 PASS

- [ ] **Step 7: 커밋**

```bash
git add app/complex_discusion_topic_agent.py app/judge_agent.py app/tests/test_fact_check.py
git commit -m "fix: fact_check 하드코딩 컨텍스트 제거하고 병합 guardrail 배선"
```

---

### Task 13: 문서화와 최종 검증

**Files:**
- Create: `app/README.md`

- [ ] **Step 1: app/README.md 작성**

다음을 포함한다: 실행 방법, 전제조건(Obsidian 실행 필요), 새 환경변수
(`OBSIDIAN_MCP_URL`, `OBSIDIAN_API_KEY`, `AGENT_*_MODEL`, `SUMMARY_TRIGGER_TOKENS`,
`SUMMARY_KEEP_MESSAGES`), 그래프 구조도, 도메인별 MOC 진입점.

- [ ] **Step 2: 전체 테스트 (LLM 제외)**

Run: `..\.venv\Scripts\python.exe -m pytest -v`
Expected: 전부 PASS

- [ ] **Step 3: LLM 테스트 전체**

Obsidian 실행 상태에서:

Run: `..\.venv\Scripts\python.exe -m pytest -v -m llm`
Expected: 전부 PASS

- [ ] **Step 4: 라우팅 회귀 확인**

개선 전 14/14 였던 의도 분류 케이스를 다시 돌린다 (complex 6 + general 6 + unknown 2).
`app/main_agent.py` 이전 버전의 `intent_examples` 목록을 쓴다.

Expected: 14/14 유지

- [ ] **Step 5: Langfuse 트레이스 확인**

`http://localhost:3300` 에서 최근 트레이스를 연다.
Expected: `domain:server_infra`, `domain:ai` 등 병렬 노드가 개별 span 으로 보임

- [ ] **Step 6: 커밋**

```bash
git add app/README.md
git commit -m "docs: app 멀티 에이전트 구조 문서화"
```

---

## 남은 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| Obsidian 앱이 꺼져 있으면 MCP 연결 실패 | 두 도메인 에이전트가 근거 없이 답변 | `get_obsidian_tools` 가 빈 리스트로 degrade 하고, 프롬프트가 "조회 실패"를 명시하게 함 (Task 4·6) |
| `server_infra` 와 `system_design` 경계가 모호해 중복 답변 | 병합 결과에 같은 내용 반복 | Task 5 Step 5 에서 확인. "지금 돌아가는 걸 고치는가 vs 무엇을 만들지 정하는가" 기준을 프롬프트에 이미 반영 |
| MOC 노트 이름이 실제와 다르면 진입 실패 | 볼트 검색 품질 저하 | Task 6 Step 5 에서 실제 실행으로 확인, 불일치 시 `search_simple` 로 실제 이름 확인 후 수정 |
| 병렬 3개 + 판단 2회로 호출 수 증가 | 비용·지연 증가 | 도메인은 Sonnet 배정. 단일 도메인이면 병합 LLM 호출 생략 |
| `InMemorySaver` 는 앱 재시작 시 소실 | 진행 중 대화 유실 | 토이프로젝트 범위에서 허용. 필요해지면 `PostgresSaver` 로 한 줄 교체 |

## 사전 검증된 전제

계획 작성 중 실제로 확인한 것들. 실행 시 재확인할 필요 없다.

| 전제 | 확인 결과 |
|---|---|
| 옵시디언 MCP 연결 | `https://127.0.0.1:27124/mcp/` initialize 200, 프로토콜 `2025-06-18`, 도구 16종 |
| 볼트 기술 자료 | 아키텍처 117건, 서버 62건, RAG 41건, 쿠버네티스 32건. MOC 노트 33개 |
| 볼트 태그 | 27개 전부 농업/경제/공고문 계열 → **태그 라우팅 불가** |
| `create_agent(middleware=...)` | 지원함 (파라미터 목록에 존재) |
| `SummarizationMiddleware` | langchain 1.4.0 포함, 인스턴스화 정상 |
| `ConversationSummaryBufferMemory` | **제거됨** — `langchain.memory` 모듈 없음 |
| `StreamableHttpConnection` | `headers`(Bearer) + `httpx_client_factory`(자체서명 우회) 지원 |
| `langgraph.types` | `Send`, `Command`, `interrupt` 전부 사용 가능 |
| Pinecone `testdb` | 벡터 2,710개. `Index(host=...)` 키워드 인자 필수 |
