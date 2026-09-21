"""app/(LangGraph) 과 app2/(Agent SDK) 가 같은 것을 모델에게 주는지 확인한다.

두 앱은 모듈 이름이 같아서(state, models, ...) 한 프로세스에 같이 못 올린다.
app/ 쪽은 별도 파이썬 프로세스에서 import 해 값을 JSON 으로 뽑아 비교한다.
실행 방식(그래프 vs 에이전트)은 다르지만, 모델이 읽는 프롬프트·도구 이름·설명·인자,
분기 기준값, 화면 문자열은 글자 하나까지 같아야 한다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

_DUMP = r'''
import json, os, re, inspect
os.environ.pop("LITELLM_BASE_URL", None)
os.environ.setdefault("ANTHROPIC_API_KEY", "x")
os.environ.setdefault("USER_AGENT", "parity")
from langchain_core.utils.function_calling import convert_to_openai_tool
import domain_agents, obsidian_tools, models, rag_tools, judge_agent, main_agent
import general_it_trend_topic_agent as trend
from state import MainQueryState

src = open("general_it_trend_topic_agent.py", encoding="utf-8").read()
i = src.index('query_expansion_template = """') + len('query_expansion_template = """')
expand_template = src[i:src.index('"""', i)]

def tool(t):
    f = convert_to_openai_tool(t)["function"]
    return {"name": f["name"], "description": f["description"], "parameters": f["parameters"]}

app_src = open("app.py", encoding="utf-8").read()
css = app_src[app_src.index('CSS = """') + 9: app_src.index('"""', app_src.index('CSS = """') + 9)]

print(json.dumps({
    "domain_prompts": domain_agents.DOMAIN_PROMPTS,
    "moc_hub": obsidian_tools.MOC_HUB,
    "moc_entrypoints": obsidian_tools.MOC_ENTRYPOINTS,
    "obsidian_tool_names": list(obsidian_tools.OBSIDIAN_TOOL_NAMES),
    "role_defaults": models._ROLE_DEFAULTS,
    "summary_trigger": models.SUMMARY_TRIGGER_TOKENS,
    "trend_system_prompt": trend.system_prompt,
    "expand_template": expand_template,
    "fact_check_prompt": rag_tools.FACT_CHECK_PROMPT.messages[0].prompt.template,
    "fact_check_schema": rag_tools.FactCheckResult.model_json_schema(),
    "verify_failed_prefix": rag_tools.VERIFY_FAILED_COMMENT_PREFIX,
    "ambiguity_rules": main_agent.AMBIGUITY_RULES,
    "classify_prompt": main_agent.CLASSIFY_PROMPT.messages[0].prompt.template,
    "decompose_prompt": judge_agent.DECOMPOSE_PROMPT.messages[0].prompt.template,
    "merge_prompt": judge_agent.MERGE_PROMPT.messages[0].prompt.template,
    "fix_prompt": judge_agent.FIX_PROMPT.messages[0].prompt.template,
    "guardrail_threshold": judge_agent.GUARDRAIL_THRESHOLD,
    "domain_labels": judge_agent._DOMAIN_LABELS,
    "general_reply": main_agent.general_node(MainQueryState(), {})["messages"][0].content,
    "finalize_reply": main_agent.finalize_node(MainQueryState(), {})["messages"][0].content,
    "tools": [tool(t) for t in (rag_tools.vector_search_tool, trend.expand_query, trend.web_search_tool, trend.fetch_webpage_content)],
    "ui_css": css,
    "ui_examples": re.search(r"EXAMPLE_QUESTIONS = \[(.*?)\]", app_src, re.S).group(1),
    "ui_page_config": re.search(r"st.set_page_config\((.*?)\)", app_src, re.S).group(1),
}, ensure_ascii=False))
'''


@pytest.fixture(scope="module")
def app():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out = subprocess.run(
        [sys.executable, "-c", _DUMP], cwd=APP, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_domain_prompts_identical(app):
    from domain_agents import DOMAIN_PROMPTS

    assert DOMAIN_PROMPTS == app["domain_prompts"]


def test_vault_constants_identical(app):
    import obsidian_tools

    assert obsidian_tools.MOC_HUB == app["moc_hub"]
    assert obsidian_tools.MOC_ENTRYPOINTS == app["moc_entrypoints"]
    assert list(obsidian_tools.OBSIDIAN_TOOL_NAMES) == app["obsidian_tool_names"]


def test_model_roles_identical(app):
    import models

    assert models._ROLE_DEFAULTS == app["role_defaults"]
    assert models.SUMMARY_TRIGGER_TOKENS == app["summary_trigger"]


def test_trend_prompts_identical(app):
    import general_it_trend_topic_agent as trend

    assert trend.system_prompt == app["trend_system_prompt"]
    assert trend.QUERY_EXPANSION_TEMPLATE == app["expand_template"]


def test_fact_check_prompt_and_schema_identical(app):
    import rag_tools

    assert rag_tools.FACT_CHECK_PROMPT == app["fact_check_prompt"]
    assert rag_tools.FactCheckResult.model_json_schema() == app["fact_check_schema"]
    assert rag_tools.VERIFY_FAILED_COMMENT_PREFIX == app["verify_failed_prefix"]


def test_orchestrator_rules_come_from_app_prompts(app):
    """분류·재질문·분해·병합·재작성 규칙 문장이 app/ 프롬프트와 같다."""
    import judge_agent
    import main_agent

    assert main_agent.AMBIGUITY_RULES == app["ambiguity_rules"]
    assert main_agent.INTENT_RULES in app["classify_prompt"]
    assert judge_agent.DECOMPOSE_RULES in app["decompose_prompt"]
    assert judge_agent.MERGE_RULES in app["merge_prompt"]
    assert judge_agent.FIX_RULES in app["fix_prompt"]
    assert judge_agent.GUARDRAIL_THRESHOLD == app["guardrail_threshold"]
    assert judge_agent._DOMAIN_LABELS == app["domain_labels"]


def test_fixed_replies_identical(app):
    import main_agent

    assert main_agent.GENERAL_REPLY == app["general_reply"]
    assert main_agent.EMPTY_REPLY == app["finalize_reply"]


def _norm(schema):
    """키 순서와 'type: object' 표기 차이만 무시하고 비교한다."""
    return json.loads(json.dumps(schema, sort_keys=True))


def test_tools_identical_to_app(app):
    """모델이 보는 도구 이름·설명·인자 스키마가 같다."""
    import general_it_trend_topic_agent as trend
    import rag_tools

    mine = [rag_tools.vector_search_tool, trend.expand_query, trend.web_search_tool, trend.fetch_webpage_content]
    for sdk_tool, lc in zip(mine, app["tools"]):
        assert sdk_tool.name == lc["name"]
        assert sdk_tool.description == lc["description"]
        assert _norm(sdk_tool.input_schema) == _norm(lc["parameters"]), sdk_tool.name


def test_ui_identical(app):
    import re

    src = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    i = src.index('CSS = """') + 9
    assert src[i:src.index('"""', i)] == app["ui_css"]
    assert re.search(r"EXAMPLE_QUESTIONS = \[(.*?)\]", src, re.S).group(1) == app["ui_examples"]
    assert re.search(r"st.set_page_config\((.*?)\)", src, re.S).group(1) == app["ui_page_config"]


def test_streamlit_theme_identical():
    a = (APP / ".streamlit" / "config.toml").read_bytes()
    b = (Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_bytes()
    assert a == b


def test_same_module_filenames_as_app():
    """app/ 과 같은 파일 구성(LLM 흐름을 담당하는 모듈 이름)을 유지한다."""
    names = {p.name for p in (Path(__file__).resolve().parents[1]).glob("*.py")}
    expected = {p.name for p in APP.glob("*.py")}
    assert expected <= names, expected - names


@pytest.mark.llm
@pytest.mark.parametrize("query", ["Gemini 1.5 long context", "residual learning deep networks"])
def test_vector_search_returns_same_docs_as_app(query):
    """LangChain(PineconeVectorStore+CohereRerank) 과 공식 클라이언트 구현이 같은 문서를 같은 순서로 준다."""
    import rag_tools

    code = (
        "import sys, json; from dotenv import load_dotenv; load_dotenv('.env'); import rag_tools;"
        "print(json.dumps([[d.page_content, round(d.metadata['relevance_score'], 6)]"
        " for d in rag_tools.vector_search_tool.invoke({'query': sys.argv[1]})], ensure_ascii=False))"
    )
    out = subprocess.run([sys.executable, "-c", code, query], cwd=APP, capture_output=True, text=True,
                         encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=120)
    theirs = json.loads(out.stdout.strip().splitlines()[-1])
    mine = [[d["page_content"], round(d["metadata"]["relevance_score"], 6)] for d in rag_tools.search_papers(query)]
    assert mine == theirs
