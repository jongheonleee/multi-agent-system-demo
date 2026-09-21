import asyncio
import json

import general_it_trend_topic_agent as trend


def _text(result):
    return result["content"][0]["text"]


def test_trend_agent_definition():
    agent = trend.build_trend_agent()
    assert agent.prompt == trend.system_prompt
    assert agent.model == "claude-sonnet-5"
    assert agent.tools == [
        "mcp__web__expand_query", "mcp__web__web_search_tool", "mcp__web__fetch_webpage_content",
    ]
    assert agent.mcpServers == ["web"]
    assert agent.background is False


def test_tool_names_and_descriptions_match_app():
    tools = {t.name: t for t in (trend.expand_query, trend.web_search_tool, trend.fetch_webpage_content)}
    assert tools["expand_query"].description == "입력된 질문을 여러 개의 유사한 질문으로 확장하는 도구"
    assert tools["web_search_tool"].description == "검색 쿼리를 사용하여 웹에서 정보를 서칭하는 도구"
    assert tools["fetch_webpage_content"].description == "웹 서칭에서 URL의 내용을 가져오는 도구"
    assert tools["web_search_tool"].input_schema["properties"]["num_results"]["default"] == 2


def test_expand_keeps_original_plus_two(monkeypatch):
    seen = {}

    async def fake_ask(role, prompt, **kw):
        seen["role"], seen["prompt"] = role, prompt
        return "확장1\n\n  확장2  \n확장3\n"

    monkeypatch.setattr(trend, "ask", fake_ask)
    out = asyncio.run(trend.expand_query.handler({"query": "원 질문"}))
    assert json.loads(_text(out)) == ["원 질문", "확장1", "확장2"]
    assert seen["role"] == "trend"
    assert "원본 질문이 주어집니다: 원 질문" in seen["prompt"]


def test_web_search_sends_same_serper_request(monkeypatch):
    calls = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"organic": [{"title": "T", "link": "https://x", "snippet": "S", "position": 1}]}

    def fake_post(url, headers, params, timeout):
        calls.update(url=url, headers=headers, params=params)
        return Resp()

    import requests

    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setattr(requests, "post", fake_post)
    out = asyncio.run(trend.web_search_tool.handler({"query": "맥북 가격"}))
    assert json.loads(_text(out)) == [{"title": "T", "url": "https://x", "snippet": "S"}]
    assert calls["url"] == "https://google.serper.dev/search"
    assert calls["params"] == {"q": "맥북 가격", "gl": "kr", "hl": "ko", "num": 2}
    assert calls["headers"]["X-API-KEY"] == "serper-key"


def test_fetch_summarizes_each_page(monkeypatch):
    prompts = []

    def fake_load(url):
        return {"page_content": f"본문:{url}", "metadata": {"source": url, "title": "제목"}}

    async def fake_ask(role, prompt, **kw):
        prompts.append((role, prompt))
        return "요약"

    monkeypatch.setattr(trend, "load_page", fake_load)
    monkeypatch.setattr(trend, "ask", fake_ask)
    out = asyncio.run(trend.fetch_webpage_content.handler({"urls": ["https://a", "https://b"]}))
    docs = json.loads(_text(out))
    assert [d["metadata"]["source"] for d in docs] == ["https://a", "https://b"]
    assert all(d["page_content"] == "요약" for d in docs)
    assert prompts[0] == ("trend", "## 임 문서를 핵심만 남겨서 200자 내외로 요약해줘, \n\n ##Document: \n본문:https://a")


def test_load_page_builds_webbaseloader_metadata(monkeypatch):
    import requests

    html = "<html lang='ko'><head><title>제목</title><meta name='description' content='설명'></head><body><p>본문</p></body></html>"

    class Resp:
        text = html
        apparent_encoding = "utf-8"
        encoding = None

    monkeypatch.setattr(requests, "get", lambda url, headers, timeout: Resp())
    page = trend.load_page("https://x")
    assert page["metadata"] == {"source": "https://x", "title": "제목", "description": "설명", "language": "ko"}
    assert "본문" in page["page_content"]
