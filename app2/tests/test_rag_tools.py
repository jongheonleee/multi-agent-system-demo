import asyncio
import json
from types import SimpleNamespace

import pytest

import rag_tools


def _match(similarity, text="청크", **meta):
    md = {"text": text, **meta} if text is not None else dict(meta)
    return SimpleNamespace(score=similarity, metadata=md)


@pytest.fixture
def fake_pinecone(monkeypatch):
    """Pinecone/OpenAI/Cohere 를 가짜로 바꾼다. 호출 인자를 기록한다."""
    import pinecone

    seen = {}

    class Index:
        def query(self, vector, top_k, include_metadata):
            seen.update(vector=vector, top_k=top_k, include_metadata=include_metadata)
            return SimpleNamespace(matches=seen["matches"])

    class Pc:
        def Index(self, host):  # noqa: N802 - SDK 이름
            seen["host"] = host
            return Index()

    def fake_rerank(query, docs):
        seen["reranked"] = [d["page_content"] for d in docs]
        return [{**d, "metadata": {**d["metadata"], "relevance_score": 0.9}} for d in docs[:3]]

    monkeypatch.setenv("PINECONE_HOST", "idx.pinecone.io")
    monkeypatch.setattr(pinecone, "Pinecone", Pc)
    monkeypatch.setattr(rag_tools, "_embed", lambda q: [0.1])
    monkeypatch.setattr(rag_tools, "_rerank", fake_rerank)
    return seen


def test_search_applies_langchain_relevance_threshold(fake_pinecone):
    """(cosine+1)/2 >= 0.7 — LangChain similarity_score_threshold 와 같은 기준."""
    fake_pinecone["matches"] = [
        _match(0.41, "통과", source="a.pdf", file_path="/tmp/a", score=1),
        _match(0.39, "탈락"),
        _match(0.9, None, source="텍스트 없음"),
    ]
    docs = rag_tools.search_papers("질의")
    assert fake_pinecone["host"] == "idx.pinecone.io"
    assert fake_pinecone["top_k"] == 10 and fake_pinecone["include_metadata"] is True
    assert fake_pinecone["reranked"] == ["통과"]
    assert docs == [{"page_content": "통과", "metadata": {"source": "a.pdf", "relevance_score": 0.9}}]


def test_search_without_host_returns_empty(monkeypatch):
    monkeypatch.delenv("PINECONE_HOST", raising=False)
    assert rag_tools.search_papers("q") == []


def test_search_failure_returns_empty(fake_pinecone, monkeypatch):
    def boom(q):
        raise RuntimeError("openai down")

    monkeypatch.setattr(rag_tools, "_embed", boom)
    assert rag_tools.search_papers("q") == []


def test_vector_search_tool_returns_json_text(fake_pinecone):
    fake_pinecone["matches"] = [_match(0.8, "RAG 는 검색증강")]
    out = asyncio.run(rag_tools.vector_search_tool.handler({"query": "RAG"}))
    assert json.loads(out["content"][0]["text"])[0]["page_content"] == "RAG 는 검색증강"


def test_fact_check_without_context_returns_unverifiable():
    out = asyncio.run(rag_tools.fact_check("아무 주장", None))
    assert out["overall_accuracy"] == 0.0
    assert out["sentences"] == []
    assert out["overall_accuracy_comment"].startswith(rag_tools.VERIFY_FAILED_COMMENT_PREFIX)


def test_fact_check_uses_judge_with_16000_tokens(monkeypatch):
    seen = {}

    async def fake(role, prompt, schema, max_tokens):
        seen.update(role=role, prompt=prompt, schema=schema, max_tokens=max_tokens)
        return schema(sentences=[], overall_accuracy=0.9, overall_accuracy_comment="ok")

    monkeypatch.setattr(rag_tools, "ask_structured", fake)
    out = asyncio.run(rag_tools.fact_check("답", "출처"))
    assert out["overall_accuracy"] == 0.9
    assert (seen["role"], seen["max_tokens"], seen["schema"]) == ("judge", 16000, rag_tools.FactCheckResult)
    assert seen["prompt"] == rag_tools.FACT_CHECK_PROMPT.format(answer="답", context="출처")


def test_fact_check_error_is_marked_unverifiable_not_inaccurate(monkeypatch):
    """검증 실패를 정확도 0.0 과 구분하지 않으면 멀쩡한 답을 재작성시킨다."""

    async def boom(*a, **k):
        raise rag_tools.LLMCallError("응답 잘림")

    monkeypatch.setattr(rag_tools, "ask_structured", boom)
    out = asyncio.run(rag_tools.fact_check("주장", "출처"))
    assert out["overall_accuracy_comment"].startswith("[검증불가] 사실 확인 중 오류:")
    assert out["sentences"][0]["verdict"] == "unverifiable"


def test_no_hardcoded_gemini_snippet_in_source():
    import inspect

    assert "본 리포트에서 소개하는" not in inspect.getsource(rag_tools)


@pytest.mark.llm
def test_live_vector_search_returns_reranked_docs():
    docs = rag_tools.search_papers("residual learning deep networks")
    assert docs and len(docs) <= 3
    assert all("relevance_score" in d["metadata"] for d in docs)
