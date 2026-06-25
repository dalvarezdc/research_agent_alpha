"""Tests for WebResearchClient priority chain behaviour."""
from unittest.mock import MagicMock, patch
import pytest

from web_research.search import WebResearchClient


def _make_result(provider="tavily"):
    return [{"title": f"{provider} result", "url": f"https://{provider}.com", "content": "snippet"}]


def test_tavily_used_when_key_set_and_returns_results(monkeypatch):
    """When TAVILY_API_KEY is set and Tavily returns results, return Tavily results only."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    mock_tavily_tool = MagicMock()
    mock_tavily_tool.return_value.invoke.return_value = _make_result("tavily")

    with patch("web_research.search.TavilySearchResults", mock_tavily_tool):
        client = WebResearchClient(providers=["tavily", "serpapi", "duckduckgo"])
        results = client.search("test query")

    assert len(results) == 1
    assert results[0].provider == "tavily"


def test_serpapi_used_when_tavily_key_absent(monkeypatch):
    """When TAVILY_API_KEY absent but SERPAPI_API_KEY set and returns results, use SerpAPI."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

    mock_serpapi_cls = MagicMock()
    mock_wrapper = MagicMock()
    mock_wrapper.results.return_value = {"organic_results": _make_result("serpapi")}
    mock_serpapi_cls.return_value = mock_wrapper

    with patch("web_research.search.SerpAPIWrapper", mock_serpapi_cls):
        client = WebResearchClient(providers=["tavily", "serpapi", "duckduckgo"])
        results = client.search("test query")

    assert len(results) == 1
    assert results[0].provider == "serpapi"


def test_duckduckgo_used_when_no_premium_keys(monkeypatch):
    """When no premium API keys set, fall back to DuckDuckGo."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    mock_ddg_tool = MagicMock()
    mock_ddg_tool.return_value.invoke.return_value = _make_result("duckduckgo")

    with patch("web_research.search.DuckDuckGoSearchResults", mock_ddg_tool):
        client = WebResearchClient(providers=["tavily", "serpapi", "duckduckgo"])
        results = client.search("test query")

    assert len(results) == 1
    assert results[0].provider == "duckduckgo"


def test_duckduckgo_fallback_when_tavily_returns_empty(monkeypatch):
    """When Tavily key is set but returns no results, fall through to DuckDuckGo."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    mock_tavily_tool = MagicMock()
    mock_tavily_tool.return_value.invoke.return_value = []  # empty results

    mock_ddg_tool = MagicMock()
    mock_ddg_tool.return_value.invoke.return_value = _make_result("duckduckgo")

    with patch("web_research.search.TavilySearchResults", mock_tavily_tool), \
         patch("web_research.search.DuckDuckGoSearchResults", mock_ddg_tool):
        client = WebResearchClient(providers=["tavily", "serpapi", "duckduckgo"])
        results = client.search("test query")

    assert len(results) == 1
    assert results[0].provider == "duckduckgo"


def test_tavily_skipped_when_not_in_providers(monkeypatch):
    """Tavily is not tried even if key is set when 'tavily' not in providers list."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    mock_tavily_tool = MagicMock()
    mock_ddg_tool = MagicMock()
    mock_ddg_tool.return_value.invoke.return_value = _make_result("duckduckgo")

    with patch("web_research.search.TavilySearchResults", mock_tavily_tool), \
         patch("web_research.search.DuckDuckGoSearchResults", mock_ddg_tool):
        client = WebResearchClient(providers=["duckduckgo"])  # no tavily
        results = client.search("test query")

    mock_tavily_tool.assert_not_called()
    assert results[0].provider == "duckduckgo"


def test_tavily_exception_falls_through_to_duckduckgo(monkeypatch):
    """When Tavily raises mid-call, the chain falls through to DuckDuckGo."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    mock_tavily_tool = MagicMock()
    mock_tavily_tool.return_value.invoke.side_effect = RuntimeError("network error")

    mock_ddg_tool = MagicMock()
    mock_ddg_tool.return_value.invoke.return_value = _make_result("duckduckgo")

    with patch("web_research.search.TavilySearchResults", mock_tavily_tool), \
         patch("web_research.search.DuckDuckGoSearchResults", mock_ddg_tool):
        client = WebResearchClient(providers=["tavily", "serpapi", "duckduckgo"])
        results = client.search("test query")

    assert len(results) == 1
    assert results[0].provider == "duckduckgo"


def test_search_returns_empty_when_providers_empty(monkeypatch):
    """Returns [] when no providers are configured."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    client = WebResearchClient(providers=[])
    results = client.search("test query")

    assert results == []


def test_duckduckgo_string_response_parsed(monkeypatch):
    """DuckDuckGo string response is parsed into results, not silently dropped."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    # Simulate DDG returning a formatted string
    ddg_string = "[cholesterol is complex] (https://pubmed.ncbi.nlm.nih.gov/12345)\n[HDL is protective] (https://heart.org/article)"

    mock_ddg_tool = MagicMock()
    mock_ddg_tool.return_value.invoke.return_value = ddg_string

    with patch("web_research.search.DuckDuckGoSearchResults", mock_ddg_tool):
        client = WebResearchClient(providers=["duckduckgo"])
        results = client.search("cholesterol")

    assert len(results) >= 1
    assert all(r.provider == "duckduckgo" for r in results)
    assert any("cholesterol" in r.snippet.lower() or "cholesterol" in r.title.lower() for r in results)


def test_build_web_context_logs_provider(monkeypatch, caplog):
    """_build_web_context logs which provider returned results."""
    import logging
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    mock_ddg_tool = MagicMock()
    mock_ddg_tool.return_value.invoke.return_value = [
        {"title": "DDG result", "url": "https://example.com", "content": "snippet"}
    ]

    with patch("web_research.search.DuckDuckGoSearchResults", mock_ddg_tool):
        from langchain_agents.base import LangChainAgentBase
        from web_research.search import WebResearchClient

        agent = LangChainAgentBase.__new__(LangChainAgentBase)
        agent.enable_web_research = True
        agent.web_research = WebResearchClient(providers=["duckduckgo"])
        agent.web_context = None

        with caplog.at_level(logging.INFO, logger="langchain_agents.base"):
            ctx = agent._build_web_context("cholesterol test")

    assert "duckduckgo" in caplog.text
    assert len(ctx) > 0


def test_search_returns_empty_when_all_providers_absent(monkeypatch):
    """Returns empty list when DuckDuckGoSearchResults is None and no keys set."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    with patch("web_research.search.TavilySearchResults", None), \
         patch("web_research.search.DuckDuckGoSearchResults", None), \
         patch("web_research.search.SerpAPIWrapper", None):
        client = WebResearchClient(providers=["tavily", "serpapi", "duckduckgo"])
        results = client.search("test query")

    assert results == []


def test_serpapi_skipped_when_empty_results_falls_to_duckduckgo(monkeypatch):
    """When SerpAPI returns empty results, DuckDuckGo is tried."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

    mock_serpapi_cls = MagicMock()
    mock_wrapper = MagicMock()
    mock_wrapper.results.return_value = {"organic_results": []}  # empty
    mock_serpapi_cls.return_value = mock_wrapper

    mock_ddg_tool = MagicMock()
    mock_ddg_tool.return_value.invoke.return_value = [
        {"title": "DDG result", "url": "https://example.com", "content": "snippet"}
    ]

    with patch("web_research.search.SerpAPIWrapper", mock_serpapi_cls), \
         patch("web_research.search.DuckDuckGoSearchResults", mock_ddg_tool):
        client = WebResearchClient(providers=["tavily", "serpapi", "duckduckgo"])
        results = client.search("test query")

    assert results[0].provider == "duckduckgo"
