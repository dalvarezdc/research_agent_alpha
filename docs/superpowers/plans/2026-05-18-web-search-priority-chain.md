# Web Search Priority Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current concatenate-all web search behaviour with a key-detection priority chain: Tavily first (if `TAVILY_API_KEY` set), then SerpAPI (if `SERPAPI_API_KEY` set), then DuckDuckGo always as the guaranteed fallback.

**Architecture:** All logic lives in `web_research/search.py`. `WebResearchClient.search()` tries providers in priority order and returns as soon as one yields results, with DuckDuckGo always attempted last if no premium provider returned anything. The `LangChainAgentConfig` default providers list is simplified to `["duckduckgo"]` since the priority chain handles key-based provider activation internally. No agent code changes needed.

**Tech Stack:** `langchain-community` (TavilySearchResults, DuckDuckGoSearchResults, SerpAPIWrapper), `python-dotenv` (already installed), existing `WebResearchClient` / `WebSearchResult` types.

---

## Task 1: Priority chain in WebResearchClient.search()

- [ ] Create `tests/test_web_research.py` with the failing tests
- [ ] Verify all tests fail (pytest confirms failures before implementation)
- [ ] Replace `search()` in `web_research/search.py` with priority chain implementation
- [ ] Verify all Task 1 tests pass

### Step 1.1 — Write failing tests

Create `tests/test_web_research.py` with the following content:

```python
"""Tests for WebResearchClient priority chain behaviour."""
from unittest.mock import MagicMock, patch
import pytest


def _make_result(provider="tavily"):
    return [{"title": f"{provider} result", "url": f"https://{provider}.com", "content": "snippet"}]


def test_tavily_used_when_key_set_and_returns_results(monkeypatch):
    """When TAVILY_API_KEY is set and Tavily returns results, return Tavily results only."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    mock_tavily_tool = MagicMock()
    mock_tavily_tool.return_value.invoke.return_value = _make_result("tavily")

    with patch("web_research.search.TavilySearchResults", mock_tavily_tool):
        from web_research.search import WebResearchClient
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
        from web_research.search import WebResearchClient
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
        from web_research.search import WebResearchClient
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
        from web_research.search import WebResearchClient
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
        from web_research.search import WebResearchClient
        client = WebResearchClient(providers=["duckduckgo"])  # no tavily
        results = client.search("test query")

    mock_tavily_tool.assert_not_called()
    assert results[0].provider == "duckduckgo"
```

### Step 1.2 — Verify tests fail

Run:
```bash
uv run python -m pytest tests/test_web_research.py -v
```

Expected: all 5 tests FAIL (the current `search()` concatenates all providers instead of using a priority chain).

### Step 1.3 — Implement the priority chain

Replace the `search()` method in `web_research/search.py` with:

```python
    def search(self, query: str) -> List[WebSearchResult]:
        """Search using a priority chain: Tavily → SerpAPI → DuckDuckGo.

        Returns results from the first provider that yields any, with
        DuckDuckGo always attempted last if no premium provider succeeded.
        """
        # Priority 1: Tavily (best quality, needs API key)
        if "tavily" in self.providers and os.getenv("TAVILY_API_KEY"):
            results = self._search_tavily(query)
            if results:
                return results

        # Priority 2: SerpAPI (Google results, needs API key)
        if "serpapi" in self.providers and os.getenv("SERPAPI_API_KEY"):
            results = self._search_serpapi(query)
            if results:
                return results

        # Priority 3: DuckDuckGo (no key needed, always the fallback)
        if "duckduckgo" in self.providers:
            return self._search_duckduckgo(query)

        return []
```

The full updated `web_research/search.py` after the change:

```python
"""
Web research client using LangChain tools (Tavily, SerpAPI, DuckDuckGo).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

try:
    from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchResults
except ImportError:
    TavilySearchResults = None
    DuckDuckGoSearchResults = None

try:
    from langchain_community.utilities import SerpAPIWrapper
except ImportError:
    SerpAPIWrapper = None


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str
    provider: str


class WebResearchClient:
    def __init__(
        self,
        providers: Optional[List[str]] = None,
        max_results: int = 5,
    ) -> None:
        self.providers = providers or ["tavily", "serpapi", "duckduckgo"]
        self.max_results = max_results

    def search(self, query: str) -> List[WebSearchResult]:
        """Search using a priority chain: Tavily → SerpAPI → DuckDuckGo.

        Returns results from the first provider that yields any, with
        DuckDuckGo always attempted last if no premium provider succeeded.
        """
        # Priority 1: Tavily (best quality, needs API key)
        if "tavily" in self.providers and os.getenv("TAVILY_API_KEY"):
            results = self._search_tavily(query)
            if results:
                return results

        # Priority 2: SerpAPI (Google results, needs API key)
        if "serpapi" in self.providers and os.getenv("SERPAPI_API_KEY"):
            results = self._search_serpapi(query)
            if results:
                return results

        # Priority 3: DuckDuckGo (no key needed, always the fallback)
        if "duckduckgo" in self.providers:
            return self._search_duckduckgo(query)

        return []

    def _search_tavily(self, query: str) -> List[WebSearchResult]:
        if TavilySearchResults is None:
            return []
        if not os.getenv("TAVILY_API_KEY"):
            return []
        try:
            tool = TavilySearchResults(max_results=self.max_results)
            raw = tool.invoke(query)
            return self._normalize_results(raw, provider="tavily")
        except Exception:
            return []

    def _search_serpapi(self, query: str) -> List[WebSearchResult]:
        if SerpAPIWrapper is None:
            return []
        if not os.getenv("SERPAPI_API_KEY"):
            return []
        try:
            wrapper = SerpAPIWrapper()
            raw: Any
            if hasattr(wrapper, "results"):
                raw = wrapper.results(query)
            else:
                raw = wrapper.run(query)
            return self._normalize_results(raw, provider="serpapi")
        except Exception:
            return []

    def _search_duckduckgo(self, query: str) -> List[WebSearchResult]:
        if DuckDuckGoSearchResults is None:
            return []
        try:
            tool = DuckDuckGoSearchResults(max_results=self.max_results)
            raw = tool.invoke(query)
            return self._normalize_results(raw, provider="duckduckgo")
        except Exception:
            return []

    def _normalize_results(self, raw: Any, provider: str) -> List[WebSearchResult]:
        results: List[WebSearchResult] = []
        if isinstance(raw, str):
            return results
        if isinstance(raw, dict):
            raw = raw.get("results") or raw.get("organic_results") or raw.get("items") or []
        for item in self._ensure_list(raw):
            title = item.get("title") or item.get("headline") or ""
            url = item.get("url") or item.get("link") or ""
            snippet = item.get("content") or item.get("snippet") or item.get("summary") or ""
            if not title and not snippet:
                continue
            results.append(
                WebSearchResult(
                    title=str(title),
                    url=str(url),
                    snippet=str(snippet),
                    source=self._infer_source(url),
                    provider=provider,
                )
            )
        return results

    def _ensure_list(self, value: Any) -> Iterable[dict]:
        if isinstance(value, list):
            return value
        return []

    def _infer_source(self, url: str) -> str:
        if not url:
            return "web"
        return url.split("/")[2] if "://" in url else url
```

### Step 1.4 — Verify tests pass

Run:
```bash
uv run python -m pytest tests/test_web_research.py::test_tavily_used_when_key_set_and_returns_results tests/test_web_research.py::test_serpapi_used_when_tavily_key_absent tests/test_web_research.py::test_duckduckgo_used_when_no_premium_keys tests/test_web_research.py::test_duckduckgo_fallback_when_tavily_returns_empty tests/test_web_research.py::test_tavily_skipped_when_not_in_providers -v
```

Expected: all 5 tests PASS.

---

## Task 2: Log active provider in _build_web_context

- [ ] Add the `test_build_web_context_logs_provider` test to `tests/test_web_research.py`
- [ ] Verify it fails before implementation
- [ ] Update `_build_web_context` in `langchain_agents/base.py` to log the active provider
- [ ] Verify the test passes

### Step 2.1 — Write failing test

Append to `tests/test_web_research.py`:

```python
def test_build_web_context_logs_provider(monkeypatch, caplog):
    """_build_web_context logs which provider returned results."""
    import logging
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    mock_ddg_tool = MagicMock()
    mock_ddg_tool.return_value.invoke.return_value = _make_result("duckduckgo")

    with patch("web_research.search.DuckDuckGoSearchResults", mock_ddg_tool):
        from web_research.search import WebResearchClient
        from langchain_agents.base import LangChainAgentBase, LangChainAgentConfig

        # Build a minimal base instance with web research enabled
        config = LangChainAgentConfig(
            primary_llm_provider="openai",
            fallback_providers=[],
            enable_web_research=True,
            enable_audit=False,
        )

        mock_llm_mgr = MagicMock()
        mock_provider = MagicMock()
        mock_llm_mgr.get_available_provider.return_value = mock_provider

        with patch("langchain_agents.base.create_llm_manager", return_value=mock_llm_mgr), \
             patch("langchain_agents.base.WebResearchClient", return_value=WebResearchClient(
                 providers=["duckduckgo"]
             )):
            agent = LangChainAgentBase.__new__(LangChainAgentBase)
            agent.config = config
            agent.enable_web_research = True
            agent.web_research = WebResearchClient(providers=["duckduckgo"])
            agent.web_context = None

            with caplog.at_level(logging.INFO, logger="langchain_agents.base"):
                ctx = agent._build_web_context("cholesterol test")

    assert "duckduckgo" in caplog.text
    assert len(ctx) > 0
```

### Step 2.2 — Verify test fails

Run:
```bash
uv run python -m pytest tests/test_web_research.py::test_build_web_context_logs_provider -v
```

Expected: FAIL — `_build_web_context` currently does not emit a log message naming the provider.

### Step 2.3 — Implement the logging

In `langchain_agents/base.py`, replace the existing `_build_web_context` method with:

```python
    def _build_web_context(self, query: str) -> str:
        if not self.web_research:
            return ""
        results = self.web_research.search(query)
        if not results:
            return ""
        provider_used = results[0].provider if results else "none"
        import logging
        logging.getLogger(__name__).info(
            f"Web research: {len(results)} results from '{provider_used}' for query: {query[:60]}"
        )
        lines = []
        for idx, item in enumerate(results, 1):
            lines.append(
                f"[{idx}] {item.title} ({item.source}) - {item.snippet} {item.url}".strip()
            )
        return "\n".join(lines)
```

### Step 2.4 — Verify test passes

Run:
```bash
uv run python -m pytest tests/test_web_research.py::test_build_web_context_logs_provider -v
```

Expected: PASS.

---

## Task 3: Integration smoke tests + full test run

- [ ] Append integration smoke tests to `tests/test_web_research.py`
- [ ] Verify new tests pass
- [ ] Run full test suite and confirm no regressions

### Step 3.1 — Write integration smoke tests

Append to `tests/test_web_research.py`:

```python
def test_search_returns_empty_when_all_providers_absent(monkeypatch):
    """Returns empty list when DuckDuckGoSearchResults is None and no keys set."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    with patch("web_research.search.TavilySearchResults", None), \
         patch("web_research.search.DuckDuckGoSearchResults", None), \
         patch("web_research.search.SerpAPIWrapper", None):
        from web_research.search import WebResearchClient
        client = WebResearchClient(providers=["tavily", "serpapi", "duckduckgo"])
        results = client.search("test query")

    assert results == []


def test_serpapi_skipped_when_empty_results_falls_to_duckduckgo(monkeypatch):
    """When SerpAPI returns empty, DuckDuckGo is tried."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

    mock_serpapi_cls = MagicMock()
    mock_wrapper = MagicMock()
    mock_wrapper.results.return_value = {"organic_results": []}  # empty
    mock_serpapi_cls.return_value = mock_wrapper

    mock_ddg_tool = MagicMock()
    mock_ddg_tool.return_value.invoke.return_value = _make_result("duckduckgo")

    with patch("web_research.search.SerpAPIWrapper", mock_serpapi_cls), \
         patch("web_research.search.DuckDuckGoSearchResults", mock_ddg_tool):
        from web_research.search import WebResearchClient
        client = WebResearchClient(providers=["tavily", "serpapi", "duckduckgo"])
        results = client.search("test query")

    assert results[0].provider == "duckduckgo"
```

### Step 3.2 — Run all web research tests

Run:
```bash
uv run python -m pytest tests/test_web_research.py -v
```

Expected: all 8 tests PASS.

### Step 3.3 — Run full suite to confirm no regressions

Run:
```bash
uv run python -m pytest tests/test_web_research.py tests/test_llm_integrations.py tests/test_observability.py tests/test_cost_tracker.py -q
```

Expected: all tests PASS, no regressions introduced.
