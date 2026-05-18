"""
Web research client using LangChain tools (Tavily, SerpAPI, DuckDuckGo).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

try:
    from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchResults
except ImportError:  # pragma: no cover
    TavilySearchResults = None
    DuckDuckGoSearchResults = None

try:
    from langchain_community.utilities import SerpAPIWrapper
except ImportError:  # pragma: no cover
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
            # DDG sometimes returns a formatted string like:
            # "[snippet] [url]\n[snippet] [url]\n..."
            # Parse it into WebSearchResult objects directly.
            if isinstance(raw, str) and raw.strip():
                return self._parse_ddg_string(raw)
            return self._normalize_results(raw, provider="duckduckgo")
        except Exception:
            return []

    def _parse_ddg_string(self, raw: str) -> List[WebSearchResult]:
        """Parse DuckDuckGo's formatted string response into WebSearchResult objects."""
        import re
        results = []
        # DuckDuckGo string format: entries separated by newlines or "snippet [url]" pattern
        # Try to extract url+snippet pairs using a regex
        pattern = re.compile(r"\[([^\]]+)\]\s*\(?(https?://[^\s\)]+)\)?")
        for match in pattern.finditer(raw):
            snippet, url = match.group(1), match.group(2)
            results.append(
                WebSearchResult(
                    title=snippet[:80],
                    url=url,
                    snippet=snippet,
                    source=self._infer_source(url),
                    provider="duckduckgo",
                )
            )
        if not results:
            # Fallback: treat each non-empty line as a snippet with no URL
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    results.append(
                        WebSearchResult(
                            title=line[:80],
                            url="",
                            snippet=line,
                            source="web",
                            provider="duckduckgo",
                        )
                    )
        return results

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
