# Web Research

A small web-search client that gives agents live web context. It uses a
**priority chain** of providers and returns normalized results, so callers don't
need to know which backend answered.

## Provider priority chain

```
Tavily  →  SerpAPI  →  DuckDuckGo
(API key)   (API key)   (no key, always the fallback)
```

`search(query)` returns results from the **first** provider that yields any.
Premium providers are only attempted when their API key is present; DuckDuckGo
needs no key and is always the last resort.

| Provider | Key env var | Notes |
|----------|-------------|-------|
| Tavily | `TAVILY_API_KEY` | Best quality; tried first |
| SerpAPI | `SERPAPI_API_KEY` | Google results; tried second |
| DuckDuckGo | — | No key; always-available fallback |

## Quick start

```python
from web_research import WebResearchClient

client = WebResearchClient(
    providers=["tavily", "serpapi", "duckduckgo"],  # order = priority
    max_results=5,
)
results = client.search("latest vitamin D supplementation guidelines")

for r in results:
    print(r.provider, "-", r.title)
    print("  ", r.url)
    print("  ", r.snippet)
```

## Result shape

```python
@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str
    source: str      # inferred domain/source
    provider: str    # tavily | serpapi | duckduckgo
```

## Module map

| File | Responsibility |
|------|----------------|
| `search.py` | `WebResearchClient` (priority-chain search + result normalization) and `WebSearchResult`. |
| `__init__.py` | Public exports. |

## How agents use it

`LangChainAgentBase` (in `langchain_agents/`) lazily constructs a
`WebResearchClient` when `enable_web_research=True`, then calls
`_build_web_context(query)` to format results into a context block appended to
prompts:

```
[1] Title (source) - snippet https://url
[2] ...
```

The router enables web research by default; disable it with
`uv run python router.py --no-web-search`.

## Behavior & resilience

- **Graceful degradation** — each provider call is wrapped in try/except and
  returns `[]` on error or missing dependency, so the chain falls through
  cleanly to the next provider.
- **DuckDuckGo parsing** — DDG sometimes returns a formatted string; the client
  parses `[snippet] (url)` pairs into `WebSearchResult` objects, falling back to
  treating each line as a URL-less snippet.
- **Optional dependencies** — `langchain_community` search tools are imported
  defensively; if unavailable, that provider is simply skipped.

## Configuration

Set whichever keys you have in `.env` / `.env.dev`:

```bash
TAVILY_API_KEY="tvly-..."
SERPAPI_API_KEY="..."
# DuckDuckGo requires no configuration.
```

Tune behavior per client via `providers=[...]` (subset/order) and
`max_results=N`.
