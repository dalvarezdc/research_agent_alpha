# GCP Vertex AI Auth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Claude-on-Vertex-AI and Gemini-on-Vertex-AI as first-class LLM providers, activated when `IS_GCP=true` in `.env.dev`, while keeping all existing API-key paths fully backwards-compatible.

**Architecture:** Two new `LLMInterface` subclasses (`ClaudeVertexLLM`, `GeminiVertexLLM`) mirror the existing provider pattern. `create_llm_manager()` inspects `IS_GCP` at runtime and instantiates Vertex variants instead of direct-API variants when the flag is set. `.env.dev` is loaded by `router.py` at startup via `load_dotenv(".env.dev", override=False)` before the existing `.env` load, so the dev file enriches but does not clobber the base config.

**Tech Stack:** `langchain-google-vertexai`, `langchain-google-genai`, `google-auth` (already installed), `python-dotenv` (already installed), existing `LLMInterface` / `LLMManager` / `LLMConfig` / `LLMProvider` types.

---

## File map

| File | Action | Purpose |
|------|--------|---------|
| `llm_integrations.py` | Modify | Add `CLAUDE_VERTEX`, `GEMINI_VERTEX` enum values; add `ClaudeVertexLLM`, `GeminiVertexLLM` classes; extend `create_llm_manager()` and `_initialize_providers()` |
| `router.py` | Modify | Load `.env.dev` at startup before `.env` |
| `pyproject.toml` | Modify | Add `langchain-google-vertexai` and `langchain-google-genai` deps |
| `tests/test_gcp_providers.py` | Create | Unit tests for both new provider classes and `IS_GCP` routing logic |

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add Google AI dependencies to pyproject.toml**

In `pyproject.toml`, inside the `dependencies = [` list, add after the `"langchain-anthropic>=0.1.0",` line:

```toml
    "langchain-google-vertexai>=2.0.0",
    "langchain-google-genai>=2.0.0",
```

- [ ] **Step 2: Install the new dependencies**

```bash
uv sync
```

Expected: resolves and installs `langchain-google-vertexai` and `langchain-google-genai` without errors.

- [ ] **Step 3: Verify imports work**

```bash
uv run python -c "from langchain_google_vertexai import ChatVertexAI; from langchain_google_genai import ChatGoogleGenerativeAI; print('OK')"
```

Expected output: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add langchain-google-vertexai and langchain-google-genai dependencies"
```

---

### Task 2: Extend LLMProvider enum

**Files:**
- Modify: `llm_integrations.py:128-139`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gcp_providers.py`:

```python
"""Tests for GCP Vertex AI provider classes and IS_GCP routing."""
import pytest
from unittest.mock import MagicMock, patch


def test_claude_vertex_provider_enum_exists():
    """CLAUDE_VERTEX and GEMINI_VERTEX must exist in LLMProvider."""
    from llm_integrations import LLMProvider
    assert hasattr(LLMProvider, "CLAUDE_VERTEX")
    assert hasattr(LLMProvider, "GEMINI_VERTEX")
    assert LLMProvider.CLAUDE_VERTEX.value == "claude-vertex"
    assert LLMProvider.GEMINI_VERTEX.value == "gemini-vertex"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_claude_vertex_provider_enum_exists -v
```

Expected: FAIL with `AttributeError: CLAUDE_VERTEX`

- [ ] **Step 3: Add the two enum values to LLMProvider**

In `llm_integrations.py`, find the `class LLMProvider(Enum):` block (lines 128–139) and add two entries after the existing ones:

```python
class LLMProvider(Enum):
    """Supported LLM providers"""
    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_OPUS = "claude-opus"
    OPENAI = "openai"
    OLLAMA = "ollama"
    # Grok 4.3 — current flagship (replaces all grok-4-1-* models, retiring May 15 2026)
    GROK_43 = "grok-4.3"
    # Legacy grok-4-1 entries kept for backwards compatibility until retirement
    GROK_41_FAST = "grok-4-1-fast"
    GROK_41_CODE = "grok-4-1-code"
    GROK_41_REASONING = "grok-4-1-reasoning"
    # GCP Vertex AI providers
    CLAUDE_VERTEX = "claude-vertex"
    GEMINI_VERTEX = "gemini-vertex"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_claude_vertex_provider_enum_exists -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llm_integrations.py tests/test_gcp_providers.py
git commit -m "feat: add CLAUDE_VERTEX and GEMINI_VERTEX to LLMProvider enum"
```

---

### Task 3: Implement ClaudeVertexLLM

**Files:**
- Modify: `llm_integrations.py` (add new class after `XaiLLM`, before `LLMManager`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gcp_providers.py`:

```python
def test_claude_vertex_llm_init_uses_project_and_location(monkeypatch):
    """ClaudeVertexLLM initialises ChatAnthropicVertex with project/location from env."""
    monkeypatch.setenv("VERTEX_PROJECT", "my-gcp-project")
    monkeypatch.setenv("VERTEX_LOCATION", "us-east5")

    mock_client_cls = MagicMock()
    mock_client_cls.return_value = MagicMock()

    with patch("llm_integrations.ChatAnthropicVertex", mock_client_cls, create=True):
        from llm_integrations import ClaudeVertexLLM, LLMConfig, LLMProvider
        config = LLMConfig(provider=LLMProvider.CLAUDE_VERTEX, model="claude-sonnet-4-6")
        llm = ClaudeVertexLLM(config)

    mock_client_cls.assert_called_once()
    call_kwargs = mock_client_cls.call_args.kwargs
    assert call_kwargs["project"] == "my-gcp-project"
    assert call_kwargs["location"] == "us-east5"
    assert call_kwargs["model"] == "claude-sonnet-4-6"


def test_claude_vertex_llm_generate_response_returns_content(monkeypatch):
    """ClaudeVertexLLM.generate_response returns (content_str, TokenUsage)."""
    monkeypatch.setenv("VERTEX_PROJECT", "proj")
    monkeypatch.setenv("VERTEX_LOCATION", "us-east5")

    mock_response = MagicMock()
    mock_response.content = "vertex response"
    mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    mock_client = MagicMock()
    mock_client.invoke.return_value = mock_response

    mock_client_cls = MagicMock(return_value=mock_client)

    with patch("llm_integrations.ChatAnthropicVertex", mock_client_cls, create=True):
        from llm_integrations import ClaudeVertexLLM, LLMConfig, LLMProvider
        config = LLMConfig(provider=LLMProvider.CLAUDE_VERTEX, model="claude-sonnet-4-6")
        llm = ClaudeVertexLLM(config)
        content, usage = llm.generate_response("hello")

    assert content == "vertex response"
    assert usage.input_tokens == 10
    assert usage.output_tokens == 5
    assert usage.total_tokens == 15


def test_claude_vertex_llm_is_available_false_when_no_client(monkeypatch):
    """ClaudeVertexLLM.is_available returns False when client is None."""
    monkeypatch.setenv("VERTEX_PROJECT", "proj")
    monkeypatch.setenv("VERTEX_LOCATION", "us-east5")

    mock_client_cls = MagicMock(side_effect=Exception("no vertex"))

    with patch("llm_integrations.ChatAnthropicVertex", mock_client_cls, create=True):
        from llm_integrations import ClaudeVertexLLM, LLMConfig, LLMProvider
        config = LLMConfig(provider=LLMProvider.CLAUDE_VERTEX, model="claude-sonnet-4-6")
        llm = ClaudeVertexLLM(config)

    assert llm.is_available() is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_claude_vertex_llm_init_uses_project_and_location tests/test_gcp_providers.py::test_claude_vertex_llm_generate_response_returns_content tests/test_gcp_providers.py::test_claude_vertex_llm_is_available_false_when_no_client -v
```

Expected: FAIL with `ImportError: cannot import name 'ClaudeVertexLLM'`

- [ ] **Step 3: Add the import guard at the top of llm_integrations.py**

After the existing `try/except` import blocks (around line 120), add:

```python
try:
    from langchain_google_vertexai import ChatVertexAI
    from langchain_google_vertexai.model_garden import ChatAnthropicVertex
except ImportError:
    ChatVertexAI = None
    ChatAnthropicVertex = None
```

- [ ] **Step 4: Implement ClaudeVertexLLM**

Insert the following class in `llm_integrations.py` after the `XaiLLM` class and before the `LLMManager` class:

```python
class ClaudeVertexLLM(LLMInterface):
    """Claude on Google Cloud Vertex AI (no ANTHROPIC_API_KEY required — uses ADC)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if ChatAnthropicVertex is None:
            raise ImportError(
                "ChatAnthropicVertex not available. "
                "Install langchain-google-vertexai: pip install langchain-google-vertexai"
            )

        project = os.getenv("VERTEX_PROJECT", "")
        location = os.getenv("VERTEX_LOCATION", "us-east5")

        try:
            self.client = ChatAnthropicVertex(
                model=config.model,
                project=project,
                location=location,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize ClaudeVertex client: {e}")
            self.client = None

    @retry_with_backoff(max_retries=1, initial_delay=1.0, backoff_factor=2.0)
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response using Claude on Vertex AI."""
        if self.client is None:
            raise RuntimeError("ClaudeVertex client not initialized")

        try:
            from cost_tracker import record_model_usage
            record_model_usage(self.config.model)
        except Exception:
            pass

        base_system = "You are a professional assistant. Respond in a formal, concise, and objective manner without humor or casual language."
        full_system = f"{base_system}\n\n{system_prompt}" if system_prompt else base_system
        professional_prompt = f"Please answer this in a professional tone: {prompt}"

        messages = [
            SystemMessage(content=full_system),
            HumanMessage(content=professional_prompt),
        ]

        try:
            response = self.client.invoke(messages)

            token_usage = TokenUsage()
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_usage.input_tokens = response.usage_metadata.get("input_tokens", 0)
                token_usage.output_tokens = response.usage_metadata.get("output_tokens", 0)
                token_usage.total_tokens = token_usage.input_tokens + token_usage.output_tokens
            elif hasattr(response, "response_metadata") and response.response_metadata:
                usage = response.response_metadata.get("usage", {})
                token_usage.input_tokens = usage.get("input_tokens", 0)
                token_usage.output_tokens = usage.get("output_tokens", 0)
                token_usage.total_tokens = token_usage.input_tokens + token_usage.output_tokens

            return response.content, token_usage

        except Exception as e:
            self.logger.error(f"ClaudeVertex API error: {e}")
            raise

    def medical_analysis(self, medical_input: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Specialized medical analysis using Claude on Vertex AI."""
        system_prompt = (
            "You are a medical reasoning AI that provides systematic analysis "
            "of medical procedures. Focus on evidence-based recommendations."
        )
        prompt = f"Medical Input: {medical_input}\nReasoning Stage: {stage}\n\nProvide analysis with confidence scores."
        response, token_usage = self.generate_response(prompt, system_prompt)
        return {"analysis": response, "confidence": 0.8, "sources_needed": [], "token_usage": token_usage}

    def is_available(self) -> bool:
        """Check if ClaudeVertex client is initialized."""
        return self.client is not None
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_claude_vertex_llm_init_uses_project_and_location tests/test_gcp_providers.py::test_claude_vertex_llm_generate_response_returns_content tests/test_gcp_providers.py::test_claude_vertex_llm_is_available_false_when_no_client -v
```

Expected: all three PASS

- [ ] **Step 6: Commit**

```bash
git add llm_integrations.py tests/test_gcp_providers.py
git commit -m "feat: add ClaudeVertexLLM provider for Claude-on-Vertex-AI"
```

---

### Task 4: Implement GeminiVertexLLM

**Files:**
- Modify: `llm_integrations.py` (add new class after `ClaudeVertexLLM`, before `LLMManager`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gcp_providers.py`:

```python
def test_gemini_vertex_llm_init_uses_project_and_location(monkeypatch):
    """GeminiVertexLLM initialises ChatVertexAI with project/location from env."""
    monkeypatch.setenv("VERTEX_PROJECT", "my-gcp-project")
    monkeypatch.setenv("VERTEX_LOCATION", "us-central1")

    mock_client_cls = MagicMock()
    mock_client_cls.return_value = MagicMock()

    with patch("llm_integrations.ChatVertexAI", mock_client_cls):
        from llm_integrations import GeminiVertexLLM, LLMConfig, LLMProvider
        config = LLMConfig(provider=LLMProvider.GEMINI_VERTEX, model="gemini-1.5-pro")
        llm = GeminiVertexLLM(config)

    mock_client_cls.assert_called_once()
    call_kwargs = mock_client_cls.call_args.kwargs
    assert call_kwargs["project"] == "my-gcp-project"
    assert call_kwargs["location"] == "us-central1"
    assert call_kwargs["model"] == "gemini-1.5-pro"


def test_gemini_vertex_llm_generate_response_returns_content(monkeypatch):
    """GeminiVertexLLM.generate_response returns (content_str, TokenUsage)."""
    monkeypatch.setenv("VERTEX_PROJECT", "proj")
    monkeypatch.setenv("VERTEX_LOCATION", "us-central1")

    mock_response = MagicMock()
    mock_response.content = "gemini response"
    mock_response.usage_metadata = {"prompt_token_count": 8, "candidates_token_count": 4}

    mock_client = MagicMock()
    mock_client.invoke.return_value = mock_response

    mock_client_cls = MagicMock(return_value=mock_client)

    with patch("llm_integrations.ChatVertexAI", mock_client_cls):
        from llm_integrations import GeminiVertexLLM, LLMConfig, LLMProvider
        config = LLMConfig(provider=LLMProvider.GEMINI_VERTEX, model="gemini-1.5-pro")
        llm = GeminiVertexLLM(config)
        content, usage = llm.generate_response("hello")

    assert content == "gemini response"
    assert usage.input_tokens == 8
    assert usage.output_tokens == 4
    assert usage.total_tokens == 12


def test_gemini_vertex_llm_is_available_false_when_no_client(monkeypatch):
    """GeminiVertexLLM.is_available returns False when client failed to init."""
    monkeypatch.setenv("VERTEX_PROJECT", "proj")
    monkeypatch.setenv("VERTEX_LOCATION", "us-central1")

    mock_client_cls = MagicMock(side_effect=Exception("no vertex"))

    with patch("llm_integrations.ChatVertexAI", mock_client_cls):
        from llm_integrations import GeminiVertexLLM, LLMConfig, LLMProvider
        config = LLMConfig(provider=LLMProvider.GEMINI_VERTEX, model="gemini-1.5-pro")
        llm = GeminiVertexLLM(config)

    assert llm.is_available() is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_gemini_vertex_llm_init_uses_project_and_location tests/test_gcp_providers.py::test_gemini_vertex_llm_generate_response_returns_content tests/test_gcp_providers.py::test_gemini_vertex_llm_is_available_false_when_no_client -v
```

Expected: FAIL with `ImportError: cannot import name 'GeminiVertexLLM'`

- [ ] **Step 3: Implement GeminiVertexLLM**

Insert the following class in `llm_integrations.py` after `ClaudeVertexLLM` and before `LLMManager`:

```python
class GeminiVertexLLM(LLMInterface):
    """Gemini on Google Cloud Vertex AI (uses ADC / service account)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        if ChatVertexAI is None:
            raise ImportError(
                "ChatVertexAI not available. "
                "Install langchain-google-vertexai: pip install langchain-google-vertexai"
            )

        project = os.getenv("VERTEX_PROJECT", "")
        location = os.getenv("VERTEX_LOCATION", "us-central1")

        try:
            self.client = ChatVertexAI(
                model=config.model,
                project=project,
                location=location,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            )
        except Exception as e:
            self.logger.warning(f"Failed to initialize GeminiVertex client: {e}")
            self.client = None

    @retry_with_backoff(max_retries=1, initial_delay=1.0, backoff_factor=2.0)
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, TokenUsage]:
        """Generate response using Gemini on Vertex AI."""
        if self.client is None:
            raise RuntimeError("GeminiVertex client not initialized")

        try:
            from cost_tracker import record_model_usage
            record_model_usage(self.config.model)
        except Exception:
            pass

        base_system = "You are a professional assistant. Respond in a formal, concise, and objective manner without humor or casual language."
        full_system = f"{base_system}\n\n{system_prompt}" if system_prompt else base_system
        professional_prompt = f"Please answer this in a professional tone: {prompt}"

        messages = [
            SystemMessage(content=full_system),
            HumanMessage(content=professional_prompt),
        ]

        try:
            response = self.client.invoke(messages)

            token_usage = TokenUsage()
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                # Gemini Vertex reports prompt_token_count / candidates_token_count
                if hasattr(meta, "prompt_token_count"):
                    token_usage.input_tokens = meta.prompt_token_count
                    token_usage.output_tokens = meta.candidates_token_count
                elif isinstance(meta, dict):
                    token_usage.input_tokens = meta.get("prompt_token_count", 0)
                    token_usage.output_tokens = meta.get("candidates_token_count", 0)
                token_usage.total_tokens = token_usage.input_tokens + token_usage.output_tokens

            return response.content, token_usage

        except Exception as e:
            self.logger.error(f"GeminiVertex API error: {e}")
            raise

    def medical_analysis(self, medical_input: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Specialized medical analysis using Gemini on Vertex AI."""
        system_prompt = (
            "You are a medical reasoning AI that provides systematic analysis "
            "of medical procedures. Focus on evidence-based recommendations."
        )
        prompt = f"Medical Input: {medical_input}\nReasoning Stage: {stage}\n\nProvide analysis with confidence scores."
        response, token_usage = self.generate_response(prompt, system_prompt)
        return {"analysis": response, "confidence": 0.8, "sources_needed": [], "token_usage": token_usage}

    def is_available(self) -> bool:
        """Check if GeminiVertex client is initialized."""
        return self.client is not None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_gemini_vertex_llm_init_uses_project_and_location tests/test_gcp_providers.py::test_gemini_vertex_llm_generate_response_returns_content tests/test_gcp_providers.py::test_gemini_vertex_llm_is_available_false_when_no_client -v
```

Expected: all three PASS

- [ ] **Step 5: Commit**

```bash
git add llm_integrations.py tests/test_gcp_providers.py
git commit -m "feat: add GeminiVertexLLM provider for Gemini-on-Vertex-AI"
```

---

### Task 5: Wire new providers into LLMManager and create_llm_manager()

**Files:**
- Modify: `llm_integrations.py` — `_initialize_providers()` and `create_llm_manager()`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gcp_providers.py`:

```python
def test_create_llm_manager_returns_claude_vertex_when_is_gcp_true(monkeypatch):
    """create_llm_manager('claude-vertex') returns a manager with ClaudeVertexLLM."""
    monkeypatch.setenv("IS_GCP", "true")
    monkeypatch.setenv("VERTEX_PROJECT", "proj")
    monkeypatch.setenv("VERTEX_LOCATION", "us-east5")

    mock_client_cls = MagicMock(return_value=MagicMock())

    with patch("llm_integrations.ChatAnthropicVertex", mock_client_cls, create=True):
        from llm_integrations import create_llm_manager, LLMProvider, ClaudeVertexLLM
        manager = create_llm_manager(primary_provider="claude-vertex")

    assert LLMProvider.CLAUDE_VERTEX in manager.providers
    assert isinstance(manager.providers[LLMProvider.CLAUDE_VERTEX], ClaudeVertexLLM)


def test_create_llm_manager_returns_gemini_vertex_when_is_gcp_true(monkeypatch):
    """create_llm_manager('gemini-vertex') returns a manager with GeminiVertexLLM."""
    monkeypatch.setenv("IS_GCP", "true")
    monkeypatch.setenv("VERTEX_PROJECT", "proj")
    monkeypatch.setenv("VERTEX_LOCATION", "us-central1")

    mock_client_cls = MagicMock(return_value=MagicMock())

    with patch("llm_integrations.ChatVertexAI", mock_client_cls):
        from llm_integrations import create_llm_manager, LLMProvider, GeminiVertexLLM
        manager = create_llm_manager(primary_provider="gemini-vertex")

    assert LLMProvider.GEMINI_VERTEX in manager.providers
    assert isinstance(manager.providers[LLMProvider.GEMINI_VERTEX], GeminiVertexLLM)


def test_create_llm_manager_api_key_path_unchanged(monkeypatch):
    """create_llm_manager('claude-sonnet') still uses ClaudeLLM when IS_GCP is not set."""
    monkeypatch.delenv("IS_GCP", raising=False)

    mock_claude_cls = MagicMock(return_value=MagicMock())

    with patch("llm_integrations.ChatAnthropic", mock_claude_cls):
        from llm_integrations import create_llm_manager, LLMProvider, ClaudeLLM
        manager = create_llm_manager(primary_provider="claude-sonnet", fallback_providers=[])

    assert LLMProvider.CLAUDE_SONNET in manager.providers
    assert isinstance(manager.providers[LLMProvider.CLAUDE_SONNET], ClaudeLLM)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_create_llm_manager_returns_claude_vertex_when_is_gcp_true tests/test_gcp_providers.py::test_create_llm_manager_returns_gemini_vertex_when_is_gcp_true tests/test_gcp_providers.py::test_create_llm_manager_api_key_path_unchanged -v
```

Expected: FAIL (provider keys not recognised yet)

- [ ] **Step 3: Extend _initialize_providers() in LLMManager**

In `llm_integrations.py`, find the `_initialize_providers` method inside `LLMManager` (around line 619). Add two new branches inside the `for config in self.configs:` loop, after the existing `elif config.provider in [LLMProvider.GROK_43, ...]` block:

```python
                elif config.provider == LLMProvider.CLAUDE_VERTEX:
                    self.providers[config.provider] = ClaudeVertexLLM(config)
                elif config.provider == LLMProvider.GEMINI_VERTEX:
                    self.providers[config.provider] = GeminiVertexLLM(config)
```

- [ ] **Step 4: Extend create_llm_manager() to recognise the new provider keys**

In `create_llm_manager()` (around line 734), add two new `elif` branches in the primary provider section, after the existing legacy grok entries:

```python
    elif primary_provider == "claude-vertex":
        configs.append(LLMConfig(
            provider=LLMProvider.CLAUDE_VERTEX,
            model="claude-sonnet-4-6",
            temperature=0.1
        ))
    elif primary_provider == "gemini-vertex":
        configs.append(LLMConfig(
            provider=LLMProvider.GEMINI_VERTEX,
            model="gemini-1.5-pro",
            temperature=0.1
        ))
```

Also add matching branches inside the `for provider in fallback_providers:` loop:

```python
        elif provider == "claude-vertex":
            configs.append(LLMConfig(
                provider=LLMProvider.CLAUDE_VERTEX,
                model="claude-sonnet-4-6",
                temperature=0.1
            ))
        elif provider == "gemini-vertex":
            configs.append(LLMConfig(
                provider=LLMProvider.GEMINI_VERTEX,
                model="gemini-1.5-pro",
                temperature=0.1
            ))
```

- [ ] **Step 5: Add entries to get_available_models()**

In `get_available_models()` (around line 838), add two entries to the returned dict:

```python
        # GCP Vertex AI models
        "claude-sonnet-4-6-vertex": "claude-vertex",
        "gemini-1.5-pro": "gemini-vertex",
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_create_llm_manager_returns_claude_vertex_when_is_gcp_true tests/test_gcp_providers.py::test_create_llm_manager_returns_gemini_vertex_when_is_gcp_true tests/test_gcp_providers.py::test_create_llm_manager_api_key_path_unchanged -v
```

Expected: all three PASS

- [ ] **Step 7: Commit**

```bash
git add llm_integrations.py tests/test_gcp_providers.py
git commit -m "feat: wire ClaudeVertexLLM and GeminiVertexLLM into LLMManager and create_llm_manager"
```

---

### Task 6: Load .env.dev in router.py

**Files:**
- Modify: `router.py:1-18`

- [ ] **Step 1: Write the failing test**

Create a new test in `tests/test_gcp_providers.py`:

```python
def test_router_loads_env_dev_before_env(tmp_path, monkeypatch):
    """router startup loads .env.dev (override=False) before the base .env."""
    import importlib
    # Simulate the dotenv load sequence: .env.dev sets IS_GCP, .env does not
    load_calls = []

    def fake_load_dotenv(path=None, override=False, **kwargs):
        load_calls.append((str(path) if path else None, override))

    monkeypatch.setattr("dotenv.load_dotenv", fake_load_dotenv, raising=False)

    # We test the load sequence directly — import the module's _load_env helper
    import router  # noqa: F401  (import to exercise module-level code)

    # .env.dev must be loaded (path ends with .env.dev)
    dev_calls = [c for c in load_calls if c[0] and c[0].endswith(".env.dev")]
    assert len(dev_calls) >= 1, f"Expected .env.dev load, got calls: {load_calls}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_router_loads_env_dev_before_env -v
```

Expected: FAIL (no `.env.dev` load happens)

- [ ] **Step 3: Add env loading to router.py**

At the top of `router.py`, after the existing imports block (after `from observability import setup_phoenix, get_tracer`), add:

```python
# Load environment variables: .env.dev first (dev overrides), then .env (base)
try:
    from dotenv import load_dotenv as _load_dotenv
    import pathlib as _pathlib

    _repo_root = _pathlib.Path(__file__).parent
    _load_dotenv(_repo_root / ".env.dev", override=False)  # dev-specific vars
    _load_dotenv(_repo_root / ".env", override=False)       # base vars (don't overwrite)
except ImportError:
    pass  # python-dotenv not installed, rely on shell environment
```

Note: `override=False` means the first file loaded wins — if `.env.dev` sets `IS_GCP=true`, the base `.env` will not overwrite it. If `.env.dev` doesn't set a var, `.env` supplies it.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_router_loads_env_dev_before_env -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add router.py tests/test_gcp_providers.py
git commit -m "feat: load .env.dev at router startup for GCP dev auth"
```

---

### Task 7: Auto-select Vertex providers when IS_GCP=true

**Files:**
- Modify: `llm_integrations.py` — `create_llm_manager()`

The goal: when `IS_GCP=true` and the caller requests `claude-sonnet` or `claude-opus`, transparently redirect to `claude-vertex`. Same for any Claude/OpenAI model → a Vertex equivalent. Grok is not available on Vertex, so it keeps its existing path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gcp_providers.py`:

```python
def test_is_gcp_redirects_claude_sonnet_to_claude_vertex(monkeypatch):
    """When IS_GCP=true, create_llm_manager('claude-sonnet') uses CLAUDE_VERTEX."""
    monkeypatch.setenv("IS_GCP", "true")
    monkeypatch.setenv("VERTEX_PROJECT", "proj")
    monkeypatch.setenv("VERTEX_LOCATION", "us-east5")

    mock_client_cls = MagicMock(return_value=MagicMock())

    with patch("llm_integrations.ChatAnthropicVertex", mock_client_cls, create=True):
        from llm_integrations import create_llm_manager, LLMProvider, ClaudeVertexLLM
        manager = create_llm_manager(primary_provider="claude-sonnet", fallback_providers=[])

    assert LLMProvider.CLAUDE_VERTEX in manager.providers
    assert isinstance(manager.providers[LLMProvider.CLAUDE_VERTEX], ClaudeVertexLLM)
    # Direct Claude API must NOT be instantiated
    assert LLMProvider.CLAUDE_SONNET not in manager.providers


def test_is_gcp_false_keeps_api_key_path(monkeypatch):
    """When IS_GCP=false, create_llm_manager('claude-sonnet') uses ClaudeLLM."""
    monkeypatch.setenv("IS_GCP", "false")

    mock_claude_cls = MagicMock(return_value=MagicMock())

    with patch("llm_integrations.ChatAnthropic", mock_claude_cls):
        from llm_integrations import create_llm_manager, LLMProvider, ClaudeLLM
        manager = create_llm_manager(primary_provider="claude-sonnet", fallback_providers=[])

    assert LLMProvider.CLAUDE_SONNET in manager.providers
    assert isinstance(manager.providers[LLMProvider.CLAUDE_SONNET], ClaudeLLM)
    assert LLMProvider.CLAUDE_VERTEX not in manager.providers
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_is_gcp_redirects_claude_sonnet_to_claude_vertex tests/test_gcp_providers.py::test_is_gcp_false_keeps_api_key_path -v
```

Expected: FAIL (no IS_GCP redirect logic yet)

- [ ] **Step 3: Add IS_GCP redirect logic to create_llm_manager()**

At the top of `create_llm_manager()` (before the `configs = []` line), add:

```python
    # When running on GCP, redirect Claude providers to Vertex AI equivalents
    _is_gcp = os.getenv("IS_GCP", "").lower() in ("true", "1", "yes")

    if _is_gcp:
        if primary_provider in ("claude-sonnet", "claude-opus"):
            primary_provider = "claude-vertex"
        # Gemini is GCP-native; openai/grok keep their own paths
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run python -m pytest tests/test_gcp_providers.py::test_is_gcp_redirects_claude_sonnet_to_claude_vertex tests/test_gcp_providers.py::test_is_gcp_false_keeps_api_key_path -v
```

Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add llm_integrations.py tests/test_gcp_providers.py
git commit -m "feat: auto-redirect claude-sonnet/opus to claude-vertex when IS_GCP=true"
```

---

### Task 8: Full test suite + smoke check

**Files:**
- Read: `tests/test_gcp_providers.py` (verify all tests collected)

- [ ] **Step 1: Run all existing tests to confirm nothing is broken**

```bash
uv run python -m pytest tests/ -q --ignore=tests/test_langchain_agents.py --ignore=tests/test_langchain_integration.py --ignore=tests/test_orchestrator_integration.py --ignore=tests/test_router_langchain_integration.py
```

(The ignored files require live API keys — skip in unit test runs.)

Expected: all tests PASS, 0 errors.

- [ ] **Step 2: Run just the new GCP test file with verbose output**

```bash
uv run python -m pytest tests/test_gcp_providers.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Smoke-check that the CLI still starts without errors**

```bash
timeout 5 uv run python router.py --models 2>&1 || true
```

Expected: prints model list including `claude-sonnet-4-6-vertex` and `gemini-1.5-pro`, then exits. No tracebacks.

- [ ] **Step 4: Commit final state**

```bash
git add .
git commit -m "feat: GCP Vertex AI auth complete — ClaudeVertex + Gemini + IS_GCP auto-routing"
```

---

## Self-Review

**Spec coverage check:**
- [x] Claude via Vertex AI — `ClaudeVertexLLM` in Task 3
- [x] Gemini via Vertex AI — `GeminiVertexLLM` in Task 4
- [x] Backwards compatible — Task 5 and 7 tests explicitly verify API-key path unchanged
- [x] IS_GCP=true trigger — Task 7
- [x] .env.dev loaded at startup — Task 6
- [x] New deps declared — Task 1

**Placeholder scan:** No TBD/TODO/etc found. All code blocks are complete.

**Type consistency:**
- `ClaudeVertexLLM` / `GeminiVertexLLM` referenced consistently across Tasks 3–7
- `LLMProvider.CLAUDE_VERTEX` / `LLMProvider.GEMINI_VERTEX` defined in Task 2, used from Task 3 onward
- `ChatAnthropicVertex` / `ChatVertexAI` imported at module level in Task 3, patched consistently in all tests
