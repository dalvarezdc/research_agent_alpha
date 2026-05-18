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


def test_router_loads_env_dev_before_env(monkeypatch):
    """router.py must call load_dotenv with .env.dev path at module level."""
    import importlib
    import sys

    load_calls = []

    def fake_load_dotenv(path=None, override=False, **kwargs):
        load_calls.append((str(path) if path else None, override))

    # Patch dotenv before importing router
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", fake_load_dotenv)

    # Remove cached router module to force re-import
    if "router" in sys.modules:
        del sys.modules["router"]

    import router  # noqa: F401

    dev_calls = [c for c in load_calls if c[0] and c[0].endswith(".env.dev")]
    assert len(dev_calls) >= 1, f"Expected .env.dev load call, got: {load_calls}"


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
