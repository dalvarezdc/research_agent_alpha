"""Tests for llm_integrations module."""
from unittest.mock import MagicMock, patch

from llm_integrations import TokenUsage


def test_call_model_does_not_invoke_is_available(monkeypatch):
    """call_model must not call is_available() — that makes an extra API ping."""
    from llm_integrations import call_model

    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = (
        "test response",
        TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    mock_llm.is_available = MagicMock(return_value=True)

    mock_manager = MagicMock()
    mock_manager.get_provider_direct.return_value = mock_llm
    mock_manager.get_available_provider.return_value = mock_llm

    with patch("llm_integrations.create_llm_manager", return_value=mock_manager):
        result = call_model(
            "gpt-4-turbo-preview",
            [{"role": "user", "content": "hello"}],
        )

    # is_available should NOT be called during call_model
    mock_llm.is_available.assert_not_called()
    assert result == "test response"


def test_get_provider_direct_returns_first_provider():
    """get_provider_direct returns the first initialized provider without health check."""
    from llm_integrations import LLMManager

    manager = LLMManager.__new__(LLMManager)
    mock_provider = MagicMock()
    # Simulate initialized providers dict
    from llm_integrations import LLMProvider
    manager.providers = {LLMProvider.OPENAI: mock_provider}
    manager.current_provider = None

    result = manager.get_provider_direct()

    assert result is mock_provider
    mock_provider.is_available.assert_not_called()


def test_get_provider_direct_returns_none_when_empty():
    """get_provider_direct returns None if no providers are initialized."""
    from llm_integrations import LLMManager

    manager = LLMManager.__new__(LLMManager)
    manager.providers = {}

    result = manager.get_provider_direct()

    assert result is None
