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


def test_call_model_creates_otel_span():
    """call_model() creates an OTEL span with model name and response attributes."""
    from llm_integrations import call_model, TokenUsage

    mock_llm = MagicMock()
    mock_llm.generate_response.return_value = (
        "routing: medication_agent",
        TokenUsage(input_tokens=50, output_tokens=5, total_tokens=55),
    )
    mock_manager = MagicMock()
    mock_manager.get_provider_direct.return_value = mock_llm

    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    with (
        patch("llm_integrations.create_llm_manager", return_value=mock_manager),
        patch("llm_integrations._get_tracer", return_value=mock_tracer),
    ):
        result = call_model(
            "gpt-4o",
            [
                {"role": "system", "content": "route"},
                {"role": "user", "content": "drug query"},
            ],
        )

    mock_tracer.start_as_current_span.assert_called_once_with("llm.call")
    mock_span.set_attribute.assert_any_call("llm.model_name", "gpt-4o")
    assert result == "routing: medication_agent"


def test_call_model_span_uses_openinference_keys():
    """call_model() must emit input.value, output.value, openinference.span.kind=LLM."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from unittest.mock import patch, MagicMock
    import llm_integrations as li

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    mock_provider = MagicMock()
    mock_provider.generate_response.return_value = (
        "the response",
        MagicMock(input_tokens=10, output_tokens=20, total_tokens=30),
    )

    with (
        patch.object(li, "_get_tracer", return_value=tracer),
        patch.object(
            li,
            "create_llm_manager",
            return_value=MagicMock(
                get_provider_direct=MagicMock(return_value=mock_provider)
            ),
        ),
    ):
        result = li.call_model(
            messages=[{"role": "user", "content": "hello"}],
            model_name="grok-4.3",
        )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    attrs = dict(spans[0].attributes)
    assert attrs.get("openinference.span.kind") == "LLM", (
        f"Got: {attrs.get('openinference.span.kind')}"
    )
    assert "input.value" in attrs, (
        f"input.value missing from attrs: {list(attrs.keys())}"
    )
    assert "output.value" in attrs, (
        f"output.value missing from attrs: {list(attrs.keys())}"
    )
    assert "llm.input_messages" not in attrs, (
        "old key llm.input_messages must be removed"
    )
    assert "llm.output.value" not in attrs, "old key llm.output.value must be removed"
    assert result == "the response"
