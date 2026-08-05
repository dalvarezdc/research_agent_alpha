"""
Tests for DeepSeek LLM provider integration, model management, and cost tracking.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from cost_tracker import PRICING, calculate_cost
from llm_integrations import (
    LLMConfig,
    LLMProvider,
    DeepSeekLLM,
    create_llm_manager,
    get_available_models,
    call_model,
)


def test_deepseek_provider_enum():
    """Verify DEEPSEEK provider enum entries exist and have expected string values."""
    assert hasattr(LLMProvider, "DEEPSEEK")
    assert hasattr(LLMProvider, "DEEPSEEK_V4_FLASH")
    assert hasattr(LLMProvider, "DEEPSEEK_V4_PRO")
    assert hasattr(LLMProvider, "DEEPSEEK_CHAT")
    assert hasattr(LLMProvider, "DEEPSEEK_REASONER")

    assert LLMProvider.DEEPSEEK.value == "deepseek"
    assert LLMProvider.DEEPSEEK_V4_FLASH.value == "deepseek-v4-flash"
    assert LLMProvider.DEEPSEEK_V4_PRO.value == "deepseek-v4-pro"
    assert LLMProvider.DEEPSEEK_CHAT.value == "deepseek-chat"
    assert LLMProvider.DEEPSEEK_REASONER.value == "deepseek-reasoner"


def test_deepseek_pricing():
    """Verify DeepSeek pricing entries are present in PRICING dictionary."""
    assert "deepseek-v4-flash" in PRICING
    assert "deepseek-v4-pro" in PRICING
    assert "deepseek-chat" in PRICING
    assert "deepseek-reasoner" in PRICING
    assert "deepseek" in PRICING

    # Test cost calculations
    flash_cost = calculate_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        model="deepseek-v4-flash",
        cache_read=1_000_000,
    )
    # $0.14 + $0.28 + $0.0028 = 0.4228
    assert pytest.approx(flash_cost, 0.0001) == 0.4228

    pro_cost = calculate_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        model="deepseek-v4-pro",
        cache_read=1_000_000,
    )
    # $0.435 + $0.87 + $0.003625 = 1.308625
    assert pytest.approx(pro_cost, 0.0001) == 1.308625


def test_deepseek_available_models():
    """Verify DeepSeek model mappings in get_available_models()."""
    models = get_available_models()
    assert "deepseek-v4-flash" in models
    assert "deepseek-v4-pro" in models
    assert "deepseek-chat" in models
    assert "deepseek-reasoner" in models

    assert models["deepseek-v4-flash"] == "deepseek-v4-flash"
    assert models["deepseek-v4-pro"] == "deepseek-v4-pro"
    assert models["deepseek-chat"] == "deepseek-v4-flash"
    assert models["deepseek-reasoner"] == "deepseek-v4-pro"


def test_deepseek_llm_init():
    """Test DeepSeekLLM initialization with custom config and environment variable."""
    config = LLMConfig(
        provider=LLMProvider.DEEPSEEK_V4_FLASH,
        model="deepseek-v4-flash",
        api_key="test-deepseek-key",
        base_url="https://api.deepseek.com",
    )
    llm = DeepSeekLLM(config)
    assert llm.config.model == "deepseek-v4-flash"
    assert llm.client is not None


def test_deepseek_generate_response_mock():
    """Test DeepSeekLLM response generation with a mocked ChatOpenAI client."""
    config = LLMConfig(
        provider=LLMProvider.DEEPSEEK_V4_FLASH,
        model="deepseek-v4-flash",
        api_key="mock-key",
    )
    llm = DeepSeekLLM(config)

    mock_response = MagicMock()
    mock_response.content = "DeepSeek test response"
    mock_response.usage_metadata = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }

    mock_client = MagicMock()
    mock_client.invoke.return_value = mock_response
    llm.client = mock_client

    content, usage = llm.generate_response("Hello", system_prompt="Be concise")
    assert content == "DeepSeek test response"
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150


def test_create_llm_manager_deepseek():
    """Test create_llm_manager with deepseek primary providers."""
    manager_flash = create_llm_manager(
        primary_provider="deepseek-v4-flash", fallback_providers=[]
    )
    assert LLMProvider.DEEPSEEK_V4_FLASH in manager_flash.providers
    assert isinstance(manager_flash.providers[LLMProvider.DEEPSEEK_V4_FLASH], DeepSeekLLM)

    manager_pro = create_llm_manager(
        primary_provider="deepseek-v4-pro", fallback_providers=[]
    )
    assert LLMProvider.DEEPSEEK_V4_PRO in manager_pro.providers
    assert isinstance(manager_pro.providers[LLMProvider.DEEPSEEK_V4_PRO], DeepSeekLLM)


def test_call_model_deepseek():
    """Test call_model function with deepseek-v4-flash using mocked response."""
    mock_response = MagicMock()
    mock_response.content = "Call model DeepSeek output"
    mock_response.usage_metadata = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    with patch("llm_integrations.ChatOpenAI") as mock_chat:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = mock_response
        mock_chat.return_value = mock_instance

        res = call_model(
            "deepseek-v4-flash",
            [{"role": "system", "content": "System"}, {"role": "user", "content": "User"}],
        )
        assert res == "Call model DeepSeek output"
