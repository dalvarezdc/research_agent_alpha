"""
Shared utilities for LangChain-based agents.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from llm_integrations import TokenUsage, create_llm_manager


@dataclass
class LangChainAgentConfig:
    """Configuration shared across LangChain agents."""

    primary_llm_provider: str = "claude-sonnet"
    fallback_providers: list[str] = field(default_factory=lambda: ["openai", "ollama"])
    enable_logging: bool = True
    enable_reference_validation: bool = False


class LangChainAgentBase:
    """Base class for LangChain-based agents using shared LLM manager."""

    def __init__(self, config: LangChainAgentConfig):
        self.config = config
        self.enable_reference_validation = config.enable_reference_validation
        self.reference_validator = None

        if self.enable_reference_validation:
            try:
                from reference_validation import ReferenceValidator, ValidationConfig

                self.reference_validator = ReferenceValidator(
                    ValidationConfig(cache_backend="sqlite", min_credibility_score=70)
                )
            except ImportError:
                self.reference_validator = None

        try:
            self.llm_manager = create_llm_manager(
                primary_provider=config.primary_llm_provider,
                fallback_providers=config.fallback_providers,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize LLM manager: {exc}") from exc

        self.llm_provider = self.llm_manager.get_available_provider()
        if not self.llm_provider:
            raise RuntimeError("No LLM provider available for LangChain agents")

        self.total_token_usage = TokenUsage()

    def _render_prompt(
        self, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> Tuple[str, str]:
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt)]
        )
        messages = prompt.format_messages(**kwargs)

        system_text = ""
        user_texts: list[str] = []
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            if msg_type == "system":
                system_text = msg.content
            else:
                user_texts.append(msg.content)

        return system_text, "\n".join(user_texts)

    def _call_llm(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        system_text, user_text = self._render_prompt(
            system_prompt, user_prompt, **kwargs
        )
        response, token_usage = self.llm_provider.generate_response(
            prompt=user_text, system_prompt=system_text
        )
        if token_usage:
            self.total_token_usage.add(token_usage)
        return response

    def _parse_json(self, text: str) -> Optional[Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
