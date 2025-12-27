"""
Shared utilities for LangChain-based agents.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
import os
from typing import Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from llm_integrations import TokenUsage, create_llm_manager

try:
    from langsmith import Client as LangSmithClient
except ImportError:  # pragma: no cover
    LangSmithClient = None

@dataclass
class LangChainAgentConfig:
    """Configuration shared across LangChain agents."""

    primary_llm_provider: str = "claude-sonnet"
    fallback_providers: list[str] = field(default_factory=lambda: ["openai", "ollama"])
    enable_logging: bool = True
    enable_reference_validation: bool = False
    enable_audit: bool = True


class LangChainAgentBase:
    """Base class for LangChain-based agents using shared LLM manager."""

    def __init__(self, config: LangChainAgentConfig):
        self.config = config
        self.enable_reference_validation = config.enable_reference_validation
        self.enable_audit = config.enable_audit
        self.reference_validator = None
        self.audit_events: list[dict[str, Any]] = []
        self.langsmith_client = None
        self.langsmith_project = os.getenv("LANGCHAIN_PROJECT") or "research-agent-alpha"

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
        self._initialize_langsmith()

    def _initialize_langsmith(self) -> None:
        if not self.enable_audit or LangSmithClient is None:
            return
        tracing_flag = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if tracing_flag not in ("1", "true", "yes") or not api_key:
            return
        try:
            self.langsmith_client = LangSmithClient()
        except Exception:
            self.langsmith_client = None

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

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        audit_step: str | None = None,
        **kwargs: Any,
    ) -> str:
        system_text, user_text = self._render_prompt(
            system_prompt, user_prompt, **kwargs
        )
        response, token_usage = self.llm_provider.generate_response(
            prompt=user_text, system_prompt=system_text
        )
        if token_usage:
            self.total_token_usage.add(token_usage)
        if self.enable_audit:
            self.audit_events.append(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "step": audit_step,
                    "system_prompt": system_text,
                    "user_prompt": user_text,
                    "response": response,
                    "token_usage": (
                        {
                            "input_tokens": token_usage.input_tokens,
                            "output_tokens": token_usage.output_tokens,
                            "total_tokens": token_usage.total_tokens,
                        }
                        if token_usage
                        else None
                    ),
                }
            )
            self._record_langsmith(audit_step, system_text, user_text, response)
        return response

    def _record_langsmith(
        self, audit_step: str | None, system_text: str, user_text: str, response: str
    ) -> None:
        if not self.langsmith_client:
            return
        try:
            self.langsmith_client.create_run(
                name=audit_step or "llm_call",
                run_type="llm",
                inputs={"system": system_text, "user": user_text},
                outputs={"response": response},
                project_name=self.langsmith_project,
            )
        except Exception:
            pass

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
